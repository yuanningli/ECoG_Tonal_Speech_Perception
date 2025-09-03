import numpy as np
import os
import scipy.io as scio
from pynwb import NWBFile, NWBHDF5IO
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_to_save_Data = os.getcwd().rstrip('/code')
sub = 'sub-01'

if not os.path.exists(os.path.join(os.getcwd(), 'trained_model')):
    os.makedirs(os.path.join(os.getcwd(), 'trained_model'))

save_path = os.path.join(os.getcwd(), 'trained_model/')
fullname = f'speech_{sub}'

speech_resp_dict = {
    'sub-01': np.array([54, 84, 53, 85, 99, 66, 83, 88, 98, 67, 68, 101, 82, 69, 70, 100, 86, 72, 73, 39, 97, 81, 38, 52, 87, 102, 71, 55, 89, 90, 56, 80, 104, 103, 93, 76, 118, 105, 94, 74, 120, 51, 109, 59, 114, 78, 125, 64]),
    'sub-02': np.array([19, 38, 20, 37, 34, 54, 23, 21, 56, 15, 16, 17, 57, 35, 9, 40, 18, 42, 13, 2, 62, 12, 36, 29, 30, 1, 39, 63, 44, 58, 11, 14, 33, 10, 55, 32, 28, 53]),
    'sub-03': np.array([40, 22, 43, 5, 60, 182, 59, 9, 24, 52, 21, 8, 27, 20, 28, 45, 181, 44, 15, 66, 46, 12, 18, 37, 30, 78, 23, 29, 63, 25, 13, 11, 42, 36, 61, 39, 77, 7, 62, 10, 53, 31, 41, 65, 34, 96, 165, 19, 4, 47, 38, 56, 26, 80, 35, 64, 17, 196, 112, 195, 32, 14, 180, 33, 79, 179, 6, 55, 223]),
    'sub-04': np.array([55, 59, 60, 89, 88, 200, 26, 73, 42, 75, 72, 43, 84, 91, 184, 56, 58, 183, 199, 90, 83, 87, 71, 63, 39, 31, 67, 74, 146, 168, 132, 40, 30, 46, 29, 45, 41, 69, 57, 70, 68, 38, 15, 139, 44, 28, 54, 131, 53, 27, 14, 47, 4, 92, 93, 3, 64, 77, 167, 13, 50, 65, 62, 49, 66, 52, 94, 95, 185, 82, 103, 201, 133, 19, 78, 11, 23, 79, 86])
    # ELectrodes that respond to speech
}

def get_ECoG_time(sub, session, sr=400):
    '''
    get_ECoG_time('sub-01', '01', ECoG_path) = 278(s)
    '''

    with NWBHDF5IO(path_to_save_Data+f'/{sub}/ieeg/{sub}_task-audio_run-{session}_ieeg.nwb', 'r') as io:
        nwbfile = io.read()
        iEEG_ts = nwbfile.acquisition['iEEG']
        iEEG_data = iEEG_ts.data[:]
        ECoG = iEEG_data.T

    ECoG_length = ECoG.shape[1]
    ECoG_time = int(ECoG_length / sr)  # high-gamma ECoG sampling rate is 400Hz
    return ECoG_time

def get_textgrid_interval_asccd(sub, session, tg_file_path=os.path.join(path_to_save_Data, 'Annotations/')):
    '''
    get a list contains several intervals from annotation file: PAxBxpinyin.npy
    get_textgrid_interval('PA1', 'B1')[0][0]      0.0
    get_textgrid_interval('PA1', 'B1')[0][1]      49.0334
    get_textgrid_interval('PA1', 'B1')[0][2]      'silence' or 'speech'
    '''
    tg_file_path = os.path.join(tg_file_path, sub, f'{sub}_task-audio_run-{session}_pinyin.npy')
    tg_file = np.load(tg_file_path, allow_pickle=True).item()
    point_times = []
    point_times.append([0, tg_file['onset_list'][0], 'silence'])
    for i in range(len(tg_file['onset_list'])-1):
        point_times.append([tg_file['onset_list'][i], tg_file['offset_list'][i], 'speech'])
        point_times.append([tg_file['offset_list'][i], tg_file['onset_list'][i + 1], 'silence'])
    point_times.append([tg_file['onset_list'][-1], tg_file['offset_list'][-1], 'speech'])
    point_times.append([tg_file['offset_list'][-1], get_ECoG_time(sub, session), 'silence'])
    return point_times

def get_duration_in_range(data, start_time, end_time):
    '''
    Calculate the speech duration and silence duration in a certain window
    '''
    speech_duration = 0
    silence_duration = 0

    for item in data:
        segment_start, segment_end, label = item
        if segment_end > start_time and segment_start < end_time:
            overlap_start = max(segment_start, start_time)
            overlap_end = min(segment_end, end_time)
            
            if overlap_end > overlap_start:
                duration = overlap_end - overlap_start
                if label == 'speech':
                    speech_duration += duration
                elif label == 'silence':
                    silence_duration += duration

    return speech_duration, silence_duration

def get_time_mark_asccd_len(sub, session):
    '''
    Determine the window belongs to speech or silence
    '''
    time_marks = []
    for time in np.arange(1, get_ECoG_time(sub, session)-1, 0.1):
        # print(time)
        start_time, end_time = time-0.5, time+0.5
        speech_duration, silence_duration = get_duration_in_range(get_textgrid_interval_asccd(sub, session), start_time, end_time)
        if speech_duration >= silence_duration:
            time_marks.append(1)
        else:
            time_marks.append(0)
    return time_marks

def get_timelocked_activity(times, hg, back, forward, hz=400):
    #times = np.array(times)
    '''
    get_timelocked_activity(np.array([2,3,4,...]), a['bands'][0], back=0.5, forward=0.5)
    '''
    if hz:
        times = (times*hz).astype(int)
        back = int(back*hz)
        forward = int(forward*hz)
    times = times[times - back > 0]
    times = times[times + forward < hg.shape[-1]]
    if hg.ndim == 3:
        Y_mat = np.zeros((len(times), hg.shape[0], hg.shape[1], int(back + forward)), dtype=float)
        for i, index in enumerate(times):
            Y_mat[i, :, :, :] = hg[:,:, int(index-back):int(index+forward)]
    elif hg.ndim == 2:
        Y_mat = np.zeros((len(times), hg.shape[0], int(back + forward)), dtype=float)
        for i, index in enumerate(times):
            Y_mat[i, :, :] = hg[:, int(index-back):int(index+forward)]
    else:
        Y_mat = np.zeros((len(times), int(back + forward)), dtype=float)
        for i, index in enumerate(times):
            Y_mat[i,  :] = hg[int(index-back):int(index+forward)]
            
    return Y_mat, back, forward

def get_speech_trainer_data(sub_dict):
    all_elecs_dataset = []
    for sub, sessions in sub_dict.items():
        for session in sessions:
            with NWBHDF5IO(path_to_save_Data+f'/{sub}/ieeg/{sub}_task-audio_run-{session}_ieeg.nwb', 'r') as io:
                nwbfile = io.read()
                iEEG_ts = nwbfile.acquisition['iEEG']
                iEEG_data = iEEG_ts.data[:]
                ECoG_DATA = iEEG_data.T
            ECoG_DATA = (ECoG_DATA - np.mean(ECoG_DATA, axis=-1, keepdims=True)) / np.std(ECoG_DATA, axis=-1, keepdims=True)
            inputs = torch.tensor(get_timelocked_activity(
                np.arange(1, get_ECoG_time(sub, session)-1, 0.1),
                ECoG_DATA,
                back=0.5,
                forward=0.5
            )[0][:,speech_resp_dict[sub],:])
            label = torch.tensor(get_time_mark_asccd_len(sub, session))
            for i in range(inputs.shape[0]):
                all_elecs_dataset.append((inputs[i], label[i]))
    return all_elecs_dataset

all_dict = {'sub-01': ['01', '02', '03', '04', '05', '06'],
            'sub-02': ['01', '02', '03', '04', '05', '06'],
            'sub-03': ['01', '02', '03', '04'],
            'sub-04': ['01', '02', '03', '04']}

if sub == 'sub-01' or sub == 'sub-02':
    for i in [1, 3, 5]:
        HS_val_Dict = {sub: [f'0{i}']}
        HS_test_Dict = {sub: [f'0{i+1}']}
        HS_train_Dict = {sub: [item for item in all_dict[sub] if (item != f'0{i}' and item != f'0{i+1}')]}

        train_dataset = get_speech_trainer_data(HS_train_Dict)
        val_dataset = get_speech_trainer_data(HS_val_Dict)
        test_dataset = get_speech_trainer_data(HS_test_Dict)

        class ECoGSpeechDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                data_item = self.data[idx][0]
                label = self.data[idx][1]

                return data_item, label
            
        class CRNN(nn.Module):
            def __init__(self, *, typeNum, in_chans,
                        num_layers=2, gruDim=128, drop_out=0.5):
                super().__init__()
                self.conv1d = nn.Conv1d(in_channels=in_chans, out_channels=gruDim, kernel_size=3,stride=1, padding=0)
                self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
                self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=None, padding=0)
                self.dropout = nn.Dropout(p=drop_out)
                gru_layers = []
                for i in range(num_layers):
                    if i == 0:
                        gru_layers.append(nn.GRU(gruDim, gruDim, 1, batch_first=True, bidirectional=True))
                    else:
                        gru_layers.append(nn.GRU(gruDim * 2, gruDim, 1, batch_first=True, bidirectional=True))
                # Create the sequential model with stacked GRU layers
                self.gru_layers = nn.Sequential(*gru_layers)
                elec_feature = int(2 * gruDim)
                self.fc1 = nn.Linear(elec_feature, typeNum)
                
            def forward(self, x):
                x = self.conv1d(x)
                x = self.leaky_relu(x)
                x = self.max_pooling(x)
                x = x.permute(0, 2, 1)
                for gru_layer in self.gru_layers:
                    x, _ = gru_layer(x)
                    x = self.dropout(x)
                x = x[:, -1, :]
                x = self.fc1(x)

                return x

        model1 = CRNN(typeNum=2, in_chans=len(speech_resp_dict[sub]), gruDim=128).to(device)
        torch.save(model1,(save_path+fullname+f'_onset_{i}.pt'))
        torch.save(model1,(save_path+fullname+f'_onset{i+1}.pt'))
        torch.save(model1,(save_path+fullname+f'_{i}.pt'))
        torch.save(model1,(save_path+fullname+f'_{i+1}.pt'))

        train_dataset = ECoGSpeechDataset(train_dataset)
        val_dataset = ECoGSpeechDataset(val_dataset)
        test_dataset = ECoGSpeechDataset(test_dataset)

        def count_labels(all_elecs_dataset):
            count_0 = 0
            count_1 = 0

            for data in all_elecs_dataset:
                label = data[1]
                if label == 0:
                    count_0 += 1
                elif label == 1:
                    count_1 += 1
            return count_0, count_1

        train_0, train_1 = count_labels(train_dataset)

        def train(batch_size, lr, EPOCH, patience, train_dataset, 
                val_dataset=False, test_dataset=False, pred_only_datasets=False, 
                class_weight=False, verbose=True):
            model = torch.load(save_path+fullname+f'_{i}.pt')
            loss_func = nn.CrossEntropyLoss() 
            if class_weight is not False:
                weight = [max(class_weight)/x for x in class_weight]
                weight = torch.FloatTensor(weight).to(device)
                loss_func = nn.CrossEntropyLoss(weight=weight)
                
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            
            early_stopping = EarlyStopping(patience)

            for epoch in range(EPOCH):
                #return acc，ground truth and predicited label for plotting confusion matrices
                sum_loss_train= 0
                total_train=0
                model.train()
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = loss_func(pred, labels)
                    sum_loss_train+= loss.item()
                    total_train+= labels.size(0)
                    loss.backward()
                    optimizer.step()
                train_loss = sum_loss_train/total_train
                if verbose:
                    print('Epoch {}:train-loss:{:.2e},'.format(epoch+1,train_loss),end='')
            
                if val_dataset is not False:
                    sum_loss_val=0
                    total_val = 0  
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                    for data in val_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device).float(), labels.to(device)
                        pred = model(inputs)
                        loss = loss_func(pred, labels)
                        sum_loss_val+=loss.item()
                        total_val+= labels.size(0)
                    val_loss = sum_loss_val/total_val
                    if verbose:
                        print('val-loss:{:.2e},'.format(val_loss,),end='')

                    early_stopping(val_loss, model)

                    if early_stopping.early_stop:
                        if verbose:
                            print("Early stopping")
                        break
                if verbose:
                    if test_dataset is not False:
                        total_test = 0 
                        sum_loss_test= 0
                        predicted=[]
                        label=[]
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        for data in test_loader:
                            inputs, labels = data
                            inputs, labels = inputs.to(device).float(), labels.to(device)
                            pred_prob = model(inputs)
                            loss = loss_func(pred_prob, labels)
                            sum_loss_test+=loss.item()
                            _, pred = torch.max(pred_prob.data, 1)
                            pred_prob = pred_prob.data
                            pred_prob = F.softmax(pred_prob, dim=1) 
                            total_test+= labels.size(0)
                            predicted.append(pred.cpu())
                            label.append(labels.cpu())
                        label = torch.cat(label,dim=0)
                        predicted = torch.cat(predicted,dim=0)
                        correct = (predicted == label).sum()
                        test_acc= correct.item()/total_test
                        test_loss=sum_loss_test/total_test
                        print('test-loss:{:.2e},Acc:{:.4f}'.format(test_loss,test_acc),end='')
                    print('')
            
            model.load_state_dict(torch.load(save_path+f'checkpoint_{fullname}_{i}.pt'))
            ##############################EVALUATION model###############################################
            predicted=[]
            predicted_prob=[]
            label=[]
            acc=[]
            if test_dataset is not False:
                total_test = 0
                sum_loss_test= 0
                model.eval()
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    pred_prob = model(inputs)
                    loss = loss_func(pred_prob, labels)
                    sum_loss_test+=loss.item()
                    _, pred = torch.max(pred_prob.data, 1)
                    pred_prob = pred_prob.data
                    pred_prob = F.softmax(pred_prob, dim=1) 
                    total_test+= labels.size(0)

                    predicted.append(pred.cpu())
                    predicted_prob.append(pred_prob.cpu())
                    label.append(labels.cpu())

                label = torch.cat(label,dim=0)
                predicted = torch.cat(predicted,dim=0)
                predicted_prob = torch.cat(predicted_prob,dim=0)
                correct = (predicted == label).sum()
                test_acc= correct.item()/total_test
                test_loss=sum_loss_test/total_test
                if verbose:
                    print('Final: test-loss:{:.4f},Acc:{:.4f}'.format(test_loss,test_acc))
                label = label.detach().numpy()
                acc = test_acc
                predicted = predicted.detach().numpy()
                predicted_prob = predicted_prob.detach().numpy()
            
            
            #############################################PRED-ONLY 
            predicted_only=[]
            predicted_only_prob=[]
                
            if pred_only_datasets is not False:
                for pred_only_dataset in pred_only_datasets:
                    pred_only_loader = DataLoader(pred_only_dataset, batch_size=batch_size, shuffle=False)
                    for inputs in pred_only_loader:
                        inputs = inputs[0].to(device)
                        pred = model(inputs)
                        pred_only_prob = pred.data
                        pred_only_prob = F.softmax(pred_only_prob, dim=1) 
                        _, pred_only = torch.max(pred.data, 1)
                        predicted_only.append(pred_only.cpu())
                        predicted_only_prob.append(pred_only_prob.cpu())
                    
                predicted_only = torch.cat(predicted_only).detach().numpy()
                predicted_only_prob = torch.cat(predicted_only_prob).detach().numpy()
                if verbose:
                    print(predicted_only_prob.shape,'#$#')

            save_file = os.path.join(save_path, f'{fullname}_cv_fold_{i}.npz')
            np.savez(save_file,
                    labels=label,
                    predicted=predicted,
                    predicted_prob=predicted_prob,
                    predicted_only=predicted_only,
                    predicted_only_prob=predicted_only_prob,
                    acc=acc)


            return label,acc,predicted,predicted_prob, predicted_only,predicted_only_prob, model,loss_func.cpu()

        class EarlyStopping:
            """Early stops the training if validation loss doesn't improve after a given patience."""
            def __init__(self, patience=7, delta=0, verbose=True):
                """
                Args:
                    patience (int): How long to wait after last time validation loss improved.
                                    Default: 7
                    verbose (bool): If True, prints a message for each validation loss improvement. 
                                    Default: False
                    delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                    Default: 0
                """
                self.patience = patience
                self.verbose = verbose
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.val_loss_min = np.Inf
                self.delta = delta

            def __call__(self, val_loss, model):

                score = -val_loss

                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                elif score < self.best_score + self.delta:
                    self.counter += 1
                    if self.verbose:
                        print(f'EarlyStopping: {self.counter}/{self.patience}',end='')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                    self.counter = 0

            def save_checkpoint(self, val_loss, model):
                '''Saves model when validation loss decrease.'''
                torch.save(model.state_dict(), save_path+f'checkpoint_{fullname}_{i}.pt')  # The best parameters
                self.val_loss_min = val_loss

        train(
            batch_size=168,
            lr=0.0001,
            EPOCH=80,
            patience=10,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            pred_only_datasets=False,
            class_weight=[train_0, train_1],
            verbose=True
        )

        def train2(batch_size, lr, EPOCH, patience, train_dataset, 
                val_dataset=False, test_dataset=False, pred_only_datasets=False, 
                class_weight=False, verbose=True):
            model = torch.load(save_path+fullname+f'_{i+1}.pt')
            loss_func = nn.CrossEntropyLoss() 
            if class_weight is not False:
                weight = [max(class_weight)/x for x in class_weight]
                weight = torch.FloatTensor(weight).to(device)
                loss_func = nn.CrossEntropyLoss(weight=weight)
                
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            
            early_stopping = EarlyStopping2(patience)

            for epoch in range(EPOCH):
                sum_loss_train= 0
                total_train=0
                model.train()
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = loss_func(pred, labels)
                    sum_loss_train+= loss.item()
                    total_train+= labels.size(0)
                    loss.backward()
                    optimizer.step()
                train_loss = sum_loss_train/total_train
                if verbose:
                    print('Epoch {}:train-loss:{:.2e},'.format(epoch+1,train_loss),end='')
            
                if val_dataset is not False:
                    sum_loss_val=0
                    total_val = 0  
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                    for data in val_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device).float(), labels.to(device)
                        pred = model(inputs)
                        loss = loss_func(pred, labels)
                        sum_loss_val+=loss.item()
                        total_val+= labels.size(0)
                    val_loss = sum_loss_val/total_val
                    if verbose:
                        print('val-loss:{:.2e},'.format(val_loss,),end='')
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        if verbose:
                            print("Early stopping")
                        break
                if verbose:
                    if test_dataset is not False:
                        total_test = 0  # 总数
                        sum_loss_test= 0
                        predicted=[]
                        label=[]
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        for data in test_loader:
                            inputs, labels = data
                            inputs, labels = inputs.to(device).float(), labels.to(device)
                            pred_prob = model(inputs)
                            loss = loss_func(pred_prob, labels)
                            sum_loss_test+=loss.item()
                            _, pred = torch.max(pred_prob.data, 1)
                            pred_prob = pred_prob.data
                            pred_prob = F.softmax(pred_prob, dim=1) 
                            total_test+= labels.size(0)
                            predicted.append(pred.cpu())
                            label.append(labels.cpu())
                        label = torch.cat(label,dim=0)
                        predicted = torch.cat(predicted,dim=0)
                        correct = (predicted == label).sum()
                        test_acc= correct.item()/total_test
                        test_loss=sum_loss_test/total_test
                        print('test-loss:{:.2e},Acc:{:.4f}'.format(test_loss,test_acc),end='')
                    print('')
            
            model.load_state_dict(torch.load(save_path+f'checkpoint_{fullname}_{i+1}.pt'))
            ##############################EVALUATION model###############################################
            predicted=[]
            predicted_prob=[]
            label=[]
            acc=[]
            if test_dataset is not False:
                total_test = 0 
                sum_loss_test= 0
                model.eval()
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    pred_prob = model(inputs)
                    loss = loss_func(pred_prob, labels)
                    sum_loss_test+=loss.item()
                    _, pred = torch.max(pred_prob.data, 1)
                    pred_prob = pred_prob.data
                    pred_prob = F.softmax(pred_prob, dim=1) 
                    total_test+= labels.size(0)

                    predicted.append(pred.cpu())
                    predicted_prob.append(pred_prob.cpu())
                    label.append(labels.cpu())

                label = torch.cat(label,dim=0)
                predicted = torch.cat(predicted,dim=0)
                predicted_prob = torch.cat(predicted_prob,dim=0)
                correct = (predicted == label).sum()
                test_acc= correct.item()/total_test
                test_loss=sum_loss_test/total_test
                if verbose:
                    print('Final: test-loss:{:.4f},Acc:{:.4f}'.format(test_loss,test_acc))
                label = label.detach().numpy()
                acc = test_acc
                predicted = predicted.detach().numpy()
                predicted_prob = predicted_prob.detach().numpy()
            
            
            #############################################PRED-ONLY 
            predicted_only=[]
            predicted_only_prob=[]
                
            if pred_only_datasets is not False:
                for pred_only_dataset in pred_only_datasets:
                    pred_only_loader = DataLoader(pred_only_dataset, batch_size=batch_size, shuffle=False)
                    for inputs in pred_only_loader:
                        inputs = inputs[0].to(device)
                        pred = model(inputs)
                        pred_only_prob = pred.data
                        pred_only_prob = F.softmax(pred_only_prob, dim=1) 
                        _, pred_only = torch.max(pred.data, 1)
                        predicted_only.append(pred_only.cpu())
                        predicted_only_prob.append(pred_only_prob.cpu())
                    
                predicted_only = torch.cat(predicted_only).detach().numpy()
                predicted_only_prob = torch.cat(predicted_only_prob).detach().numpy()
                if verbose:
                    print(predicted_only_prob.shape,'#$#')

            save_file = os.path.join(save_path, f'{fullname}_cv_fold_{i+1}.npz')
            np.savez(save_file,
                    labels=label,
                    predicted=predicted,
                    predicted_prob=predicted_prob,
                    predicted_only=predicted_only,
                    predicted_only_prob=predicted_only_prob,
                    acc=acc)

            return label,acc,predicted,predicted_prob, predicted_only,predicted_only_prob, model,loss_func.cpu()

        class EarlyStopping2:
            """Early stops the training if validation loss doesn't improve after a given patience."""
            def __init__(self, patience=7, delta=0, verbose=True):
                """
                Args:
                    patience (int): How long to wait after last time validation loss improved.
                                    Default: 7
                    verbose (bool): If True, prints a message for each validation loss improvement. 
                                    Default: False
                    delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                    Default: 0
                """
                self.patience = patience
                self.verbose = verbose
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.val_loss_min = np.Inf
                self.delta = delta

            def __call__(self, val_loss, model):

                score = -val_loss

                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                elif score < self.best_score + self.delta:
                    self.counter += 1
                    if self.verbose:
                        print(f'EarlyStopping: {self.counter}/{self.patience}',end='')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                    self.counter = 0

            def save_checkpoint(self, val_loss, model):
                '''Saves model when validation loss decrease.'''
                torch.save(model.state_dict(), save_path+f'checkpoint_{fullname}_{i+1}.pt')
                self.val_loss_min = val_loss

        train2(
            batch_size=168,
            lr=0.0001,
            EPOCH=80,
            patience=10,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            test_dataset=val_dataset,
            pred_only_datasets=False,
            class_weight=[train_0, train_1],
            verbose=True 
        )

elif sub == 'sub-03' or sub == 'sub-04':
    for i in [1, 2, 3, 4]:
        HS_val_test_Dict = {sub: [f'0{i}']}
        HS_train_Dict = {sub: [item for item in all_dict[sub] if item != f'0{i}']}

        train_dataset = get_speech_trainer_data(HS_train_Dict)
        val_test_dataset = get_speech_trainer_data(HS_val_test_Dict)

        class ECoGSpeechDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                data_item = self.data[idx][0]
                label = self.data[idx][1]

                return data_item, label
            
        class CRNN(nn.Module):
            def __init__(self, *, typeNum, in_chans,
                        num_layers=2, gruDim=256, drop_out=0.5):
                super().__init__()
                self.conv1d = nn.Conv1d(in_channels=in_chans, out_channels=gruDim, kernel_size=3,stride=1, padding=0)
                self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
                self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=None, padding=0)
                self.dropout = nn.Dropout(p=drop_out)
                gru_layers = []
                for i in range(num_layers):
                    if i == 0:
                        gru_layers.append(nn.GRU(gruDim, gruDim, 1, batch_first=True, bidirectional=True))
                    else:
                        gru_layers.append(nn.GRU(gruDim * 2, gruDim, 1, batch_first=True, bidirectional=True))
                # Create the sequential model with stacked GRU layers
                self.gru_layers = nn.Sequential(*gru_layers)
                elec_feature = int(2 * gruDim)
                self.fc1 = nn.Linear(elec_feature, typeNum)
                
            def forward(self, x):
                x = self.conv1d(x)
                x = self.leaky_relu(x)
                x = self.max_pooling(x)
                x = x.permute(0, 2, 1)
                for gru_layer in self.gru_layers:
                    x, _ = gru_layer(x)
                    x = self.dropout(x)
                x = x[:, -1, :]
                x = self.fc1(x)

                return x

        model1 = CRNN(typeNum=2, in_chans=len(speech_resp_dict[sub]), gruDim=128).to(device)
        torch.save(model1,(save_path+fullname+f'_onset_{2*i-1}.pt'))
        torch.save(model1,(save_path+fullname+f'_onset_{2*i}.pt'))
        torch.save(model1,(save_path+fullname+f'_{2*i-1}.pt'))
        torch.save(model1,(save_path+fullname+f'_{2*i}.pt'))

        train_dataset = ECoGSpeechDataset(train_dataset)
        val_test_dataset = ECoGSpeechDataset(val_test_dataset)

        total_size = len(val_test_dataset)
        val_size = int(0.5 * total_size)
        test_size = total_size - val_size
        # Split the val and test dataset, and note that they harly have any overlapping
        val_dataset = torch.utils.data.Subset(val_test_dataset, range(0, val_size))
        test_dataset = torch.utils.data.Subset(val_test_dataset, range(val_size, total_size))

        def count_labels(elecs_dataset):
            count_0 = 0
            count_1 = 0

            for data in elecs_dataset:
                label = data[1]
                if label == 0:
                    count_0 += 1
                elif label == 1:
                    count_1 += 1
            return count_0, count_1

        train_0, train_1 = count_labels(train_dataset)

        def train(batch_size, lr, EPOCH, patience, train_dataset, 
                val_dataset=False, test_dataset=False, pred_only_datasets=False, 
                class_weight=False, verbose=True):
            model = torch.load(save_path+fullname+f'_{2*i-1}.pt')
            loss_func = nn.CrossEntropyLoss() 
            if class_weight is not False:
                weight = [max(class_weight)/x for x in class_weight]
                weight = torch.FloatTensor(weight).to(device)
                loss_func = nn.CrossEntropyLoss(weight=weight)
                
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            
            early_stopping = EarlyStopping(patience)

            for epoch in range(EPOCH):
                sum_loss_train= 0
                total_train=0
                model.train()
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = loss_func(pred, labels)
                    sum_loss_train+= loss.item()
                    total_train+= labels.size(0)
                    loss.backward()
                    optimizer.step()
                train_loss = sum_loss_train/total_train
                if verbose:
                    print('Epoch {}:train-loss:{:.2e},'.format(epoch+1,train_loss),end='')
            
                if val_dataset is not False:
                    sum_loss_val=0
                    total_val = 0  
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                    for data in val_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device).float(), labels.to(device)
                        pred = model(inputs)
                        loss = loss_func(pred, labels)
                        sum_loss_val+=loss.item()
                        total_val+= labels.size(0)
                    val_loss = sum_loss_val/total_val
                    if verbose:
                        print('val-loss:{:.2e},'.format(val_loss,),end='')
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        if verbose:
                            print("Early stopping")
                        break
                if verbose:
                    if test_dataset is not False:
                        total_test = 0
                        sum_loss_test= 0
                        predicted=[]
                        label=[]
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        for data in test_loader:
                            inputs, labels = data
                            inputs, labels = inputs.to(device).float(), labels.to(device)
                            pred_prob = model(inputs)
                            loss = loss_func(pred_prob, labels)
                            sum_loss_test+=loss.item()
                            _, pred = torch.max(pred_prob.data, 1)
                            pred_prob = pred_prob.data
                            pred_prob = F.softmax(pred_prob, dim=1) 
                            total_test+= labels.size(0)
                            predicted.append(pred.cpu())
                            label.append(labels.cpu())
                        label = torch.cat(label,dim=0)
                        predicted = torch.cat(predicted,dim=0)
                        correct = (predicted == label).sum()
                        test_acc= correct.item()/total_test
                        test_loss=sum_loss_test/total_test
                        print('test-loss:{:.2e},Acc:{:.4f}'.format(test_loss,test_acc),end='')
                    print('')
            
            model.load_state_dict(torch.load(save_path+f'checkpoint_{fullname}_{2*i-1}.pt'))
            ##############################EVALUATION model###############################################
            predicted=[]
            predicted_prob=[]
            label=[]
            acc=[]
            if test_dataset is not False:
                total_test = 0 
                sum_loss_test= 0
                model.eval()
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    pred_prob = model(inputs)
                    loss = loss_func(pred_prob, labels)
                    sum_loss_test+=loss.item()
                    _, pred = torch.max(pred_prob.data, 1)
                    pred_prob = pred_prob.data
                    pred_prob = F.softmax(pred_prob, dim=1) 
                    total_test+= labels.size(0)

                    predicted.append(pred.cpu())
                    predicted_prob.append(pred_prob.cpu())
                    label.append(labels.cpu())

                label = torch.cat(label,dim=0)
                predicted = torch.cat(predicted,dim=0)
                predicted_prob = torch.cat(predicted_prob,dim=0)
                correct = (predicted == label).sum()
                test_acc= correct.item()/total_test
                test_loss=sum_loss_test/total_test
                if verbose:
                    print('Final: test-loss:{:.4f},Acc:{:.4f}'.format(test_loss,test_acc))
                label = label.detach().numpy()
                acc = test_acc
                predicted = predicted.detach().numpy()
                predicted_prob = predicted_prob.detach().numpy()
            
            
            #############################################PRED-ONLY 
            predicted_only=[]
            predicted_only_prob=[]
                
            if pred_only_datasets is not False:
                for pred_only_dataset in pred_only_datasets:
                    pred_only_loader = DataLoader(pred_only_dataset, batch_size=batch_size, shuffle=False)
                    for inputs in pred_only_loader:
                        inputs = inputs[0].to(device)
                        pred = model(inputs)
                        pred_only_prob = pred.data
                        pred_only_prob = F.softmax(pred_only_prob, dim=1) 
                        _, pred_only = torch.max(pred.data, 1)
                        predicted_only.append(pred_only.cpu())
                        predicted_only_prob.append(pred_only_prob.cpu())
                    
                predicted_only = torch.cat(predicted_only).detach().numpy()
                predicted_only_prob = torch.cat(predicted_only_prob).detach().numpy()
                if verbose:
                    print(predicted_only_prob.shape,'#$#')

            save_file = os.path.join(save_path, f'{fullname}_cv_fold_{2*i-1}.npz')
            np.savez(save_file,
                    labels=label,
                    predicted=predicted,
                    predicted_prob=predicted_prob,
                    predicted_only=predicted_only,
                    predicted_only_prob=predicted_only_prob,
                    acc=acc)

            return label,acc,predicted,predicted_prob, predicted_only,predicted_only_prob, model,loss_func.cpu()

        class EarlyStopping:
            """Early stops the training if validation loss doesn't improve after a given patience."""
            def __init__(self, patience=7, delta=0, verbose=True):
                """
                Args:
                    patience (int): How long to wait after last time validation loss improved.
                                    Default: 7
                    verbose (bool): If True, prints a message for each validation loss improvement. 
                                    Default: False
                    delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                    Default: 0
                """
                self.patience = patience
                self.verbose = verbose
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.val_loss_min = np.Inf
                self.delta = delta

            def __call__(self, val_loss, model):

                score = -val_loss

                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                elif score < self.best_score + self.delta:
                    self.counter += 1
                    if self.verbose:
                        print(f'EarlyStopping: {self.counter}/{self.patience}',end='')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                    self.counter = 0

            def save_checkpoint(self, val_loss, model):
                '''Saves model when validation loss decrease.'''
                torch.save(model.state_dict(), save_path+f'checkpoint_{fullname}_{2*i-1}.pt')
                self.val_loss_min = val_loss

        train(
            batch_size=168,
            lr=0.0001,
            EPOCH=80,
            patience=10,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            pred_only_datasets=False,
            class_weight=[train_0, train_1],
            verbose=True
        )

        def train2(batch_size, lr, EPOCH, patience, train_dataset, 
                val_dataset=False, test_dataset=False, pred_only_datasets=False, 
                class_weight=False, verbose=True):
            model = torch.load(save_path+fullname+f'_{2*i}.pt')
            loss_func = nn.CrossEntropyLoss() 
            if class_weight is not False:
                weight = [max(class_weight)/x for x in class_weight]
                weight = torch.FloatTensor(weight).to(device)
                loss_func = nn.CrossEntropyLoss(weight=weight)
                
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            
            early_stopping = EarlyStopping2(patience)

            for epoch in range(EPOCH):
                sum_loss_train= 0
                total_train=0
                model.train()
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                for data in train_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    
                    optimizer.zero_grad()
                    pred = model(inputs)
                    loss = loss_func(pred, labels)
                    sum_loss_train+= loss.item()
                    total_train+= labels.size(0)
                    loss.backward()
                    optimizer.step()
                train_loss = sum_loss_train/total_train
                if verbose:
                    print('Epoch {}:train-loss:{:.2e},'.format(epoch+1,train_loss),end='')
            
                if val_dataset is not False:
                    sum_loss_val=0
                    total_val = 0  
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                    for data in val_loader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device).float(), labels.to(device)
                        pred = model(inputs)
                        loss = loss_func(pred, labels)
                        sum_loss_val+=loss.item()
                        total_val+= labels.size(0)
                    val_loss = sum_loss_val/total_val
                    if verbose:
                        print('val-loss:{:.2e},'.format(val_loss,),end='')
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        if verbose:
                            print("Early stopping")
                        break
                if verbose:
                    if test_dataset is not False:
                        total_test = 0 
                        sum_loss_test= 0
                        predicted=[]
                        label=[]
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        for data in test_loader:
                            inputs, labels = data
                            inputs, labels = inputs.to(device).float(), labels.to(device)
                            pred_prob = model(inputs)
                            loss = loss_func(pred_prob, labels)
                            sum_loss_test+=loss.item()
                            _, pred = torch.max(pred_prob.data, 1)
                            pred_prob = pred_prob.data
                            pred_prob = F.softmax(pred_prob, dim=1) 
                            total_test+= labels.size(0)
                            predicted.append(pred.cpu())
                            label.append(labels.cpu())
                        label = torch.cat(label,dim=0)
                        predicted = torch.cat(predicted,dim=0)
                        correct = (predicted == label).sum()
                        test_acc= correct.item()/total_test
                        test_loss=sum_loss_test/total_test
                        print('test-loss:{:.2e},Acc:{:.4f}'.format(test_loss,test_acc),end='')
                    print('')
            
            model.load_state_dict(torch.load(save_path+f'checkpoint_{fullname}_{2*i}.pt'))
            ##############################EVALUATION model###############################################
            predicted=[]
            predicted_prob=[]
            label=[]
            acc=[]
            if test_dataset is not False:
                total_test = 0
                sum_loss_test= 0
                model.eval()
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device).float(), labels.to(device)
                    pred_prob = model(inputs)
                    loss = loss_func(pred_prob, labels)
                    sum_loss_test+=loss.item()
                    _, pred = torch.max(pred_prob.data, 1)
                    pred_prob = pred_prob.data
                    pred_prob = F.softmax(pred_prob, dim=1) 
                    total_test+= labels.size(0)

                    predicted.append(pred.cpu())
                    predicted_prob.append(pred_prob.cpu())
                    label.append(labels.cpu())

                label = torch.cat(label,dim=0)
                predicted = torch.cat(predicted,dim=0)
                predicted_prob = torch.cat(predicted_prob,dim=0)
                correct = (predicted == label).sum()
                test_acc= correct.item()/total_test
                test_loss=sum_loss_test/total_test
                if verbose:
                    print('Final: test-loss:{:.4f},Acc:{:.4f}'.format(test_loss,test_acc))
                label = label.detach().numpy()
                acc = test_acc
                predicted = predicted.detach().numpy()
                predicted_prob = predicted_prob.detach().numpy()
            
            
            #############################################PRED-ONLY 
            predicted_only=[]
            predicted_only_prob=[]
                
            if pred_only_datasets is not False:
                for pred_only_dataset in pred_only_datasets:
                    pred_only_loader = DataLoader(pred_only_dataset, batch_size=batch_size, shuffle=False)
                    for inputs in pred_only_loader:
                        inputs = inputs[0].to(device)
                        pred = model(inputs)
                        pred_only_prob = pred.data
                        pred_only_prob = F.softmax(pred_only_prob, dim=1) 
                        _, pred_only = torch.max(pred.data, 1)
                        predicted_only.append(pred_only.cpu())
                        predicted_only_prob.append(pred_only_prob.cpu())
                    
                predicted_only = torch.cat(predicted_only).detach().numpy()
                predicted_only_prob = torch.cat(predicted_only_prob).detach().numpy()
                if verbose:
                    print(predicted_only_prob.shape,'#$#')

            save_file = os.path.join(save_path, f'{fullname}_cv_fold_{2*i}.npz')
            np.savez(save_file,
                    labels=label,
                    predicted=predicted,
                    predicted_prob=predicted_prob,
                    predicted_only=predicted_only,
                    predicted_only_prob=predicted_only_prob,
                    acc=acc)

            return label,acc,predicted,predicted_prob, predicted_only,predicted_only_prob, model,loss_func.cpu()

        class EarlyStopping2:
            """Early stops the training if validation loss doesn't improve after a given patience."""
            def __init__(self, patience=7, delta=0, verbose=True):
                """
                Args:
                    patience (int): How long to wait after last time validation loss improved.
                                    Default: 7
                    verbose (bool): If True, prints a message for each validation loss improvement. 
                                    Default: False
                    delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                    Default: 0
                """
                self.patience = patience
                self.verbose = verbose
                self.counter = 0
                self.best_score = None
                self.early_stop = False
                self.val_loss_min = np.Inf
                self.delta = delta

            def __call__(self, val_loss, model):

                score = -val_loss

                if self.best_score is None:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                elif score < self.best_score + self.delta:
                    self.counter += 1
                    if self.verbose:
                        print(f'EarlyStopping: {self.counter}/{self.patience}',end='')
                    if self.counter >= self.patience:
                        self.early_stop = True
                else:
                    self.best_score = score
                    self.save_checkpoint(val_loss, model)
                    self.counter = 0

            def save_checkpoint(self, val_loss, model):
                '''Saves model when validation loss decrease.'''
                torch.save(model.state_dict(), save_path+f'checkpoint_{fullname}_{2*i}.pt')
                self.val_loss_min = val_loss

        train2(
            batch_size=168,
            lr=0.0001,
            EPOCH=80,
            patience=10,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            test_dataset=val_dataset,
            pred_only_datasets=False,
            class_weight=[train_0, train_1],
            verbose=True 
        )