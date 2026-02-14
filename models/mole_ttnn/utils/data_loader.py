"""
Data Loading Utilities for Time Series
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    Generic time series dataset
    
    Args:
        data: Time series data [timesteps, features]
        seq_len: Input sequence length
        pred_len: Prediction length
        flag: 'train', 'val', or 'test'
        scale: Whether to normalize data
    """
    def __init__(self, data, seq_len=96, pred_len=96, flag='train', scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag
        
        # Split data
        n = len(data)
        if flag == 'train':
            self.data = data[:int(0.7 * n)]
        elif flag == 'val':
            self.data = data[int(0.7 * n):int(0.8 * n)]
        else:  # test
            self.data = data[int(0.8 * n):]
        
        # Normalization
        self.scale = scale
        if scale:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            self.scaler = None
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]  # [seq_len, features]
        seq_y = self.data[r_begin:r_end]  # [pred_len, features]
        
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)


class ETTHDataset(TimeSeriesDataset):
    """ETT (Electricity Transformer Temperature) Dataset"""
    def __init__(self, root_path='./data', data_path='ETTh1.csv', 
                 seq_len=96, pred_len=96, flag='train', scale=True):
        # Load data
        df = pd.read_csv(os.path.join(root_path, data_path))
        
        # Remove date column
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
        
        data = df.values
        
        super().__init__(data, seq_len, pred_len, flag, scale)


class WeatherDataset(TimeSeriesDataset):
    """Weather Dataset"""
    def __init__(self, root_path='./data', data_path='weather.csv',
                 seq_len=96, pred_len=96, flag='train', scale=True):
        df = pd.read_csv(os.path.join(root_path, data_path))
        
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
        
        data = df.values
        
        super().__init__(data, seq_len, pred_len, flag, scale)


class ElectricityDataset(TimeSeriesDataset):
    """Electricity Consumption Dataset"""
    def __init__(self, root_path='./data', data_path='electricity.csv',
                 seq_len=96, pred_len=96, flag='train', scale=True):
        df = pd.read_csv(os.path.join(root_path, data_path))
        
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
        
        data = df.values
        
        super().__init__(data, seq_len, pred_len, flag, scale)


class StandardScaler:
    """Standard scaler for time series normalization"""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit_transform(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean
    
    def transform(self, data):
        return (data - self.mean) / self.std


def get_dataloader(dataset_name='ETTh1', root_path='./data', 
                  seq_len=96, pred_len=96, batch_size=32, 
                  flag='train', scale=True, num_workers=0):
    """
    Get data loader for a dataset
    
    Args:
        dataset_name: Name of dataset ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity')
        root_path: Path to data directory
        seq_len: Input sequence length
        pred_len: Prediction length
        batch_size: Batch size
        flag: 'train', 'val', or 'test'
        scale: Whether to normalize data
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader
    """
    data_map = {
        'ETTh1': ('ETTh1.csv', ETTHDataset),
        'ETTh2': ('ETTh2.csv', ETTHDataset),
        'ETTm1': ('ETTm1.csv', ETTHDataset),
        'ETTm2': ('ETTm2.csv', ETTHDataset),
        'weather': ('weather.csv', WeatherDataset),
        'electricity': ('electricity.csv', ElectricityDataset),
    }
    
    if dataset_name not in data_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data_path, DatasetClass = data_map[dataset_name]
    
    dataset = DatasetClass(
        root_path=root_path,
        data_path=data_path,
        seq_len=seq_len,
        pred_len=pred_len,
        flag=flag,
        scale=scale
    )
    
    shuffle = (flag == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
    
    return dataloader


def load_data(dataset_name='ETTh1', root_path='./data', 
              seq_len=96, pred_len=96, batch_size=32, scale=True):
    """
    Load train, val, and test data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = get_dataloader(
        dataset_name, root_path, seq_len, pred_len, batch_size, 'train', scale
    )
    val_loader = get_dataloader(
        dataset_name, root_path, seq_len, pred_len, batch_size, 'val', scale
    )
    test_loader = get_dataloader(
        dataset_name, root_path, seq_len, pred_len, batch_size, 'test', scale
    )
    
    return train_loader, val_loader, test_loader
