"""
Evaluation Metrics for Time Series Forecasting
"""
import torch
import numpy as np


def RSE(pred, true):
    """Root Squared Error"""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """Correlation"""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """Mean Squared Error"""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """Mean Squared Percentage Error"""
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    """
    Compute all metrics
    
    Args:
        pred: Predictions [batch, pred_len, features]
        true: Ground truth [batch, pred_len, features]
    
    Returns:
        dict of metrics
    """
    # Convert to numpy if torch tensors
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(true):
        true = true.detach().cpu().numpy()
    
    return {
        'mae': MAE(pred, true),
        'mse': MSE(pred, true),
        'rmse': RMSE(pred, true),
        'mape': MAPE(pred, true),
        'mspe': MSPE(pred, true)
    }


class MetricsTracker:
    """Track metrics during training and evaluation"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'mae': [],
            'mse': [],
            'rmse': [],
            'mape': [],
            'mspe': []
        }
        self.total_samples = 0
    
    def update(self, pred, true):
        """Update metrics with batch predictions"""
        # Convert to numpy
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(true):
            true = true.detach().cpu().numpy()
        
        batch_size = pred.shape[0]
        self.total_samples += batch_size
        
        self.metrics['mae'].append(MAE(pred, true) * batch_size)
        self.metrics['mse'].append(MSE(pred, true) * batch_size)
        self.metrics['rmse'].append(RMSE(pred, true) * batch_size)
        self.metrics['mape'].append(MAPE(pred, true) * batch_size)
        self.metrics['mspe'].append(MSPE(pred, true) * batch_size)
    
    def get_metrics(self):
        """Get average metrics"""
        if self.total_samples == 0:
            return {k: 0.0 for k in self.metrics.keys()}
        
        return {
            k: sum(v) / self.total_samples 
            for k, v in self.metrics.items()
        }
    
    def print_metrics(self, prefix=''):
        """Print formatted metrics"""
        metrics = self.get_metrics()
        print(f"{prefix}MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}, "
              f"RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.4f}, "
              f"MSPE: {metrics['mspe']:.4f}")


def compute_expert_specialization(router_weights, num_experts):
    """
    Compute expert specialization metrics
    
    Args:
        router_weights: [batch, num_experts] router weight distribution
        num_experts: Number of experts
    
    Returns:
        dict of specialization metrics
    """
    if torch.is_tensor(router_weights):
        router_weights = router_weights.detach().cpu().numpy()
    
    # Average usage per expert
    avg_usage = router_weights.mean(axis=0)  # [num_experts]
    
    # Entropy of usage (higher = more uniform)
    epsilon = 1e-10
    entropy = -np.sum(avg_usage * np.log(avg_usage + epsilon))
    max_entropy = np.log(num_experts)
    normalized_entropy = entropy / max_entropy
    
    # Gini coefficient (0 = uniform, 1 = single expert)
    sorted_usage = np.sort(avg_usage)
    n = len(sorted_usage)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_usage)) / (n * np.sum(sorted_usage)) - (n + 1) / n
    
    return {
        'avg_usage': avg_usage.tolist(),
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'gini': float(gini),
        'max_expert': int(np.argmax(avg_usage)),
        'min_expert': int(np.argmin(avg_usage))
    }
