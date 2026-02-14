"""
Training Utilities for MoLE
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class Trainer:
    """
    Trainer for MoLE models
    
    Args:
        model: MoLE model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: torch device
        lr: Learning rate
        epochs: Number of training epochs
        patience: Early stopping patience
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device='cpu', lr=0.001, epochs=100, patience=10,
                 use_aux_loss=True, aux_loss_coef=0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.use_aux_loss = use_aux_loss
        self.aux_loss_coef = aux_loss_coef
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            
            # Main loss
            loss = self.criterion(outputs, batch_y)
            
            # Auxiliary loss (load balancing)
            if self.use_aux_loss and hasattr(self.model, 'router'):
                if hasattr(self.model.router, 'top_k') and self.model.router.top_k is not None:
                    _, weights, aux_loss = self.model(batch_x, return_weights=True)
                    if aux_loss is not None:
                        loss = loss + self.aux_loss_coef * aux_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def test(self):
        """Test the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def train(self):
        """Full training loop"""
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            self.early_stopping(val_loss, self.model, 'checkpoint.pth')
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('checkpoint.pth'))
        
        # Final test
        test_loss = self.test()
        self.history['test_loss'].append(test_loss)
        print(f"\nFinal Test Loss: {test_loss:.6f}")
        
        # Print expert usage
        if hasattr(self.model, 'get_expert_usage'):
            usage = self.model.get_expert_usage()
            print(f"\nExpert Usage: {usage}")
        
        return self.history
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")


class TTNNTrainer(Trainer):
    """
    Trainer with TT-NN support for Tenstorrent hardware
    """
    def __init__(self, model, train_loader, val_loader, test_loader,
                 device='cpu', ttnn_device=None, lr=0.001, epochs=100, patience=10):
        super().__init__(model, train_loader, val_loader, test_loader, 
                        device, lr, epochs, patience)
        self.ttnn_device = ttnn_device
        self.use_ttnn = ttnn_device is not None
    
    def train_epoch(self):
        """Train for one epoch with TT-NN support"""
        if not self.use_ttnn:
            return super().train_epoch()
        
        # For TT-NN, we need to handle training differently
        # Currently, TT-NN inference is supported but training needs special handling
        return super().train_epoch()  # Fall back to PyTorch for training
