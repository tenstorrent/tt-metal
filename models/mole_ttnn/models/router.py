"""
Router Model for MoLE
Learns to dynamically weight and combine expert outputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn


class Router(nn.Module):
    """
    Router model for MoLE.
    
    A small MLP that learns expert mixing weights based on input features.
    
    Args:
        seq_len: Input sequence length
        enc_in: Number of input features
        num_experts: Number of experts
        hidden_dim: Hidden dimension for MLP
        num_layers: Number of MLP layers (default: 2)
        dropout: Dropout rate
    """
    def __init__(self, seq_len, enc_in, num_experts, hidden_dim=None, num_layers=2, dropout=0.1):
        super(Router, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_experts = num_experts
        
        if hidden_dim is None:
            hidden_dim = max(64, num_experts * 2)
        
        # Feature extraction: simple statistics
        self.feature_dim = enc_in * 4  # mean, std, min, max per feature
        
        # MLP layers
        layers = []
        in_dim = self.feature_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer outputs expert weights
                layers.append(nn.Linear(in_dim, num_experts))
            else:
                # Hidden layers
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def extract_features(self, x):
        """
        Extract simple statistical features from input.
        
        Args:
            x: [batch, seq_len, enc_in]
        Returns:
            features: [batch, feature_dim]
        """
        # Compute statistics along time dimension
        mean = x.mean(dim=1)  # [batch, enc_in]
        std = x.std(dim=1)    # [batch, enc_in]
        min_val = x.min(dim=1)[0]  # [batch, enc_in]
        max_val = x.max(dim=1)[0]  # [batch, enc_in]
        
        # Concatenate features
        features = torch.cat([mean, std, min_val, max_val], dim=1)  # [batch, feature_dim]
        return features
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, enc_in]
        Returns:
            weights: [batch, num_experts] (softmax normalized)
        """
        # Extract features
        features = self.extract_features(x)  # [batch, feature_dim]
        
        # MLP forward
        logits = self.mlp(features)  # [batch, num_experts]
        
        # Softmax normalization
        weights = F.softmax(logits, dim=-1)  # [batch, num_experts]
        
        return weights


class RouterTTNN(nn.Module):
    """
    Router model with TT-NN backend support
    """
    def __init__(self, seq_len, enc_in, num_experts, hidden_dim=None, num_layers=2, 
                 dropout=0.1, device=None):
        super(RouterTTNN, self).__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.num_experts = num_experts
        self.device = device
        
        # Store model for CPU fallback
        self.router = Router(seq_len, enc_in, num_experts, hidden_dim, num_layers, dropout)
        self.use_ttnn = False
        
        if device is not None and self._check_ttnn_available():
            self.use_ttnn = True
            self._init_ttnn_layers()
    
    def _check_ttnn_available(self):
        """Check if TT-NN is available"""
        try:
            import ttnn
            test_tensor = ttnn.ones([1, 1], device=self.device)
            return True
        except Exception as e:
            print(f"TT-NN not available, using PyTorch fallback: {e}")
            return False
    
    def _init_ttnn_layers(self):
        """Initialize TT-NN layers from PyTorch model"""
        self.ttnn_layers = []
        
        # Extract MLP layers
        for module in self.router.mlp:
            if isinstance(module, nn.Linear):
                weight = ttnn.from_torch(module.weight.data.T, dtype=ttnn.bfloat16, device=self.device)
                bias = ttnn.from_torch(module.bias.data, dtype=ttnn.bfloat16, device=self.device) if module.bias is not None else None
                self.ttnn_layers.append({
                    'type': 'linear',
                    'weight': weight,
                    'bias': bias
                })
            elif isinstance(module, nn.ReLU):
                self.ttnn_layers.append({'type': 'relu'})
    
    def extract_features_ttnn(self, x):
        """
        Extract features using TT-NN operations
        
        Args:
            x: TT-NN tensor [batch, seq_len, enc_in]
        Returns:
            features: TT-NN tensor [batch, feature_dim]
        """
        # For now, use PyTorch for feature extraction (complex reductions)
        x_torch = ttnn.to_torch(x)
        features = self.router.extract_features(x_torch)
        return ttnn.from_torch(features, dtype=ttnn.bfloat16, device=self.device)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, enc_in] or TT-NN tensor
        Returns:
            weights: [batch, num_experts] (softmax normalized)
        """
        if not self.use_ttnn:
            return self.router(x)
        
        if isinstance(x, ttnn.Tensor):
            # Extract features
            features = self.extract_features_ttnn(x)
            
            # MLP forward with TT-NN
            for layer in self.ttnn_layers:
                if layer['type'] == 'linear':
                    features = ttnn.linear(features, layer['weight'], bias=layer['bias'])
                elif layer['type'] == 'relu':
                    features = ttnn.relu(features)
            
            # Softmax
            # TT-NN softmax: ttnn.softmax(features, dim=-1)
            weights = ttnn.softmax(features, dim=-1)
            return weights
        else:
            # PyTorch fallback
            return self.router(x)


class TopKRouter(Router):
    """
    Router with Top-K expert selection for efficiency
    """
    def __init__(self, seq_len, enc_in, num_experts, top_k=2, **kwargs):
        super(TopKRouter, self).__init__(seq_len, enc_in, num_experts, **kwargs)
        self.top_k = min(top_k, num_experts)
        
        # Load balancing loss coefficient
        self.aux_loss_coef = 0.01
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, enc_in]
        Returns:
            weights: [batch, num_experts] (sparse, only top-k non-zero)
            aux_loss: load balancing auxiliary loss
        """
        # Get full softmax weights
        weights = super().forward(x)  # [batch, num_experts]
        
        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # Renormalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Create sparse weight tensor
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        # Compute auxiliary load balancing loss
        # Encourage uniform expert usage across batch
        router_prob = weights.mean(dim=0)  # [num_experts]
        aux_loss = self.num_experts * (router_prob ** 2).mean() - 1.0
        aux_loss = self.aux_loss_coef * aux_loss
        
        return sparse_weights, aux_loss


class NoisyTopKRouter(Router):
    """
    Router with noise injection for exploration during training
    """
    def __init__(self, seq_len, enc_in, num_experts, top_k=2, noise_std=0.1, **kwargs):
        super(NoisyTopKRouter, self).__init__(seq_len, enc_in, num_experts, **kwargs)
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std
        
        # Noise generation network
        self.noise_linear = nn.Linear(self.feature_dim, num_experts)
    
    def forward(self, x, training=True):
        """
        Args:
            x: [batch, seq_len, enc_in]
            training: whether in training mode
        Returns:
            weights: [batch, num_experts]
            aux_loss: load balancing loss
        """
        # Extract features
        features = self.extract_features(x)
        
        # Get logits
        logits = self.mlp(features)
        
        # Add noise during training
        if training:
            noise_logits = self.noise_linear(features)
            noise = torch.randn_like(noise_logits) * self.noise_std
            logits = logits + noise * F.softplus(noise_logits)
        
        # Softmax
        weights = F.softmax(logits, dim=-1)
        
        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Sparse weights
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        # Load balancing loss
        router_prob = weights.mean(dim=0)
        aux_loss = self.num_experts * (router_prob ** 2).mean() - 1.0
        
        return sparse_weights, aux_loss
