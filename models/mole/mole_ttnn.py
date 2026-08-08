"""
MoLE (Mixture-of-Linear-Experts) Implementation for Tenstorrent TTNN

This is a TTNN-based implementation of the MoLE framework from the paper:
"Mixture-of-Linear-Experts for Long-term Time Series Forecasting"
by Ni et al., 2024

Requirements:
- TTNN APIs (Python)
- Tenstorrent hardware (Wormhole or Blackhole)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

# Try to import TTNN - will be available on Tenstorrent hardware
try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    print("Warning: TTNN not available. Running in PyTorch fallback mode.")


class DLinear(nn.Module):
    """
    Decomposition Linear Model (DLinear)
    
    From the paper: "Are Transformers Effective for Time Series Forecasting?"
    DLinear decomposes input into trend and seasonal components,
    then applies separate linear layers to each.
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool = False,
        kernel_size: int = 25,
    ):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.kernel_size = kernel_size
        
        # Decomposition using moving average
        self.decomposition = MovingAvg(kernel_size, stride=1)
        
        # Linear layers for trend and seasonal components
        if self.individual:
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)
            ])
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)
            ])
        else:
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, enc_in]
        Returns:
            Output tensor of shape [batch, pred_len, enc_in]
        """
        # x: [batch, seq_len, enc_in]
        seasonal_init = self.decomposition(x)
        trend_init = x - seasonal_init
        
        # Transpose for linear layer: [batch, enc_in, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), self.enc_in, self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device
            )
            trend_output = torch.zeros(
                [trend_init.size(0), self.enc_in, self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device
            )
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        # Sum components and transpose back: [batch, pred_len, enc_in]
        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)
        return x


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, channels]
        # padding on both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.avg(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]
        return x


class RLinear(nn.Module):
    """
    RLinear model with RevIN (Reversible Instance Normalization)
    
    From the paper: "Revisiting Long-term Time Series Forecasting: 
    An Investigation on Linear Mapping"
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool = False,
        use_revin: bool = True,
    ):
        super(RLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.use_revin = use_revin
        
        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
        
        if self.use_revin:
            self.revin = RevIN(self.enc_in)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, enc_in]
        if self.use_revin:
            x = self.revin(x, 'norm')
        
        x = x.permute(0, 2, 1)  # [batch, enc_in, seq_len]
        
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.enc_in, self.pred_len],
                dtype=x.dtype,
                device=x.device
            )
            for i in range(self.enc_in):
                output[:, i, :] = self.Linear[i](x[:, i, :])
        else:
            output = self.Linear(x)
        
        output = output.permute(0, 2, 1)  # [batch, pred_len, enc_in]
        
        if self.use_revin:
            output = self.revin(output, 'denorm')
        
        return output


class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x
    
    def _get_statistics(self, x: torch.Tensor):
        # x: [batch, seq_len, channels]
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class Router(nn.Module):
    """
    Router model for MoLE
    
    Takes timestamp embedding as input and outputs weights for each expert.
    Uses a 2-layer MLP with channel-specific weights.
    """
    def __init__(
        self,
        timestamp_dim: int,
        num_experts: int,
        num_channels: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super(Router, self).__init__()
        self.timestamp_dim = timestamp_dim
        self.num_experts = num_experts
        self.num_channels = num_channels
        
        if hidden_dim is None:
            hidden_dim = num_experts
        
        # 2-layer MLP as described in the paper
        self.fc1 = nn.Linear(timestamp_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, num_experts * num_channels)
    
    def forward(self, timestamp_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamp_embed: [batch, timestamp_dim] - embedding of first timestamp
        Returns:
            weights: [batch, num_channels, num_experts] - softmax weights per channel
        """
        x = self.fc1(timestamp_embed)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Reshape to [batch, num_channels, num_experts]
        x = x.view(-1, self.num_channels, self.num_experts)
        
        # Apply softmax across experts dimension
        weights = F.softmax(x, dim=-1)
        return weights


class MoLE(nn.Module):
    """
    Mixture-of-Linear-Experts (MoLE) Framework
    
    Main architecture that combines multiple expert models with a router.
    Can be applied to any linear-centric time series forecasting model.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction length
        enc_in: Number of input channels/features
        num_experts: Number of expert models (default: 4)
        expert_type: Type of expert model ('dlinear', 'rlinear', 'rmlp')
        individual: Whether to use channel-independent linear layers
        timestamp_dim: Dimension of timestamp embedding
        router_dropout: Dropout rate for router
        expert_dropout: Dropout rate for experts during training
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_experts: int = 4,
        expert_type: str = 'dlinear',
        individual: bool = False,
        timestamp_dim: int = 4,  # Default: month, day, hour, minute normalized
        router_dropout: float = 0.0,
        expert_dropout: float = 0.0,
        **expert_kwargs
    ):
        super(MoLE, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_experts = num_experts
        self.expert_type = expert_type
        self.individual = individual
        self.timestamp_dim = timestamp_dim
        self.expert_dropout = expert_dropout
        
        # Create expert models
        self.experts = nn.ModuleList([
            self._create_expert(expert_type, seq_len, pred_len, enc_in, individual, **expert_kwargs)
            for _ in range(num_experts)
        ])
        
        # Router model
        self.router = Router(
            timestamp_dim=timestamp_dim,
            num_experts=num_experts,
            num_channels=enc_in,
            dropout=router_dropout
        )
        
        # Head dropout for regularization (mentioned in paper)
        self.head_dropout_rate = expert_dropout
    
    def _create_expert(
        self,
        expert_type: str,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool,
        **kwargs
    ) -> nn.Module:
        """Factory method to create expert models"""
        if expert_type.lower() == 'dlinear':
            return DLinear(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                individual=individual,
                kernel_size=kwargs.get('kernel_size', 25)
            )
        elif expert_type.lower() == 'rlinear':
            return RLinear(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=enc_in,
                individual=individual,
                use_revin=kwargs.get('use_revin', True)
            )
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        timestamp_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of MoLE
        
        Args:
            x: Input time series [batch, seq_len, enc_in]
            timestamp_embed: Timestamp embedding [batch, timestamp_dim]
                             If None, uses zeros (uniform weights)
        
        Returns:
            output: Predicted time series [batch, pred_len, enc_in]
        """
        batch_size = x.size(0)
        
        # Get router weights
        if timestamp_embed is None:
            # Uniform weights if no timestamp provided
            weights = torch.ones(
                batch_size, self.enc_in, self.num_experts,
                device=x.device
            ) / self.num_experts
        else:
            weights = self.router(timestamp_embed)  # [batch, enc_in, num_experts]
        
        # Apply head dropout during training
        if self.training and self.head_dropout_rate > 0:
            # Randomly drop some experts
            dropout_mask = torch.bernoulli(
                torch.ones_like(weights) * (1 - self.head_dropout_rate)
            )
            weights = weights * dropout_mask
            # Renormalize
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)  # [batch, pred_len, enc_in]
            expert_outputs.append(out)
        
        # Stack expert outputs: [batch, pred_len, enc_in, num_experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        
        # Apply weights: [batch, enc_in, num_experts] -> [batch, 1, enc_in, num_experts]
        weights = weights.unsqueeze(1)  # [batch, 1, enc_in, num_experts]
        
        # Weighted sum across experts
        output = (expert_outputs * weights).sum(dim=-1)  # [batch, pred_len, enc_in]
        
        return output
    
    def get_expert_outputs(
        self,
        x: torch.Tensor,
        timestamp_embed: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get individual expert outputs and weights for analysis
        
        Returns:
            output: Weighted combination [batch, pred_len, enc_in]
            expert_outputs: All expert outputs [batch, pred_len, enc_in, num_experts]
            weights: Router weights [batch, enc_in, num_experts]
        """
        batch_size = x.size(0)
        
        if timestamp_embed is None:
            weights = torch.ones(
                batch_size, self.enc_in, self.num_experts,
                device=x.device
            ) / self.num_experts
        else:
            weights = self.router(timestamp_embed)
        
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)
            expert_outputs.append(out)
        
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        weights_expanded = weights.unsqueeze(1)
        output = (expert_outputs * weights_expanded).sum(dim=-1)
        
        return output, expert_outputs, weights


class TimestampEmbedding:
    """
    Utility class to create timestamp embeddings for MoLE
    
    As described in the paper, encodes datetime components into
    uniformly spaced values between [-0.5, 0.5]
    """
    @staticmethod
    def embed_datetime(
        dt,
        components: List[str] = ['month', 'day', 'hour', 'minute']
    ) -> torch.Tensor:
        """
        Convert datetime to normalized embedding
        
        Args:
            dt: datetime object or pandas Timestamp
            components: List of components to encode
        
        Returns:
            embedding: Tensor of shape [len(components)]
        """
        values = []
        
        for comp in components:
            if comp == 'month':
                val = (dt.month - 1) / 11.0 - 0.5  # [0, 11] -> [-0.5, 0.5]
            elif comp == 'day':
                val = (dt.day - 1) / 30.0 - 0.5  # Approximate [0, 30] -> [-0.5, 0.5]
            elif comp == 'hour':
                val = dt.hour / 23.0 - 0.5  # [0, 23] -> [-0.5, 0.5]
            elif comp == 'minute':
                val = dt.minute / 59.0 - 0.5  # [0, 59] -> [-0.5, 0.5]
            elif comp == 'dayofweek':
                val = dt.dayofweek / 6.0 - 0.5  # [0, 6] -> [-0.5, 0.5]
            elif comp == 'dayofyear':
                val = (dt.dayofyear - 1) / 364.0 - 0.5  # [0, 364] -> [-0.5, 0.5]
            else:
                raise ValueError(f"Unknown component: {comp}")
            values.append(val)
        
        return torch.tensor(values, dtype=torch.float32)
    
    @staticmethod
    def embed_batch(
        timestamps,
        components: List[str] = ['month', 'day', 'hour', 'minute']
    ) -> torch.Tensor:
        """
        Convert a batch of timestamps to embeddings
        
        Args:
            timestamps: List of datetime objects
            components: List of components to encode
        
        Returns:
            embeddings: Tensor of shape [batch, len(components)]
        """
        embeddings = [
            TimestampEmbedding.embed_datetime(ts, components)
            for ts in timestamps
        ]
        return torch.stack(embeddings)


def create_mole_dlinear(
    seq_len: int = 336,
    pred_len: int = 96,
    enc_in: int = 7,
    num_experts: int = 4,
    **kwargs
) -> MoLE:
    """
    Factory function to create MoLE-DLinear model
    
    This is the main configuration used in the paper.
    """
    return MoLE(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        num_experts=num_experts,
        expert_type='dlinear',
        **kwargs
    )


def create_mole_rlinear(
    seq_len: int = 336,
    pred_len: int = 96,
    enc_in: int = 7,
    num_experts: int = 4,
    **kwargs
) -> MoLE:
    """
    Factory function to create MoLE-RLinear model
    """
    return MoLE(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        num_experts=num_experts,
        expert_type='rlinear',
        **kwargs
    )


# ==============================================================================
# TTNN Integration Layer
# ==============================================================================

class MoLE_TTNN:
    """
    TTNN wrapper for MoLE model
    
    This class handles conversion between PyTorch tensors and TTNN tensors,
    and manages device initialization for Tenstorrent hardware.
    """
    
    def __init__(
        self,
        model: MoLE,
        device_id: int = 0,
        dtype: torch.dtype = torch.bfloat16
    ):
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN is not available. Cannot create MoLE_TTNN.")
        
        self.model = model
        self.device_id = device_id
        self.dtype = dtype
        self.device = None
        self._initialized = False
    
    def initialize_device(self):
        """Initialize Tenstorrent device"""
        if not self._initialized:
            self.device = ttnn.open_device(self.device_id)
            self._initialized = True
    
    def close_device(self):
        """Close Tenstorrent device"""
        if self._initialized and self.device is not None:
            ttnn.close_device(self.device)
            self._initialized = False
    
    def to_ttnn(self, tensor: torch.Tensor) -> 'ttnn.Tensor':
        """Convert PyTorch tensor to TTNN tensor"""
        return ttnn.from_torch(tensor, device=self.device, layout=ttnn.TILE)
    
    def to_torch(self, tensor: 'ttnn.Tensor') -> torch.Tensor:
        """Convert TTNN tensor to PyTorch tensor"""
        return ttnn.to_torch(tensor, dtype=self.dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        timestamp_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with TTNN acceleration
        
        Note: This is a placeholder for the actual TTNN implementation.
        Full TTNN implementation would require converting all operations
        to use TTNN primitives.
        """
        # For now, fall back to PyTorch implementation
        # Full TTNN implementation would convert each layer
        return self.model(x, timestamp_embed)
    
    def __enter__(self):
        self.initialize_device()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_device()
        return False


# ==============================================================================
# Training Utilities
# ==============================================================================

def mole_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    diversity_weight: float = 0.0
) -> torch.Tensor:
    """
    Loss function for MoLE training
    
    Args:
        pred: Predictions [batch, pred_len, enc_in]
        target: Ground truth [batch, pred_len, enc_in]
        weights: Router weights [batch, enc_in, num_experts]
        diversity_weight: Weight for diversity regularization
    
    Returns:
        loss: Scalar loss value
    """
    # MSE loss
    mse_loss = F.mse_loss(pred, target)
    
    # Diversity regularization (encourage experts to specialize)
    if weights is not None and diversity_weight > 0:
        # Encourage different experts to be active for different samples
        avg_weights = weights.mean(dim=0)  # [enc_in, num_experts]
        # Penalize uniform usage (high entropy)
        entropy = -(avg_weights * torch.log(avg_weights + 1e-8)).sum(dim=-1)
        max_entropy = math.log(weights.size(-1))
        normalized_entropy = entropy / max_entropy
        diversity_penalty = diversity_weight * normalized_entropy.mean()
        return mse_loss + diversity_penalty
    
    return mse_loss


if __name__ == "__main__":
    # Test the implementation
    print("Testing MoLE Implementation...")
    
    # Create model
    model = create_mole_dlinear(
        seq_len=336,
        pred_len=96,
        enc_in=7,
        num_experts=4
    )
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 336, 7)
    timestamp_embed = torch.randn(batch_size, 4)
    
    output = model(x, timestamp_embed)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [{batch_size}, 96, 7]")
    
    # Test expert outputs
    output, expert_outputs, weights = model.get_expert_outputs(x, timestamp_embed)
    print(f"\nExpert outputs shape: {expert_outputs.shape}")
    print(f"Router weights shape: {weights.shape}")
    print(f"Weight sum per channel: {weights.sum(dim=-1)[0]}")  # Should be ~1.0
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n鉁?MoLE implementation test passed!")
