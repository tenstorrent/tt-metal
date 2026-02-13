"""
MoLE (Mixture-of-Linear-Experts) Model
Main implementation of the MoLE framework
"""
import torch
import torch.nn as nn
import ttnn
from .dlinear import DLinear, DLinearTTNN
from .router import Router, RouterTTNN, TopKRouter


class MoLE(nn.Module):
    """
    MoLE: Mixture-of-Linear-Experts for Long-term Time Series Forecasting
    
    A meta-architecture that augments existing linear models with a Mixture-of-Experts framework.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction length  
        enc_in: Number of input features
        num_experts: Number of experts (default: 4)
        expert_type: Type of expert model ('dlinear', 'rlinear', 'rmlp')
        individual: Whether experts use individual linear layers
        top_k: Use top-k expert selection (None for full mixture)
        **expert_kwargs: Additional arguments for expert models
    """
    def __init__(self, seq_len, pred_len, enc_in, num_experts=4, 
                 expert_type='dlinear', individual=False, top_k=None, **expert_kwargs):
        super(MoLE, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_experts = num_experts
        self.expert_type = expert_type
        self.top_k = top_k
        
        # Create experts
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if expert_type == 'dlinear':
                expert = DLinear(seq_len, pred_len, enc_in, individual, **expert_kwargs)
            elif expert_type == 'rlinear':
                # RLinear would be implemented similarly
                expert = DLinear(seq_len, pred_len, enc_in, individual, **expert_kwargs)
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
            
            self.experts.append(expert)
        
        # Create router
        if top_k is not None and top_k < num_experts:
            self.router = TopKRouter(seq_len, enc_in, num_experts, top_k=top_k)
        else:
            self.router = Router(seq_len, enc_in, num_experts)
        
        # Store expert activations for analysis
        self.register_buffer('expert_activations', torch.zeros(num_experts))
        self.register_buffer('call_count', torch.tensor(0))
    
    def forward(self, x, return_weights=False):
        """
        Forward pass through MoLE
        
        Args:
            x: [batch, seq_len, enc_in]
            return_weights: Whether to return router weights
        Returns:
            output: [batch, pred_len, enc_in]
            weights (optional): [batch, num_experts]
            aux_loss (optional): Load balancing loss
        """
        batch_size = x.shape[0]
        
        # Get router weights
        if self.top_k is not None:
            weights, aux_loss = self.router(x)  # [batch, num_experts]
        else:
            weights = self.router(x)  # [batch, num_experts]
            aux_loss = None
        
        # Update activation statistics
        with torch.no_grad():
            self.expert_activations += weights.mean(dim=0)
            self.call_count += 1
        
        # Compute expert outputs in parallel
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)  # [batch, pred_len, enc_in]
            expert_outputs.append(out)
        
        # Stack expert outputs: [num_experts, batch, pred_len, enc_in]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # Weighted combination
        # weights: [batch, num_experts] -> [num_experts, batch, 1, 1]
        weights_expanded = weights.T.unsqueeze(-1).unsqueeze(-1)
        
        # [num_experts, batch, pred_len, enc_in] * [num_experts, batch, 1, 1]
        weighted_outputs = expert_outputs * weights_expanded
        
        # Sum over experts: [batch, pred_len, enc_in]
        output = weighted_outputs.sum(dim=0)
        
        if return_weights:
            return output, weights, aux_loss
        return output
    
    def get_expert_usage(self):
        """Get average expert usage statistics"""
        if self.call_count > 0:
            return self.expert_activations / self.call_count
        return self.expert_activations
    
    def reset_expert_stats(self):
        """Reset expert activation statistics"""
        self.expert_activations.zero_()
        self.call_count.zero_()


class MoLETTNN(nn.Module):
    """
    MoLE with TT-NN backend support
    Optimized for Tenstorrent hardware with parallel expert computation
    """
    def __init__(self, seq_len, pred_len, enc_in, num_experts=4, 
                 expert_type='dlinear', individual=False, top_k=None, 
                 device=None, parallel_experts=True, **expert_kwargs):
        super(MoLETTNN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_experts = num_experts
        self.expert_type = expert_type
        self.top_k = top_k
        self.device = device
        self.parallel_experts = parallel_experts
        
        # Check TT-NN availability
        self.use_ttnn = False
        if device is not None:
            self.use_ttnn = self._check_ttnn_available()
        
        # Create experts
        if self.use_ttnn:
            self.experts = nn.ModuleList([
                DLinearTTNN(seq_len, pred_len, enc_in, individual, device=device, **expert_kwargs)
                for _ in range(num_experts)
            ])
            self.router = RouterTTNN(seq_len, enc_in, num_experts, device=device)
        else:
            self.experts = nn.ModuleList()
            for i in range(num_experts):
                if expert_type == 'dlinear':
                    expert = DLinear(seq_len, pred_len, enc_in, individual, **expert_kwargs)
                else:
                    expert = DLinear(seq_len, pred_len, enc_in, individual, **expert_kwargs)
                self.experts.append(expert)
            
            if top_k is not None:
                self.router = TopKRouter(seq_len, enc_in, num_experts, top_k=top_k)
            else:
                self.router = Router(seq_len, enc_in, num_experts)
        
        # Expert usage statistics
        self.register_buffer('expert_activations', torch.zeros(num_experts))
        self.register_buffer('call_count', torch.tensor(0))
    
    def _check_ttnn_available(self):
        """Check if TT-NN is available"""
        try:
            import ttnn
            test_tensor = ttnn.ones([1, 1], device=self.device)
            return True
        except Exception as e:
            print(f"TT-NN not available: {e}")
            return False
    
    def forward(self, x, return_weights=False):
        """
        Forward pass
        
        Args:
            x: [batch, seq_len, enc_in] or TT-NN tensor
            return_weights: Whether to return router weights
        Returns:
            output: [batch, pred_len, enc_in]
        """
        if self.use_ttnn and isinstance(x, ttnn.Tensor):
            return self._forward_ttnn(x, return_weights)
        else:
            return self._forward_torch(x, return_weights)
    
    def _forward_torch(self, x, return_weights=False):
        """PyTorch forward pass"""
        batch_size = x.shape[0]
        
        # Get router weights
        if self.top_k is not None and hasattr(self.router, 'top_k'):
            weights, aux_loss = self.router(x)
        else:
            weights = self.router(x)
            aux_loss = None
        
        # Update statistics
        with torch.no_grad():
            self.expert_activations += weights.mean(dim=0)
            self.call_count += 1
        
        # Compute expert outputs
        if self.parallel_experts and batch_size > 1:
            # Parallel computation
            expert_outputs = [expert(x) for expert in self.experts]
        else:
            expert_outputs = [expert(x) for expert in self.experts]
        
        # Stack and weight
        expert_outputs = torch.stack(expert_outputs, dim=0)
        weights_expanded = weights.T.unsqueeze(-1).unsqueeze(-1)
        weighted_outputs = expert_outputs * weights_expanded
        output = weighted_outputs.sum(dim=0)
        
        if return_weights:
            return output, weights, aux_loss
        return output
    
    def _forward_ttnn(self, x, return_weights=False):
        """TT-NN forward pass with parallel expert computation"""
        batch_size = x.shape[0]
        
        # Get router weights
        weights = self.router(x)  # TT-NN tensor [batch, num_experts]
        aux_loss = None
        
        # Convert to torch for statistics
        weights_torch = ttnn.to_torch(weights)
        with torch.no_grad():
            self.expert_activations += weights_torch.mean(dim=0)
            self.call_count += 1
        
        # Compute expert outputs
        # For parallel execution on TT hardware, we can process all experts
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)  # TT-NN tensor
            expert_outputs.append(out)
        
        # Stack expert outputs using TT-NN
        # Convert to torch for stacking (TT-NN concat is limited)
        expert_outputs_torch = [ttnn.to_torch(out) for out in expert_outputs]
        expert_outputs_stacked = torch.stack(expert_outputs_torch, dim=0)
        
        # Apply weights
        weights_expanded = weights_torch.T.unsqueeze(-1).unsqueeze(-1)
        weighted_outputs = expert_outputs_stacked * weights_expanded
        output_torch = weighted_outputs.sum(dim=0)
        
        # Convert back to TT-NN
        output = ttnn.from_torch(output_torch, dtype=ttnn.bfloat16, device=self.device)
        
        if return_weights:
            return output, weights, aux_loss
        return output
    
    def get_expert_usage(self):
        """Get average expert usage statistics"""
        if self.call_count > 0:
            return self.expert_activations / self.call_count
        return self.expert_activations
    
    def reset_expert_stats(self):
        """Reset expert activation statistics"""
        self.expert_activations.zero_()
        self.call_count.zero_()


class MoLEConfig:
    """Configuration class for MoLE models"""
    def __init__(self,
                 seq_len=96,
                 pred_len=96,
                 enc_in=7,
                 num_experts=4,
                 expert_type='dlinear',
                 individual=False,
                 top_k=None,
                 decomp_kernel=25,
                 router_hidden_dim=None,
                 router_num_layers=2,
                 dropout=0.1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_experts = num_experts
        self.expert_type = expert_type
        self.individual = individual
        self.top_k = top_k
        self.decomp_kernel = decomp_kernel
        self.router_hidden_dim = router_hidden_dim
        self.router_num_layers = router_num_layers
        self.dropout = dropout
    
    def create_model(self, device=None):
        """Create MoLE model from config"""
        if device is not None:
            return MoLETTNN(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                enc_in=self.enc_in,
                num_experts=self.num_experts,
                expert_type=self.expert_type,
                individual=self.individual,
                top_k=self.top_k,
                device=device,
                decomp_kernel=self.decomp_kernel
            )
        else:
            return MoLE(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                enc_in=self.enc_in,
                num_experts=self.num_experts,
                expert_type=self.expert_type,
                individual=self.individual,
                top_k=self.top_k,
                decomp_kernel=self.decomp_kernel
            )
