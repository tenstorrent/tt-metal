"""
DLinear: Decomposition Linear Model for Time Series Forecasting
Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023
"""
import torch
import torch.nn as nn
import ttnn


class DLinear(nn.Module):
    """
    DLinear model with seasonal-trend decomposition.
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction length
        enc_in: Number of input features
        individual: Whether to use individual linear layers per feature
        decomp_kernel: Kernel size for moving average decomposition
    """
    def __init__(self, seq_len, pred_len, enc_in, individual=False, decomp_kernel=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        
        # Decomposition
        self.decomposition = SeriesDecomp(decomp_kernel)
        
        # Linear layers for seasonal and trend components
        if individual:
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, enc_in]
        Returns:
            forecast: [batch, pred_len, enc_in]
        """
        # Decomposition
        seasonal_init, trend_init = self.decomposition(x)  # both [batch, seq_len, enc_in]
        
        if self.individual:
            seasonal_output = torch.zeros_like(seasonal_init[:, -self.pred_len:, :])
            trend_output = torch.zeros_like(trend_init[:, -self.pred_len:, :])
            
            for i in range(self.enc_in):
                seasonal_output[:, :, i] = self.Linear_Seasonal[i](seasonal_init[:, :, i])
                trend_output[:, :, i] = self.Linear_Trend[i](trend_init[:, :, i])
            
            x = seasonal_output + trend_output
        else:
            # [batch, enc_in, seq_len] -> [batch, enc_in, pred_len]
            seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1))
            seasonal_output = seasonal_output.permute(0, 2, 1)
            
            trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1))
            trend_output = trend_output.permute(0, 2, 1)
            
            x = seasonal_output + trend_output
        
        # Return prediction
        return x[:, -self.pred_len:, :]


class SeriesDecomp(nn.Module):
    """
    Series decomposition block using moving average
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, channels]
        Returns:
            seasonal: [batch, seq_len, channels]
            trend: [batch, seq_len, channels]
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MovingAvg(nn.Module):
    """
    Moving average block
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, channels]
        Returns:
            smoothed: [batch, seq_len, channels]
        """
        # Padding on both ends
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        # Apply moving average
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.avg(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]
        return x


class DLinearTTNN(nn.Module):
    """
    DLinear model with TT-NN backend support
    """
    def __init__(self, seq_len, pred_len, enc_in, individual=False, decomp_kernel=25, device=None):
        super(DLinearTTNN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.device = device
        
        # Store model for CPU fallback
        self.dlinear = DLinear(seq_len, pred_len, enc_in, individual, decomp_kernel)
        self.use_ttnn = False
        
        if device is not None and self._check_ttnn_available():
            self.use_ttnn = True
            self._init_ttnn_layers()
    
    def _check_ttnn_available(self):
        """Check if TT-NN is available and device is ready"""
        try:
            import ttnn
            # Check if we can create a tensor
            test_tensor = ttnn.ones([1, 1], device=self.device)
            return True
        except Exception as e:
            print(f"TT-NN not available, using PyTorch fallback: {e}")
            return False
    
    def _init_ttnn_layers(self):
        """Initialize TT-NN layers"""
        import ttnn
        
        # Convert PyTorch linear layers to TT-NN format
        if self.individual:
            self.ttnn_linears_seasonal = []
            self.ttnn_linears_trend = []
            for i in range(self.enc_in):
                # Extract weights and biases
                w_seasonal = self.dlinear.Linear_Seasonal[i].weight.data
                b_seasonal = self.dlinear.Linear_Seasonal[i].bias.data
                w_trend = self.dlinear.Linear_Trend[i].weight.data
                b_trend = self.dlinear.Linear_Trend[i].bias.data
                
                # Create TT-NN tensors
                self.ttnn_linears_seasonal.append({
                    'weight': ttnn.from_torch(w_seasonal, dtype=ttnn.bfloat16, device=self.device),
                    'bias': ttnn.from_torch(b_seasonal, dtype=ttnn.bfloat16, device=self.device)
                })
                self.ttnn_linears_trend.append({
                    'weight': ttnn.from_torch(w_trend, dtype=ttnn.bfloat16, device=self.device),
                    'bias': ttnn.from_torch(b_trend, dtype=ttnn.bfloat16, device=self.device)
                })
        else:
            w_seasonal = self.dlinear.Linear_Seasonal.weight.data
            b_seasonal = self.dlinear.Linear_Seasonal.bias.data
            w_trend = self.dlinear.Linear_Trend.weight.data
            b_trend = self.dlinear.Linear_Trend.bias.data
            
            self.ttnn_linear_seasonal = {
                'weight': ttnn.from_torch(w_seasonal, dtype=ttnn.bfloat16, device=self.device),
                'bias': ttnn.from_torch(b_seasonal, dtype=ttnn.bfloat16, device=self.device)
            }
            self.ttnn_linear_trend = {
                'weight': ttnn.from_torch(w_trend, dtype=ttnn.bfloat16, device=self.device),
                'bias': ttnn.from_torch(b_trend, dtype=ttnn.bfloat16, device=self.device)
            }
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, enc_in] or TT-NN tensor
        Returns:
            forecast: [batch, pred_len, enc_in] or TT-NN tensor
        """
        if not self.use_ttnn or not isinstance(x, ttnn.Tensor):
            # Fallback to PyTorch
            if isinstance(x, ttnn.Tensor):
                x = ttnn.to_torch(x)
            return self.dlinear(x)
        
        # TT-NN forward pass
        return self._forward_ttnn(x)
    
    def _forward_ttnn(self, x):
        """TT-NN forward pass"""
        # Convert to torch for decomposition (TT-NN doesn't have moving avg yet)
        x_torch = ttnn.to_torch(x)
        seasonal_init, trend_init = self.dlinear.decomposition(x_torch)
        
        # Convert back to TT-NN
        seasonal_ttnn = ttnn.from_torch(seasonal_init, dtype=ttnn.bfloat16, device=self.device)
        trend_ttnn = ttnn.from_torch(trend_init, dtype=ttnn.bfloat16, device=self.device)
        
        if self.individual:
            outputs = []
            for i in range(self.enc_in):
                # Get feature slice
                seasonal_i = ttnn.slice(seasonal_ttnn, [0, 0, i], [seasonal_ttnn.shape[0], seasonal_ttnn.shape[1], i+1])
                trend_i = ttnn.slice(trend_ttnn, [0, 0, i], [trend_ttnn.shape[0], trend_ttnn.shape[1], i+1])
                
                # Linear operations
                seasonal_out = ttnn.linear(seasonal_i, self.ttnn_linears_seasonal[i]['weight'], 
                                          bias=self.ttnn_linears_seasonal[i]['bias'])
                trend_out = ttnn.linear(trend_i, self.ttnn_linears_trend[i]['weight'],
                                       bias=self.ttnn_linears_trend[i]['bias'])
                
                outputs.append(ttnn.add(seasonal_out, trend_out))
            
            # Concatenate outputs
            x = ttnn.concat(outputs, dim=2)
        else:
            # Permute for linear: [batch, seq_len, enc_in] -> [batch, enc_in, seq_len]
            seasonal_ttnn = ttnn.permute(seasonal_ttnn, (0, 2, 1))
            trend_ttnn = ttnn.permute(trend_ttnn, (0, 2, 1))
            
            # Apply linear layers
            seasonal_out = ttnn.linear(seasonal_ttnn, self.ttnn_linear_seasonal['weight'],
                                      bias=self.ttnn_linear_seasonal['bias'])
            trend_out = ttnn.linear(trend_ttnn, self.ttnn_linear_trend['weight'],
                                   bias=self.ttnn_linear_trend['bias'])
            
            # Add and permute back
            x = ttnn.add(seasonal_out, trend_out)
            x = ttnn.permute(x, (0, 2, 1))
        
        return x
