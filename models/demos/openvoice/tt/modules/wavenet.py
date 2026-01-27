# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
WaveNet-style dilated convolution module for TTNN.

Used in PosteriorEncoder and ResidualCouplingLayer.
Implements gated activations with dilated convolutions.
"""

from typing import Optional, Any, List

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d, Conv1dLayer


def fused_add_tanh_sigmoid_multiply(
    input_a: Any,
    input_b: Any,
    n_channels: int,
) -> Any:
    """
    Fused operation: tanh(a[:n]) * sigmoid(a[n:]) where a = input_a + input_b

    This is the core gating mechanism in WaveNet.

    Args:
        input_a: First input tensor [B, 2*n_channels, L]
        input_b: Second input tensor [B, 2*n_channels, L]
        n_channels: Number of channels for split

    Returns:
        Gated output [B, n_channels, L]
    """
    # Check if inputs are PyTorch tensors
    is_torch = isinstance(input_a, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch:
        in_act = input_a + input_b
        t_act = torch.tanh(in_act[:, :n_channels, :])
        s_act = torch.sigmoid(in_act[:, n_channels:, :])
        return t_act * s_act

    # TTNN implementation
    in_act = ttnn.add(input_a, input_b)

    # Split channels
    t_input = in_act[:, :n_channels, :]
    s_input = in_act[:, n_channels:, :]

    # Apply activations
    t_act = ttnn.tanh(t_input)
    s_act = ttnn.sigmoid(s_input)

    # Multiply
    return ttnn.multiply(t_act, s_act)


class WaveNetModule:
    """
    WaveNet-style dilated convolution module.

    Architecture:
        For each layer i:
            1. Dilated conv with dilation = dilation_rate^i
            2. Add conditioning (if provided)
            3. Gated activation: tanh(.) * sigmoid(.)
            4. Residual + skip connection

    Used in:
        - PosteriorEncoder (enc_q)
        - ResidualCouplingLayer
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
        in_layer_weights: Optional[List[Any]] = None,
        in_layer_biases: Optional[List[Any]] = None,
        res_skip_weights: Optional[List[Any]] = None,
        res_skip_biases: Optional[List[Any]] = None,
        cond_layer_weight: Optional[Any] = None,
        cond_layer_bias: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        """
        Initialize WaveNet module.

        Args:
            hidden_channels: Number of hidden channels
            kernel_size: Convolution kernel size
            dilation_rate: Base dilation rate (dilation = dilation_rate^layer)
            n_layers: Number of layers
            gin_channels: Conditioning channels (0 = no conditioning)
            p_dropout: Dropout probability (ignored in inference)
            in_layer_weights: List of dilated conv weights
            in_layer_biases: List of dilated conv biases
            res_skip_weights: List of residual/skip conv weights
            res_skip_biases: List of residual/skip conv biases
            cond_layer_weight: Conditioning layer weight
            cond_layer_bias: Conditioning layer bias
            device: TTNN device
        """
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.device = device

        # Store weights
        self.in_layer_weights = in_layer_weights or []
        self.in_layer_biases = in_layer_biases or []
        self.res_skip_weights = res_skip_weights or []
        self.res_skip_biases = res_skip_biases or []
        self.cond_layer_weight = cond_layer_weight
        self.cond_layer_bias = cond_layer_bias

    def __call__(
        self,
        x: Any,
        x_mask: Any,
        g: Optional[Any] = None,
    ) -> Any:
        """
        Forward pass.

        Args:
            x: Input tensor [B, hidden_channels, L]
            x_mask: Mask tensor [B, 1, L]
            g: Conditioning tensor [B, gin_channels, 1] (optional)

        Returns:
            Output tensor [B, hidden_channels, L]
        """
        # Check if input is PyTorch tensor
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, g)
        return self._forward_ttnn(x, x_mask, g)

    def _forward_pytorch(self, x, x_mask, g):
        """PyTorch fallback implementation."""
        # Helper to convert TTNN tensors to PyTorch (and match dtype)
        def to_torch(t, dtype=torch.float32):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.to(dtype) if t.dtype != dtype else t
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t).to(dtype)
            return t

        output = torch.zeros_like(x)

        # Process conditioning
        if g is not None and self.gin_channels > 0 and self.cond_layer_weight is not None:
            cond_w = to_torch(self.cond_layer_weight)
            cond_w = cond_w.squeeze(2) if cond_w.dim() == 4 else cond_w
            cond_b = to_torch(self.cond_layer_bias) if self.cond_layer_bias is not None else None
            g = F.conv1d(g, cond_w, cond_b)

        for i in range(self.n_layers):
            dilation = self.dilation_rate ** i
            padding = int((self.kernel_size * dilation - dilation) / 2)

            # Dilated convolution
            weight = to_torch(self.in_layer_weights[i])
            weight = weight.squeeze(2) if weight.dim() == 4 else weight  # [2H, H, K]
            x_in = F.conv1d(x, weight, to_torch(self.in_layer_biases[i]), padding=padding, dilation=dilation)

            # Add conditioning
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2*self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            # Gated activation
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)

            # Residual + skip
            res_skip_weight = to_torch(self.res_skip_weights[i])
            res_skip_weight = res_skip_weight.squeeze(2) if res_skip_weight.dim() == 4 else res_skip_weight
            res_skip_acts = F.conv1d(acts, res_skip_weight, to_torch(self.res_skip_biases[i]))

            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def _forward_ttnn(self, x, x_mask, g):
        """TTNN implementation."""
        output = ttnn.zeros_like(x)

        # Process conditioning
        if g is not None and self.gin_channels > 0 and self.cond_layer_weight is not None:
            g = ttnn_conv1d(g, self.cond_layer_weight, self.cond_layer_bias, device=self.device)

        for i in range(self.n_layers):
            dilation = self.dilation_rate ** i
            padding = int((self.kernel_size * dilation - dilation) / 2)

            # Dilated convolution
            x_in = ttnn_conv1d(
                x,
                self.in_layer_weights[i],
                self.in_layer_biases[i],
                padding=padding,
                dilation=dilation,
                device=self.device,
            )

            # Add conditioning
            if g is not None and self.gin_channels > 0:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2*self.hidden_channels, :]
            else:
                g_l = ttnn.zeros_like(x_in)

            # Gated activation
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)

            # Residual + skip connection
            res_skip_acts = ttnn_conv1d(
                acts,
                self.res_skip_weights[i],
                self.res_skip_biases[i],
                device=self.device,
            )

            if i < self.n_layers - 1:
                # Split into residual and skip
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                skip_acts = res_skip_acts[:, self.hidden_channels:, :]

                # Update x with residual
                x = ttnn.add(x, res_acts)
                x = ttnn.multiply(x, x_mask)

                # Accumulate skip
                output = ttnn.add(output, skip_acts)
            else:
                # Last layer: all goes to output
                output = ttnn.add(output, res_skip_acts)

        # Apply final mask
        return ttnn.multiply(output, x_mask)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        device: Optional[Any] = None,
    ) -> "WaveNetModule":
        """
        Create WaveNet module from a state dict.

        Args:
            state_dict: Model state dict
            prefix: Key prefix (e.g., "enc_q.enc")
            hidden_channels: Number of hidden channels
            kernel_size: Kernel size
            dilation_rate: Dilation rate
            n_layers: Number of layers
            gin_channels: Conditioning channels
            device: TTNN device

        Returns:
            Initialized WaveNetModule
        """
        def get_weight(prefix_key):
            """Get weight, handling weight normalization (weight_g, weight_v)."""
            # Try direct weight first
            w = state_dict.get(f"{prefix_key}.weight")
            if w is not None:
                return w

            # Try weight normalization format
            g = state_dict.get(f"{prefix_key}.weight_g")
            v = state_dict.get(f"{prefix_key}.weight_v")
            if g is not None and v is not None:
                # Compute normalized weight: w = g * (v / ||v||)
                # v shape: [out, in, kernel], g shape: [out, 1, 1]
                v_norm = v / (torch.norm(v, dim=(1, 2), keepdim=True) + 1e-7)
                return g * v_norm

            return None

        in_layer_weights = []
        in_layer_biases = []
        res_skip_weights = []
        res_skip_biases = []

        for i in range(n_layers):
            # In layer weights (with weight normalization)
            in_layer_weights.append(get_weight(f"{prefix}.in_layers.{i}"))
            in_layer_biases.append(state_dict.get(f"{prefix}.in_layers.{i}.bias"))

            # Res skip weights (with weight normalization)
            res_skip_weights.append(get_weight(f"{prefix}.res_skip_layers.{i}"))
            res_skip_biases.append(state_dict.get(f"{prefix}.res_skip_layers.{i}.bias"))

        # Conditioning layer (with weight normalization)
        cond_weight = get_weight(f"{prefix}.cond_layer")
        cond_bias = state_dict.get(f"{prefix}.cond_layer.bias")

        return cls(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            gin_channels=gin_channels,
            in_layer_weights=in_layer_weights,
            in_layer_biases=in_layer_biases,
            res_skip_weights=res_skip_weights,
            res_skip_biases=res_skip_biases,
            cond_layer_weight=cond_weight,
            cond_layer_bias=cond_bias,
            device=device,
        )
