# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Posterior Encoder for TTNN.

Encodes mel spectrograms into latent representations using WaveNet.
Used in voice conversion to encode source audio.
"""

from typing import Optional, Any

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d
from models.demos.openvoice.tt.modules.wavenet import WaveNetModule


def get_weight_from_state_dict(state_dict: dict, prefix_key: str):
    """Get weight from state dict, handling weight normalization."""
    w = state_dict.get(f"{prefix_key}.weight")
    if w is not None:
        return w

    g = state_dict.get(f"{prefix_key}.weight_g")
    v = state_dict.get(f"{prefix_key}.weight_v")
    if g is not None and v is not None:
        dims = tuple(range(1, v.dim()))
        v_norm = v / (torch.norm(v, dim=dims, keepdim=True) + 1e-7)
        return g * v_norm

    return None


def sequence_mask(length: Any, max_length: Optional[int] = None, device: Any = None) -> Any:
    """
    Create a sequence mask from lengths.

    Args:
        length: Tensor of lengths [B]
        max_length: Maximum length (default: max of length tensor)
        device: TTNN device for placing tensors

    Returns:
        Boolean mask [B, max_length]
    """
    # Check if input is PyTorch tensor - use isinstance for reliability
    is_torch = isinstance(length, torch.Tensor)

    if not TTNN_AVAILABLE or is_torch:
        if max_length is None:
            max_length = int(length.max().item())
        tensor_device = length.device if hasattr(length, 'device') else 'cpu'
        x = torch.arange(max_length, dtype=length.dtype, device=tensor_device)
        return x.unsqueeze(0) < length.unsqueeze(1)

    # TTNN implementation - tensors must be on device
    if max_length is None:
        # Move length to host to get max value
        length_host = ttnn.to_torch(length)
        max_length = int(length_host.max().item())

    # Create mask using PyTorch and convert to TTNN
    # This is more reliable than creating TTNN tensors from scratch
    batch = length.shape[0] if hasattr(length, 'shape') else 1
    length_host = ttnn.to_torch(length) if not isinstance(length, torch.Tensor) else length

    x = torch.arange(max_length, dtype=length_host.dtype)
    mask = x.unsqueeze(0) < length_host.unsqueeze(1)

    # Convert to TTNN if device provided
    if device is not None:
        mask = ttnn.from_torch(mask.float(), dtype=ttnn.bfloat16, device=device)

    return mask


class TTNNPosteriorEncoder:
    """
    Posterior Encoder (enc_q).

    Encodes mel spectrograms to latent space using:
        1. Pre-conv to hidden channels
        2. WaveNet for temporal modeling
        3. Projection to mean and log-variance

    Used for encoding source audio in voice conversion.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int,
        pre_weight: Any,
        pre_bias: Any,
        wavenet: WaveNetModule,
        proj_weight: Any,
        proj_bias: Any,
        device: Optional[Any] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.device = device

        # Pre-conv
        self.pre_weight = pre_weight
        self.pre_bias = pre_bias

        # WaveNet encoder
        self.wavenet = wavenet

        # Projection to mean + log-variance
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias

    def __call__(
        self,
        x: Any,
        x_lengths: Any,
        g: Optional[Any] = None,
        tau: float = 1.0,
    ) -> tuple:
        """
        Encode mel spectrogram to latent representation.

        Args:
            x: Mel spectrogram [B, in_channels, T]
            x_lengths: Lengths tensor [B]
            g: Speaker conditioning [B, gin_channels, 1]
            tau: Temperature for sampling (1.0 = standard)

        Returns:
            Tuple of (z, m, logs, x_mask):
                z: Sampled latent [B, out_channels, T]
                m: Mean [B, out_channels, T]
                logs: Log variance [B, out_channels, T]
                x_mask: Sequence mask [B, 1, T]
        """
        # Check if input is PyTorch tensor - use isinstance for reliability
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_lengths, g, tau)
        return self._forward_ttnn(x, x_lengths, g, tau)

    def _forward_pytorch(self, x, x_lengths, g, tau):
        # Helper to convert TTNN tensors to PyTorch (and match dtype)
        def to_torch(t, dtype=torch.float32):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.to(dtype) if t.dtype != dtype else t
            # TTNN tensor - convert to PyTorch
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t).to(dtype)
            return t

        # Create mask [B, 1, T]
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        # Pre-conv - convert weights from TTNN if needed
        pre_w = to_torch(self.pre_weight)
        pre_b = to_torch(self.pre_bias)
        w = pre_w.squeeze(2) if pre_w.dim() == 4 else pre_w
        x = F.conv1d(x, w, pre_b)
        x = x * x_mask

        # WaveNet
        x = self.wavenet(x, x_mask, g=g)

        # Project to mean + log-variance
        proj_w = to_torch(self.proj_weight)
        proj_b = to_torch(self.proj_bias)
        w = proj_w.squeeze(2) if proj_w.dim() == 4 else proj_w
        stats = F.conv1d(x, w, proj_b)
        stats = stats * x_mask

        # Split into mean and log-variance
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # Sample z = m + randn * exp(logs) * tau
        z = m + torch.randn_like(m) * tau * torch.exp(logs)
        z = z * x_mask

        return z, m, logs, x_mask

    def _forward_ttnn(self, x, x_lengths, g, tau):
        # Create mask [B, 1, T]
        seq_len = x.shape[2]
        x_mask = sequence_mask(x_lengths, seq_len, device=self.device)
        x_mask = ttnn.unsqueeze(x_mask, 1)  # [B, 1, T]
        x_mask = ttnn.to_dtype(x_mask, x.dtype)
        # Convert mask to TILE layout for compatibility with conv output
        x_mask = ttnn.to_layout(x_mask, ttnn.TILE_LAYOUT)

        # Pre-conv
        x = ttnn_conv1d(x, self.pre_weight, self.pre_bias, device=self.device)
        x = ttnn.multiply(x, x_mask)

        # WaveNet
        x = self.wavenet(x, x_mask, g=g)

        # Project to mean + log-variance
        stats = ttnn_conv1d(x, self.proj_weight, self.proj_bias, device=self.device)
        stats = ttnn.multiply(stats, x_mask)

        # Split into mean and log-variance
        m = stats[:, :self.out_channels, :]
        logs = stats[:, self.out_channels:, :]

        # Sample z
        # Generate random noise on host and transfer
        batch, _, length = m.shape
        noise = torch.randn(batch, self.out_channels, length)
        noise_tt = ttnn.from_torch(noise, dtype=m.dtype, device=self.device)

        # z = m + noise * tau * exp(logs)
        scaled_noise = ttnn.multiply(noise_tt, tau)
        exp_logs = ttnn.exp(logs)
        scaled_noise = ttnn.multiply(scaled_noise, exp_logs)
        z = ttnn.add(m, scaled_noise)
        z = ttnn.multiply(z, x_mask)

        return z, m, logs, x_mask

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        device: Optional[Any] = None,
    ) -> "TTNNPosteriorEncoder":
        """Create PosteriorEncoder from state dict."""

        # Pre-conv (with weight normalization support)
        pre_weight = get_weight_from_state_dict(state_dict, f"{prefix}.pre")
        pre_bias = state_dict.get(f"{prefix}.pre.bias")

        # WaveNet
        wavenet = WaveNetModule.from_state_dict(
            state_dict,
            prefix=f"{prefix}.enc",
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            gin_channels=gin_channels,
            device=device,
        )

        # Projection (with weight normalization support)
        proj_weight = get_weight_from_state_dict(state_dict, f"{prefix}.proj")
        proj_bias = state_dict.get(f"{prefix}.proj.bias")

        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            gin_channels=gin_channels,
            pre_weight=pre_weight,
            pre_bias=pre_bias,
            wavenet=wavenet,
            proj_weight=proj_weight,
            proj_bias=proj_bias,
            device=device,
        )
