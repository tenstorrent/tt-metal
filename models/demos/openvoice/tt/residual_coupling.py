# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Residual Coupling Block for normalizing flows.

Transforms latent representations between speaker identities
using invertible coupling layers.
"""

from typing import Optional, Any, List

import torch
import torch.nn.functional as F

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d
from models.demos.openvoice.tt.modules.wavenet import WaveNetModule


class Flip:
    """
    Flip operation for normalizing flows.

    Reverses the channel dimension to alternate which half of channels
    is transformed in coupling layers.

    Note: Uses CPU roundtrip because TTNN lacks native flip operation.
    Impact is minimal (~0.01ms) as this is a simple memory copy.
    See TRADEOFFS.md section 10.2 for details.
    """

    def __call__(self, x: Any, *args, reverse: bool = False, **kwargs):
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            x = torch.flip(x, [1])
            if not reverse:
                logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                return x, logdet
            return x

        # CPU roundtrip required - TTNN has no native flip operation
        # and slicing with negative step is not supported
        was_on_device = ttnn.is_tensor_storage_on_device(x)
        device = x.device() if was_on_device else None
        orig_layout = x.get_layout()

        x_torch = ttnn.to_torch(x)
        x_flipped = torch.flip(x_torch, [1])
        x = ttnn.from_torch(x_flipped, dtype=ttnn.bfloat16, layout=orig_layout)

        if was_on_device and device is not None:
            x = ttnn.to_device(x, device)

        if not reverse:
            batch = x.shape[0]
            logdet = ttnn.zeros((batch,), dtype=x.dtype)
            return x, logdet
        return x


class ResidualCouplingLayer:
    """
    Single residual coupling layer.

    Splits input into two halves, uses one half to predict
    transformation parameters for the other half.

    Forward: x1' = m + x1 * exp(logs)
    Reverse: x1 = (x1' - m) * exp(-logs)
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = False,
        pre_weight: Any = None,
        pre_bias: Any = None,
        wavenet: Optional[WaveNetModule] = None,
        post_weight: Any = None,
        post_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.device = device

        self.pre_weight = pre_weight
        self.pre_bias = pre_bias
        self.wavenet = wavenet
        self.post_weight = post_weight
        self.post_bias = post_bias

    def __call__(
        self,
        x: Any,
        x_mask: Any,
        g: Optional[Any] = None,
        reverse: bool = False,
    ):
        # Check if input is PyTorch tensor
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask, g, reverse)
        return self._forward_ttnn(x, x_mask, g, reverse)

    def _forward_pytorch(self, x, x_mask, g, reverse):
        # Helper to convert TTNN tensors to PyTorch (and match dtype)
        def to_torch(t, dtype=torch.float32):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.to(dtype) if t.dtype != dtype else t
            if TTNN_AVAILABLE:
                return ttnn.to_torch(t).to(dtype)
            return t

        # Split channels
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)

        # Transform x0 through network to get parameters for x1
        pre_w = to_torch(self.pre_weight)
        pre_w = pre_w.squeeze(2) if pre_w.dim() == 4 else pre_w
        h = F.conv1d(x0, pre_w, to_torch(self.pre_bias))
        h = h * x_mask
        h = self.wavenet(h, x_mask, g=g)
        post_w = to_torch(self.post_weight)
        post_w = post_w.squeeze(2) if post_w.dim() == 4 else post_w
        stats = F.conv1d(h, post_w, to_torch(self.post_bias))
        stats = stats * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            # Forward: x1' = m + x1 * exp(logs)
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            # Reverse: x1 = (x1' - m) * exp(-logs)
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], dim=1)
            return x

    def _forward_ttnn(self, x, x_mask, g, reverse):
        # Split channels
        x0 = x[:, :self.half_channels, :]
        x1 = x[:, self.half_channels:, :]

        # Transform x0 through network
        h = ttnn_conv1d(x0, self.pre_weight, self.pre_bias, device=self.device)
        h = ttnn.multiply(h, x_mask)
        h = self.wavenet(h, x_mask, g=g)
        stats = ttnn_conv1d(h, self.post_weight, self.post_bias, device=self.device)
        stats = ttnn.multiply(stats, x_mask)

        if not self.mean_only:
            m = stats[:, :self.half_channels, :]
            logs = stats[:, self.half_channels:, :]
        else:
            m = stats
            logs = ttnn.zeros_like(m)

        if not reverse:
            # Forward
            exp_logs = ttnn.exp(logs)
            x1_new = ttnn.add(m, ttnn.multiply(x1, exp_logs))
            x1_new = ttnn.multiply(x1_new, x_mask)
            x = ttnn.concat([x0, x1_new], dim=1)
            logdet = ttnn.sum(logs, dim=[1, 2])
            return x, logdet
        else:
            # Reverse
            neg_logs = ttnn.neg(logs)
            exp_neg_logs = ttnn.exp(neg_logs)
            x1_new = ttnn.multiply(ttnn.subtract(x1, m), exp_neg_logs)
            x1_new = ttnn.multiply(x1_new, x_mask)
            x = ttnn.concat([x0, x1_new], dim=1)
            return x

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = True,
        device: Optional[Any] = None,
    ) -> "ResidualCouplingLayer":
        """Create ResidualCouplingLayer from state dict."""

        pre_weight = state_dict.get(f"{prefix}.pre.weight")
        pre_bias = state_dict.get(f"{prefix}.pre.bias")

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

        post_weight = state_dict.get(f"{prefix}.post.weight")
        post_bias = state_dict.get(f"{prefix}.post.bias")

        return cls(
            channels=channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            gin_channels=gin_channels,
            mean_only=mean_only,
            pre_weight=pre_weight,
            pre_bias=pre_bias,
            wavenet=wavenet,
            post_weight=post_weight,
            post_bias=post_bias,
            device=device,
        )


class TTNNResidualCouplingBlock:
    """
    Residual Coupling Block for normalizing flows.

    Contains multiple coupling layers alternating with flip operations.
    Used to transform latent representations between speaker identities.

    Forward: source speaker → latent space
    Reverse: latent space → target speaker
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        flows: Optional[List] = None,
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.device = device

        # flows is a list alternating [CouplingLayer, Flip, CouplingLayer, Flip, ...]
        self.flows = flows or []

    def __call__(
        self,
        x: Any,
        x_mask: Any,
        g: Optional[Any] = None,
        reverse: bool = False,
    ) -> Any:
        """
        Apply flow transformation.

        Args:
            x: Input tensor [B, channels, T]
            x_mask: Mask [B, 1, T]
            g: Speaker conditioning [B, gin_channels, 1]
            reverse: If True, run flows in reverse order

        Returns:
            Transformed tensor [B, channels, T]
        """
        if not reverse:
            # Forward pass
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=False)
        else:
            # Reverse pass
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)

        return x

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        prefix: str,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
        device: Optional[Any] = None,
    ) -> "TTNNResidualCouplingBlock":
        """Create ResidualCouplingBlock from state dict."""

        flows = []
        for i in range(n_flows):
            # Coupling layer
            coupling = ResidualCouplingLayer.from_state_dict(
                state_dict,
                prefix=f"{prefix}.flows.{i * 2}",  # Even indices are coupling layers
                channels=channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                n_layers=n_layers,
                gin_channels=gin_channels,
                mean_only=True,
                device=device,
            )
            flows.append(coupling)

            # Flip
            flows.append(Flip())

        return cls(
            channels=channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            n_flows=n_flows,
            gin_channels=gin_channels,
            flows=flows,
            device=device,
        )
