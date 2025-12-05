# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
AdaLayerNorm implementations for SD3.5 Medium
"""

import ttnn
from .module import Module
from .linear import Linear
from .normalization import LayerNorm
from models.experimental.tt_dit.utils.substate import substate


class AdaLayerNormZero(Module):
    """AdaLayerNormZero with 6x scaling for SD3.5 Medium."""

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        bias: bool = True,
        mesh_device=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.conditioning_size = conditioning_size
        self.mesh_device = mesh_device

        # Linear layer to process conditioning
        self.linear = Linear(hidden_size, conditioning_size, bias=bias, mesh_device=mesh_device)

        # SiLU activation
        self.silu = ttnn.silu

        # Layer norm
        self.norm = LayerNorm(
            hidden_size,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
        )

    def forward(self, x, c):
        """
        Args:
            x: Input tensor [1, B, seq_len, hidden_size]
            c: Conditioning tensor [1, B, hidden_size]
        Returns:
            normalized_x: Normalized input
            scale: Scale tensors for modulation
        """
        # Apply layer norm
        normalized_x = self.norm(x)

        # Process conditioning through linear and SiLU
        c_processed = self.linear(c)
        scale = self.silu(c_processed)

        # Split scale into chunks for attention and feedforward
        # For AdaLayerNormZero, scale is split into 6 parts
        scale = ttnn.reshape(scale, (1, scale.shape[1], 6, self.hidden_size))

        return normalized_x, scale

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.linear.load_torch_state_dict(substate(state_dict, "linear"))


class AdaLayerNormContinuous(Module):
    """AdaLayerNormContinuous with 2x scaling for SD3.5 Medium block 23."""

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        bias: bool = True,
        mesh_device=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.conditioning_size = conditioning_size
        self.mesh_device = mesh_device

        # Linear layer to process conditioning
        self.linear = Linear(hidden_size, conditioning_size, bias=bias, mesh_device=mesh_device)

        # SiLU activation
        self.silu = ttnn.silu

        # Layer norm
        self.norm = LayerNorm(
            hidden_size,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
        )

    def forward(self, x, c):
        """
        Args:
            x: Input tensor [1, B, seq_len, hidden_size]
            c: Conditioning tensor [1, B, hidden_size]
        Returns:
            normalized_x: Normalized input
            scale: Scale tensors for modulation
        """
        # Apply layer norm
        normalized_x = self.norm(x)

        # Process conditioning through linear and SiLU
        c_processed = self.linear(c)
        scale = self.silu(c_processed)

        # Split scale into chunks (2x for continuous)
        scale = ttnn.reshape(scale, (1, scale.shape[1], 2, self.hidden_size))

        return normalized_x, scale

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.linear.load_torch_state_dict(substate(state_dict, "linear"))


class SD35AdaLayerNormZeroX(Module):
    """SD35AdaLayerNormZeroX with 9x scaling for SD3.5 Medium blocks 0-12."""

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int,
        bias: bool = True,
        mesh_device=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.conditioning_size = conditioning_size
        self.mesh_device = mesh_device

        # Linear layer to process conditioning
        self.linear = Linear(hidden_size, conditioning_size, bias=bias, mesh_device=mesh_device)

        # SiLU activation
        self.silu = ttnn.silu

        # Layer norm
        self.norm = LayerNorm(
            hidden_size,
            norm_eps=eps,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
        )

    def forward(self, x, c):
        """
        Args:
            x: Input tensor [1, B, seq_len, hidden_size]
            c: Conditioning tensor [1, B, hidden_size]
        Returns:
            normalized_x: Normalized input
            scale: Scale tensors for modulation
        """
        # Apply layer norm
        normalized_x = self.norm(x)

        # Process conditioning through linear and SiLU
        c_processed = self.linear(c)
        scale = self.silu(c_processed)

        # Split scale into chunks (9x for SD35AdaLayerNormZeroX)
        scale = ttnn.reshape(scale, (1, scale.shape[1], 9, self.hidden_size))

        return normalized_x, scale

    def load_torch_state_dict(self, state_dict):
        """Load weights from PyTorch state dict."""
        self.linear.load_torch_state_dict(substate(state_dict, "linear"))
