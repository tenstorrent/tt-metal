# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Building block components for mLSTM and other architectures.

This module provides reusable components including:
- CausalConv1d: Causal 1D convolution for temporal sequences
- MultiHeadLayerNorm: Layer normalization for multi-head tensors
- LinearHeadwiseExpand: Headwise linear projection layer
"""

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple

import numpy as np
import ttnn

import ttml
from ttml.modules import AbstractModuleBase, Parameter


@dataclass
class CausalConv1dConfig:
    """Configuration for CausalConv1d."""

    feature_dim: int
    kernel_size: int = 4
    bias: bool = True


class CausalConv1d(AbstractModuleBase):
    """Causal 1D depthwise convolution.

    Implements causal depthwise convolution of a time series tensor.
    Input: Tensor of shape (B, S, D), i.e. (batch, sequence, features)
    Output: Tensor of shape (B, S, D)

    Args:
        config: CausalConv1dConfig with feature_dim, kernel_size, bias
    """

    def __init__(self, config: CausalConv1dConfig) -> None:
        super().__init__()
        self.config = config

        if config.kernel_size == 0:
            self.weight = None
            self.bias_param = None
        else:
            # Weight shape: (kernel_size, feature_dim) for depthwise conv
            # Initialize with uniform distribution
            k = config.kernel_size
            d = config.feature_dim
            bound = 1.0 / sqrt(k)

            weight_data = np.random.uniform(-bound, bound, (k, d)).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(weight_data))

            if config.bias:
                bias_data = np.zeros((d,), dtype=np.float32)
                self.bias_param = Parameter(ttml.autograd.Tensor.from_numpy(bias_data))
            else:
                self.bias_param = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of causal conv1d.

        Args:
            x: Input tensor of shape (B, S, D)

        Returns:
            Output tensor of shape (B, S, D)
        """
        if self.config.kernel_size == 0:
            return x

        # Simple implementation using manual convolution
        # For a full implementation, would use ttnn.conv1d with proper padding
        B, S, D = x.get_value().shape

        # Pad on the left (causal padding)
        pad_size = self.config.kernel_size - 1

        # Create padded tensor
        x_val = x.get_value()

        # Transpose to (B, D, S) for conv
        x_transposed = ttnn.transpose(x_val, 1, 2)  # (B, D, S)

        # Pad left with zeros
        if pad_size > 0:
            device = ttml.autograd.AutoContext.get_instance().get_device()
            zeros_shape = (B, D, pad_size)
            zeros = ttml.core.zeros(zeros_shape, device)
            x_padded = ttnn.concat([zeros, x_transposed], dim=2)  # (B, D, S+pad)
        else:
            x_padded = x_transposed

        # Apply depthwise conv manually
        # weight: (K, D)
        weight_val = self.weight.tensor.get_value()

        # Compute conv output
        outputs = []
        for t in range(S):
            # Window: [t, t+K)
            window = x_padded[:, :, t : t + self.config.kernel_size]  # (B, D, K)
            window_t = ttnn.transpose(window, 1, 2)  # (B, K, D)

            # Element-wise multiply and sum over kernel dimension
            weighted = ttnn.multiply(window_t, weight_val)  # (B, K, D)
            summed = ttnn.sum(weighted, dim=1, keepdim=True)  # (B, 1, D)
            outputs.append(summed)

        # Stack outputs
        out = ttnn.concat(outputs, dim=1)  # (B, S, D)

        # Add bias
        if self.bias_param is not None:
            bias_val = self.bias_param.tensor.get_value()
            out = ttnn.add(out, bias_val)

        return ttml.autograd.create_tensor(out, requires_grad=True)


@dataclass
class LinearHeadwiseExpandConfig:
    """Configuration for LinearHeadwiseExpand."""

    in_features: int
    num_heads: int
    expand_factor: float = 1.0
    bias: bool = True

    def __post_init__(self):
        assert self.num_heads > 0, "num_heads must be > 0"
        assert (
            self.in_features % self.num_heads == 0
        ), "in_features must be divisible by num_heads"


class LinearHeadwiseExpand(AbstractModuleBase):
    """Headwise linear projection layer.

    Projects input features where each head is projected independently.
    This is useful for Q/K/V projections in attention mechanisms.

    Args:
        config: LinearHeadwiseExpandConfig
    """

    def __init__(self, config: LinearHeadwiseExpandConfig) -> None:
        super().__init__()
        self.config = config

        in_features = config.in_features
        num_heads = config.num_heads
        out_features = round(config.expand_factor * in_features)
        out_per_head = out_features // num_heads
        in_per_head = in_features // num_heads

        # Weight shape: (num_heads, out_per_head, in_per_head)
        # Small init: normal with std = sqrt(2/5/in_per_head)
        std = sqrt(2.0 / 5.0 / in_per_head)
        weight_data = np.random.normal(
            0, std, (num_heads, out_per_head, in_per_head)
        ).astype(np.float32)
        self.weight = Parameter(ttml.autograd.Tensor.from_numpy(weight_data))

        self.out_features = out_features
        self.num_heads = num_heads
        self.in_per_head = in_per_head
        self.out_per_head = out_per_head

        if config.bias:
            bias_data = np.zeros((out_features,), dtype=np.float32)
            self.bias_param = Parameter(ttml.autograd.Tensor.from_numpy(bias_data))
        else:
            self.bias_param = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of headwise linear projection.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        x_val = x.get_value()
        shape = x_val.shape  # (..., in_features)

        # Reshape to (..., num_heads, in_per_head)
        new_shape = list(shape[:-1]) + [self.num_heads, self.in_per_head]
        x_heads = ttnn.reshape(x_val, new_shape)

        # Apply einsum: ...hd,hod->...ho
        # Manual implementation since ttnn may not have einsum
        weight_val = self.weight.tensor.get_value()  # (NH, out_per_head, in_per_head)

        # For simplicity, use matmul with proper reshaping
        # x_heads: (..., NH, in_per_head) -> (..., NH, 1, in_per_head)
        # weight: (NH, out_per_head, in_per_head) -> needs transpose to (NH, in_per_head, out_per_head)
        weight_t = ttnn.transpose(weight_val, -2, -1)  # (NH, in_per_head, out_per_head)

        # Expand dims for matmul
        x_expanded = ttnn.reshape(
            x_heads, list(shape[:-1]) + [self.num_heads, 1, self.in_per_head]
        )

        # Batched matmul
        # This is tricky with varying batch dims - simplified version
        # For now, flatten batch dims
        batch_shape = shape[:-1]
        batch_size = 1
        for s in batch_shape:
            batch_size *= s

        x_flat = ttnn.reshape(x_heads, (batch_size, self.num_heads, self.in_per_head))
        # x_flat: (B, NH, in_per_head) @ weight_t: (NH, in_per_head, out_per_head)
        # Need batched matmul per head

        # Transpose to (B, NH, 1, in_per_head) @ (NH, in_per_head, out_per_head) broadcast
        x_for_mm = ttnn.reshape(
            x_flat, (batch_size, self.num_heads, 1, self.in_per_head)
        )

        # Broadcast weight across batch: need to tile it
        # For now, do a simple loop (can be optimized)
        # Actually, use the fact that weight is (NH, in_per_head, out_per_head)
        # and we need (1, NH, in_per_head, out_per_head) for broadcasting
        weight_broadcast = ttnn.reshape(
            weight_t, (1, self.num_heads, self.in_per_head, self.out_per_head)
        )

        # matmul: (B, NH, 1, in_per_head) @ (1, NH, in_per_head, out_per_head) -> (B, NH, 1, out_per_head)
        out = ttnn.matmul(x_for_mm, weight_broadcast)

        # Reshape back to (..., num_heads, out_per_head) -> (..., out_features)
        out = ttnn.reshape(out, (batch_size, self.num_heads, self.out_per_head))
        out = ttnn.reshape(out, list(batch_shape) + [self.out_features])

        # Add bias
        if self.bias_param is not None:
            bias_val = self.bias_param.tensor.get_value()
            out = ttnn.add(out, bias_val)

        return ttml.autograd.create_tensor(out, requires_grad=True)


class LayerNorm(AbstractModuleBase):
    """Layer normalization with optional bias.

    Args:
        ndim: Normalized dimension
        weight: Whether to use learnable weight
        bias: Whether to use learnable bias
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        ndim: int,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.eps = eps
        self.has_weight = weight
        self.has_bias = bias

        if weight:
            # Initialize to ones (via 1 + zeros pattern)
            weight_data = np.ones((ndim,), dtype=np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(weight_data))
        else:
            self.weight = None

        if bias:
            bias_data = np.zeros((ndim,), dtype=np.float32)
            self.bias_param = Parameter(ttml.autograd.Tensor.from_numpy(bias_data))
        else:
            self.bias_param = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of layer norm.

        Args:
            x: Input tensor of shape (..., ndim)

        Returns:
            Normalized output tensor
        """
        x_val = x.get_value()

        # Compute mean and variance over last dimension
        mean = ttnn.mean(x_val, dim=-1, keepdim=True)
        x_centered = ttnn.subtract(x_val, mean)
        var = ttnn.mean(ttnn.multiply(x_centered, x_centered), dim=-1, keepdim=True)

        # Normalize
        inv_std = ttnn.rsqrt(ttnn.add(var, self.eps))
        out = ttnn.multiply(x_centered, inv_std)

        # Apply weight and bias
        if self.weight is not None:
            weight_val = self.weight.tensor.get_value()
            out = ttnn.multiply(out, weight_val)

        if self.bias_param is not None:
            bias_val = self.bias_param.tensor.get_value()
            out = ttnn.add(out, bias_val)

        return ttml.autograd.create_tensor(out, requires_grad=True)


class MultiHeadLayerNorm(AbstractModuleBase):
    """Multi-head layer normalization using group norm.

    Applies layer normalization per head using group normalization.
    Input shape: (B, NH, S, DH)
    Output shape: (B, NH, S, DH)

    Args:
        ndim: Total embedding dimension (NH * DH)
        num_heads: Number of heads
        weight: Whether to use learnable weight
        bias: Whether to use learnable bias
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        ndim: int,
        num_heads: int,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.num_heads = num_heads
        self.eps = eps

        if weight:
            weight_data = np.ones((ndim,), dtype=np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(weight_data))
        else:
            self.weight = None

        if bias:
            bias_data = np.zeros((ndim,), dtype=np.float32)
            self.bias_param = Parameter(ttml.autograd.Tensor.from_numpy(bias_data))
        else:
            self.bias_param = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of multi-head layer norm.

        Args:
            x: Input tensor of shape (B, NH, S, DH)

        Returns:
            Normalized output tensor of shape (B, NH, S, DH)
        """
        x_val = x.get_value()
        assert len(x_val.shape) == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x_val.shape

        # Per-head normalization over DH dimension
        mean = ttnn.mean(x_val, dim=-1, keepdim=True)
        x_centered = ttnn.subtract(x_val, mean)
        var = ttnn.mean(ttnn.multiply(x_centered, x_centered), dim=-1, keepdim=True)

        # Normalize
        inv_std = ttnn.rsqrt(ttnn.add(var, self.eps))
        out = ttnn.multiply(x_centered, inv_std)

        # Apply weight and bias (reshape for broadcasting)
        if self.weight is not None:
            weight_val = self.weight.tensor.get_value()
            # Reshape weight to (1, NH, 1, DH)
            weight_reshaped = ttnn.reshape(weight_val, (1, NH, 1, DH))
            out = ttnn.multiply(out, weight_reshaped)

        if self.bias_param is not None:
            bias_val = self.bias_param.tensor.get_value()
            bias_reshaped = ttnn.reshape(bias_val, (1, NH, 1, DH))
            out = ttnn.add(out, bias_reshaped)

        return ttml.autograd.create_tensor(out, requires_grad=True)
