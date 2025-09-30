# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Normalization modules - Layer normalization and RMS normalization"""

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn


@dataclass
class NormConfig:
    """Configuration for normalization modules"""

    normalized_shape: int
    eps: float = 1e-6
    elementwise_affine: bool = True


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.

    Commonly used in modern transformer models like LLaMA.
    More efficient than standard LayerNorm as it doesn't center the activations.
    """

    def __init__(
        self,
        config: NormConfig,
        device: ttnn.Device,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.eps = config.eps

        # Weight parameter (gamma)
        self.weight = None

    def setup_weight(self, weight: ttnn.Tensor):
        """Set pre-loaded weight for the norm module"""
        self.weight = weight

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor

        Returns:
            output: Normalized tensor
        """
        # Calculate RMS
        variance = ttnn.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * ttnn.rsqrt(variance + self.eps)

        # Apply learned scale if available
        if self.weight is not None and self.config.elementwise_affine:
            x_normed = x_normed * self.weight

        return x_normed


class LayerNorm(torch.nn.Module):
    """
    Layer Normalization module.

    Standard layer normalization as used in the original transformer.
    Normalizes across the feature dimension.
    """

    def __init__(
        self,
        config: NormConfig,
        device: ttnn.Device,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.eps = config.eps

        # Affine parameters
        self.weight = None  # gamma
        self.bias = None  # beta

    def setup_parameters(self, weight: ttnn.Tensor, bias: Optional[ttnn.Tensor] = None):
        """Set pre-loaded parameters for the norm module"""
        self.weight = weight
        self.bias = bias

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor

        Returns:
            output: Normalized tensor
        """
        # Calculate mean and variance
        mean = ttnn.mean(x, dim=-1, keepdim=True)
        variance = ttnn.mean((x - mean) ** 2, dim=-1, keepdim=True)

        # Normalize
        x_normed = (x - mean) * ttnn.rsqrt(variance + self.eps)

        # Apply affine transformation if available
        if self.config.elementwise_affine:
            if self.weight is not None:
                x_normed = x_normed * self.weight
            if self.bias is not None:
                x_normed = x_normed + self.bias

        return x_normed


class GroupNorm(torch.nn.Module):
    """
    Group Normalization module.

    Divides channels into groups and normalizes within each group.
    Useful for vision transformers and multimodal models.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
        device: ttnn.Device = None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.device = device

        assert (
            num_channels % num_groups == 0
        ), f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"

        # Affine parameters
        self.weight = None
        self.bias = None

    def setup_parameters(self, weight: ttnn.Tensor, bias: Optional[ttnn.Tensor] = None):
        """Set pre-loaded parameters for the norm module"""
        self.weight = weight
        self.bias = bias

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply group normalization.

        Args:
            x: Input tensor of shape [batch_size, num_channels, height, width]

        Returns:
            output: Normalized tensor
        """
        batch_size, num_channels, height, width = x.shape
        channels_per_group = num_channels // self.num_groups

        # Reshape to separate groups
        x = ttnn.reshape(x, [batch_size, self.num_groups, channels_per_group, height, width])

        # Calculate mean and variance per group
        mean = ttnn.mean(x, dim=[2, 3, 4], keepdim=True)
        variance = ttnn.mean((x - mean) ** 2, dim=[2, 3, 4], keepdim=True)

        # Normalize
        x_normed = (x - mean) * ttnn.rsqrt(variance + self.eps)

        # Reshape back
        x_normed = ttnn.reshape(x_normed, [batch_size, num_channels, height, width])

        # Apply affine transformation if available
        if self.affine and self.weight is not None:
            # Reshape weight and bias for broadcasting
            weight = ttnn.reshape(self.weight, [1, num_channels, 1, 1])
            x_normed = x_normed * weight

            if self.bias is not None:
                bias = ttnn.reshape(self.bias, [1, num_channels, 1, 1])
                x_normed = x_normed + bias

        return x_normed
