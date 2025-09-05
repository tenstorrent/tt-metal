# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch.nn import functional as F
import warnings


def _check_if_dynamo_compiling() -> bool:
    """Check if running under torch.compile dynamo compilation"""
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version >= (2, 1):
        from torch._dynamo import is_compiling

        return is_compiling()
    else:
        return False


class Conv2d(torch.nn.Conv2d):
    """
    PyTorch Conv2d wrapper with normalization and activation support.

    This is a cleaner implementation of a conv2d wrapper that supports:
    - Optional normalization layers
    - Optional activation functions
    - Empty input handling (for compatibility)

    Similar in structure to the TTNN wrapper but maintains backward compatibility
    with the existing torch.nn.Conv2d interface.

    Usage examples:

    # Basic convolution with norm and activation
    conv = Conv2d(in_channels, out_channels, kernel_size=3,
                  norm=nn.BatchNorm2d(out_channels), activation=F.relu)

    # Simple convolution
    conv = Conv2d(in_channels, out_channels, kernel_size=1)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize Conv2d with optional normalization and activation.

        Args:
            norm (nn.Module, optional): normalization layer to apply after convolution
            activation (callable, optional): activation function to apply after normalization

        All other arguments are passed to torch.nn.Conv2d.

        Note: Normalization is applied before activation.
        """
        # Extract our custom arguments before calling parent init
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)

        # Initialize the base Conv2d first
        super().__init__(*args, **kwargs)

        # Now assign the custom attributes after parent init
        self.norm = norm
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conv2d, optional norm, and optional activation.

        Args:
            x: Input tensor

        Returns:
            Output tensor after convolution, normalization, and activation
        """
        # Handle empty inputs for compatibility (from original implementation)
        if not torch.jit.is_scripting():
            is_dynamo_compiling = _check_if_dynamo_compiling()
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(
                            self.norm, torch.nn.SyncBatchNorm
                        ), "SyncBatchNorm does not support empty inputs!"

        # Standard conv2d operation
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Apply normalization if provided
        if self.norm is not None:
            x = self.norm(x)

        # Apply activation if provided
        if self.activation is not None:
            x = self.activation(x)

        return x
