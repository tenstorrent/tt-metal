# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python implementation of linear regression using ttml operations.

This module provides a pure Python implementation of linear regression that
uses ttml's C++ operations (via _ttml) for computation while maintaining
a Python-friendly interface.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

import ttml
from ttml.modules import AbstractModuleBase, Parameter


class LinearRegression(AbstractModuleBase):
    """Linear regression model implemented in Python using ttml operations.

    This class implements a linear regression model using ttml's autograd
    system and operations. It follows the PyTorch-like interface provided by
    AbstractModuleBase.

    Args:
        in_features: Number of input features
        out_features: Number of output features (default: 1 for regression)
        bias: Whether to include a bias term (default: True)
        init_scale: Scale factor for weight initialization (default: None, uses sqrt(1/in_features))

    Example:
        >>> model = LinearRegression(in_features=10, out_features=1)
        >>> x = ttml.autograd.Tensor.from_numpy(np.random.randn(32, 1, 1, 10).astype(np.float32))
        >>> y = model(x)
        >>> print(y.shape)  # [32, 1, 1, 1]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        bias: bool = True,
        init_scale: Optional[float] = None,
    ) -> None:
        """Initialize the linear regression model.

        Args:
            in_features: Number of input features
            out_features: Number of output features (default: 1)
            bias: Whether to include a bias term (default: True)
            init_scale: Scale factor for weight initialization. If None, uses sqrt(1/in_features)
        """
        super().__init__()
        # Module name is automatically set to class name ("LinearRegression") by AbstractModuleBase
        # Can be overridden by calling self.create_name("custom_name") if needed

        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight and bias tensors
        init_k = init_scale if init_scale is not None else math.sqrt(1.0 / in_features)

        # Create weight tensor: shape [1, 1, out_features, in_features]
        # Parameter wrapper automatically registers this via __setattr__
        weight_shape = (1, 1, out_features, in_features)
        weight_np = np.random.uniform(
            low=-init_k, high=init_k, size=weight_shape
        ).astype(np.float32)
        weight_tensor = ttml.autograd.Tensor.from_numpy(weight_np)
        self.weight = Parameter(weight_tensor)

        # Create bias tensor if needed: shape [1, 1, 1, out_features]
        # Parameter wrapper automatically registers this via __setattr__
        if bias:
            bias_shape = (1, 1, 1, out_features)
            bias_np = np.random.uniform(
                low=-init_k, high=init_k, size=bias_shape
            ).astype(np.float32)
            bias_tensor = ttml.autograd.Tensor.from_numpy(bias_np)
            self.bias = Parameter(bias_tensor)
        else:
            # Set to None explicitly - won't be registered as parameter
            self.bias = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of the linear regression model.

        Args:
            x: Input tensor of shape [batch_size, 1, 1, in_features]

        Returns:
            Output tensor of shape [batch_size, 1, 1, out_features]
        """
        # Use ttml's linear operation
        # Access underlying tensor from Parameter wrapper via .tensor attribute
        bias_tensor = self.bias.tensor if self.bias is not None else None
        return ttml.ops.linear.linear(x, self.weight.tensor, bias_tensor)


def create_linear_regression_model(
    in_features: int, out_features: int = 1, bias: bool = True
) -> LinearRegression:
    """Factory function to create a linear regression model.

    This function provides a convenient way to create a LinearRegression model,
    matching the interface of the C++ model factory.

    Args:
        in_features: Number of input features
        out_features: Number of output features (default: 1)
        bias: Whether to include a bias term (default: True)

    Returns:
        A LinearRegression model instance

    Example:
        >>> model = create_linear_regression_model(in_features=10, out_features=1)
        >>> x = ttml.autograd.Tensor.from_numpy(np.random.randn(32, 1, 1, 10).astype(np.float32))
        >>> y = model(x)
    """
    return LinearRegression(
        in_features=in_features, out_features=out_features, bias=bias
    )
