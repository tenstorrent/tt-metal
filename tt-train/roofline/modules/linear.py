# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Linear layer module for roofline modeling.

This module provides MockLinearLayer for roofline estimation of
linear/fully-connected layers.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockLinearOp
from .module import MockModule, MockParameter

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockLinearLayer(MockModule):
    """Linear layer module for roofline estimation.

    Mirrors ttml.modules.LinearLayer for roofline modeling.

    Example:
        >>> layer = MockLinearLayer(4096, 11008)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 8192, 4096))
        >>> y = layer(ctx, x)
        >>> print(ctx.summary())
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            has_bias: Whether to include bias (default: True)
            dtype: Data type for parameters (default: BFLOAT16)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

        # Weight: [1, 1, out_features, in_features]
        self.weight = MockParameter(
            MockTensor((1, 1, out_features, in_features), dtype=dtype)
        )

        # Bias: [1, 1, 1, out_features]
        if has_bias:
            self.bias = MockParameter(MockTensor((1, 1, 1, out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Forward pass through linear layer.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [*, in_features]

        Returns:
            Output tensor [*, out_features]
        """
        bias_tensor = self.bias.tensor if self.bias is not None else None
        return MockLinearOp.apply(ctx, x, self.weight.tensor, bias_tensor)

    def __repr__(self) -> str:
        return (
            f"MockLinearLayer(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.has_bias})"
        )
