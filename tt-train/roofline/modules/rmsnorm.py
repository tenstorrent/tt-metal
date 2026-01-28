# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RMS normalization module for roofline modeling.

This module provides MockRMSNormLayer for roofline estimation of
RMS normalization operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockRMSNormOp
from .module import MockModule, MockParameter

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockRMSNormLayer(MockModule):
    """RMS normalization module for roofline estimation.

    Mirrors ttml.modules.RMSNormLayer for roofline modeling.
    Unlike LayerNorm, RMSNorm does not subtract mean or use beta parameter.

    Example:
        >>> ln = MockRMSNormLayer(4096)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 4096))
        >>> output = ln(ctx, x)
    """

    def __init__(
        self,
        features: int,
        epsilon: float = 1e-6,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize RMS normalization.

        Args:
            features: Dimension to normalize over
            epsilon: Small constant for numerical stability
            dtype: Data type for parameters
        """
        super().__init__()

        self.features = features
        self.epsilon = epsilon

        # Gamma (scale): [1, 1, 1, features]
        self.gamma = MockParameter(MockTensor((1, 1, 1, features), dtype=dtype))

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Forward pass: apply RMS normalization.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [..., features]

        Returns:
            Normalized output tensor
        """
        return MockRMSNormOp.apply(ctx, x, self.gamma.tensor, self.epsilon)

    def __repr__(self) -> str:
        return f"MockRMSNormLayer(features={self.features}, epsilon={self.epsilon})"
