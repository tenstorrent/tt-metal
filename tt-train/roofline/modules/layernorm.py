# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layer normalization module for roofline modeling.

This module provides MockLayerNorm for roofline estimation of
layer normalization operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockLayerNormOp
from .module import MockModule, MockParameter

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockLayerNorm(MockModule):
    """Layer normalization module for roofline estimation.

    Mirrors ttml.models.nanogpt.LayerNorm for roofline modeling.

    Example:
        >>> ln = MockLayerNorm(768)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 768))
        >>> output = ln(ctx, x)
    """

    def __init__(
        self,
        embedding_dim: int,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize layer normalization.

        Args:
            embedding_dim: Dimension to normalize over
            dtype: Data type for parameters
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Gamma (scale): [1, 1, 1, embedding_dim]
        self.gamma = MockParameter((1, 1, 1, embedding_dim), dtype=dtype, name="gamma")

        # Beta (shift): [1, 1, 1, embedding_dim]
        self.beta = MockParameter((1, 1, 1, embedding_dim), dtype=dtype, name="beta")

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Forward pass: apply layer normalization.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [..., embedding_dim]

        Returns:
            Normalized output tensor
        """
        return MockLayerNormOp.apply(ctx, x, self.gamma.tensor, self.beta.tensor)

    def __repr__(self) -> str:
        return f"MockLayerNorm(embedding_dim={self.embedding_dim})"
