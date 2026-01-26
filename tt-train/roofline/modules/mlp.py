# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MLP modules for roofline modeling.

This module provides MockGPTMLP for roofline estimation of
feed-forward network operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockGELUOp, MockDropoutOp
from .module import MockModule
from .linear import MockLinearLayer

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockGPTMLP(MockModule):
    """GPT-style MLP module for roofline estimation.

    Mirrors ttml.models.nanogpt.GPTMLP for roofline modeling.

    This module performs:
    1. Up-projection: embedding_dim -> 4 * embedding_dim
    2. GELU activation
    3. Down-projection: 4 * embedding_dim -> embedding_dim
    4. Optional dropout

    Example:
        >>> mlp = MockGPTMLP(768, dropout=0.1)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 768))
        >>> output = mlp(ctx, x)
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize GPT MLP.

        Args:
            embedding_dim: Dimension of embeddings
            dropout: Dropout probability
            dtype: Data type for parameters
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        # Up-projection: embedding_dim -> 4 * embedding_dim
        self.fc1 = MockLinearLayer(
            embedding_dim, embedding_dim * 4, has_bias=True, dtype=dtype
        )

        # Down-projection: 4 * embedding_dim -> embedding_dim
        self.fc2 = MockLinearLayer(
            embedding_dim * 4, embedding_dim, has_bias=True, dtype=dtype
        )

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Forward pass: apply MLP.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [..., embedding_dim]

        Returns:
            Output tensor [..., embedding_dim]
        """
        # Up-projection
        x = self.fc1(ctx, x)

        # GELU activation
        x = MockGELUOp.apply(ctx, x)

        # Down-projection
        x = self.fc2(ctx, x)

        # Apply dropout if probability > 0
        if self.dropout_prob > 0:
            x = MockDropoutOp.apply(ctx, x, self.dropout_prob)

        return x

    def __repr__(self) -> str:
        return (
            f"MockGPTMLP(embedding_dim={self.embedding_dim}, "
            f"hidden_dim={self.embedding_dim * 4}, dropout={self.dropout_prob})"
        )
