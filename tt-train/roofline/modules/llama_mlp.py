# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama MLP (SwiGLU) module for roofline modeling.

This module provides MockLlamaMLP for roofline estimation of
the SwiGLU MLP architecture used in Llama models.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockMulOp, MockDropoutOp
from ..operations.silu import MockSiLUOp
from .module import MockModule
from .linear import MockLinearLayer

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockLlamaMLP(MockModule):
    """Llama MLP (SwiGLU) module for roofline estimation.

    Mirrors ttml.modules.LlamaMLP for roofline modeling.
    Implements SwiGLU: silu(w1(x)) * w3(x), then w2(...).

    The hidden dimension is computed as:
        hidden_size = 4 * embedding_size * (2/3), rounded to multiple of 256

    All linear layers have no bias.

    Example:
        >>> mlp = MockLlamaMLP(embedding_size=4096)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 4096))
        >>> output = mlp(ctx, x)
    """

    def __init__(
        self,
        embedding_size: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.0,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize Llama MLP.

        Args:
            embedding_size: Input/output dimension
            intermediate_dim: Hidden dimension (if None, computed from embedding_size)
            dropout: Dropout probability
            dtype: Data type for parameters
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.dropout_prob = dropout

        # Compute hidden size following Llama convention
        if intermediate_dim is None:
            multiple_of = 256
            hidden_size = int(4 * embedding_size * (2.0 / 3.0))
            hidden_size = ((hidden_size + multiple_of - 1) // multiple_of) * multiple_of
        else:
            hidden_size = intermediate_dim

        self.hidden_size = hidden_size

        # Gate projection (w1): used with SiLU
        self.w1 = MockLinearLayer(
            embedding_size, hidden_size, has_bias=False, dtype=dtype
        )

        # Up projection (w3): multiplied with gated output
        self.w3 = MockLinearLayer(
            embedding_size, hidden_size, has_bias=False, dtype=dtype
        )

        # Down projection (w2): projects back to embedding_size
        self.w2 = MockLinearLayer(
            hidden_size, embedding_size, has_bias=False, dtype=dtype
        )

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Forward pass: SwiGLU computation.

        SwiGLU: w2(silu(w1(x)) * w3(x))

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [batch, 1, seq_len, embedding_size]

        Returns:
            Output tensor [batch, 1, seq_len, embedding_size]
        """
        # Gate path: silu(w1(x))
        gate = self.w1(ctx, x)
        swished = MockSiLUOp.apply(ctx, gate)

        # Up path: w3(x)
        up = self.w3(ctx, x)

        # Multiply gated output with up projection
        gated = MockMulOp.apply(ctx, swished, up)

        # Down projection
        out = self.w2(ctx, gated)

        # Apply dropout if probability > 0
        if self.dropout_prob > 0:
            out = MockDropoutOp.apply(ctx, out, self.dropout_prob)

        return out

    def __repr__(self) -> str:
        return (
            f"MockLlamaMLP(embedding_size={self.embedding_size}, "
            f"hidden_size={self.hidden_size}, dropout={self.dropout_prob})"
        )
