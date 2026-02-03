# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused Llama MLP (SwiGLU) module for roofline modeling.

This module provides MockLlamaMLPFused for roofline estimation of
the fused SwiGLU MLP architecture used in Llama models.

Two implementation variants are supported:
- swiglu_fused_row_mcast: Input read once, weights read multiple times
- swiglu_fused_mcast: Everything read once (optimal)
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockDropoutOp
from ..operations.swiglu_fused import MockSwiGLUFusedOp, SwiGLUFusedImpl
from .module import MockModule, MockParameter

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockLlamaMLPFused(MockModule):
    """Fused Llama MLP (SwiGLU) module for roofline estimation.

    Implements fused SwiGLU: silu(w1(x)) * w2(x), then w3(...)
    as a single fused operation for accurate roofline modeling.

    The hidden dimension is computed as:
        hidden_size = 4 * embedding_size * (2/3), rounded to multiple of 256

    All linear layers have no bias.

    Example:
        >>> mlp = MockLlamaMLPFused(
        ...     embedding_size=4096,
        ...     impl=SwiGLUFusedImpl.ROW_MCAST
        ... )
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
        impl: SwiGLUFusedImpl = SwiGLUFusedImpl.ROW_MCAST,
    ):
        """Initialize Fused Llama MLP.

        Args:
            embedding_size: Input/output dimension
            intermediate_dim: Hidden dimension (if None, computed from embedding_size)
            dropout: Dropout probability
            dtype: Data type for parameters
            impl: Which fused SwiGLU implementation to model
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.dropout_prob = dropout
        self.impl = impl

        # Compute hidden size following Llama convention
        if intermediate_dim is None:
            multiple_of = 256
            hidden_size = int(4 * embedding_size * (2.0 / 3.0))
            hidden_size = ((hidden_size + multiple_of - 1) // multiple_of) * multiple_of
        else:
            hidden_size = intermediate_dim

        self.hidden_size = hidden_size

        # Weight matrices stored as parameters
        # w1: gate projection (used with SiLU) - [1, 1, hidden_size, embedding_size]
        self.w1 = MockParameter(
            (1, 1, hidden_size, embedding_size), dtype=dtype, name="w1"
        )

        # w2: up projection - [1, 1, hidden_size, embedding_size]
        self.w2 = MockParameter(
            (1, 1, hidden_size, embedding_size), dtype=dtype, name="w2"
        )

        # w3: down projection - [1, 1, embedding_size, hidden_size]
        self.w3 = MockParameter(
            (1, 1, embedding_size, hidden_size), dtype=dtype, name="w3"
        )

    def forward(self, ctx: "RooflineContext", x: MockTensor) -> MockTensor:
        """Forward pass: Fused SwiGLU computation.

        Fused: w3(silu(w1(x)) * w2(x))

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [batch, 1, seq_len, embedding_size]

        Returns:
            Output tensor [batch, 1, seq_len, embedding_size]
        """
        # Single fused operation for the entire MLP
        out = MockSwiGLUFusedOp.apply(
            ctx,
            x,
            self.w1.tensor,
            self.w2.tensor,
            self.w3.tensor,
            self.impl,
        )

        # Apply dropout if probability > 0
        if self.dropout_prob > 0:
            out = MockDropoutOp.apply(ctx, out, self.dropout_prob)

        return out

    def __repr__(self) -> str:
        return (
            f"MockLlamaMLPFused(embedding_size={self.embedding_size}, "
            f"hidden_size={self.hidden_size}, dropout={self.dropout_prob}, "
            f"impl={self.impl.value})"
        )
