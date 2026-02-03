# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GPT transformer block module for roofline modeling.

This module provides MockGPTBlock for roofline estimation of
transformer block operations.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockAddOp
from .module import MockModule
from .layernorm import MockLayerNorm
from .attention import MockMultiHeadAttention
from .mlp import MockGPTMLP

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockGPTBlock(MockModule):
    """GPT transformer block module for roofline estimation.

    Mirrors ttml.models.nanogpt.GPTBlock for roofline modeling.

    This module performs:
    1. Pre-LayerNorm + Multi-Head Attention + Residual
    2. Pre-LayerNorm + MLP + Residual

    Example:
        >>> block = MockGPTBlock(768, 12, dropout=0.1)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 768))
        >>> output = block(ctx, x)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize GPT block.

        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            dtype: Data type for parameters
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Layer norms
        self.ln1 = MockLayerNorm(embedding_dim, dtype=dtype)
        self.ln2 = MockLayerNorm(embedding_dim, dtype=dtype)

        # Attention
        self.attention = MockMultiHeadAttention(
            embedding_dim, num_heads, dropout=dropout, dtype=dtype
        )

        # MLP
        self.mlp = MockGPTMLP(embedding_dim, dropout=dropout, dtype=dtype)

    def forward(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: apply transformer block.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [batch, 1, seq_len, embedding_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, 1, seq_len, embedding_dim]
        """
        # Pre-norm attention with residual
        # residual = x; x = ln1(x); x = attention(x); x = x + residual
        residual = x
        x = self.ln1(ctx, x)
        x = self.attention(ctx, x, mask)
        x = MockAddOp.apply(ctx, x, residual)

        # Pre-norm MLP with residual
        # residual = x; x = ln2(x); x = mlp(x); x = x + residual
        residual = x
        x = self.ln2(ctx, x)
        x = self.mlp(ctx, x)
        x = MockAddOp.apply(ctx, x, residual)

        return x

    def __repr__(self) -> str:
        return (
            f"MockGPTBlock(embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads})"
        )
