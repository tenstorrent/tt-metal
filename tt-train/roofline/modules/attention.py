# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-head attention module for roofline modeling.

This module provides MockMultiHeadAttention for roofline estimation of
multi-head attention operations.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import (
    MockLinearOp,
    MockHeadsCreationOp,
    MockHeadsFusionOp,
    MockScaledDotProductAttentionOp,
    MockDropoutOp,
)
from .module import MockModule, MockParameter
from .linear import MockLinearLayer

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockMultiHeadAttention(MockModule):
    """Multi-head attention module for roofline estimation.

    Mirrors ttml.models.nanogpt.MultiHeadAttention for roofline modeling.

    This module performs:
    1. QKV projection: input -> Q, K, V via single linear layer
    2. Split into heads
    3. Scaled dot-product attention
    4. Merge heads
    5. Output projection
    6. Optional dropout

    Example:
        >>> mha = MockMultiHeadAttention(768, 12, dropout=0.1)
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 768))
        >>> output = mha(ctx, x)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize multi-head attention.

        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            dtype: Data type for parameters
        """
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout_prob = dropout

        # QKV projection: [embedding_dim, 3 * embedding_dim]
        self.qkv_linear = MockLinearLayer(
            embedding_dim, embedding_dim * 3, has_bias=True, dtype=dtype
        )

        # Output projection: [embedding_dim, embedding_dim]
        self.out_linear = MockLinearLayer(
            embedding_dim, embedding_dim, has_bias=True, dtype=dtype
        )

    def forward(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: compute multi-head attention.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [batch, 1, seq_len, embedding_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, 1, seq_len, embedding_dim]
        """
        # QKV projection
        qkv = self.qkv_linear(ctx, x)

        # Split into heads: [B, 1, S, 3*H*d] -> 3x [B, H, S, d]
        query, key, value = MockHeadsCreationOp.apply(ctx, qkv, self.num_heads)

        # Scaled dot-product attention
        attn_out = MockScaledDotProductAttentionOp.apply(ctx, query, key, value, mask)

        # Merge heads: [B, H, S, d] -> [B, 1, S, H*d]
        merged = MockHeadsFusionOp.apply(ctx, attn_out)

        # Output projection
        out = self.out_linear(ctx, merged)

        # Apply dropout if probability > 0
        if self.dropout_prob > 0:
            out = MockDropoutOp.apply(ctx, out, self.dropout_prob)

        return out

    def __repr__(self) -> str:
        return (
            f"MockMultiHeadAttention(embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, dropout={self.dropout_prob})"
        )
