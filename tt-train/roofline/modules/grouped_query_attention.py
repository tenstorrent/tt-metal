# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Grouped Query Attention module for roofline modeling.

This module provides MockGroupedQueryAttention for roofline estimation of
grouped query attention operations used in Llama models.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import (
    MockGroupedHeadsCreationOp,
    MockHeadsFusionOp,
    MockScaledDotProductAttentionOp,
    MockDropoutOp,
)
from ..operations.rope import MockRoPEOp
from .module import MockModule
from .linear import MockLinearLayer
from .rope import MockRotaryEmbedding

if TYPE_CHECKING:
    from ..roofline import RooflineContext


@dataclass
class RoPEParams:
    """Parameters for Rotary Position Embedding."""

    head_dim: int
    max_seq_len: int = 2048
    theta: float = 10000.0


class MockGroupedQueryAttention(MockModule):
    """Grouped Query Attention module for roofline estimation.

    Mirrors ttml.modules.GroupedQueryAttention for roofline modeling.
    GQA differs from MHA in that K and V have fewer heads than Q.

    This module performs:
    1. Q projection: input -> Q via linear layer
    2. KV projection: input -> K, V via single linear layer
    3. Split into heads (grouped for K/V)
    4. Apply RoPE to Q and K (if enabled)
    5. Scaled dot-product attention (with K/V broadcasting)
    6. Merge heads
    7. Output projection
    8. Optional dropout

    Example:
        >>> gqa = MockGroupedQueryAttention(
        ...     embedding_dim=4096, num_heads=32, num_groups=8, dropout=0.1
        ... )
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 4096))
        >>> output = gqa(ctx, x)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_groups: int,
        dropout: float = 0.0,
        rope_params: Optional[RoPEParams] = None,
        bias: bool = False,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize grouped query attention.

        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads for Q
            num_groups: Number of KV groups (fewer heads for K/V)
            dropout: Dropout probability
            rope_params: Optional RoPE parameters (if None, no RoPE)
            bias: Whether to use bias in linear layers
            dtype: Data type for parameters
        """
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embedding_dim // num_heads
        self.dropout_prob = dropout

        # Q projection: [embedding_dim, embedding_dim]
        self.q_linear = MockLinearLayer(
            embedding_dim, embedding_dim, has_bias=bias, dtype=dtype
        )

        # KV projection: [embedding_dim, 2 * num_groups * head_dim]
        kv_dim = 2 * num_groups * self.head_dim
        self.kv_linear = MockLinearLayer(
            embedding_dim, kv_dim, has_bias=bias, dtype=dtype
        )

        # Output projection: [embedding_dim, embedding_dim]
        self.out_linear = MockLinearLayer(
            embedding_dim, embedding_dim, has_bias=bias, dtype=dtype
        )

        # RoPE (optional)
        if rope_params is not None:
            self.rope = MockRotaryEmbedding(
                head_dim=rope_params.head_dim,
                max_seq_len=rope_params.max_seq_len,
                theta=rope_params.theta,
                dtype=dtype,
            )
        else:
            self.rope = None

    def forward(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass: compute grouped query attention.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [batch, 1, seq_len, embedding_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, 1, seq_len, embedding_dim]
        """
        # Q projection
        q = self.q_linear(ctx, x)

        # KV projection
        kv = self.kv_linear(ctx, x)

        # Split into heads: Q [B, num_heads, S, d], K/V [B, num_groups, S, d]
        query, key, value = MockGroupedHeadsCreationOp.apply(
            ctx, q, kv, self.num_heads, self.num_groups
        )

        # Apply RoPE to Q and K (if enabled)
        if self.rope is not None:
            query = self.rope(ctx, query)
            key = self.rope(ctx, key)

        # For GQA, we need to broadcast K/V to match Q's number of heads
        # This is handled inside SDPA - we just need to use the right shapes
        # The attention computation internally handles the broadcasting

        # Scaled dot-product attention
        # Note: For GQA, the roofline estimate should account for K/V broadcasting
        # The actual SDPA implementation handles this, and we pass the shapes as-is
        attn_out = MockScaledDotProductAttentionOp.apply(ctx, query, key, value, mask)

        # Merge heads: [B, num_heads, S, d] -> [B, 1, S, E]
        merged = MockHeadsFusionOp.apply(ctx, attn_out)

        # Output projection
        out = self.out_linear(ctx, merged)

        # Apply dropout if probability > 0
        if self.dropout_prob > 0:
            out = MockDropoutOp.apply(ctx, out, self.dropout_prob)

        return out

    def __repr__(self) -> str:
        return (
            f"MockGroupedQueryAttention(embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, num_groups={self.num_groups}, "
            f"dropout={self.dropout_prob}, rope={'enabled' if self.rope else 'disabled'})"
        )
