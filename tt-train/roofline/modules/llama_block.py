# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama transformer block module for roofline modeling.

This module provides MockLlamaBlock for roofline estimation of
the transformer block architecture used in Llama models.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ..mock_tensor import MockTensor
from ..hardware import DataType
from ..operations import MockAddOp
from .module import MockModule
from .rmsnorm import MockRMSNormLayer
from .grouped_query_attention import MockGroupedQueryAttention, RoPEParams
from .llama_mlp import MockLlamaMLP
from .llama_mlp_fused import MockLlamaMLPFused
from ..operations.swiglu_fused import SwiGLUFusedImpl

if TYPE_CHECKING:
    from ..roofline import RooflineContext


class MockLlamaBlock(MockModule):
    """Llama transformer block module for roofline estimation.

    Mirrors ttml.modules.LlamaBlock for roofline modeling.
    Implements pre-norm architecture with residual connections:
        1. attention_norm -> attention -> add residual
        2. mlp_norm -> mlp -> add residual

    Example:
        >>> block = MockLlamaBlock(
        ...     embedding_size=4096, num_heads=32, num_groups=8
        ... )
        >>> ctx = RooflineContext(WORMHOLE_N150)
        >>> x = MockTensor((1, 1, 1024, 4096))
        >>> output = block(ctx, x)
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        num_groups: int,
        rope_params: Optional[RoPEParams] = None,
        dropout: float = 0.0,
        intermediate_dim: Optional[int] = None,
        dtype: DataType = DataType.BFLOAT16,
    ):
        """Initialize Llama transformer block.

        Args:
            embedding_size: Embedding dimension
            num_heads: Number of attention heads for Q
            num_groups: Number of KV groups (for GQA)
            rope_params: Optional RoPE parameters (if None, no RoPE)
            dropout: Dropout probability
            intermediate_dim: MLP hidden dimension (if None, computed)
            dtype: Data type for parameters
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_groups = num_groups

        # Pre-attention normalization
        self.attention_norm = MockRMSNormLayer(embedding_size, dtype=dtype)

        # Grouped query attention
        self.attention = MockGroupedQueryAttention(
            embedding_dim=embedding_size,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
            rope_params=rope_params,
            bias=False,  # Llama uses no bias
            dtype=dtype,
        )

        # Pre-MLP normalization
        self.mlp_norm = MockRMSNormLayer(embedding_size, dtype=dtype)

        # SwiGLU MLP (non-fused version)
        self.mlp = MockLlamaMLP(
            embedding_size=embedding_size,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            dtype=dtype,
        )

        # Fused SwiGLU MLP
        # self.mlp = MockLlamaMLPFused(
        #     embedding_size=embedding_size,
        #     intermediate_dim=intermediate_dim,
        #     dropout=dropout,
        #     dtype=dtype,
        #     impl=SwiGLUFusedImpl.MCAST,
        # )

    def forward(
        self,
        ctx: "RooflineContext",
        x: MockTensor,
        mask: Optional[MockTensor] = None,
    ) -> MockTensor:
        """Forward pass through Llama block.

        Args:
            ctx: Roofline context for estimates
            x: Input tensor [batch, 1, seq_len, embedding_size]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, 1, seq_len, embedding_size]
        """
        # Attention block with residual
        residual = x
        h = self.attention_norm(ctx, x)
        h = self.attention(ctx, h, mask)
        h = MockAddOp.apply(ctx, h, residual)

        # MLP block with residual
        residual = h
        out = self.mlp_norm(ctx, h)
        out = self.mlp(ctx, out)
        out = MockAddOp.apply(ctx, out, residual)

        return out

    def __repr__(self) -> str:
        return (
            f"MockLlamaBlock(\n"
            f"  embedding_size={self.embedding_size},\n"
            f"  num_heads={self.num_heads},\n"
            f"  num_groups={self.num_groups}\n"
            f")"
        )
