# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-head attention layer for NanoGPT."""

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, RunMode


class MultiHeadAttention(AbstractModuleBase):
    """Multi-head attention layer using C++ LinearLayer for better performance.

    This implementation uses C++ LinearLayer modules which provide:
    - Optimized forward/backward passes
    - Pickle support for checkpoint save/load
    - Proper parameter registration for optimizer integration
    """

    def __init__(
        self, embedding_dim: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        """Initialize multi-head attention layer.

        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout_prob = dropout
        # Note: RunMode is managed by AbstractModuleBase (defaults to TRAIN)
        # Note: scaling by 1/sqrt(head_dim) is handled inside scaled_dot_product_attention

        # QKV projection: embedding_dim -> embedding_dim * 3 (no bias for attention projections)
        self.qkv = LinearLayer(embedding_dim, embedding_dim * 3, False)

        # Output projection: embedding_dim -> embedding_dim (no bias)
        self.out_proj = LinearLayer(embedding_dim, embedding_dim, False)

    # train() and eval() are inherited from AbstractModuleBase

    def forward(
        self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        """Forward pass of multi-head attention.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Output tensor after attention
        """
        # QKV projection using C++ LinearLayer
        qkv = self.qkv(x)

        # Split into heads using ttml's heads_creation
        # Output: query, key, value each have shape (B, H, S, head_dim)
        query, key, value = ttml.ops.multi_head_utils.heads_creation(
            qkv, self.num_heads
        )

        # Use the C++ scaled_dot_product_attention which handles:
        # - Scaling by 1/sqrt(head_dim)
        # - Q @ K^T
        # - Softmax
        # - Attention @ V
        # - Proper backward pass for 4D tensors
        attn_out = ttml.ops.attention.scaled_dot_product_attention(
            query, key, value, mask
        )

        # Fuse heads back
        attention_out = ttml.ops.multi_head_utils.heads_fusion(attn_out)

        # Output projection using C++ LinearLayer
        out = self.out_proj(attention_out)

        # Apply dropout if in training mode (using RunMode from AbstractModuleBase)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)

        return out

    def __call__(
        self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        """Call the forward method."""
        return self.forward(x, mask)
