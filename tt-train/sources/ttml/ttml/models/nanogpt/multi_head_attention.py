# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-head attention layer for NanoGPT."""

import math
from typing import Optional

import numpy as np
import ml_dtypes  # Used for bfloat16 type in fallback initialization

import ttml
from ttml.modules import AbstractModuleBase, Parameter


class MultiHeadAttention(AbstractModuleBase):
    """Multi-head attention layer implemented in Python using ttml operations.

    This implementation uses ttml operations to build attention from scratch.
    For better performance, consider using the C++ MultiHeadAttention module
    if it becomes available through the Python API.
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
        # Note: scaling by 1/sqrt(head_dim) is handled inside scaled_dot_product_attention

        # Try to use C++ LinearLayer if available (via recursive import)
        # Otherwise, use Parameter with linear_op
        try:
            # Check if C++ LinearLayer is available
            if hasattr(ttml.modules, "LinearLayer"):
                LinearLayer = ttml.modules.LinearLayer
                # Use C++ LinearLayer for better performance
                self.qkv_layer = LinearLayer(
                    embedding_dim, embedding_dim * 3, has_bias=False
                )
                self.out_layer = LinearLayer(
                    embedding_dim, embedding_dim, has_bias=False
                )
                self._use_cpp_layers = True
            else:
                self._use_cpp_layers = False
                raise AttributeError("C++ LinearLayer not available")
        except (AttributeError, TypeError):
            # Fallback to Python implementation with Parameter
            self._use_cpp_layers = False
            # QKV projection: embedding_dim -> embedding_dim * 3
            # Linear weights must be in TILE layout
            qkv_shape = (1, 1, embedding_dim * 3, embedding_dim)
            qkv_np = np.random.normal(0.0, 0.02, size=qkv_shape).astype(
                ml_dtypes.bfloat16
            )
            qkv_tensor = ttml.autograd.Tensor.from_numpy(
                qkv_np, layout=ttml.Layout.TILE
            )
            self.qkv = Parameter(qkv_tensor)

            # Output projection: embedding_dim -> embedding_dim
            out_shape = (1, 1, embedding_dim, embedding_dim)
            out_np = np.random.normal(0.0, 0.02, size=out_shape).astype(
                ml_dtypes.bfloat16
            )
            out_tensor = ttml.autograd.Tensor.from_numpy(
                out_np, layout=ttml.Layout.TILE
            )
            self.out_proj = Parameter(out_tensor)

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
        # QKV projection
        if self._use_cpp_layers:
            qkv = self.qkv_layer(x)
        else:
            qkv = ttml.ops.linear.linear_op(x, self.qkv.tensor, None)

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

        # Output projection
        if self._use_cpp_layers:
            out = self.out_layer(attention_out)
        else:
            out = ttml.ops.linear.linear_op(attention_out, self.out_proj.tensor, None)

        # Apply dropout if needed
        if self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)

        return out

    def __call__(
        self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        """Call the forward method."""
        return self.forward(x, mask)
