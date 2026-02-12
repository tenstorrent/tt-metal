# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GPT transformer block for NanoGPT."""

from __future__ import annotations

from typing import Optional

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter

from .gpt_mlp import GPTMLP
from .multi_head_attention import MultiHeadAttention


class LayerNorm(AbstractModuleBase):
    """Layer normalization module with gamma and beta parameters."""

    def __init__(self, embedding_dim: int, bias: bool = True) -> None:
        """Initialize layer norm.

        Args:
            embedding_dim: Dimension of embeddings
            bias: Whether to use bias (beta) parameter
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Layer norm requires gamma (scale) and beta (shift) parameters
        ln_shape = (1, 1, 1, embedding_dim)
        gamma_np = np.ones(ln_shape, dtype=ml_dtypes.bfloat16)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(
            gamma_np, layout=ttnn.Layout.TILE
        )
        self.gamma = Parameter(gamma_tensor)

        if bias:
            beta_np = np.zeros(ln_shape, dtype=ml_dtypes.bfloat16)
            beta_tensor = ttml.autograd.Tensor.from_numpy(
                beta_np, layout=ttnn.Layout.TILE
            )
            self.beta = Parameter(beta_tensor)
        else:
            self.beta = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of layer norm.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """
        return ttml.ops.layernorm.layernorm(
            x, self.gamma.tensor, self.beta.tensor if self.beta else None
        )


class GPTBlock(AbstractModuleBase):
    """GPT transformer block with attention and MLP."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize GPT block.

        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in layer norm
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        # Note: RunMode is managed by AbstractModuleBase (defaults to TRAIN)

        # Layer norms
        self.ln1 = LayerNorm(embedding_dim, bias)
        self.ln2 = LayerNorm(embedding_dim, bias)

        # Attention and MLP
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)

        # MLP (matching C++ mlp = GPTMLP(embedding_size, dropout_prob))
        self.mlp = GPTMLP(embedding_dim, dropout)

    # train() and eval() are inherited from AbstractModuleBase
    # They automatically propagate RunMode to all registered submodules

    def forward(
        self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        """Forward pass of GPT block.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Output tensor after transformer block
        """
        # Pre-norm attention with residual (matching C++)
        # residual = input; x = (*ln1)(input); x = (*attention)(x, mask); x = ops::add(x, residual)
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = ttml.ops.binary.add(x, residual)

        # Pre-norm MLP with residual (matching C++)
        # residual = x; x = (*ln2)(x); x = (*mlp)(x); x = ops::add(x, residual)
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = ttml.ops.binary.add(x, residual)

        return x
