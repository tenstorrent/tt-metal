# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GPT transformer block for NanoGPT."""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, Parameter

from .gpt_mlp import GPTMLP
from .multi_head_attention import MultiHeadAttention


class LayerNorm(AbstractModuleBase):
    """Layer normalization module with gamma and beta parameters."""

    def __init__(
        self,
        embedding_dim: int,
        bias: bool = True,
        use_composite_layernorm: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_composite_layernorm = use_composite_layernorm

        ln_shape = (1, 1, 1, embedding_dim)
        self.gamma = Parameter(ttml.init.ones()(ln_shape))

        if bias:
            self.beta = Parameter(ttml.init.zeros()(ln_shape))
        else:
            self.beta = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of layer norm.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """

        if self.use_composite_layernorm:
            layernorm_op = ttml.ops.layernorm.composite_layernorm
        else:
            layernorm_op = ttml.ops.layernorm.layernorm

        return layernorm_op(x, self.gamma.tensor, self.beta.tensor if self.beta else None)


class GPTBlock(AbstractModuleBase):
    """GPT transformer block with attention and MLP."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_composite_layernorm: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        # Layer norms
        self.ln1 = LayerNorm(embedding_dim, bias, use_composite_layernorm)
        self.ln2 = LayerNorm(embedding_dim, bias, use_composite_layernorm)

        # Attention and MLP
        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout,
        )
        self.mlp = GPTMLP(
            embedding_dim,
            dropout,
        )

    # train() and eval() are inherited from AbstractModuleBase
    # They automatically propagate RunMode to all registered submodules

    def forward(self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None) -> ttml.autograd.Tensor:
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
