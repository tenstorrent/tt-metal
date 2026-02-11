# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama transformer block."""

from __future__ import annotations

from typing import Optional

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter

from .gqattn import GroupedQueryAttention


class RMSNormLayer(AbstractModuleBase):
    def __init__(
        self,
        features: int,
        epsilon: float = 1e-5,
        use_composite: bool = False,
    ) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.use_composite = use_composite

        gamma_shape = (1, 1, 1, features)
        gamma_np = np.ones(gamma_shape, dtype=ml_dtypes.bfloat16)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(
            gamma_np, layout=ttnn.Layout.TILE
        )
        self.gamma = Parameter(gamma_tensor)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of RMSNorm.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """

        if self.use_composite:
            rmsnorm_op = ttml.ops.rmsnorm.rmsnorm_composite
        else:
            rmsnorm_op = ttml.ops.rmsnorm.rmsnorm

        return rmsnorm_op(x, self.gamma.tensor, self.epsilon)


class LlamaMLP(AbstractModuleBase):
    """Llama-style MLP (feed-forward) layer."""

    def __init__(
        self,
        embedding_size: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize Llama MLP layer.

        Args:
            embedding_size: Dimension of embeddings
            intermediate_dim: Intermediate dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.dropout_prob = dropout

        multiple_of = 256

        if intermediate_size is None:
            unrounded_size = (4 * embedding_size * 2) // 3
            intermediate_size = (
                (unrounded_size + multiple_of - 1) // multiple_of
            ) * multiple_of

        self.w1 = LinearLayer(embedding_size, intermediate_size, False)
        self.w3 = LinearLayer(embedding_size, intermediate_size, False)
        self.w2 = LinearLayer(intermediate_size, embedding_size, False)

    def forward(self, input: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor after MLP
        """
        swished = ttml.ops.unary.silu(self.w1(input))
        gate = self.w3(input)
        gated = ttml.ops.binary.mul(swished, gate)
        x = self.w2(gated)

        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            x = ttml.ops.dropout.dropout(x, self.dropout_prob)

        return x


class LlamaBlock(AbstractModuleBase):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_params: ttml.ops.rope.RotaryEmbeddingParams,
        attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        num_groups = num_attention_heads // num_key_value_heads

        self.mlp = LlamaMLP(hidden_size, intermediate_size, mlp_dropout)
        self.attention_norm = RMSNormLayer(hidden_size)
        self.mlp_norm = RMSNormLayer(hidden_size)
        self.attention = GroupedQueryAttention(
            embedding_size=hidden_size,
            num_heads=num_attention_heads,
            num_groups=num_groups,
            dropout=attention_dropout,
            rope_params=rope_params,
        )

    def forward(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
    ) -> ttml.autograd.Tensor:
        residual = input
        h = self.attention_norm(input)
        h = self.attention(h, mask)
        h = ttml.ops.binary.add(h, residual)

        residual = h
        x = self.mlp_norm(h)
        x = self.mlp(x)
        x = ttml.ops.binary.add(x, residual)

        return x
