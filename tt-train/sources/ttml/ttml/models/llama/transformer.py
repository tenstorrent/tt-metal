# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama transformer block."""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, Parameter, RunMode, LinearLayer
from ttml.modules.parameter import TensorMetadata

from .gqattn import GroupedQueryAttention


class RMSNormLayer(AbstractModuleBase):
    def __init__(
        self,
        features: int,
        epsilon: float = 1e-5,
        use_composite: bool = False,
        **kwargs,
    ) -> None:
        self.epsilon = epsilon
        self.use_composite = use_composite

        self.gamma = Parameter(
            TensorMetadata(
                shape=(1, 1, 1, features),
                init_fn=ttml.init.ones(),
            )
        )

        super().__init__(**kwargs)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
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
        **kwargs,
    ) -> None:
        self.embedding_size = embedding_size
        self.dropout_prob = dropout

        multiple_of = 256

        if intermediate_size is None:
            unrounded_size = (4 * embedding_size * 2) // 3
            intermediate_size = (
                (unrounded_size + multiple_of - 1) // multiple_of
            ) * multiple_of

        self.w1 = LinearLayer(embedding_size, intermediate_size, False, **kwargs)
        self.w3 = LinearLayer(embedding_size, intermediate_size, False, **kwargs)
        self.w2 = LinearLayer(intermediate_size, embedding_size, False, **kwargs)

        super().__init__(**kwargs)

    def forward(self, input: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
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
        attention_bias: bool = False,
        **kwargs,
    ) -> None:
        self.mlp = LlamaMLP(hidden_size, intermediate_size, mlp_dropout, **kwargs)
        self.attention_norm = RMSNormLayer(hidden_size, **kwargs)
        self.mlp_norm = RMSNormLayer(hidden_size, **kwargs)
        self.attention = GroupedQueryAttention(
            embedding_size=hidden_size,
            num_heads=num_attention_heads,
            num_groups=num_key_value_heads,
            dropout=attention_dropout,
            rope_params=rope_params,
            bias_linears=attention_bias,
            **kwargs,
        )

        super().__init__(**kwargs)

    def forward(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        layer_idx: Optional[int] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        residual = input
        h = self.attention_norm(input)
        h = self.attention(h, mask, kv_cache, layer_idx, new_tokens)
        h = ttml.ops.binary.add(h, residual)

        residual = h
        x = self.mlp_norm(h)
        x = self.mlp(x)
        x = ttml.ops.binary.add(x, residual)

        return x
