# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama transformer block."""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, Parameter, RunMode, LinearLayer, ColumnParallelLinear, RowParallelLinear

from .gqattn import GroupedQueryAttention


def compute_swiglu_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """Compute the default MLP intermediate size for Llama.

    Meta's Llama uses SwiGLU which has 3 matrices (w1, w2, w3) instead of 2 in a
    standard MLP. To match the parameter count of a conventional 4x MLP, the
    intermediate size is scaled to 2/3 of 4*hidden = 8/3*hidden, then rounded up
    to ``multiple_of`` for hardware alignment.
    """
    unrounded = (4 * hidden_size * 2) // 3
    return ((unrounded + multiple_of - 1) // multiple_of) * multiple_of


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
        self.gamma = Parameter(ttml.init.ones()(gamma_shape))

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
        use_tp: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.dropout_prob = dropout

        if intermediate_size is None:
            intermediate_size = compute_swiglu_intermediate_size(embedding_size)

        if use_tp:
            # Gate (w1) and up (w3) projections are column-parallel (output sharded);
            # down projection (w2) is row-parallel (input sharded).  The pair
            # eliminates a gather/scatter round-trip between w1/w3 and w2.
            self.w1 = ColumnParallelLinear(
                embedding_size,
                intermediate_size,
                has_bias=False,
                gather_output=False,
                axis_name="tp",
            )
            self.w3 = ColumnParallelLinear(
                embedding_size,
                intermediate_size,
                has_bias=False,
                gather_output=False,
                axis_name="tp",
            )
            self.w2 = RowParallelLinear(
                intermediate_size,
                embedding_size,
                has_bias=False,
                input_is_parallel=True,
                axis_name="tp",
            )
        else:
            self.w1 = LinearLayer(
                embedding_size,
                intermediate_size,
                False,
            )
            self.w3 = LinearLayer(
                embedding_size,
                intermediate_size,
                False,
            )
            self.w2 = LinearLayer(
                intermediate_size,
                embedding_size,
                False,
            )

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
    """Pre-norm residual transformer block (attention + MLP)."""

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
        use_tp: bool = False,
    ) -> None:
        super().__init__()

        self.mlp = LlamaMLP(
            hidden_size,
            intermediate_size,
            mlp_dropout,
            use_tp=use_tp,
        )
        self.attention_norm = RMSNormLayer(hidden_size)
        self.mlp_norm = RMSNormLayer(hidden_size)
        self.attention = GroupedQueryAttention(
            embedding_size=hidden_size,
            num_heads=num_attention_heads,
            num_groups=num_key_value_heads,
            dropout=attention_dropout,
            rope_params=rope_params,
            bias_linears=attention_bias,
            use_tp=use_tp,
        )

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
