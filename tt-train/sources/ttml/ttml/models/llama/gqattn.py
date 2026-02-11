# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Grouped-query attention layer for Llama."""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, RunMode


class GroupedQueryAttention(AbstractModuleBase):
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        num_groups: int,
        dropout: float,
        rope_params: ttml.ops.rope.RotaryEmbeddingParams,
        bias_linears: bool = False,
    ) -> None:
        super().__init__()

        assert (
            embedding_size % num_heads == 0
        ), "embedding_size must be divisible by num_heads"

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.dropout_prob = dropout
        self.rope_params = rope_params

        concat_kv_dim = 2 * num_groups * (embedding_size // num_heads)

        self.q_linear = LinearLayer(embedding_size, embedding_size, bias_linears)
        self.kv_linear = LinearLayer(embedding_size, concat_kv_dim, bias_linears)
        self.out_linear = LinearLayer(embedding_size, embedding_size, bias_linears)

    def forward(
        self, input: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        q = self.q_linear(input)
        kv = self.kv_linear(input)

        q_heads, k_heads, v_heads = ttml.ops.multi_head_utils.grouped_heads_creation(
            q, kv, self.num_heads, self.num_groups
        )

        q_heads = ttml.ops.rope.rope(q_heads, self.rope_params)
        k_heads = ttml.ops.rope.rope(k_heads, self.rope_params)

        attention = ttml.ops.attention.scaled_dot_product_attention(
            q_heads, k_heads, v_heads, mask
        )

        out = self.out_linear(attention)

        # Apply dropout if in training mode (using RunMode from AbstractModuleBase)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)

        return out
