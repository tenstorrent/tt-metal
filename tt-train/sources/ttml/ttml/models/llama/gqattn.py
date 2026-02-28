# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Grouped-query attention layer for Llama."""

from __future__ import annotations

from typing import Optional

import ttnn
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

        if embedding_size % num_heads != 0:
            raise ValueError(
                "Embedding size must be divisible by the number of attention heads. "
                f"Provided embedding_size={embedding_size}, num_heads={num_heads}"
            )

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.dropout_prob = dropout
        self.rope_params = rope_params

        concat_kv_dim = 2 * num_groups * (embedding_size // num_heads)

        self.q_linear = LinearLayer(embedding_size, embedding_size, bias_linears)
        self.kv_linear = LinearLayer(embedding_size, concat_kv_dim, bias_linears)
        self.out_linear = LinearLayer(embedding_size, embedding_size, bias_linears)

    def forward_no_kv(
        self, input: ttml.autograd.Tensor, mask: ttml.autograd.Tensor
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
        attention = ttml.ops.multi_head_utils.heads_fusion(attention)

        out = self.out_linear(attention)

        # Apply dropout if in training mode (using RunMode from AbstractModuleBase)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)

        return out

    def forward_kv(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: ttml.models.KvCache,
        layer_idx: int,
        new_tokens: int,
    ) -> ttml.autograd.Tensor:
        q = self.q_linear(input)
        kv = self.kv_linear(input)

        q_heads, k_heads, v_heads = ttml.ops.multi_head_utils.grouped_heads_creation(
            q, kv, self.num_heads, self.num_groups
        )

        token_pos = kv_cache.get_cache_position()

        q_heads = ttml.ops.rope.rope(q_heads, self.rope_params, token_pos)
        k_heads = ttml.ops.rope.rope(k_heads, self.rope_params, token_pos)

        kv_cache.update(layer_idx, k_heads.get_value(), v_heads.get_value(), new_tokens)

        k_cache = kv_cache.get_k_cache(layer_idx)
        v_cache = kv_cache.get_v_cache(layer_idx)

        token_end = [
            k_cache.shape[0],
            k_cache.shape[1],
            mask.shape()[-1],
            k_cache.shape[3],
        ]

        step = [1, 1, 1, 1]
        k_cache_slice = ttnn.slice(k_cache, [0, 0, 0, 0], token_end, step)
        v_cache_slice = ttnn.slice(v_cache, [0, 0, 0, 0], token_end, step)

        k_cache_to_process = ttml.autograd.create_tensor(k_cache_slice)
        v_cache_to_process = ttml.autograd.create_tensor(v_cache_slice)

        attention = ttml.ops.attention.scaled_dot_product_attention(
            q_heads, k_cache_to_process, v_cache_to_process, mask
        )
        attention = ttml.ops.multi_head_utils.heads_fusion(attention)

        out = self.out_linear(attention)

        # Apply dropout if in training mode (using RunMode from AbstractModuleBase)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)

        return out

    def forward(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        layer_idx: Optional[int] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        if kv_cache is None:
            return self.forward_no_kv(input, mask)
        if layer_idx is None or new_tokens is None:
            raise ValueError(
                "forward with kv_cache requires layer_idx and new_tokens to be set"
            )
        return self.forward_kv(input, mask, kv_cache, layer_idx, new_tokens)
