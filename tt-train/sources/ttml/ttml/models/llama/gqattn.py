# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Grouped-query attention layer for Llama."""

from __future__ import annotations

from typing import Callable, Optional

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ColumnParallelLinear, RowParallelLinear, RunMode


class GroupedQueryAttention(AbstractModuleBase):
    """Grouped-query attention (GQA) with optional tensor-parallel linear layers.

    Under tensor parallelism the fused QKV projection uses ``ColumnParallelLinear``
    (output features sharded) with ``gather_output=False``, and the output projection
    uses ``RowParallelLinear`` with ``input_is_parallel=True``. This avoids redundant
    communication between the two matmuls.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        num_groups: int,
        dropout: float,
        rope_params: ttml.ops.rope.RotaryEmbeddingParams,
        bias_linears: bool = False,
        use_tp: bool = False,
        out_proj_init: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError(
                "Embedding size must be divisible by the number of attention heads. "
                f"Provided embedding_size={embedding_size}, num_heads={num_heads}"
            )

        self.embedding_size = embedding_size
        self.dropout_prob = dropout
        self.rope_params = rope_params
        # Distinct mask per device only when each holds distinct data. Too coarse under DP+TP:
        # ttnn offsets the seed by flat device id, so sharing within TP also shares across DP.
        self.dropout_per_device_seed = not use_tp

        head_dim = embedding_size // num_heads
        qkv_dim = (num_heads + 2 * num_groups) * head_dim  # == embedding_size + 2 * num_groups * head_dim

        if use_tp:
            tp_size = ttml.mesh().axis_size("tp")
            if num_heads % tp_size != 0:
                raise ValueError(f"num_heads ({num_heads}) must be divisible by the tensor-parallel size ({tp_size})")
            if num_groups % tp_size != 0:
                raise ValueError(f"num_groups ({num_groups}) must be divisible by the tensor-parallel size ({tp_size})")
            self.num_heads = num_heads // tp_size
            self.num_groups = num_groups // tp_size
        else:
            self.num_heads = num_heads
            self.num_groups = num_groups

        if use_tp:
            self.qkv_linear = ColumnParallelLinear(
                embedding_size,
                qkv_dim,
                has_bias=bias_linears,
                bias_init=ttml.init.zeros(),
                gather_output=False,
                axis_name="tp",
            )
            self.out_linear = RowParallelLinear(
                embedding_size,
                embedding_size,
                has_bias=bias_linears,
                weight_init=out_proj_init,
                bias_init=ttml.init.zeros(),
                input_is_parallel=True,
                axis_name="tp",
            )
        else:
            self.qkv_linear = LinearLayer(
                embedding_size,
                qkv_dim,
                bias_linears,
                bias_init=ttml.init.zeros(),
            )
            self.out_linear = LinearLayer(
                embedding_size,
                embedding_size,
                bias_linears,
                weight_init=out_proj_init,
                bias_init=ttml.init.zeros(),
            )

    def sdpa(
        self,
        q_heads: ttml.autograd.Tensor,
        k_heads: ttml.autograd.Tensor,
        v_heads: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
    ) -> ttml.autograd.Tensor:
        """The attention kernel; assign per instance or override to swap it."""
        return ttml.ops.attention.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

    def forward_no_kv(self, input: ttml.autograd.Tensor, mask: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        qkv = self.qkv_linear(input)

        q_heads, k_heads, v_heads = ttml.ops.multi_head_utils.heads_creation(qkv, self.num_heads, self.num_groups)

        q_heads = ttml.ops.rope.rope(q_heads, self.rope_params)
        k_heads = ttml.ops.rope.rope(k_heads, self.rope_params)

        attention = self.sdpa(q_heads, k_heads, v_heads, mask)
        attention = ttml.ops.multi_head_utils.heads_fusion(attention)

        out = self.out_linear(attention)

        # Apply dropout if in training mode (using RunMode from AbstractModuleBase)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob, use_per_device_seed=self.dropout_per_device_seed)

        return out

    def forward_kv(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: ttml.models.KvCache,
        layer_idx: int,
        new_tokens: int,
    ) -> ttml.autograd.Tensor:
        qkv = self.qkv_linear(input)

        q_heads, k_heads, v_heads = ttml.ops.multi_head_utils.heads_creation(qkv, self.num_heads, self.num_groups)

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

        attention = self.sdpa(q_heads, k_cache_to_process, v_cache_to_process, mask)
        attention = ttml.ops.multi_head_utils.heads_fusion(attention)

        out = self.out_linear(attention)

        # Apply dropout if in training mode (using RunMode from AbstractModuleBase)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob, use_per_device_seed=self.dropout_per_device_seed)

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
            raise ValueError("forward with kv_cache requires layer_idx and new_tokens to be set")
        return self.forward_kv(input, mask, kv_cache, layer_idx, new_tokens)
