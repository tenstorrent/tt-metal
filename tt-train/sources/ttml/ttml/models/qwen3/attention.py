# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 attention with QK-Norm and separate Q/K/V projections.

Key differences from Llama's GroupedQueryAttention:
  - Separate Q, K, V linear projections (not fused KV)
  - QK-Norm: RMSNorm applied per-head on Q and K before RoPE
  - Explicit head_dim that can differ from hidden_size / num_heads
    (e.g. Qwen3-0.6B: hidden=1024, heads=16, head_dim=128)
  - Optional attention bias on all projections
"""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter

from .autograd_ops import ConcatLastDim, RMSNormFunction


class _QKNorm(AbstractModuleBase):
    """RMSNorm for QK normalization (per-head, on head_dim)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, hidden_size)))

    def forward(self, hidden_states):
        return RMSNormFunction.apply(hidden_states, self.weight.tensor, self.eps)


class Qwen3Attention(AbstractModuleBase):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        rope_scaling_params = ttml.ops.rope.RopeScalingParams()
        rs = config.rope_scaling
        if rs.scaling_factor != 0.0 and rs.original_context_length != 0:
            rope_scaling_params.scaling_factor = rs.scaling_factor
            rope_scaling_params.high_freq_factor = rs.high_freq_factor
            rope_scaling_params.low_freq_factor = rs.low_freq_factor
            rope_scaling_params.original_context_length = rs.original_context_length

        self.rope_params = ttml.ops.rope.build_rope_params(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
            rope_scaling_params,
        )

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim

        self.q_proj = LinearLayer(
            self.hidden_size,
            q_out,
            config.attention_bias,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )
        self.k_proj = LinearLayer(
            self.hidden_size,
            kv_out,
            config.attention_bias,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )
        self.v_proj = LinearLayer(
            self.hidden_size,
            kv_out,
            config.attention_bias,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )
        self.o_proj = LinearLayer(
            q_out,
            self.hidden_size,
            config.attention_bias,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )

        self.q_norm = _QKNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _QKNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
        position_offset: int = 0,
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape = q.shape()
        k_shape = k.shape()
        B, S = q_shape[0], q_shape[2]

        # Reshape to (B, 1, S*num_heads, head_dim) for per-head QK-Norm
        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.num_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(k, [B, 1, S * self.num_kv_heads, self.head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        kvs = ConcatLastDim.apply(k, v)

        query_heads, key_heads, value_heads = ttml.ops.multi_head_utils.grouped_heads_creation(
            q,
            kvs,
            self.num_heads,
            self.num_kv_heads,
        )

        query_heads = ttml.ops.rope.rope(query_heads, self.rope_params, position_offset)
        key_heads = ttml.ops.rope.rope(key_heads, self.rope_params, position_offset)

        if past_key_values is not None:
            key_heads, value_heads = past_key_values.update(self.layer_idx, key_heads, value_heads)

        q_seq = query_heads.shape()[2]
        k_seq = key_heads.shape()[2]
        # TODO: #39558
        sdpa_fn = (
            ttml.ops.attention.scaled_dot_product_attention
            if q_seq == k_seq
            else ttml.ops.attention.scaled_dot_product_attention_composite
        )
        attn = sdpa_fn(query_heads, key_heads, value_heads, attention_mask)

        attn_output = ttml.ops.multi_head_utils.heads_fusion(attn)
        return self.o_proj(attn_output)
