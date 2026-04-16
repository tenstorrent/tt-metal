# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 attention with separate Q/K/V projections and QK-Norm."""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

from .autograd_ops import ConcatLastDim, Qwen3RMSNorm


class Qwen3Attention(AbstractModuleBase):
    """Qwen3 grouped-query attention with QK-Norm.

    Key differences from Llama-style GQA:
      - Separate Q, K, V projections (not fused KV)
      - QK-Norm (RMSNorm on Q and K per head_dim)
      - Configurable attention bias
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim

        self.q_proj = LinearLayer(
            self.hidden_size,
            q_out,
            has_bias=config.attention_bias,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )
        self.k_proj = LinearLayer(
            self.hidden_size,
            kv_out,
            has_bias=config.attention_bias,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )
        self.v_proj = LinearLayer(
            self.hidden_size,
            kv_out,
            has_bias=config.attention_bias,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )
        self.o_proj = LinearLayer(
            q_out,
            self.hidden_size,
            has_bias=config.attention_bias,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Build RoPE params per layer (matching existing Qwen3 design)
        rope_scaling = ttml.ops.rope.RopeScalingParams()
        if config.rope_scaling_factor != 0.0 and config.rope_original_context_length != 0:
            rope_scaling.original_context_length = config.rope_original_context_length
            rope_scaling.scaling_factor = config.rope_scaling_factor
            rope_scaling.high_freq_factor = config.rope_high_freq_factor
            rope_scaling.low_freq_factor = config.rope_low_freq_factor

        self.rope_params = ttml.ops.rope.build_rope_params(
            sequence_length=config.max_position_embeddings,
            head_dim=self.head_dim,
            theta=config.rope_theta,
            rope_scaling_params=rope_scaling,
        )

    def forward(
        self,
        hidden_states: ttml.autograd.Tensor,
        attention_mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
        position_offset: int = 0,
    ) -> ttml.autograd.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape = q.shape()
        k_shape = k.shape()
        B, S = q_shape[0], q_shape[2]

        # QK-Norm: reshape to (B, 1, S*num_heads, head_dim), normalize, reshape back
        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.num_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(k, [B, 1, S * self.num_kv_heads, self.head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        # Concatenate K and V, then split into multi-head format
        kvs = ConcatLastDim.apply(k, v)
        (
            query_heads,
            key_heads,
            value_heads,
        ) = ttml.ops.multi_head_utils.grouped_heads_creation(q, kvs, self.num_heads, self.num_kv_heads)

        # RoPE positional encoding
        query_heads = ttml.ops.rope.rope(query_heads, self.rope_params, position_offset)
        key_heads = ttml.ops.rope.rope(key_heads, self.rope_params, position_offset)

        # KV cache: append new K/V and use full history for attention
        if past_key_values is not None:
            key_heads, value_heads = past_key_values.update(self.layer_idx, key_heads, value_heads)

        q_seq = query_heads.shape()[2]
        k_seq = key_heads.shape()[2]
        sdpa_fn = (
            ttml.ops.attention.scaled_dot_product_attention
            if q_seq == k_seq
            else ttml.ops.attention.scaled_dot_product_attention_composite
        )
        attn = sdpa_fn(query_heads, key_heads, value_heads, attention_mask)

        attn_output = ttml.ops.multi_head_utils.heads_fusion(attn)
        return self.o_proj(attn_output)
