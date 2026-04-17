# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Grouped-query attention for Qwen3.

Differs from ttml's standard GQA in three places:
  - Separate Q / K / V projections (HF layout), not a fused KV matrix.
  - QK-Norm: per-head RMSNorm applied to Q and K before RoPE.
  - Explicit ``head_dim`` (not ``hidden_size // num_heads``) and configurable
    attention bias.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

from .rmsnorm import Qwen3RMSNorm


class _ConcatLastDim(ttml.autograd.Function):
    """Autograd-aware concat along the last dim (K, V fusion before head split)."""

    @staticmethod
    def forward(ctx, a, b):
        a_shape = a.shape()
        b_shape = b.shape()
        ctx.save_for_backward(a_shape, b_shape)
        return ttml.autograd.create_tensor(ttnn.concat([a.get_value(), b.get_value()], dim=-1))

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        a_last = a_shape[-1]
        b_last = b_shape[-1]
        grad_a = ttnn.slice(grad_output, [0, 0, 0, 0], [a_shape[0], a_shape[1], a_shape[2], a_last])
        grad_b = ttnn.slice(
            grad_output,
            [0, 0, 0, a_last],
            [b_shape[0], b_shape[1], b_shape[2], a_last + b_last],
        )
        return grad_a, grad_b


class Qwen3Attention(AbstractModuleBase):
    """Grouped-query attention with QK-Norm.

    Shapes assume 4-D tensors ``(B, 1, S, ·)``. ``num_kv_heads`` divides
    ``num_heads``; the grouped-head split is handled by
    ``ttml.ops.multi_head_utils.grouped_heads_creation``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_params: ttml.ops.rope.RotaryEmbeddingParams,
        attention_bias: bool = True,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_params = rope_params

        q_out = num_attention_heads * head_dim
        kv_out = num_key_value_heads * head_dim

        # Separate Q/K/V/O projections (HF layout). A fused kv matrix would
        # break the HF safetensors key map used by the loader.
        self.q_proj = LinearLayer(hidden_size, q_out, has_bias=attention_bias)
        self.k_proj = LinearLayer(hidden_size, kv_out, has_bias=attention_bias)
        self.v_proj = LinearLayer(hidden_size, kv_out, has_bias=attention_bias)
        self.o_proj = LinearLayer(q_out, hidden_size, has_bias=attention_bias)

        # QK-Norm: per-head RMSNorm over head_dim, applied to Q and K before RoPE.
        self.q_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        layer_idx: Optional[int] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        # Project Q/K/V from hidden (B, 1, S, hidden) to (B, 1, S, num_heads·head_dim)
        # and (B, 1, S, num_kv_heads·head_dim) respectively.
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape = q.shape()
        k_shape = k.shape()
        B, S = q_shape[0], q_shape[2]

        # Per-head QK-Norm: fold the head dim into the seq dim so a single
        # RMSNorm over the last axis normalizes each head independently.
        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.num_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(k, [B, 1, S * self.num_kv_heads, self.head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        # Fuse K and V along the last dim so grouped_heads_creation can split
        # Q plus a single KV tensor into (q_heads, k_heads, v_heads).
        kvs = _ConcatLastDim.apply(k, v)
        query_heads, key_heads, value_heads = ttml.ops.multi_head_utils.grouped_heads_creation(
            q, kvs, self.num_heads, self.num_kv_heads
        )

        # RoPE: offset by the current KV-cache position so decode steps line
        # up with the cached prefix.
        position_offset = kv_cache.get_cache_position() if kv_cache is not None else 0
        query_heads = ttml.ops.rope.rope(query_heads, self.rope_params, position_offset)
        key_heads = ttml.ops.rope.rope(key_heads, self.rope_params, position_offset)

        # KV-cache update convention:
        #   - Prefill (cache_position == 0): write the fresh K/V and attend
        #     over just this step — caller passed a square causal mask.
        #   - Decode (cache_position > 0):   write the new row, then attend
        #     over the full cached K/V — caller passed a slab mask.
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("forward with kv_cache requires layer_idx")
            is_prefill = kv_cache.get_cache_position() == 0
            tokens_to_write = new_tokens if new_tokens is not None else key_heads.shape()[2]
            kv_cache.update(layer_idx, key_heads.get_value(), value_heads.get_value(), tokens_to_write)
            if not is_prefill:
                key_heads = ttml.autograd.create_tensor(kv_cache.get_k_cache(layer_idx), requires_grad=False)
                value_heads = ttml.autograd.create_tensor(kv_cache.get_v_cache(layer_idx), requires_grad=False)

        # Full SDPA op fuses softmax+matmul and requires q_seq == k_seq.
        # Decode (q_seq=1, k_seq=max_seq_len) falls back to the composite variant.
        q_seq = query_heads.shape()[2]
        k_seq = key_heads.shape()[2]
        sdpa_fn = (
            ttml.ops.attention.scaled_dot_product_attention
            if q_seq == k_seq
            else ttml.ops.attention.scaled_dot_product_attention_composite
        )
        attn = sdpa_fn(query_heads, key_heads, value_heads, mask)
        attn = ttml.ops.multi_head_utils.heads_fusion(attn)
        return self.o_proj(attn)


def build_attn_mask(
    seq_len: int,
    *,
    kv_cache: Optional[ttml.models.KvCache] = None,
    max_seq_len: Optional[int] = None,
) -> ttml.autograd.Tensor:
    """Build a Qwen3 attention mask.

    Without ``kv_cache`` (or at position 0): square causal mask ``[1, 1, seq_len, seq_len]``.
    With ``kv_cache`` at position > 0 (decode step): slab mask
    ``[1, 1, 1, max_seq_len]`` with unmasked positions ``[0, cache_pos]``.
    """
    if kv_cache is None or kv_cache.get_cache_position() == 0:
        mask = np.tril(np.ones((seq_len, seq_len), dtype=ml_dtypes.bfloat16)).reshape(1, 1, seq_len, seq_len)
        return ttml.autograd.Tensor.from_numpy(mask, layout=ttnn.Layout.TILE)

    if max_seq_len is None:
        raise ValueError("build_attn_mask requires max_seq_len during decode steps")
    cache_pos = kv_cache.get_cache_position()
    row = np.zeros((max_seq_len,), dtype=ml_dtypes.bfloat16)
    row[: cache_pos + 1] = 1
    mask = row.reshape(1, 1, 1, max_seq_len)
    return ttml.autograd.Tensor.from_numpy(mask, layout=ttnn.Layout.TILE)
