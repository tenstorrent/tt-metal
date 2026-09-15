# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gated Attention, the ``full_attention`` layer of Qwen3.8 (16 of 64 layers).

Standard grouped-query attention (24 query heads over 4 KV heads) with two
Qwen3.8-specific twists:

**Output gate.** ``q_proj`` is twice as wide as the query needs: each head's
512-wide block is ``[query(256) | gate(256)]``.  The gate does not enter the
attention computation at all -- it is a sigmoid applied to the attention output
just before ``o_proj``.

**Partial RoPE.** ``partial_rotary_factor`` is 0.25, so only the first 64 of
each head's 256 dims are rotated; the remaining 192 pass through untouched.

Q/K normalization is Qwen3.8's zero-centered RMSNorm, ``x_normed * (1 + w)``.
That is a plain RMSNorm against ``1 + w``, so no separate op is needed: the
checkpoint loader folds the ``+1`` into the stored weight (see
:mod:`ttml.models.qwen38.loading`).

Only text is supported, so the checkpoint's interleaved mRoPE degenerates to
ordinary RoPE: with no image or video tokens all three mRoPE sections index the
same monotonic text positions.
"""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter

from .autograd_ops import autograd_concat, autograd_slice
from .parallel import make_column_linear, make_row_linear, tp_size

__all__ = ["Qwen38GatedAttention"]

_mul = ttml.ops.binary.mul
_reshape = ttml.ops.reshape.reshape
_sigmoid = ttml.ops.unary.sigmoid
_transpose = ttml.ops.unary.transpose
_rmsnorm = ttml.ops.rmsnorm.rmsnorm


class _QKNorm(AbstractModuleBase):
    """Per-head RMSNorm over ``head_dim``, applied to Q and K before RoPE.

    Qwen3.8's variant is zero-centered (``* (1 + w)``); the loader stores
    ``1 + w`` so this stays a plain RMSNorm.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, head_dim)))

    def forward(self, hidden_states):
        return _rmsnorm(hidden_states, self.weight.tensor, self.eps)


class Qwen38GatedAttention(AbstractModuleBase):
    """The ``self_attn`` submodule of a Qwen3.8 full-attention layer."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        # Per-chip head counts. TP is capped at 4 by num_key_value_heads=4;
        # ttml's distributed GQA requires the KV groups to divide the TP width.
        tp = tp_size(config)
        self.tp = tp
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = config.num_key_value_heads // tp
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.output_gate = config.attn_output_gate
        self.rotary_dim = config.rotary_dim

        self.rope_params = ttml.ops.rope.build_rope_params(
            config.max_position_embeddings,
            self.rotary_dim,
            config.rope_theta,
            ttml.ops.rope.RopeScalingParams(),
        )

        init = ttml.init.normal(0.0, 0.02)
        q_width = config.num_attention_heads * self.head_dim * (2 if self.output_gate else 1)
        kv_width = config.num_key_value_heads * self.head_dim

        # q_proj's output is laid out per head as [query | gate], so a
        # contiguous column shard keeps every head with its own gate and needs
        # no reordering (unlike the DeltaNet's fused QKV).
        self.q_proj = make_column_linear(config, self.hidden_size, q_width, init)
        self.k_proj = make_column_linear(config, self.hidden_size, kv_width, init)
        self.v_proj = make_column_linear(config, self.hidden_size, kv_width, init)
        self.o_proj = make_row_linear(config, config.num_attention_heads * self.head_dim, self.hidden_size, init)

        self.q_norm = _QKNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _QKNorm(self.head_dim, eps=config.rms_norm_eps)

    def _split_heads(self, x, num_heads: int):
        """``[B, 1, T, H * D]`` -> ``[B, H, T, D]``, the layout ttml's SDPA wants."""
        batch, _, seq, _ = [int(d) for d in x.shape()]
        x = _reshape(x, [batch, seq, num_heads, self.head_dim])
        return _transpose(x, 1, 2)

    def _partial_rope(self, x, position_offset: int):
        """Rotate only the leading ``rotary_dim`` features, pass the rest through."""
        batch, heads, seq, dim = [int(d) for d in x.shape()]
        if self.rotary_dim == dim:
            return ttml.ops.rope.rope(x, self.rope_params, position_offset)

        rotated = autograd_slice(x, [0, 0, 0, 0], [batch, heads, seq, self.rotary_dim])
        passthrough = autograd_slice(x, [0, 0, 0, self.rotary_dim], [batch, heads, seq, dim])
        rotated = ttml.ops.rope.rope(rotated, self.rope_params, position_offset)
        return autograd_concat([rotated, passthrough], 3)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
        position_offset: int = 0,
    ):
        batch, _, seq, _ = [int(d) for d in hidden_states.shape()]

        # q_proj is 2x wide: each head's block is [query | gate]. Give the heads
        # their own axis so the split is a slice on the feature dim.
        qg = self.q_proj(hidden_states)
        if self.output_gate:
            per_head = _reshape(qg, [batch, seq, self.num_heads, 2 * self.head_dim])
            query = autograd_slice(per_head, [0, 0, 0, 0], [batch, seq, self.num_heads, self.head_dim])
            gate = autograd_slice(
                per_head,
                [0, 0, 0, self.head_dim],
                [batch, seq, self.num_heads, 2 * self.head_dim],
            )
            # The gate is consumed in [B, 1, T, H * D] form, matching the output.
            gate = _reshape(gate, [batch, 1, seq, self.num_heads * self.head_dim])
            query_heads = _transpose(query, 1, 2)  # [B, H, T, D]
        else:
            gate = None
            query_heads = self._split_heads(qg, self.num_heads)

        key_heads = self._split_heads(self.k_proj(hidden_states), self.num_kv_heads)
        value_heads = self._split_heads(self.v_proj(hidden_states), self.num_kv_heads)

        # QK-Norm before RoPE, matching HF ordering. V is left unnormed.
        query_heads = self.q_norm(query_heads)
        key_heads = self.k_norm(key_heads)

        query_heads = self._partial_rope(query_heads, position_offset)
        key_heads = self._partial_rope(key_heads, position_offset)

        if past_key_values is not None:
            key_heads, value_heads = past_key_values.update(self.layer_idx, key_heads, value_heads)

        # ttml's SDPA broadcasts the 4 KV heads across the 24 query heads itself.
        q_seq = int(query_heads.shape()[2])
        k_seq = int(key_heads.shape()[2])
        sdpa_fn = (
            ttml.ops.attention.scaled_dot_product_attention
            if q_seq == k_seq
            else ttml.ops.attention.scaled_dot_product_attention_composite
        )
        attn = sdpa_fn(query_heads, key_heads, value_heads, attention_mask)

        attn_output = ttml.ops.multi_head_utils.heads_fusion(attn)  # [B, 1, T, H * D]
        if gate is not None:
            attn_output = _mul(attn_output, _sigmoid(gate))
        return self.o_proj(attn_output)
