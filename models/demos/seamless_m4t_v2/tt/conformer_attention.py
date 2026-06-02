# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT port of `SeamlessM4Tv2ConformerSelfAttention` (Phase 2).

Multi-head self-attention with optional Shaw `relative_key` position bias.
Reuses the proven SpeechT5-encoder pattern: fused QKV projection ->
`nlp_create_qkv_heads` -> scaled Q@K^T -> (optional) relative-position matmul ->
mask -> softmax -> @V -> `nlp_concat_heads` -> output projection.

Query is scaled by 1/sqrt(head_dim) once, so both the Q@K^T term and the
relative-position term carry the 1/sqrt(head_dim) factor, matching the reference.

The relative-position table (seq, seq, head_dim) is data-independent and is
precomputed on host per seq_len (see rel_pos_bias.build_position_bias); the
einsum itself runs on device as a matmul. head_dim must be 64 for the matmul.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.seamless_m4t_v2.tt.rel_pos_bias import build_position_bias


class TtConformerSelfAttention:
    def __init__(self, weights: dict[str, torch.Tensor], config, device, use_position_embeddings=True,
                 dtype=ttnn.bfloat16):
        """
        weights: state-dict slice for one `self_attn.` (prefix stripped):
            linear_q.{weight,bias}, linear_k.{weight,bias}, linear_v.{weight,bias},
            linear_out.{weight,bias}, [distance_embedding.weight]
        """
        self.device = device
        self.dtype = dtype
        self.num_heads = config.speech_encoder_attention_heads
        self.head_dim = config.head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.use_pos = use_position_embeddings and config.position_embeddings_type == "relative_key"
        self.left = config.left_max_position_embeddings
        self.right = config.right_max_position_embeddings

        def to_tt(t):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        # Fuse q/k/v into a single projection: weight (3C, C) -> linear weight (C, 3C)
        qkv_w = torch.cat([weights["linear_q.weight"], weights["linear_k.weight"], weights["linear_v.weight"]], dim=0)
        qkv_b = torch.cat([weights["linear_q.bias"], weights["linear_k.bias"], weights["linear_v.bias"]], dim=0)
        self.qkv_w = to_tt(qkv_w.t().contiguous())
        self.qkv_b = to_tt(qkv_b)

        self.out_w = to_tt(weights["linear_out.weight"].t().contiguous())
        self.out_b = to_tt(weights["linear_out.bias"])

        self._dist_emb = weights["distance_embedding.weight"].float() if self.use_pos else None
        self._pos_bias_cache: dict[int, object] = {}

    def _position_bias(self, seq_len):
        if seq_len not in self._pos_bias_cache:
            pb = build_position_bias(seq_len, self._dist_emb, self.left, self.right)  # (S, S, d)
            self._pos_bias_cache[seq_len] = ttnn.from_torch(
                pb, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._pos_bias_cache[seq_len]

    def __call__(self, hidden, attention_mask=None):
        """hidden: (1, seq, C) TILE. attention_mask: optional additive (1,1,seq) or broadcastable."""
        seq_len = hidden.shape[1]
        H, d = self.num_heads, self.head_dim

        qkv = ttnn.linear(hidden, self.qkv_w, bias=self.qkv_b)  # (1, seq, 3C)
        if len(qkv.shape) == 3:
            qkv = ttnn.unsqueeze(qkv, dim=1)  # (1, 1, seq, 3C)

        q, k_t, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=True
        )
        ttnn.deallocate(qkv)
        # q: (1,H,seq,d)  k_t: (1,H,d,seq)  v: (1,H,seq,d)
        q = ttnn.multiply(q, self.scaling)

        q = ttnn.reshape(q, [H, seq_len, d])
        k_t = ttnn.reshape(k_t, [H, d, seq_len])
        v = ttnn.reshape(v, [H, seq_len, d])

        attn = ttnn.matmul(q, k_t)  # (H, seq, seq)

        if self.use_pos:
            pos = self._position_bias(seq_len)  # (seq, seq, d)
            reshape_q = ttnn.permute(q, [1, 0, 2])  # (seq, H, d)
            pos_t = ttnn.permute(pos, [0, 2, 1])  # (seq, d, seq)
            rel = ttnn.matmul(reshape_q, pos_t)  # (seq, H, seq)
            rel = ttnn.permute(rel, [1, 0, 2])  # (H, seq, seq)
            attn = ttnn.add(attn, rel)

        if attention_mask is not None:
            attn = ttnn.add(attn, attention_mask)

        attn = ttnn.softmax(attn, dim=-1)
        out = ttnn.matmul(attn, v)  # (H, seq, d)

        out = ttnn.reshape(out, [1, H, seq_len, d])
        out = ttnn.experimental.nlp_concat_heads(out)  # (1, 1, seq, C)
        out = ttnn.squeeze(out, dim=1)  # (1, seq, C)
        out = ttnn.linear(out, self.out_w, bias=self.out_b)
        return out
