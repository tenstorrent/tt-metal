# tt/attention/cross_attention.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
Cross-attention: decoder queries attend over encoder hidden states.

tst_cross_attention projects K/V from encoder_hidden inline every call.
precompute_cross_attn_kv + tst_cross_attention_with_kv split that in two:
K/V are projected once (encoder hidden state never changes across the
autoregressive decode loop), then only Q is projected per step.
tst_cross_attention delegates to both, so the K/V projection logic exists
in exactly one place.
"""

import ttnn

from ..tst_config import HEAD_DIM_PADDED, NUM_HEADS
from .ops import attend, fused_unmasked_softmax


def precompute_cross_attn_kv(encoder_hidden, w):
    """K/V projected from encoder_hidden, shaped for matmul against Q."""
    B = encoder_hidden.shape[0]
    T_enc = encoder_hidden.shape[1]
    kv_half = NUM_HEADS * HEAD_DIM_PADDED

    fused_kv = ttnn.linear(encoder_hidden, w["kv_weight"], bias=w["kv_bias"])
    k_proj = ttnn.slice(fused_kv, slice_start=[0, 0, 0], slice_end=[B, T_enc, kv_half])
    v_proj = ttnn.slice(fused_kv, slice_start=[0, 0, kv_half], slice_end=[B, T_enc, 2 * kv_half])

    k = ttnn.reshape(k_proj, (B, T_enc, NUM_HEADS, HEAD_DIM_PADDED))
    k = ttnn.permute(k, (0, 2, 1, 3))
    v = ttnn.reshape(v_proj, (B, T_enc, NUM_HEADS, HEAD_DIM_PADDED))
    v = ttnn.permute(v, (0, 2, 1, 3))
    k = ttnn.permute(k, (0, 1, 3, 2))  # pre-transpose for Q @ K^T
    return k, v


def _project_query(decoder_hidden, w):
    B = decoder_hidden.shape[0]
    T_dec = decoder_hidden.shape[1]
    query_proj = ttnn.linear(decoder_hidden, w["q_proj_weight"], bias=w["q_proj_bias"])
    q = ttnn.reshape(query_proj, (B, T_dec, NUM_HEADS, HEAD_DIM_PADDED))
    return ttnn.permute(q, (0, 2, 1, 3))


def tst_cross_attention_with_kv(decoder_hidden, k, v, w):
    """Cross-attention using precomputed K/V from precompute_cross_attn_kv."""
    q = _project_query(decoder_hidden, w)
    return attend(q, k, v, w, fused_unmasked_softmax)


def tst_cross_attention(decoder_hidden, encoder_hidden, w):
    """
    decoder_hidden: ttnn [B, T_dec, NUM_HEADS*32].
    encoder_hidden: ttnn [B, T_enc, NUM_HEADS*32].
    """
    k, v = precompute_cross_attn_kv(encoder_hidden, w)
    return tst_cross_attention_with_kv(decoder_hidden, k, v, w)
