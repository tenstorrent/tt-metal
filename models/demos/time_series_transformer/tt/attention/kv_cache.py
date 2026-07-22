# tt/attention/kv_cache.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
KV-cache lifecycle and single-token self-attention for autoregressive decode.
"""

import torch

import ttnn

from ..tst_config import HEAD_DIM_PADDED, NUM_HEADS
from .ops import attend, scaled_masked_softmax


def allocate_kv_cache(device, B, T_max=24):
    """
    Allocate zeroed K/V cache tensors for one decoder layer's self-attention.

    K cache: [B, NUM_HEADS, HEAD_DIM_PADDED, T_max], ROW_MAJOR. K comes out of
    split_query_key_value_and_split_heads already transposed ([B,H,D,1]);
    this layout is incompatible with ttnn.update_cache's fixed seq-at-dim(-2)
    contract, so K always uses the slice_write path.

    V cache: [B, NUM_HEADS, T_max, HEAD_DIM_PADDED]. Allocated TILE_LAYOUT
    only when B == 1, to use ttnn.update_cache directly — it hard-asserts
    padded_shape()[0] == 1, so for B > 1 V stays ROW_MAJOR on slice_write.
    """
    k_cache = ttnn.from_torch(
        torch.zeros(B, NUM_HEADS, HEAD_DIM_PADDED, T_max, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    v_layout = ttnn.TILE_LAYOUT if B == 1 else ttnn.ROW_MAJOR_LAYOUT
    v_cache = ttnn.from_torch(
        torch.zeros(B, NUM_HEADS, T_max, HEAD_DIM_PADDED, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=v_layout,
        device=device,
    )
    return k_cache, v_cache


def tst_self_attention_cached(hidden_1token, w, k_cache, v_cache, step, causal_mask_1tok):
    """
    Single-token self-attention against a KV-cache.

    hidden_1token: ttnn [B, 1, PADDED_WIDTH] — the new token only.
    k_cache: ttnn [B, NUM_HEADS, HEAD_DIM_PADDED, T_max], ROW_MAJOR.
    v_cache: ttnn [B, NUM_HEADS, T_max, HEAD_DIM_PADDED], ROW_MAJOR (or TILE at B==1).
    step: 0-indexed current decode step.
    causal_mask_1tok: ttnn [1, 1, 1, T_max] — 0 for positions 0..step, NEG_INF beyond.

    No dedicated PCC test exists for this call site — do not change the
    softmax path here without adding one.

    Fixed shapes every step: enables TTNN tracing.
    """
    B = hidden_1token.shape[0]

    fused_qkv = ttnn.linear(hidden_1token, w["qkv_weight"], bias=w["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)
    # query: [B, H, 1, D], key: [B, H, D, 1] (pre-transposed by TTNN), value: [B, H, 1, D]

    key_rm = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
    value_rm = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
    _step = [1, 1, 1, 1]
    ttnn.experimental.slice_write(key_rm, k_cache, [0, 0, 0, step], [B, NUM_HEADS, HEAD_DIM_PADDED, step + 1], _step)
    ttnn.experimental.slice_write(value_rm, v_cache, [0, 0, step, 0], [B, NUM_HEADS, step + 1, HEAD_DIM_PADDED], _step)

    k_tile = ttnn.to_layout(k_cache, ttnn.TILE_LAYOUT)
    v_tile = ttnn.to_layout(v_cache, ttnn.TILE_LAYOUT)

    softmax_fn = lambda scores: scaled_masked_softmax(scores, causal_mask_1tok)
    return attend(query, k_tile, v_tile, w, softmax_fn)
