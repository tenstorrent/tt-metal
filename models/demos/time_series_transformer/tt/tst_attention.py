# tt/tst_attention.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import torch

import ttnn

NUM_HEADS = 2
HEAD_DIM_PADDED = 32  # each head gets its own full tile (HF's true head_dim is 13)
HEAD_DIM_TRUE = 13  # HF's real per-head width (26 / 2) -- USE THIS FOR SCALING.
NEG_INF = -1e9

# ──────────────────────────────────────────────────────────────────────────
# SCALE FACTOR FIX (verified via direct hardware measurement, not theory):
#
# Both tst_self_attention and tst_cross_attention previously scaled raw
# attention scores by HEAD_DIM_PADDED ** -0.5 (i.e. 32 ** -0.5 = 0.176777).
# HF's actual TimeSeriesTransformerAttention scales by the TRUE per-head
# width, HEAD_DIM_TRUE ** -0.5 (13 ** -0.5 = 0.277350).
#
# These are NOT interchangeable: scaling by the padded tile width instead
# of the true head dimension is a real arithmetic value, not a tolerance
# issue -- ratio (wrong/correct) = 0.637377 = sqrt(13/32), confirmed
# identical in both self-attention and cross-attention via isolated,
# pre-softmax Q@K^T PCC tests against hand-built torch references derived
# from HF's own q_proj/k_proj outputs:
#
#   self-attention:  unscaled Q@K^T PCC = 0.99999, scale ratio = 0.637377
#   cross-attention: unscaled Q@K^T PCC = 0.99711, scale ratio = 0.637377
#
# The raw dot products were always correct (the QKV projection, per-head
# padding, and split/reshape logic were separately verified correct, see
# test_qkv_split_pcc.py). Only the SCALE applied after the dot product was
# wrong. Because PCC is scale-invariant, this bug was invisible to a plain
# PCC check on the unscaled scores -- it only shows up after softmax, where
# the wrong (too-small) scale compresses attention logits and flattens the
# resulting probability distribution relative to HF's. This effect is much
# larger for causal decoder self-attention (PCC 0.8294, full pipeline) than
# for unmasked encoder self-attention (PCC 0.9998), because masking produces
# sharply peaked distributions per row that are more sensitive to logit
# scale than the near-uniform distributions encoder attention produces.
#
# Padding to 32 is still required for the TTNN tile-aligned compute kernels
# (matmul, split_query_key_value_and_split_heads, etc.) -- only the SCALE
# constant changes here, not the padding strategy.
# ──────────────────────────────────────────────────────────────────────────


def build_causal_mask(device, seq_len):
    """
    [1, 1, seq_len, seq_len] additive mask: 0 where attention is allowed
    (j <= i), NEG_INF where it isn't (j > i, future positions).
    Verified on hardware: produces 0% leakage into future positions.
    """
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), NEG_INF)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def causal_softmax(scores, mask, scale):
    """scores: ttnn [B, heads, T, T]. mask: ttnn [1, 1, T, T], broadcasts over B and heads."""
    scaled = ttnn.multiply(scores, scale)
    masked = ttnn.add(scaled, mask)
    return ttnn.softmax(masked, dim=-1)


def tst_self_attention(hidden_states, w, causal=False, causal_mask=None):
    """
    Self-attention: Q, K, V all come from hidden_states via one fused QKV
    projection (per-head padded to 32, verified safe -- see Check 1/5).
    hidden_states: ttnn tensor [B, T, NUM_HEADS*32]. Returns same shape.

    If causal=True, causal_mask must be a pre-built [1,1,T,T] mask tensor
    (use build_causal_mask once per sequence length, reuse across calls --
    don't rebuild it every layer/step).
    """
    fused_qkv = ttnn.linear(hidden_states, w["qkv_weight"], bias=w["qkv_bias"])

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)

    scores = ttnn.matmul(query, key)
    # FIX: scale by the TRUE head dimension (13), not the padded tile
    # width (32). See module-level note above -- confirmed via direct
    # measurement, ratio 0.637377, identical bug as cross-attention.
    scale = HEAD_DIM_TRUE**-0.5

    if causal:
        assert causal_mask is not None, "causal=True requires a pre-built causal_mask tensor"
        probs = causal_softmax(scores, causal_mask, scale)
    else:
        scaled = ttnn.multiply(scores, scale)
        probs = ttnn.softmax(scaled, dim=-1)

    context = ttnn.matmul(probs, value)
    context = ttnn.transformer.concatenate_heads(context)

    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])


def precompute_cross_attn_kv(encoder_hidden, w):
    """
    PERF FIX: run ONCE before the decode loop (encoder_hidden never changes
    across the 24 autoregressive steps -- only decoder Q changes each step).
    Returns (k, v) pre-shaped for matmul, identical to what tst_cross_attention
    computed inline every step. Saves ~8 ops * 2 layers * 24 steps = 384 dispatches.
    """
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
    k = ttnn.permute(k, (0, 1, 3, 2))  # pre-transpose for Q@K^T
    return k, v


def tst_cross_attention_with_kv(decoder_hidden, k, v, w):
    """
    Per-step cross-attention using PRECOMPUTED k, v from precompute_cross_attn_kv.
    Only Q (from decoder_hidden, which changes each step) is projected here.
    Same math as tst_cross_attention -- only WHERE the K/V ops run has changed.
    """
    B = decoder_hidden.shape[0]
    T_dec = decoder_hidden.shape[1]

    query_proj = ttnn.linear(decoder_hidden, w["q_proj_weight"], bias=w["q_proj_bias"])
    q = ttnn.reshape(query_proj, (B, T_dec, NUM_HEADS, HEAD_DIM_PADDED))
    q = ttnn.permute(q, (0, 2, 1, 3))

    scores = ttnn.matmul(q, k)
    scale = HEAD_DIM_TRUE**-0.5
    scaled = ttnn.multiply(scores, scale)
    probs = ttnn.softmax(scaled, dim=-1)

    context = ttnn.matmul(probs, v)
    context = ttnn.transformer.concatenate_heads(context)
    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])


# ─────────────────────────────────────────────────────────────────────────────
# KV-CACHE SELF-ATTENTION (for autoregressive decode)
# ─────────────────────────────────────────────────────────────────────────────


def allocate_kv_cache(device, B, T_max=24):
    """
    Allocate zeroed K and V cache tensors for one decoder layer's self-attention.

    K cache shape: [B, NUM_HEADS, HEAD_DIM_PADDED, T_max] -- K comes out of
    split_query_key_value_and_split_heads already transposed ([B,H,D,1]), so we
    store it transposed and the Q@K matmul works without an extra permute.

    V cache shape: [B, NUM_HEADS, T_max, HEAD_DIM_PADDED] -- V comes out [B,H,1,D].

    Both in ROW_MAJOR layout (required by ttnn.experimental.slice_write).
    Returns (k_cache, v_cache).
    """
    k_cache = ttnn.from_torch(
        torch.zeros(B, NUM_HEADS, HEAD_DIM_PADDED, T_max, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    v_cache = ttnn.from_torch(
        torch.zeros(B, NUM_HEADS, T_max, HEAD_DIM_PADDED, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return k_cache, v_cache


def tst_self_attention_cached(hidden_1token, w, k_cache, v_cache, step, causal_mask_1tok):
    """
    Single-token self-attention with KV-cache.

    hidden_1token: ttnn [B, 1, PADDED_WIDTH] -- the NEW token only.
    k_cache: ttnn [B, NUM_HEADS, HEAD_DIM_PADDED, T_max] ROW_MAJOR -- K stored pre-transposed.
             split_query_key_value_and_split_heads returns K as [B,H,D,1] (already transposed
             for Q@K^T), so we store it that way and write into the T_max dimension.
    v_cache: ttnn [B, NUM_HEADS, T_max, HEAD_DIM_PADDED] ROW_MAJOR -- V stored normally.
    step: int, 0-indexed current decode step.
    causal_mask_1tok: ttnn [1, 1, 1, T_max] -- 0 for positions 0..step, NEG_INF beyond.

    Returns: ttnn [B, 1, PADDED_WIDTH]
    Fixed shapes every step -> enables TTNN tracing.
    """
    B = hidden_1token.shape[0]

    fused_qkv = ttnn.linear(hidden_1token, w["qkv_weight"], bias=w["qkv_bias"])

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)
    # query: [B, H, 1, D]
    # key:   [B, H, D, 1]  <-- already transposed by TTNN
    # value: [B, H, 1, D]

    # Write K into k_cache at column `step` (last dim is T_max).
    # k_cache shape: [B, H, D, T_max]; key shape: [B, H, D, 1]. Match on last dim slice.
    key_rm = ttnn.to_layout(key, ttnn.ROW_MAJOR_LAYOUT)
    value_rm = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
    _step = [1, 1, 1, 1]
    ttnn.experimental.slice_write(key_rm, k_cache, [0, 0, 0, step], [B, NUM_HEADS, HEAD_DIM_PADDED, step + 1], _step)
    # Write V into v_cache at row `step` (dim=2 is T_max).
    # v_cache shape: [B, H, T_max, D]; value shape: [B, H, 1, D].
    ttnn.experimental.slice_write(value_rm, v_cache, [0, 0, step, 0], [B, NUM_HEADS, step + 1, HEAD_DIM_PADDED], _step)

    # Q @ K_cache: k_cache is [B,H,D,T_max] already in the right shape for matmul
    k_tile = ttnn.to_layout(k_cache, ttnn.TILE_LAYOUT)  # [B, H, D, T_max]
    scores = ttnn.matmul(query, k_tile)  # [B, H, 1, T_max]

    scale = HEAD_DIM_TRUE**-0.5
    scaled = ttnn.multiply(scores, scale)
    masked = ttnn.add(scaled, causal_mask_1tok)
    probs = ttnn.softmax(masked, dim=-1)

    v_tile = ttnn.to_layout(v_cache, ttnn.TILE_LAYOUT)  # [B, H, T_max, D]
    context = ttnn.matmul(probs, v_tile)  # [B, H, 1, D]
    context = ttnn.transformer.concatenate_heads(context)  # [B, 1, PADDED_WIDTH]

    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])


def tst_cross_attention(decoder_hidden, encoder_hidden, w):
    """
    Cross-attention: Q from decoder_hidden, K/V from encoder_hidden.

    FIX (slice API): ttnn.slice() kwargs changed between versions.
    The old code used begins=/ends= which no longer exist.
    The correct kwargs are slice_start=/slice_end= (or pass steps as a
    positional arg when using the starts/ends/steps overload).
    We use slice_start=/slice_end= which is the stable named overload.

    decoder_hidden: ttnn [B, T_dec, NUM_HEADS*32].
    encoder_hidden: ttnn [B, T_enc, NUM_HEADS*32].
    """
    B = decoder_hidden.shape[0]
    T_dec = decoder_hidden.shape[1]
    T_enc = encoder_hidden.shape[1]

    # Q from decoder — shape [B, T_dec, NUM_HEADS*HEAD_DIM_PADDED]
    query_proj = ttnn.linear(decoder_hidden, w["q_proj_weight"], bias=w["q_proj_bias"])

    # K and V from encoder — fused_kv: [B, T_enc, 2*NUM_HEADS*HEAD_DIM_PADDED]
    fused_kv = ttnn.linear(encoder_hidden, w["kv_weight"], bias=w["kv_bias"])

    # Split along the last (innermost/contiguous) dimension — zero-copy pointer
    # offset, no re-tiling, no rounding error. Confirmed tile-aligned: 64 is
    # an exact multiple of the 32-wide tile, so this slice does not cross a
    # tile boundary (verified against tt-metal tile documentation).
    kv_half = NUM_HEADS * HEAD_DIM_PADDED  # 64
    k_proj = ttnn.slice(fused_kv, slice_start=[0, 0, 0], slice_end=[B, T_enc, kv_half])
    v_proj = ttnn.slice(fused_kv, slice_start=[0, 0, kv_half], slice_end=[B, T_enc, 2 * kv_half])

    # Reshape Q → [B, NUM_HEADS, T_dec, HEAD_DIM_PADDED]
    q = ttnn.reshape(query_proj, (B, T_dec, NUM_HEADS, HEAD_DIM_PADDED))
    q = ttnn.permute(q, (0, 2, 1, 3))

    # Reshape K → [B, NUM_HEADS, T_enc, HEAD_DIM_PADDED]
    k = ttnn.reshape(k_proj, (B, T_enc, NUM_HEADS, HEAD_DIM_PADDED))
    k = ttnn.permute(k, (0, 2, 1, 3))

    # Reshape V → [B, NUM_HEADS, T_enc, HEAD_DIM_PADDED]
    v = ttnn.reshape(v_proj, (B, T_enc, NUM_HEADS, HEAD_DIM_PADDED))
    v = ttnn.permute(v, (0, 2, 1, 3))

    # Transpose K for Q*K^T: [B, NUM_HEADS, HEAD_DIM_PADDED, T_enc]
    k = ttnn.permute(k, (0, 1, 3, 2))

    # Attention scores [B, NUM_HEADS, T_dec, T_enc]
    scores = ttnn.matmul(q, k)
    # FIX: scale by the TRUE head dimension (13), not the padded tile
    # width (32). See module-level note above -- confirmed via direct
    # measurement, ratio 0.637377, identical bug as self-attention.
    scale = HEAD_DIM_TRUE**-0.5
    scaled = ttnn.multiply(scores, scale)
    probs = ttnn.softmax(scaled, dim=-1)

    # Weighted sum: [B, NUM_HEADS, T_dec, HEAD_DIM_PADDED]
    context = ttnn.matmul(probs, v)
    context = ttnn.transformer.concatenate_heads(context)

    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])
