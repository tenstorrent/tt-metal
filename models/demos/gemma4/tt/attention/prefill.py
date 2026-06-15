# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-mode attention forward pass for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (no transformation matrices).
"""

import ttnn

from .operations import (
    PREFILL_SDPA_MAX_SEQ,
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    chunked_prefill_sdpa,
    chunked_prefill_sdpa_sliding,
    concat_heads,
    effective_block_size,
    prefill_sdpa_program_config,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights


def prefill_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    page_table=None,
    user_id=0,
    ccl_manager=None,
    shared_kv=None,
    keep_kv=False,
    valid_seq_len=None,
):
    """
    Multi-token prefill attention, fully on device.

    Args:
        hidden_states: [1, 1, seq_len, hidden_size] on device
        cos_cache: [1, 1, max_seq_len, head_dim] - full cos cache on device
        sin_cache: [1, 1, max_seq_len, head_dim] - full sin cache on device
        weights: AttentionWeights container
        kv_cache: [k_cache, v_cache] TT tensors
        config: Gemma4AttentionConfig
        mesh_config: MeshConfig
        mesh_device: TT device
        page_table: optional paged attention table
        user_id: batch index for cache fill
        ccl_manager: optional CCL manager for TP > 1
        shared_kv: optional (tt_k, tt_v) from source layer for KV sharing
        keep_kv: if True, don't deallocate K/V (for KV source layers that share with later layers)
    """
    tp = mesh_config.tp if mesh_config else 1

    # 1. Fused QKV projection
    xqkv = apply_qkv_projection(hidden_states, weights)

    # 2. Split into Q, K, V heads
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )

    # 3. Per-head norms (only for Q if using shared K/V)
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)

    if shared_kv is not None:
        # KV-shared layer: discard own K/V, use source layer's K/V
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_k, tt_v = shared_kv
    else:
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    # 4. RoPE (skip for shared K — already RoPE'd by source layer)
    tt_q = apply_rope(tt_q, cos_cache, sin_cache)
    if shared_kv is None:
        tt_k = apply_rope(tt_k, cos_cache, sin_cache)

    # 5. Fill KV cache (skip for KV-shared layers)
    if kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        if page_table is not None:
            # Per-device kv-head count of the layer's input view. Needed for
            # eff_bs when the cache was allocated for a different layer type
            # under HMA cross-group sharing (e.g. full-attention layer writing
            # into a sliding-allocated buffer on Gemma4-26B-A4B at TP=1).
            # Mirrors split_qkv_heads_prefill's local head count.
            num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
            eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
            # Bounded sliding-window cache: when set, paged_fill_cache wraps the
            # per-tile virtual position into a circular buffer of
            # ``cache_position_modulo`` tokens before the page_table lookup. Earlier
            # tile writes that would wrap past the later writes are skipped on-device.
            # ``None`` = legacy unbounded behavior.
            paged_modulo_kwargs = (
                {"cache_position_modulo": config.cache_position_modulo}
                if config.cache_position_modulo is not None
                else {}
            )
            # Bounded sliding cache + padded single-chunk prefill: the prompt is
            # padded up to the next power of 2, and writing those padding tokens
            # into the 1024-slot circular cache WRAPS and overwrites the real
            # recent window — so decode reads padding and emits garbage once the
            # padding exceeds the window (real prompt < seq_len - window). Cap the
            # bounded-cache fill to the real (unpadded) prompt length so the
            # circular buffer ends on real tokens. Full (unbounded) layers are
            # unaffected: their padding lands at positions decode never reads.
            k_fill, v_fill = tt_k, tt_v
            if config.cache_position_modulo is not None and valid_seq_len is not None:
                fill_len = ((min(valid_seq_len, tt_k.shape[-2]) + eff_bs - 1) // eff_bs) * eff_bs
                if 0 < fill_len < tt_k.shape[-2]:
                    k_fill = ttnn.slice(tt_k, [0, 0, 0, 0], [tt_k.shape[0], tt_k.shape[1], fill_len, tt_k.shape[3]])
                    v_fill = ttnn.slice(tt_v, [0, 0, 0, 0], [tt_v.shape[0], tt_v.shape[1], fill_len, tt_v.shape[3]])
            ttnn.experimental.paged_fill_cache(
                k_cache, k_fill, page_table, batch_idx=user_id, block_size=eff_bs, **paged_modulo_kwargs
            )
            ttnn.experimental.paged_fill_cache(
                v_cache, v_fill, page_table, batch_idx=user_id, block_size=eff_bs, **paged_modulo_kwargs
            )
            if k_fill is not tt_k:
                k_fill.deallocate(True)
            if v_fill is not tt_v:
                v_fill.deallocate(True)
        else:
            ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
            ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

    # 6. SDPA (causal prefill, scale=1.0)
    # The non-chunked SDPA silently returns WRONG results for seq_len > 32768
    # (2^15) — generation degrades to garbage. For long context we chunk prefill:
    #   - full-attention layers: chunk Q and attend the full K prefix from the
    #     (already-filled) paged cache via chunked_scaled_dot_product_attention.
    #   - sliding-window layers: that op is causal-only, so use an overlapping
    #     windowed chunking over the in-memory K/V (each slice stays <=32768).
    # Both stay correct past 32768 and reduce to the non-chunked op at <=32768.
    seq_len = tt_q.shape[-2]
    long_seq = seq_len > PREFILL_SDPA_MAX_SEQ
    sliding_window = config.sliding_window if config.is_sliding else None
    if long_seq and config.is_sliding and sliding_window is not None:
        tt_sdpa = chunked_prefill_sdpa_sliding(tt_q, tt_k, tt_v, sliding_window, config.head_dim, scale=1.0)
    elif long_seq and not config.is_sliding and page_table is not None and kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        tt_sdpa = chunked_prefill_sdpa(tt_q, k_cache, v_cache, page_table, user_id, config.head_dim, scale=1.0)
    else:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=1.0,
            sliding_window_size=sliding_window,
            program_config=prefill_sdpa_program_config(config.head_dim, seq_len),
        )
    tt_q.deallocate(True)
    # Keep K/V alive for source layers; shared layers don't own them
    kept_kv = None
    if shared_kv is None and not keep_kv:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    elif keep_kv:
        kept_kv = (tt_k, tt_v)

    # 7. Concat heads + output projection + allreduce
    tt_out = concat_heads(tt_sdpa, is_decode_mode=False)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    return tt_out, kept_kv
