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

TILE_SIZE = 32


def _validate_q_rope_offset(q_rope_offset: int, *, batch_size: int) -> None:
    if q_rope_offset % TILE_SIZE != 0:
        raise ValueError(f"q_rope_offset must be a multiple of {TILE_SIZE}, got {q_rope_offset}")
    if batch_size > 1 and q_rope_offset != 0:
        raise ValueError("nonzero q_rope_offset is only supported for single-user prefill")


def _slice_rope_cache(cache, start, length):
    if start % TILE_SIZE != 0:
        raise ValueError(f"RoPE cache start must be a multiple of {TILE_SIZE}, got {start}")
    if start + length > cache.shape[-2]:
        raise ValueError(f"RoPE cache slice [{start}, {start + length}) exceeds cache length {cache.shape[-2]}")
    # Fast path is correct only when callers pre-slice RoPE caches to the active seq_len.
    if start == 0 and cache.shape[-2] == length:
        return cache
    return ttnn.slice(cache, [0, 0, start, 0], [cache.shape[0], cache.shape[1], start + length, cache.shape[3]])


def _prefill_forward_single(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    page_table=None,
    user_id=0,
    ccl_manager=None,
    shared_kv=None,
    keep_kv=False,
    valid_seq_len=None,
    write_kv_cache=True,
    attn_mask=None,
    kv_hidden_states=None,
    prefix_kv=None,
    q_rope_offset=0,
    is_causal=True,
):
    """Single-user prefill — matches arg/gemma4_optimizations."""
    tp = mesh_config.tp if mesh_config else 1

    if kv_hidden_states is not None and (shared_kv is not None or write_kv_cache):
        raise ValueError("kv_hidden_states is only supported for readonly masked denoise prefill")
    if prefix_kv is not None and (shared_kv is not None or write_kv_cache):
        raise ValueError("prefix_kv is only supported for readonly masked denoise prefill")

    xqkv = apply_qkv_projection(hidden_states, weights)
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )
    if kv_hidden_states is not None:
        xqkv_kv = apply_qkv_projection(kv_hidden_states, weights)
        tt_kv_q, tt_k_from_kv, tt_v_from_kv = split_qkv_heads_prefill(
            xqkv_kv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
        )
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_kv_q.deallocate(True)
        tt_k, tt_v = tt_k_from_kv, tt_v_from_kv

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)

    if shared_kv is not None:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_k, tt_v = shared_kv
    else:
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    q_cos_cache = _slice_rope_cache(cos_cache, q_rope_offset, tt_q.shape[-2])
    q_sin_cache = _slice_rope_cache(sin_cache, q_rope_offset, tt_q.shape[-2])
    tt_q = apply_rope(tt_q, q_cos_cache, q_sin_cache)
    if q_cos_cache is not cos_cache:
        q_cos_cache.deallocate(True)
    if q_sin_cache is not sin_cache:
        q_sin_cache.deallocate(True)
    if shared_kv is None:
        k_rope_offset = q_rope_offset if prefix_kv is not None else 0
        k_cos_cache = _slice_rope_cache(cos_cache, k_rope_offset, tt_k.shape[-2])
        k_sin_cache = _slice_rope_cache(sin_cache, k_rope_offset, tt_k.shape[-2])
        tt_k = apply_rope(tt_k, k_cos_cache, k_sin_cache)
        if k_cos_cache is not cos_cache:
            k_cos_cache.deallocate(True)
        if k_sin_cache is not sin_cache:
            k_sin_cache.deallocate(True)

    if prefix_kv is not None:
        prefix_k, prefix_v = prefix_kv
        canvas_k, canvas_v = tt_k, tt_v
        prefix_k_concat = ttnn.to_memory_config(prefix_k, canvas_k.memory_config())
        prefix_v_concat = ttnn.to_memory_config(prefix_v, canvas_v.memory_config())
        tt_k = ttnn.concat([prefix_k_concat, canvas_k], dim=2)
        tt_v = ttnn.concat([prefix_v_concat, canvas_v], dim=2)
        if prefix_k_concat is not prefix_k:
            prefix_k_concat.deallocate(True)
        if prefix_v_concat is not prefix_v:
            prefix_v_concat.deallocate(True)
        canvas_k.deallocate(True)
        canvas_v.deallocate(True)

    if write_kv_cache and kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        if page_table is not None:
            num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
            eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
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

    # 6. SDPA (prefill, scale=1.0)
    # The non-chunked SDPA silently returns WRONG results for seq_len > 32768
    # (2^15) — generation degrades to garbage. For long context we chunk prefill:
    #   - full-attention layers: chunk Q and attend the full K prefix from the
    #     (already-filled) paged cache via chunked_scaled_dot_product_attention.
    #   - sliding-window layers: that op is causal-only, so use an overlapping
    #     windowed chunking over the in-memory K/V (each slice stays <=32768).
    # Both stay correct past 32768 and reduce to the non-chunked op at <=32768.
    seq_len = tt_q.shape[-2]
    k_seq_len = tt_k.shape[-2]
    long_seq = k_seq_len > PREFILL_SDPA_MAX_SEQ
    sliding_window = config.sliding_window if is_causal and config.is_sliding else None
    if attn_mask is not None:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            program_config=prefill_sdpa_program_config(config.head_dim, seq_len, tt_k.shape[-2]),
        )
    elif not is_causal:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=False,
            scale=1.0,
            program_config=prefill_sdpa_program_config(config.head_dim, seq_len, k_seq_len),
        )
    elif long_seq and config.is_sliding and sliding_window is not None:
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
    kept_kv = None
    if shared_kv is None and not keep_kv:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    elif keep_kv:
        kept_kv = (tt_k, tt_v)

    tt_out = concat_heads(tt_sdpa, is_decode_mode=False)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    return tt_out, kept_kv


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
    batch_size=1,
    valid_seq_len=None,
    write_kv_cache=True,
    attn_mask=None,
    kv_hidden_states=None,
    prefix_kv=None,
    q_rope_offset=0,
    is_causal=True,
):
    """
    Multi-token prefill attention, fully on device.

    Args:
        hidden_states: [1, 1, seq_len, hidden_size] or [B, 1, S, hidden_size] on device
        batch_size: padded batch for batched prefill (1 for single-user / test_full_model)
    """
    _validate_q_rope_offset(q_rope_offset, batch_size=batch_size)
    if batch_size <= 1:
        return _prefill_forward_single(
            hidden_states,
            cos_cache,
            sin_cache,
            weights,
            kv_cache,
            config,
            mesh_config,
            page_table=page_table,
            user_id=user_id,
            ccl_manager=ccl_manager,
            shared_kv=shared_kv,
            keep_kv=keep_kv,
            valid_seq_len=valid_seq_len,
            write_kv_cache=write_kv_cache,
            attn_mask=attn_mask,
            kv_hidden_states=kv_hidden_states,
            prefix_kv=prefix_kv,
            q_rope_offset=q_rope_offset,
            is_causal=is_causal,
        )

    if kv_hidden_states is not None:
        raise ValueError("kv_hidden_states is only supported for single-user prefill")
    if prefix_kv is not None:
        raise ValueError("prefix_kv is only supported for single-user prefill")

    tp = mesh_config.tp if mesh_config else 1
    hidden_states = ttnn.reshape(
        hidden_states, [1, 1, hidden_states.shape[-2] * hidden_states.shape[-3] * hidden_states.shape[0], -1]
    )

    seq_len = hidden_states.shape[-2]
    original_seq_len = seq_len

    xqkv = apply_qkv_projection(hidden_states, weights)
    ttnn.deallocate(hidden_states)

    xqkv = ttnn.reshape(xqkv, [batch_size, 1, seq_len // batch_size, -1])
    seq_len_per_user = seq_len // batch_size

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )
    ttnn.deallocate(xqkv)

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)

    if shared_kv is not None:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_k, tt_v = shared_kv
    else:
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    tt_q = apply_rope(tt_q, cos_cache, sin_cache)
    if shared_kv is None:
        tt_k = apply_rope(tt_k, cos_cache, sin_cache)

    if write_kv_cache and kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        if page_table is not None:
            num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
            eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
            paged_modulo_kwargs = (
                {"cache_position_modulo": config.cache_position_modulo}
                if config.cache_position_modulo is not None
                else {}
            )
            page_len = page_table.shape[1] * eff_bs
            valid_slots = user_id if isinstance(user_id, (list, tuple)) else list(range(batch_size))
            for slot_idx in valid_slots:
                k_user = tt_k[slot_idx : slot_idx + 1, :, :, :]
                v_user = tt_v[slot_idx : slot_idx + 1, :, :, :]
                k_user_sliced = k_user[:, :, :page_len, :] if page_len < seq_len_per_user else k_user
                v_user_sliced = v_user[:, :, :page_len, :] if page_len < seq_len_per_user else v_user
                ttnn.experimental.paged_fill_cache(
                    k_cache,
                    k_user_sliced,
                    page_table,
                    batch_idx=slot_idx,
                    block_size=eff_bs,
                    **paged_modulo_kwargs,
                )
                ttnn.experimental.paged_fill_cache(
                    v_cache,
                    v_user_sliced,
                    page_table,
                    batch_idx=slot_idx,
                    block_size=eff_bs,
                    **paged_modulo_kwargs,
                )
        else:
            valid_slots = user_id if isinstance(user_id, (list, tuple)) else list(range(batch_size))
            for slot_idx in valid_slots:
                ttnn.fill_cache(k_cache, tt_k[slot_idx : slot_idx + 1], batch_idx=slot_idx)
                ttnn.fill_cache(v_cache, tt_v[slot_idx : slot_idx + 1], batch_idx=slot_idx)

    sliding_window = config.sliding_window if is_causal and config.is_sliding else None
    if attn_mask is not None:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
        )
    else:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=is_causal,
            scale=1.0,
            sliding_window_size=sliding_window,
        )
    tt_q.deallocate(True)
    kept_kv = None
    if shared_kv is None and not keep_kv:
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    elif keep_kv:
        kept_kv = (tt_k, tt_v)

    tt_out = concat_heads(tt_sdpa, is_decode_mode=False)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    tt_out = ttnn.reshape(tt_out, [1, 1, original_seq_len, -1])

    return tt_out, kept_kv
