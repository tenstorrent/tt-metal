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
    page_table=None,
    user_id=0,
    ccl_manager=None,
    shared_kv=None,
    keep_kv=False,
    valid_seq_len=None,
):
    """
    Multi-token prefill attention, fully on device. Handles single-user (B=1)
    and batched (B>1) prefill in one path.

    Batch is read from hidden_states.shape[0], never a separate arg, so a
    [B, 1, S, H] input can't silently fill only user 0's KV cache.

    Args:
        hidden_states: [B, 1, seq_len, hidden_size] on device (B=1 for single-user)
    Returns:
        (tt_out [B, 1, seq_len, hidden_size], kept_kv or None)
    """
    tp = mesh_config.tp if mesh_config else 1
    batch_size = hidden_states.shape[0]

    # No reshape round-trip: ttnn.linear and every downstream op (head split,
    # per-head norm, RoPE, SDPA, out-proj) already act on the leading batch dim,
    # so packing to [1, 1, B*S, H] and back is unnecessary.
    xqkv = apply_qkv_projection(hidden_states, weights)

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

    if kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        if page_table is not None:
            num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
            eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
            paged_modulo_kwargs = (
                {"cache_position_modulo": config.cache_position_modulo}
                if config.cache_position_modulo is not None
                else {}
            )
            if batch_size == 1:
                # Bounded sliding cache + padded prefill: padding tokens written
                # into the 1024-slot circular cache WRAP and overwrite the recent
                # window, so cap the fill to the real (unpadded) prompt so the
                # buffer ends on real tokens. Full layers are unaffected (their
                # padding lands at positions decode never reads).
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
                # paged_fill_cache is per-user; fill one batch row per slot and
                # cap each to the page-table capacity.
                page_len = page_table.shape[1] * eff_bs
                seq_per_user = tt_k.shape[-2]
                valid_slots = user_id if isinstance(user_id, (list, tuple)) else list(range(batch_size))
                for slot_idx in valid_slots:
                    k_user = tt_k[slot_idx : slot_idx + 1, :, :, :]
                    v_user = tt_v[slot_idx : slot_idx + 1, :, :, :]
                    k_user = k_user[:, :, :page_len, :] if page_len < seq_per_user else k_user
                    v_user = v_user[:, :, :page_len, :] if page_len < seq_per_user else v_user
                    ttnn.experimental.paged_fill_cache(
                        k_cache, k_user, page_table, batch_idx=slot_idx, block_size=eff_bs, **paged_modulo_kwargs
                    )
                    ttnn.experimental.paged_fill_cache(
                        v_cache, v_user, page_table, batch_idx=slot_idx, block_size=eff_bs, **paged_modulo_kwargs
                    )
        else:
            if batch_size == 1:
                ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
                ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)
            else:
                valid_slots = user_id if isinstance(user_id, (list, tuple)) else list(range(batch_size))
                for slot_idx in valid_slots:
                    ttnn.fill_cache(k_cache, tt_k[slot_idx : slot_idx + 1], batch_idx=slot_idx)
                    ttnn.fill_cache(v_cache, tt_v[slot_idx : slot_idx + 1], batch_idx=slot_idx)

    # SDPA (causal prefill, scale=1.0). seq_len is per-user (batch is dim 0).
    # The non-chunked op silently returns WRONG results past 32768 tokens, so
    # long single-user prompts are chunked: full layers attend the paged-cache K
    # prefix; sliding layers use overlapping windowed chunks over in-memory K/V.
    # The chunked ops are single-user (they index [0, ...]); batched prefill runs
    # short per-user seqs and never needs them — assert rather than degrade.
    seq_len = tt_q.shape[-2]
    long_seq = seq_len > PREFILL_SDPA_MAX_SEQ
    sliding_window = config.sliding_window if config.is_sliding else None
    assert not (long_seq and batch_size > 1), "batched prefill with seq_len > PREFILL_SDPA_MAX_SEQ is unsupported"
    if long_seq and config.is_sliding and sliding_window is not None:
        tt_sdpa = chunked_prefill_sdpa_sliding(tt_q, tt_k, tt_v, sliding_window, config.head_dim, scale=1.0)
    elif long_seq and not config.is_sliding and page_table is not None and kv_cache is not None and shared_kv is None:
        k_cache, v_cache = kv_cache
        tt_sdpa = chunked_prefill_sdpa(tt_q, k_cache, v_cache, page_table, user_id, config.head_dim, scale=1.0)
    else:
        # The tuned program_config is wired for the single-user path; batched
        # prefill keeps the op default (its validated config) for now.
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            scale=1.0,
            sliding_window_size=sliding_window,
            program_config=prefill_sdpa_program_config(config.head_dim, seq_len) if batch_size == 1 else None,
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
