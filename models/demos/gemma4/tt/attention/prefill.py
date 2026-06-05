# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-mode attention forward pass for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (no transformation matrices).
"""

import ttnn

from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    effective_block_size,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights


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
):
    """Single-user prefill — matches arg/gemma4_optimizations."""
    tp = mesh_config.tp if mesh_config else 1

    xqkv = apply_qkv_projection(hidden_states, weights)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )

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
            ttnn.experimental.paged_fill_cache(
                k_cache, tt_k, page_table, batch_idx=user_id, block_size=eff_bs, **paged_modulo_kwargs
            )
            ttnn.experimental.paged_fill_cache(
                v_cache, tt_v, page_table, batch_idx=user_id, block_size=eff_bs, **paged_modulo_kwargs
            )
        else:
            ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
            ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

    sliding_window = config.sliding_window if config.is_sliding else None
    tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
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
):
    """
    Multi-token prefill attention, fully on device.

    Args:
        hidden_states: [1, 1, seq_len, hidden_size] or [B, 1, S, hidden_size] on device
        batch_size: padded batch for batched prefill (1 for single-user / test_full_model)
    """
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
        )

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
            block_size = k_cache.padded_shape[2]
            page_len = page_table.shape[1] * block_size

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

    sliding_window = config.sliding_window if config.is_sliding else None
    tt_sdpa = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
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
