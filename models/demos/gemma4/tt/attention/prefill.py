# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    """
    tp = mesh_config.tp if mesh_config else 1

    # 1. Fused QKV projection
    xqkv = apply_qkv_projection(hidden_states, weights)

    # 2. Split into Q, K, V heads
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv, config, weights.is_global, tp=tp)

    # 3. Per-head norms
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    # 4. RoPE (HF-style — cos/sin cache handles partial rotation via identity padding)
    tt_q = apply_rope(tt_q, cos_cache, sin_cache)
    tt_k = apply_rope(tt_k, cos_cache, sin_cache)

    # 5. Fill KV cache
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        if page_table is not None:
            ttnn.experimental.paged_fill_cache(k_cache, tt_k, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(v_cache, tt_v, page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
            ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

    # 6. SDPA (causal prefill, scale=1.0)
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
    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # 7. Concat heads + output projection + allreduce
    tt_out = concat_heads(tt_sdpa, is_decode_mode=False)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    return tt_out
