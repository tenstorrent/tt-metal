# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decode-mode attention forward pass for Gemma4.

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
    split_qkv_heads_decode,
)
from .weights import AttentionWeights


def decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_config,
    mesh_device,
    position_idx,
    token_index,
    page_table=None,
    ccl_manager=None,
):
    """
    Single-token decode attention, fully on device.

    Args:
        hidden_states: [1, 1, batch, hidden_size] on device
        cos_cache: [1, 1, max_seq_len, head_dim] - full cos cache on device
        sin_cache: [1, 1, max_seq_len, head_dim] - full sin cache on device
        weights: AttentionWeights container
        kv_cache: [k_cache, v_cache] TT tensors
        config: Gemma4AttentionConfig
        mesh_config: MeshConfig
        mesh_device: TT device
        position_idx: current position tensor for KV cache update
        token_index: int position for RoPE cache slicing
        page_table: optional paged attention table
        ccl_manager: optional CCL manager for TP > 1
    """
    tp = mesh_config.tp if mesh_config else 1

    # 1. Fused QKV projection
    xqkv = apply_qkv_projection(hidden_states, weights)

    # 2. Split into Q, K, V heads
    tt_q, tt_k, tt_v = split_qkv_heads_decode(xqkv, config, weights.is_global, tp=tp)

    # 3. Per-head norms (move to DRAM for rms_norm, restore sharded for RoPE)
    q_sharded_mem = tt_q.memory_config()
    tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
    tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)

    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
    tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    # 4. RoPE (HF-style — cos/sin cache already handles partial rotation via identity padding)
    tt_q = apply_rope(tt_q, cos_cache, sin_cache, token_index=token_index)
    tt_k = apply_rope(tt_k, cos_cache, sin_cache, token_index=token_index)

    # 5. KV cache update — paged_update_cache needs HEIGHT_SHARDED input
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        # After HF-style RoPE, tensors may be in DRAM. Move to HEIGHT_SHARDED for cache update.
        tt_k = ttnn.to_memory_config(tt_k, q_sharded_mem)
        tt_v = ttnn.to_memory_config(tt_v, q_sharded_mem)

        if page_table is not None:
            ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=position_idx, page_table=page_table)
            ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=position_idx, page_table=page_table)
        else:
            ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=position_idx)
            ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=position_idx)
    else:
        k_cache = tt_k
        v_cache = tt_v

    # 6. SDPA (scale=1.0)
    sliding_window = config.sliding_window if config.is_sliding else None

    # For large head_dim (e.g. 512 on global layers), use a smaller compute grid
    # to avoid L1 overflow on mesh devices where fabric reserves L1 space
    sdpa_program_config = None
    if config.head_dim >= 512:
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
        )

    if page_table is not None:
        tt_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=position_idx,
            page_table_tensor=page_table,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    else:
        tt_sdpa = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=position_idx,
            scale=1.0,
            sliding_window_size=sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_program_config,
        )
    tt_q.deallocate(True)

    # 7. Concat heads + output projection + allreduce
    tt_out = concat_heads(tt_sdpa, is_decode_mode=True)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    return tt_out
