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
    is_kv_shared=False,
    position_idx_cache=None,
):
    """
    Single-token decode attention, fully on device.

    Args:
        hidden_states: [1, 1, batch, hidden_size] on device
        cos_cache: [max_seq_len, head_dim] 2D cache for embedding lookup, or [1,1,max_seq_len,head_dim] 4D
        sin_cache: same format as cos_cache
        weights: AttentionWeights container
        kv_cache: [k_cache, v_cache] TT tensors (for shared layers, this is the source layer's cache)
        config: Gemma4AttentionConfig
        mesh_config: MeshConfig
        mesh_device: TT device
        position_idx: [batch] tensor of current positions for KV cache update + RoPE embedding lookup
        token_index: int position for legacy RoPE slicing (unused when cos_cache is 2D)
        page_table: optional paged attention table
        ccl_manager: optional CCL manager for TP > 1
        is_kv_shared: if True, skip K/V projection and cache update (use source layer's KV cache)
    """
    tp = mesh_config.tp if mesh_config else 1

    # 1. Fused QKV projection
    xqkv = apply_qkv_projection(hidden_states, weights)

    # 2. Split into Q, K, V heads
    tt_q, tt_k, tt_v = split_qkv_heads_decode(xqkv, config, weights.is_global, tp=tp)

    # 3. Per-head norms (move to DRAM for rms_norm, restore sharded for RoPE)
    q_sharded_mem = tt_q.memory_config()
    tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    tt_q = apply_per_head_norm(tt_q, weights.q_norm_weight, config.rms_norm_eps, with_scale=True)

    if is_kv_shared:
        # KV-shared layer: discard own K/V, use source layer's KV cache directly
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    else:
        tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
        tt_v = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
        tt_k = apply_per_head_norm(tt_k, weights.k_norm_weight, config.rms_norm_eps, with_scale=True)
        tt_v = apply_per_head_norm(tt_v, None, config.rms_norm_eps, with_scale=False)

    # 4. RoPE — use on-device embedding lookup for trace compatibility
    use_embedding_rope = len(cos_cache.shape) == 2  # 2D cache = embedding lookup mode
    if use_embedding_rope:
        # Gather position-specific cos/sin via ttnn.embedding (fully on-device, trace-safe)
        # position_idx: [1, 32] uint32 padded tensor for embedding lookup
        cos_pos = ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)  # [1, batch_pad, head_dim]
        sin_pos = ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
        cos_pos = ttnn.unsqueeze_to_4D(cos_pos)  # [1, 1, batch_pad, head_dim]
        sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
        # rotary_embedding expects cos/sin as [1, 1, *, head_dim] — token_index=0 indexes position 0
        # which holds the data for the actual current position (gathered by embedding above)
        tt_q = apply_rope(tt_q, cos_pos, sin_pos, token_index=0)
        if not is_kv_shared:
            tt_k = apply_rope(tt_k, cos_pos, sin_pos, token_index=0)
    else:
        # Legacy path: full 4D cache with Python int token_index
        tt_q = apply_rope(tt_q, cos_cache, sin_cache, token_index=token_index)
        if not is_kv_shared:
            tt_k = apply_rope(tt_k, cos_cache, sin_cache, token_index=token_index)

    # 5. KV cache update — skip for KV-shared layers (source layer already updated the cache)
    # Use position_idx_cache (int32) for cache ops when position_idx is uint32 (embedding lookup format)
    cache_pos = position_idx_cache if position_idx_cache is not None else position_idx
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        if not is_kv_shared:
            # After HF-style RoPE, tensors may be in DRAM. Move to HEIGHT_SHARDED for cache update.
            tt_k = ttnn.to_memory_config(tt_k, q_sharded_mem)
            tt_v = ttnn.to_memory_config(tt_v, q_sharded_mem)

            if page_table is not None:
                ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=cache_pos, page_table=page_table)
                ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=cache_pos, page_table=page_table)
            else:
                ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=cache_pos)
                ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=cache_pos)
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
            cur_pos_tensor=cache_pos,
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
            cur_pos_tensor=cache_pos,
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
