# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decode-mode attention forward pass for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (no transformation matrices).
"""

import ttnn

from .operations import (
    _batch_to_corerange,
    apply_allreduce,
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    concat_heads_decode,
    effective_block_size,
    split_qkv_heads_decode,
)
from .weights import AttentionWeights


def _build_decode_rope_mats(position_idx, cos_cache, sin_cache, batch_size, mesh_device):
    """Build per-user decode cos/sin for ``rotary_embedding_hf``.

    Gathers position-specific cos/sin from the 2D caches via ``ttnn.embedding``
    (one row per user, taken from the first ``batch_size`` slots of the padded
    ``position_idx``), reshapes to ``[1, batch, 1, head_dim]`` so users sit on
    dim 1 (matching the Q/K decode layout), and height-shards onto the same
    one-user-per-core grid as Q/K. Mirrors tt_transformers ``HfRotarySetup``.
    """
    cos = ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)  # [1, 32, head_dim]
    sin = ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, 32, head_dim]
    sin = ttnn.unsqueeze_to_4D(sin)
    cos = ttnn.transpose(cos, 1, 2)  # [1, 32, 1, head_dim]
    sin = ttnn.transpose(sin, 1, 2)
    cos = cos[:, :batch_size, :, :]  # [1, batch, 1, head_dim]
    sin = sin[:, :batch_size, :, :]

    head_dim = cos.shape[-1]
    grid_size = mesh_device.compute_with_storage_grid_size()
    batch_grid = _batch_to_corerange(batch_size, grid_size)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    cos = ttnn.interleaved_to_sharded(cos, mem_config)
    sin = ttnn.interleaved_to_sharded(sin, mem_config)
    return cos, sin


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

    # Decode layout is [1, 1, batch, hidden_size]; users live on dim 2.
    batch_size = hidden_states.shape[2]

    # 1. Fused QKV projection
    xqkv = apply_qkv_projection(hidden_states, weights)

    # 2. Split into Q, K, V heads
    tt_q, tt_k, tt_v = split_qkv_heads_decode(
        xqkv, config, weights.is_global, tp=tp, kv_replicated=weights.kv_replicated
    )

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
    if use_embedding_rope and batch_size > 1:
        # Batched decode: each user has its own position, so a single broadcast
        # rotation (the legacy rotary_embedding op) is wrong. Build per-user
        # cos/sin shaped [1, batch, 1, head_dim] height-sharded onto the same
        # one-user-per-core grid as Q/K, and apply rotary_embedding_hf (matches
        # the tt_transformers HfRotarySetup decode path).
        tt_q = ttnn.to_memory_config(tt_q, q_sharded_mem)
        if not is_kv_shared:
            tt_k = ttnn.to_memory_config(tt_k, q_sharded_mem)
        cos_pos, sin_pos = _build_decode_rope_mats(position_idx, cos_cache, sin_cache, batch_size, mesh_device)
        tt_q = ttnn.experimental.rotary_embedding_hf(tt_q, cos_pos, sin_pos, is_decode_mode=True)
        if not is_kv_shared:
            tt_k = ttnn.experimental.rotary_embedding_hf(tt_k, cos_pos, sin_pos, is_decode_mode=True)
    elif use_embedding_rope:
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
                # Per-device kv-head count of the layer's input view. When the cache
                # was allocated for a different layer type under HMA cross-group
                # sharing (Gemma4-26B-A4B sliding kv=8 / full kv=2 on multi-device
                # TP) cache.padded_shape[1] disagrees with what the kernel needs to
                # write — see paged_update_cache num_kv_heads kwarg. Mirrors
                # split_qkv_heads_decode's local head count.
                num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
                eff_bs = effective_block_size(k_cache, config.head_dim, num_local_kv_heads)
                ttnn.experimental.paged_update_cache(
                    k_cache,
                    tt_k,
                    update_idxs_tensor=cache_pos,
                    page_table=page_table,
                    block_size=eff_bs,
                    num_kv_heads=num_local_kv_heads,
                )
                ttnn.experimental.paged_update_cache(
                    v_cache,
                    tt_v,
                    update_idxs_tensor=cache_pos,
                    page_table=page_table,
                    block_size=eff_bs,
                    num_kv_heads=num_local_kv_heads,
                )
            else:
                ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=cache_pos)
                ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=cache_pos)
    else:
        k_cache = tt_k
        v_cache = tt_v

    # 6. SDPA (scale=1.0)
    sliding_window = config.sliding_window if config.is_sliding else None

    # Always pass an SDPAProgramConfig so num_cores_per_head stays within
    # MAX_TREE_REDUCTION_ROUNDS=6 (=> 64 cores/head). With program_config=None,
    # the SDPA op falls back to the full device grid, which exceeds 64 cores
    # on Blackhole (>=110 cores) when num_kv_heads is small. The struct's
    # default max_cores_per_head_batch=16 caps the per-head reduction tree.
    if config.head_dim >= 512:
        # Global layers: smaller grid — head_dim=512 needs more L1 per core.
        sdpa_grid = ttnn.CoreCoord(8, 4)
    else:
        # Sliding layers: use the full device compute grid.
        device_grid = mesh_device.compute_with_storage_grid_size()
        sdpa_grid = ttnn.CoreCoord(device_grid.x, device_grid.y)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_grid,
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
    )

    if page_table is not None:
        sdpa_num_local_kv_heads = 1 if weights.kv_replicated else config.num_key_value_heads // tp
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
            block_size=effective_block_size(k_cache, config.head_dim, sdpa_num_local_kv_heads),
            # Tell SDPA the layer's view of the cache when the buffer was allocated
            # for a different layer type under HMA cross-group sharing — same
            # rationale as the num_kv_heads override on paged_update_cache.
            num_kv_heads=sdpa_num_local_kv_heads,
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
    if batch_size > 1:
        # Batched decode: SDPA output is [1, batch, num_local_heads, head_dim] in
        # DRAM. Reshard one-user-per-core and fold heads with nlp_concat_heads_decode.
        num_local_heads = config.num_attention_heads // tp
        tt_out = concat_heads_decode(tt_sdpa, num_local_heads, batch_size, mesh_device)
    else:
        tt_out = concat_heads(tt_sdpa, is_decode_mode=True)
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    return tt_out
