# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    apply_rope_decode_peruser,
    concat_heads,
    effective_block_size,
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
    sequential_kv_write=False,
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
    if use_embedding_rope:
        # Gather position-specific cos/sin via ttnn.embedding (fully on-device, trace-safe)
        # position_idx: [1, 32] uint32 padded tensor for embedding lookup
        cos_pos = ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT)  # [1, batch_pad, head_dim]
        sin_pos = ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT)
        cos_pos = ttnn.unsqueeze_to_4D(cos_pos)  # [1, 1, batch_pad, head_dim]
        sin_pos = ttnn.unsqueeze_to_4D(sin_pos)
        # RoPE. batch=1 uses the fused single-position rotary_embedding (one core
        # but cheap, no slice/tilize churn). batch>1 needs per-user positions,
        # which that op can't express, so fall back to the manual elementwise
        # q*cos + rotate_half(q)*sin (numerically equivalent — isolation PCC
        # ~0.99999 vs the fused op and the HF reference — but a few ops costlier).
        batch = tt_q.shape[1]
        if batch == 1:
            tt_q = apply_rope(tt_q, cos_pos, sin_pos, token_index=0)
            if not is_kv_shared:
                tt_k = apply_rope(tt_k, cos_pos, sin_pos, token_index=0)
        else:
            cos_b = ttnn.transpose(cos_pos, 1, 2)  # [1, batch_pad, 1, head_dim]
            sin_b = ttnn.transpose(sin_pos, 1, 2)
            cos_b = cos_b[:, :batch, :, :]
            sin_b = sin_b[:, :batch, :, :]
            tt_q = apply_rope_decode_peruser(tt_q, cos_b, sin_b)
            if not is_kv_shared:
                tt_k = apply_rope_decode_peruser(tt_k, cos_b, sin_b)
    else:
        # Legacy path: full 4D cache with Python int token_index
        tt_q = apply_rope(tt_q, cos_cache, sin_cache, token_index=token_index)
        if not is_kv_shared:
            tt_k = apply_rope(tt_k, cos_cache, sin_cache, token_index=token_index)

    # 5. KV cache update — skip for KV-shared layers (source layer already updated the cache)
    # Use position_idx_cache (int32) for cache ops when position_idx is uint32 (embedding lookup format)
    cache_pos = position_idx_cache if position_idx_cache is not None else position_idx
    # Bounded sliding-window cache: when set, the op wraps absolute positions into a
    # circular buffer of ``cache_position_modulo`` tokens before the page_table lookup.
    # ``None`` = legacy unbounded behavior (set on full-attention layers or when the
    # bounded-cache mode isn't wired up). Empty kwargs dict on the legacy path keeps
    # kernel signatures and program-cache keys for existing callers byte-identical.
    paged_modulo_kwargs = (
        {"cache_position_modulo": config.cache_position_modulo} if config.cache_position_modulo is not None else {}
    )
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
                batch = tt_k.shape[1]
                if sequential_kv_write and batch > 1:
                    # Speculative VERIFY: the B candidates sit at consecutive
                    # positions that share ONE paged block (all batch rows of the
                    # page table point to the same physical blocks). A single
                    # batched paged_update_cache then has multiple batch entries
                    # read-modify-write the SAME block tile concurrently and race
                    # (only some writes survive -> corrupt KV). Writing each
                    # position with its own ordered op serializes the block RMW.
                    # Matmuls + SDPA still run once on the full batch, so the
                    # verify speedup is preserved (only these tiny writes loop).
                    # paged_update_cache wants a sharded update tensor with
                    # num_shards == num_users. Slice each candidate to a batch=1
                    # [1,1,nkv,hd] tensor (from DRAM, where slicing is clean) and
                    # reshard onto a single core. q_sharded_mem shards one user per
                    # core (shard shape = [32, head_dim], head_dim shared by Q/K),
                    # so a 1-core config with that same shard shape is exactly the
                    # per-user layout the op expects.
                    _shard_shape = list(q_sharded_mem.shard_spec.shape)
                    _one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
                    single_user_mem = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(_one_core, _shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    k_seq = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)
                    v_seq = ttnn.to_memory_config(tt_v, ttnn.DRAM_MEMORY_CONFIG)
                    nkv, hd = k_seq.shape[2], k_seq.shape[3]
                    for b in range(batch):
                        kb = ttnn.slice(k_seq, [0, b, 0, 0], [1, b + 1, nkv, hd])
                        vb = ttnn.slice(v_seq, [0, b, 0, 0], [1, b + 1, nkv, hd])
                        kb = ttnn.to_memory_config(kb, single_user_mem)
                        vb = ttnn.to_memory_config(vb, single_user_mem)
                        pos_b = ttnn.slice(cache_pos, [b], [b + 1])
                        pt_b = ttnn.slice(page_table, [b, 0], [b + 1, page_table.shape[1]])
                        ttnn.experimental.paged_update_cache(
                            k_cache,
                            kb,
                            update_idxs_tensor=pos_b,
                            page_table=pt_b,
                            block_size=eff_bs,
                            num_kv_heads=num_local_kv_heads,
                            **paged_modulo_kwargs,
                        )
                        ttnn.experimental.paged_update_cache(
                            v_cache,
                            vb,
                            update_idxs_tensor=pos_b,
                            page_table=pt_b,
                            block_size=eff_bs,
                            num_kv_heads=num_local_kv_heads,
                            **paged_modulo_kwargs,
                        )
                        for t in (kb, vb, pos_b, pt_b):
                            t.deallocate(True)
                    k_seq.deallocate(True)
                    v_seq.deallocate(True)
                else:
                    ttnn.experimental.paged_update_cache(
                        k_cache,
                        tt_k,
                        update_idxs_tensor=cache_pos,
                        page_table=page_table,
                        block_size=eff_bs,
                        num_kv_heads=num_local_kv_heads,
                        **paged_modulo_kwargs,
                    )
                    ttnn.experimental.paged_update_cache(
                        v_cache,
                        tt_v,
                        update_idxs_tensor=cache_pos,
                        page_table=page_table,
                        block_size=eff_bs,
                        num_kv_heads=num_local_kv_heads,
                        **paged_modulo_kwargs,
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
            **paged_modulo_kwargs,
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
    num_local_heads = config.num_attention_heads // tp
    tt_out = concat_heads(
        tt_sdpa, is_decode_mode=True, num_heads=num_local_heads, head_dim=config.head_dim, mesh_device=mesh_device
    )
    tt_out = apply_output_projection(tt_out, weights)
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, config.hidden_size)

    return tt_out
