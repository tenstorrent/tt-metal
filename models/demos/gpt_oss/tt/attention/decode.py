# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger

import ttnn

from .config import AttentionConfig, ProgramConfig
from .operations import apply_rope
from .weights import AttentionWeights


def decode_forward(
    hidden_states,
    rope_mats,
    weights: AttentionWeights,
    kv_cache,
    config: AttentionConfig,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    transformation_mat,
    kv_mem_cfg,
    position_idx,
    page_table,
    ccl_manager,
):
    """
    Decode forward pass - optimized for single token (seq_len=1).

    Args:
        hidden_states: Input tensor [batch, 1, hidden_size]
        rope_mats: Tuple of (cos, sin) matrices for RoPE
        weights: Attention weights
        kv_cache: KV cache [k_cache, v_cache]
        config: Attention configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        program_config: Model-specific program configs
        transformation_mat: Transformation matrix for RoPE
        kv_mem_cfg: Memory config for KV tensors
        position_idx: Current position index
        page_table: Page table for paged attention (optional)
        ccl_manager: Communication manager

    Returns:
        Attention output [batch, 1, hidden_size]
    """
    # batch_size, seq_len, hidden_size = hidden_states.shape
    _, seq_len, batch_size, hidden_size = hidden_states.shape

    # Validate decode mode
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

    # QKV projection. With TP>1 the per-device QKV is small enough to fit in
    # an L1 width-sharded layout that nlp_create_qkv_heads_decode consumes
    # directly. With TP=1 (e.g. single Blackhole card) the per-device QKV is
    # TP× larger and overflows the per-core CB if width-sharded, so we use
    # DRAM interleaved instead. The kernel-side aligned-read fix in
    # nlp_create_qkv_heads_decode (PR #43292) handles the BH NOC alignment
    # constraint for DRAM-interleaved inputs; before that fix this path
    # produced silent corruption (every odd Q/K/V head returned the previous
    # user's row).
    qkv_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mesh_config.tp > 1 else ttnn.DRAM_MEMORY_CONFIG
    xqkv_fused = ttnn.matmul(hidden_states, weights.wqkv, dtype=ttnn.bfloat16, memory_config=qkv_memory_config)
    ttnn.add(xqkv_fused, weights.wqkv_bias, output_tensor=xqkv_fused)

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    head_dim = config.head_dim

    # One user per core on the grid RoPE and SDPA decode expect (see ProgramConfig.get_decode_user_grid).
    # This placement is load-bearing: with a bare L1_HEIGHT_SHARDED_MEMORY_CONFIG the op falls back to
    # the *device* compute grid, which is 8 wide on Wormhole but 13 wide on Blackhole, so for a batch
    # that is a multiple of 32 user b would land on (b % 13, b // 13) while RotarySetup's cos/sin and
    # the paged SDPA reducer live at (b % 8, b // 8): every downstream op silently reads another user's
    # Q/K/V (no TT_FATAL). Batch 1 only worked because core (0, 0) coincides.
    batch_grid, _ = program_config.get_decode_user_grid(mesh_device, batch_size)
    qkv_heads_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        memory_config=qkv_heads_mem_config,
    )

    xqkv_fused.deallocate(True)

    # Apply RoPE
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
    tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # DEBUG: Check for NaN/Inf in Q/K after rope (enable with DEBUG_ATTENTION=1)
    # Disabled by default to avoid performance impact

    # DIAGNOSTIC: with identical activations across users (e.g. one prompt broadcast to all),
    # position_idx must be identical for every user at every step -- it is just each user's
    # token count so far. GPT_OSS_DUMP_POSITION_IDX=1 reports cross-user agreement on it right
    # before the cache write, separating "wrong position bookkeeping" from "wrong cache
    # content/attention math at a correct position".
    if os.getenv("GPT_OSS_DUMP_POSITION_IDX") == "1":
        import collections

        # position_idx is row-sharded, column-replicated (ShardTensor2dMesh dims=(0, None)):
        # take one device per mesh row (col 0) and concat, avoiding the 2D mesh composer's
        # "no replicated axis" restriction entirely.
        _mesh_cols = mesh_device.shape[1]
        _row_shards = [
            ttnn.to_torch(t) for i, t in enumerate(ttnn.get_device_tensors(position_idx)) if i % _mesh_cols == 0
        ]
        _pos_t = torch.cat([s.flatten() for s in _row_shards])
        _n = _pos_t.shape[0]
        _by = collections.defaultdict(set)
        for _u in range(_n):
            _by[_u % 8].add(int(_pos_t[_u]))
        logger.warning(
            f"POSITION_IDX: {_n} entries, distinct values overall {sorted(set(_pos_t.tolist()))} | "
            "by idx%8 (should each be a single value): " + " ".join(f"{c}:{sorted(_by[c])}" for c in sorted(_by))
        )

    # Update KV cache
    k_cache, v_cache = kv_cache
    tt_k = ttnn.to_memory_config(tt_k, kv_mem_cfg)
    tt_v = ttnn.to_memory_config(tt_v, kv_mem_cfg)

    # DIAGNOSTIC: compare the K about to be written against what paged_update_cache actually
    # stores, for a healthy user (local idx 0) and a broken one (local idx 4), row 0 only.
    # Isolates the WRITE: if the stored content already differs from tt_k for the broken user,
    # the defect is in paged_update_cache/page_table addressing, not in a later read (SDPA).
    _dump_kv = os.getenv("GPT_OSS_DUMP_KV") == "1"
    if _dump_kv:
        if not hasattr(decode_forward, "_dump_kv_calls"):
            decode_forward._dump_kv_calls = 0
        decode_forward._dump_kv_calls += 1
        # decode_forward is called once per layer per decode step, layers in order 0..N-1.
        # GPT_OSS_DUMP_KV_STRIDE = num_layers selects which LAYER to dump via the remainder
        # (GPT_OSS_DUMP_KV_LAYER_OFFSET, 0 = layer 0, 1 = layer 1, ...); cap total dumps to
        # GPT_OSS_DUMP_KV_MAX_CALLS steps of that one layer.
        _stride = int(os.getenv("GPT_OSS_DUMP_KV_STRIDE", "1"))
        _layer_offset = int(os.getenv("GPT_OSS_DUMP_KV_LAYER_OFFSET", "0"))
        _max_calls = int(os.getenv("GPT_OSS_DUMP_KV_MAX_CALLS", "6")) * _stride
        _dump_kv = (decode_forward._dump_kv_calls <= _max_calls) and (
            (decode_forward._dump_kv_calls - 1 - _layer_offset) % _stride == 0
        )
    if _dump_kv:
        _mesh_cols = mesh_device.shape[1]
        _row = int(os.getenv("GPT_OSS_DUMP_KV_ROW", "0"))
        _row_dev = _row * mesh_device.shape[1]
        _tt_k_row0 = ttnn.to_torch(list(ttnn.get_device_tensors(tt_k))[_row_dev])
        _block_size = int(os.getenv("GPT_OSS_KV_BLOCK_SIZE", "64"))
        _pt_row0 = (
            ttnn.to_torch(list(ttnn.get_device_tensors(page_table))[_row_dev]) if page_table is not None else None
        )
        _posidx_row0 = ttnn.to_torch(list(ttnn.get_device_tensors(position_idx))[_row_dev]).flatten()
        logger.warning(f"DUMP_KV: tt_k per-device shape {tuple(_tt_k_row0.shape)}")
        if _pt_row0 is not None:
            logger.warning(f"DUMP_KV: page_table per-device shape {tuple(_pt_row0.shape)}")

    ttnn.experimental.paged_update_cache(
        k_cache,
        tt_k,
        update_idxs_tensor=position_idx,
        page_table=page_table,
    )
    ttnn.experimental.paged_update_cache(
        v_cache,
        tt_v,
        update_idxs_tensor=position_idx,
        page_table=page_table,
    )

    if _dump_kv:
        _kcache_row0 = ttnn.to_torch(list(ttnn.get_device_tensors(k_cache))[_row_dev])
        logger.warning(f"DUMP_KV: k_cache per-device shape {tuple(_kcache_row0.shape)}")
        for _u in (0, 4):
            _pos = int(_posidx_row0[_u])
            _blk_in_seq = _pos // _block_size
            _off = _pos % _block_size
            _blk_id = int(_pt_row0[_u, _blk_in_seq]) if _pt_row0 is not None else _u
            try:
                _written = _tt_k_row0[..., _u, :, :].flatten()[:8].tolist()
            except Exception as e:
                _written = f"<index error: {e}>"
            try:
                _stored = _kcache_row0[_blk_id, :, _off, :].flatten()[:8].tolist()
            except Exception as e:
                _stored = f"<index error: {e}>"
            logger.warning(
                f"DUMP_KV: user {_u} pos={_pos} block_id={_blk_id} offset={_off} | "
                f"tt_k(about to write)[:8]={_written} | k_cache(stored)[:8]={_stored}"
            )

    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # Calculate padded heads (must be tile-aligned, e.g., 32)
    # Use local heads per device, not global heads
    padded_heads = ((num_local_heads + 31) // 32) * 32

    # SDPA writes to DRAM; reshard onto the same one-core-per-user grid RoPE/SDPA use (user b on the b-th core,
    # row-major) for nlp_concat_heads_decode. For an input grid that is not one rectangle from (0, 0) (e.g. 13 + 9
    # cores for 22 users on Blackhole's 13-wide grid) the op requires sub_core_grids (passed below) and then reads
    # the users in row-major order of the input CoreRangeSet; with it, any 1 <= batch <= 32 works, so no separate
    # rectangular 'concat grid' (and no batch sizes it could not hold) is needed.
    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),  # Shape per shard (tile-aligned)
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    device_grid = mesh_device.compute_with_storage_grid_size()
    concat_sub_core_grids = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid.x - 1, device_grid.y - 1))}
    )  # output: the first num_local_heads cores of the compute grid, as in the op's default program
    # Scaled dot-product attention
    if page_table is not None:
        tt_sdpa_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=position_idx,
            sliding_window_size=config.sliding_window,
            attention_sink=weights.decode_sinks,
            page_table_tensor=page_table,
            scale=config.scaling,
            program_config=program_config.get_decode_sdpa_config(mesh_device, batch_size),
            compute_kernel_config=program_config.get_compute_kernel_config(),
            # memory_config=height_sharded_mem_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
    else:
        # GQA (num_kv_heads > 1) rejects sharded output in the SDPA decode
        # device op — match the paged path: write to DRAM, then to_memory_config
        # into the height-sharded layout that downstream concat_heads needs.
        tt_sdpa_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=position_idx,
            sliding_window_size=config.sliding_window,
            attention_sink=weights.decode_sinks,
            scale=config.scaling,
            program_config=program_config.get_decode_sdpa_config(mesh_device, batch_size),
            compute_kernel_config=program_config.get_compute_kernel_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
    if _dump_kv:
        _sdpa_row0 = ttnn.to_torch(list(ttnn.get_device_tensors(tt_sdpa_tensor))[_row_dev])
        logger.warning(
            f"DUMP_KV: sdpa output per-device shape {tuple(_sdpa_row0.shape)} (seq, batch, heads_local, head_dim)"
        )
        for _u in (0, 4):
            try:
                _val = _sdpa_row0[0, _u].flatten()[:8].tolist()  # all heads for user _u, first 8 values
            except Exception as e:
                _val = f"<index error over shape {tuple(_sdpa_row0.shape)}: {e}>"
            logger.warning(f"DUMP_KV: sdpa_out user {_u} (all heads, first 8 vals)={_val}")

    tt_q.deallocate(True)

    # Concat heads and apply output projection
    tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(
        tt_sdpa_tensor, num_heads=num_local_heads, sub_core_grids=concat_sub_core_grids
    )
    tt_sdpa_tensor.deallocate(True)

    if _dump_kv:
        _concat_row0 = ttnn.to_torch(list(ttnn.get_device_tensors(tt_sdpa_out))[_row_dev])
        logger.warning(f"DUMP_KV: concat_heads output per-device shape {tuple(_concat_row0.shape)}")
        _flat = _concat_row0.reshape(-1, _concat_row0.shape[-1])
        for _u in (0, 4):
            try:
                _val = _flat[_u].flatten()[:8].tolist()
            except Exception as e:
                _val = f"<index error: {e}>"
            logger.warning(f"DUMP_KV: concat_out user {_u}[:8]={_val}")

    tt_out = ttnn.linear(
        tt_sdpa_out, weights.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )

    if _dump_kv:
        _oproj_row0 = ttnn.to_torch(list(ttnn.get_device_tensors(tt_out))[_row_dev])
        logger.warning(f"DUMP_KV: o_proj output per-device shape {tuple(_oproj_row0.shape)}")
        _flat2 = _oproj_row0.reshape(-1, _oproj_row0.shape[-1])
        for _u in (0, 4):
            try:
                _val = _flat2[_u].flatten()[:8].tolist()
            except Exception as e:
                _val = f"<index error: {e}>"
            logger.warning(f"DUMP_KV: oproj_out user {_u}[:8]={_val}")

    tt_sdpa_out.deallocate(True)
    tt_out = ttnn.add(tt_out, weights.o_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_out = ttnn.typecast(tt_out, ttnn.bfloat8_b)

    # Calculate padded hidden size for tile-aligned CCL operations.
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    padded_hidden = padded_local_hidden * mesh_config.tp if mesh_config.tp > 1 else hidden_size

    tt_out = ttnn.reshape(
        tt_out,
        (1, 1, batch_size, padded_hidden),
        (1, 1, 32, padded_hidden),
    )

    # Slice padding AFTER bias: [1,1,B,3072] -> [1,1,B,2880].
    # Bias already applied on padded tensor; padding columns are zeros.
    if padded_hidden != hidden_size and mesh_config.tp > 1:
        tt_out = ttnn.slice(
            tt_out,
            starts=[0, 0, 0, 0],
            ends=[1, 1, batch_size, hidden_size],
            steps=[1, 1, 1, 1],
        )
        # Workaround: ttnn.slice on bf8 TILE produces a buffer with non-standard
        # page layout under L1 fragmentation that CCL all_reduce misreads.
        # DRAM round-trip normalizes the buffer. See #41640 for repro and root cause analysis.
        tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)
        tt_out = ttnn.to_memory_config(tt_out, ttnn.L1_MEMORY_CONFIG)

    if _dump_kv:
        _pre_ar_row0 = ttnn.to_torch(list(ttnn.get_device_tensors(tt_out))[_row_dev])
        _flat3 = _pre_ar_row0.reshape(-1, _pre_ar_row0.shape[-1])
        for _u in (0, 4):
            logger.warning(f"DUMP_KV: pre-all_reduce user {_u}[:8]={_flat3[_u].flatten()[:8].tolist()}")

    # Tensor parallel all-reduce (AllBroadcast, ~80μs vs RS+AG ~138μs).
    if mesh_config.tp > 1:
        tt_out = ttnn.all_reduce(
            tt_out,
            num_links=ccl_manager.num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=mesh_config.tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    if _dump_kv:
        _post_ar_row0 = ttnn.to_torch(list(ttnn.get_device_tensors(tt_out))[_row_dev])
        _flat4 = _post_ar_row0.reshape(-1, _post_ar_row0.shape[-1])
        for _u in (0, 4):
            logger.warning(f"DUMP_KV: post-all_reduce user {_u}[:8]={_flat4[_u].flatten()[:8].tolist()}")

    return tt_out
