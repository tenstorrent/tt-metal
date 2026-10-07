# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from ..fused_decode import (
    OPROJ_DECODE_WEIGHT_DTYPE,
    OPROJ_STREAM_READERS,
    QKV_DECODE_WEIGHT_DTYPE,
    QKV_STREAM_READERS,
    sdpa_decode_program_config,
)
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
    fused=False,
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
        fused: Fused decode path (fused_decode.py): hidden_states is the flat normed hidden of the layer boundary
            (decode_boundary.py, one token); returns this device's flat o_proj partial sum, which the next boundary
            all-reduces

    Returns:
        Attention output [batch, 1, hidden_size]
    """
    if fused:
        batch_size, hidden_size = 1, config.hidden_size
    else:
        _, seq_len, batch_size, hidden_size = hidden_states.shape
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

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    head_dim = config.head_dim
    k_cache, v_cache = kv_cache

    if fused:
        # rope_mats / transformation_mat are laid out for rotary_embedding_llama_fused_qk: cos/sin rows for the Q
        # users then the K users (Model use_qk_fused).
        assert rope_mats[0].shape[1] == 2 * batch_size, "fused decode needs the fused-QK RoPE layout (use_qk_fused)"
        # Streamed QKV + bias (experts/stream.py) from the flat norm output, written straight into the shared
        # Q / K / V head tensors: Q and K on disjoint cores (V shares Q's), the layout the fused QK RoPE and the fused
        # K/V cache update require (what nlp_create_qkv_heads_decode(overlap_qk_coregrid=False) produced).
        qkv_stream = ccl_manager.get_decode_linear_stream(
            "qkv",
            hidden_size // ttnn.TILE_SIZE,
            (num_local_heads + 2 * num_local_kv_heads) * head_dim,
            QKV_DECODE_WEIGHT_DTYPE,
            readers=QKV_STREAM_READERS,
            out_mode=3,
            heads=(num_local_heads * head_dim // ttnn.TILE_SIZE, num_local_kv_heads * head_dim // ttnn.TILE_SIZE),
        )
        tt_q, tt_k, tt_v = qkv_stream(
            hidden_states,
            weights.wqkv_stream,
            ccl_manager.get_decode_qkv_heads(num_local_heads, num_local_kv_heads, head_dim),
        )
        tt_q, tt_k = ttnn.experimental.rotary_embedding_llama_fused_qk(
            tt_q, tt_k, rope_mats[0], rope_mats[1], transformation_mat
        )
        ttnn.experimental.paged_fused_update_cache(
            k_cache, tt_k, v_cache, tt_v, update_idxs_tensor=position_idx, page_table=page_table
        )
        tt_v = None  # shared head buffer, not owned here
    else:
        xqkv_fused = ttnn.matmul(hidden_states, weights.wqkv, dtype=ttnn.bfloat16, memory_config=qkv_memory_config)
        ttnn.add(xqkv_fused, weights.wqkv_bias, output_tensor=xqkv_fused)

        tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=num_local_heads,
            num_kv_heads=num_local_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

        xqkv_fused.deallocate(True)

        # Apply RoPE
        tt_q_orig = tt_q
        tt_k_orig = tt_k
        tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
        tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)
        tt_q_orig.deallocate(True)
        tt_k_orig.deallocate(True)

        # Update KV cache
        tt_k = ttnn.to_memory_config(tt_k, kv_mem_cfg)
        tt_v = ttnn.to_memory_config(tt_v, kv_mem_cfg)

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

    tt_k.deallocate(True)
    if tt_v is not None:
        tt_v.deallocate(True)
    grid_size = ttnn.CoreCoord(8, 8)
    batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)

    # Calculate padded heads (must be tile-aligned, e.g., 32)
    # Use local heads per device, not global heads
    padded_heads = ((num_local_heads + 31) // 32) * 32

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),  # Shape per shard (tile-aligned)
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
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
            program_config=(
                sdpa_decode_program_config(config.sliding_window)
                if fused
                else program_config.get_decode_sdpa_config(mesh_device)
            ),
            compute_kernel_config=program_config.get_compute_kernel_config(),
            # memory_config=height_sharded_mem_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
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
            program_config=program_config.get_decode_sdpa_config(mesh_device),
            compute_kernel_config=program_config.get_compute_kernel_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    tt_q.deallocate(True)

    if fused:
        # Streamed o_proj + bias (on the first TP device only) reading the heads straight out of the DRAM SDPA output
        # (concat heads = head-major row order) and writing this device's flat partial sum (decode_boundary.py),
        # which the layer's next boundary all-reduces.
        partial = ccl_manager.get_decode_partial(hidden_size)
        o_stream = ccl_manager.get_decode_linear_stream(
            "o_proj",
            num_local_heads * head_dim // ttnn.TILE_SIZE,
            hidden_size,
            OPROJ_DECODE_WEIGHT_DTYPE,
            readers=OPROJ_STREAM_READERS,
            x_pages=head_dim // ttnn.TILE_SIZE,
            out_mode=5,
        )
        o_stream(
            tt_sdpa_tensor, weights.o_proj_stream, partial, send=ccl_manager.decode_boundary_send(hidden_size, "attn")
        )
        tt_sdpa_tensor.deallocate(True)
        return partial

    tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
    # Concat heads and apply output projection
    tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(tt_sdpa_tensor, num_heads=num_local_heads)
    tt_sdpa_tensor.deallocate(True)

    tt_out = ttnn.linear(
        tt_sdpa_out, weights.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )

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

    # Tensor parallel all-reduce (AllBroadcast, ~80μs vs RS+AG ~138μs).
    if mesh_config.tp > 1:
        tt_out = ttnn.all_reduce(
            tt_out,
            num_links=ccl_manager.num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=mesh_config.tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    return tt_out
