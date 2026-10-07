# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from ..fused_decode import auto_matmul_compute_config, matmul_1d_program_config, width_sharded_memory_config
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
        fused: Fused decode path (fused_decode.py); returns the all-reduced BF16 output in the residual layout

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

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    head_dim = config.head_dim
    k_cache, v_cache = kv_cache

    if fused:
        # rope_mats / transformation_mat are laid out for rotary_embedding_llama_fused_qk: cos/sin rows for the Q
        # users then the K users (Model use_qk_fused).
        assert rope_mats[0].shape[1] == 2 * batch_size, "fused decode needs the fused-QK RoPE layout (use_qk_fused)"
        # QKV projection with the bias fused into the matmul, written width-sharded (one tile per core) for the
        # head split. The 1D multicast matmul reads its activation interleaved.
        qkv_input = hidden_states
        if hidden_states.is_sharded():
            qkv_input = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        qkv_width = weights.wqkv.shape[-1]
        qkv_sharded = width_sharded_memory_config(mesh_device, qkv_width, qkv_width // ttnn.TILE_SIZE)
        xqkv_fused = ttnn.linear(
            qkv_input,
            weights.wqkv,
            bias=weights.wqkv_bias,
            program_config=matmul_1d_program_config(qkv_sharded, hidden_size, in0_block_w=15),
            # LoFi as in the unfused graph, whose BFP8 x BFP8 QKV matmul took the low-precision default.
            compute_kernel_config=auto_matmul_compute_config(mesh_device, operands_low_precision=True),
            dtype=ttnn.bfloat16,
            memory_config=qkv_sharded,
        )
        if qkv_input is not hidden_states:
            qkv_input.deallocate(True)
        # Q and K on disjoint cores (K right after Q; V shares Q's cores): the layout the fused QK RoPE and the
        # fused K/V cache update require.
        tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=num_local_heads,
            num_kv_heads=num_local_kv_heads,
            overlap_qk_coregrid=False,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv_fused.deallocate(True)
        tt_q_orig, tt_k_orig = tt_q, tt_k
        tt_q, tt_k = ttnn.experimental.rotary_embedding_llama_fused_qk(
            tt_q, tt_k, rope_mats[0], rope_mats[1], transformation_mat
        )
        tt_q_orig.deallocate(True)
        tt_k_orig.deallocate(True)
        ttnn.experimental.paged_fused_update_cache(
            k_cache, tt_k, v_cache, tt_v, update_idxs_tensor=position_idx, page_table=page_table
        )
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
            program_config=program_config.get_decode_sdpa_config(mesh_device),
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
        # Concat heads as one reshape: [1, B, heads, head_dim] -> [1, 1, B, heads * head_dim] (row-major order is
        # exactly the head concatenation), landing interleaved in L1 for the o_proj matmul.
        tt_sdpa_in = ttnn.reshape(
            tt_sdpa_tensor, (1, 1, batch_size, num_local_heads * head_dim), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tt_sdpa_tensor.deallocate(True)
        # Row-parallel o_proj with the bias fused (device 0 only), written straight into the fused all-reduce's
        # width-sharded layout; the all-reduce returns the BF16 sum in the residual layout.
        all_reduce = ccl_manager.get_decode_all_reduce(hidden_size, mesh_config.tp_axis)
        tt_out = ttnn.linear(
            tt_sdpa_in,
            weights.o_proj_decode,
            bias=weights.o_proj_bias_decode,
            program_config=matmul_1d_program_config(all_reduce.memory_config, tt_sdpa_in.shape[-1], in0_block_w=8),
            compute_kernel_config=auto_matmul_compute_config(mesh_device, operands_low_precision=False),
            dtype=ttnn.bfloat8_b,
            memory_config=all_reduce.memory_config,
        )
        tt_sdpa_in.deallocate(True)
        tt_out = ttnn.reshape(tt_out, (1, 1, batch_size, hidden_size), (1, 1, 32, hidden_size))
        return all_reduce(tt_out, "attn")

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
