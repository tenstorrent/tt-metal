# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.model_config import num_to_corerange

from .config import AttentionConfig, ProgramConfig
from .operations import apply_allreduce, apply_rope
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
    fused_transformation_mat,
    kv_mem_cfg,
    position_idx,
    page_table,
    ccl_manager,
):
    """
    Decode forward pass - optimized for single token (seq_len=1).
    """
    _, seq_len, batch_size, hidden_size = hidden_states.shape

    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

    # QKV projection
    xqkv_fused = ttnn.matmul(
        hidden_states, weights.wqkv, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )
    xqkv_fused = ttnn.add(xqkv_fused, weights.wqkv_bias, output_tensor=xqkv_fused)

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    head_dim = config.head_dim

    tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    xqkv_fused.deallocate(True)

    # Apply RoPE - use fused QK op when batch allows (requires 2*batch cores, max 32 on 8x8 grid)
    use_fused_qk = fused_transformation_mat is not None and batch_size <= 32

    if use_fused_qk:
        # Prepare doubled rope_mats for fused RoPE
        cos, sin = rope_mats
        cos_interleaved = ttnn.to_memory_config(cos, ttnn.DRAM_MEMORY_CONFIG)
        sin_interleaved = ttnn.to_memory_config(sin, ttnn.DRAM_MEMORY_CONFIG)
        doubled_cos = ttnn.repeat(cos_interleaved, ttnn.Shape([1, 2, 1, 1]))
        doubled_sin = ttnn.repeat(sin_interleaved, ttnn.Shape([1, 2, 1, 1]))
        cos_interleaved.deallocate(True)
        sin_interleaved.deallocate(True)

        grid_size = mesh_device.compute_with_storage_grid_size()
        doubled_batch_grid = ttnn.num_cores_to_corerangeset(batch_size * 2, grid_size, row_wise=True)
        doubled_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=doubled_batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        doubled_cos = ttnn.to_memory_config(doubled_cos, doubled_mem_config)
        doubled_sin = ttnn.to_memory_config(doubled_sin, doubled_mem_config)

        # Move Q and K to disjoint core regions for fused RoPE
        row_size = 8
        k_start_core = ttnn.CoreCoord(batch_size % row_size, batch_size // row_size)
        q_core_grid = ttnn.CoreRangeSet({num_to_corerange(batch_size)})
        k_core_grid = ttnn.CoreRangeSet({num_to_corerange(batch_size, start_core=k_start_core)})

        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(num_local_heads), head_dim),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(num_local_kv_heads), head_dim),
            core_grid=k_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tt_q = ttnn.to_memory_config(tt_q, q_mem_config)
        tt_k = ttnn.to_memory_config(tt_k, k_mem_config)

        # Fused RoPE applies rotation to both Q and K in a single kernel
        tt_q, tt_k = ttnn.experimental.rotary_embedding_llama_fused_qk(
            tt_q, tt_k, doubled_cos, doubled_sin, fused_transformation_mat
        )
        doubled_cos.deallocate(True)
        doubled_sin.deallocate(True)

        # Fused KV cache update (K and V already on disjoint cores)
        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_fused_update_cache(
            k_cache, tt_k, v_cache, tt_v, update_idxs_tensor=position_idx, page_table=page_table
        )
    else:
        # Fallback to separate RoPE calls
        tt_q_orig, tt_k_orig = tt_q, tt_k
        tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
        tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)
        tt_q_orig.deallocate(True)
        tt_k_orig.deallocate(True)

        # Separate KV cache updates
        k_cache, v_cache = kv_cache
        tt_k = ttnn.to_memory_config(tt_k, kv_mem_cfg)
        tt_v = ttnn.to_memory_config(tt_v, kv_mem_cfg)
        ttnn.experimental.paged_update_cache(k_cache, tt_k, update_idxs_tensor=position_idx, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, tt_v, update_idxs_tensor=position_idx, page_table=page_table)

    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # Scaled dot-product attention
    grid_size = mesh_device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid_size, row_wise=True)
    padded_heads = ((num_local_heads + 31) // 32) * 32

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
    else:
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
            memory_config=height_sharded_mem_config,
        )
    tt_q.deallocate(True)

    # Concat heads and apply output projection
    tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(tt_sdpa_tensor, num_heads=num_local_heads)
    tt_sdpa_tensor.deallocate(True)

    tt_out = ttnn.linear(
        tt_sdpa_out, weights.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )
    tt_sdpa_out.deallocate(True)

    tt_out = ttnn.add(tt_out, weights.o_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_out = ttnn.typecast(tt_out, ttnn.bfloat8_b)

    # Reshape for CCL
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    padded_hidden = padded_local_hidden * mesh_config.tp if mesh_config.tp > 1 else hidden_size

    tt_out = ttnn.reshape(tt_out, (1, 1, batch_size, padded_hidden), (1, 1, 32, padded_hidden))

    # Tensor parallel allreduce
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, batch_size, seq_len, hidden_size)

    return tt_out
