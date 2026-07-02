# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .config import AttentionConfig, ProgramConfig
from .operations import (
    apply_allgather_and_slice,
    apply_allreduce,
    apply_output_projection,
    apply_output_projection_fused_rs,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    is_shape_fused_mm_rs_supported,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights


def prefill_forward(
    hidden_states,
    rope_mats,
    weights: AttentionWeights,
    kv_cache,
    config: AttentionConfig,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    transformation_mat,
    position_idx,
    page_table,
    ccl_manager,
    user_id=0,
    batch_size=1,
    persistent_k=None,
    persistent_v=None,
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        rope_mats: Tuple of (cos, sin) matrices for RoPE
        weights: Attention weights
        kv_cache: KV cache [k_cache, v_cache]
        config: Attention configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        program_config: Model-specific program configs
        transformation_mat: Transformation matrix for RoPE
        position_idx: Position indices (unused in prefill)
        page_table: Page table for paged attention (optional)
        ccl_manager: Communication manager

    Returns:
        Attention output [batch, seq_len, hidden_size]
    """
    activation_dtype = ttnn.bfloat16
    total_seq_len = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    seq_len = total_seq_len // batch_size  # Per-user sequence length
    if seq_len > 32 * 1024:
        activation_dtype = ttnn.bfloat8_b
    else:
        activation_dtype = ttnn.bfloat16

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # QKV projection
    xqkv_fused = apply_qkv_projection(hidden_states, weights)
    hidden_states.deallocate(True)  # Free input activations after projection

    # Reshape for batch: [1, 1, B*S, QKV] -> [B, 1, S, QKV]
    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Apply RoPE (use per-user seq_len positions)
    if batch_size > 1:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats_sliced, transformation_mat, is_decode_mode=False)
    tt_k = apply_rope(tt_k, rope_mats_sliced, transformation_mat, is_decode_mode=False)
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Fill KV cache. When kv_cache is None the SDPA below still operates on the
    # freshly-computed tt_k / tt_v directly, so activations are correct; only the
    # persistent cache is skipped.  Activation-accuracy tests use this path to
    # avoid allocating cache for all 94 layers.
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        tt_k_pre_cast = tt_k
        tt_v_pre_cast = tt_v
        tt_k = ttnn.typecast(tt_k, k_cache.dtype)
        tt_v = ttnn.typecast(tt_v, v_cache.dtype)
        tt_k_pre_cast.deallocate(True)
        tt_v_pre_cast.deallocate(True)

        if page_table is not None:
            block_size = k_cache.shape[2]
            page_len = page_table.shape[-1] * block_size
            if batch_size > 1:
                # Per-user paged cache fill. The flattened approach (reshape batch into seq
                # + flattened page_table) produces wrong cache for users beyond the first —
                # paged_fill_cache doesn't correctly handle positions beyond the original
                # page_table's block count. Use per-user calls with batch_idx=0 instead.
                for b in range(batch_size):
                    k_b = tt_k[b : b + 1, :, :, :]
                    v_b = tt_v[b : b + 1, :, :, :]
                    pt_b = page_table[b : b + 1, :]
                    k_b_fill = k_b[:, :, :page_len, :] if page_len < k_b.shape[2] else k_b
                    v_b_fill = v_b[:, :, :page_len, :] if page_len < v_b.shape[2] else v_b
                    ttnn.experimental.paged_fill_cache(k_cache, k_b_fill, pt_b, batch_idx=0)
                    ttnn.experimental.paged_fill_cache(v_cache, v_b_fill, pt_b, batch_idx=0)
            else:
                tt_k_sliced = tt_k[:, :, :page_len, :] if page_len < tt_k.shape[2] else tt_k
                tt_v_sliced = tt_v[:, :, :page_len, :] if page_len < tt_v.shape[2] else tt_v
                ttnn.experimental.paged_fill_cache(k_cache, tt_k_sliced, page_table, batch_idx=user_id)
                ttnn.experimental.paged_fill_cache(v_cache, tt_v_sliced, page_table, batch_idx=user_id)
                if page_len < tt_k.shape[2]:
                    tt_k_sliced.deallocate(True)
                if page_len < tt_v.shape[2]:
                    tt_v_sliced.deallocate(True)

        else:
            # Non-paged attention
            if batch_size > 1:
                for b in range(batch_size):
                    k_b = ttnn.slice(tt_k, (b, 0, 0, 0), (b + 1, tt_k.shape[1], tt_k.shape[2], tt_k.shape[3]))
                    v_b = ttnn.slice(tt_v, (b, 0, 0, 0), (b + 1, tt_v.shape[1], tt_v.shape[2], tt_v.shape[3]))
                    ttnn.fill_cache(k_cache, k_b, batch_idx=b)
                    ttnn.fill_cache(v_cache, v_b, batch_idx=b)
                    k_b.deallocate(True)
                    v_b.deallocate(True)
            else:
                ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
                ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

    # Scaled dot-product attention — three paths based on SP factor and sliding window
    sp_factor = mesh_config.prefill.sp
    sdpa_program_config = program_config.get_prefill_sdpa_config(mesh_device, seq_len)
    sdpa_compute_config = program_config.get_compute_kernel_config()

    if sp_factor > 1 and config.sliding_window is not None:
        # Ring-shift the last actual_pad tokens of K and V from each device's
        # predecessor and prepend them. Device 0 receives zeros (Linear topology).
        # Clamp to seq_len: neighbor_pad_async requires padding_left <= tensor_dim,
        # and when seq_len < window the total sequence fits within one window anyway.
        actual_pad = min(config.sliding_window, seq_len)

        # neighbor_pad_async requires ROW_MAJOR; convert in, convert back out.
        tt_k_tile = tt_k
        tt_k = ttnn.to_layout(tt_k_tile, ttnn.ROW_MAJOR_LAYOUT)
        tt_k_tile.deallocate(True)

        neighbor_sem, barrier_sem = ccl_manager.get_neighbor_pad_semaphores()
        tt_k_rm = tt_k
        tt_k = ttnn.experimental.neighbor_pad_async(
            tt_k_rm,
            [2],
            [actual_pad],
            [0],
            "zeros",
            [mesh_config.sp_axis],
            [neighbor_sem],
            [barrier_sem],
            num_links=[1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        tt_k_rm.deallocate(True)
        tt_k_padded = tt_k
        tt_k = ttnn.to_layout(tt_k_padded, ttnn.TILE_LAYOUT)
        tt_k_padded.deallocate(True)

        tt_v_tile = tt_v
        tt_v = ttnn.to_layout(tt_v_tile, ttnn.ROW_MAJOR_LAYOUT)
        tt_v_tile.deallocate(True)

        neighbor_sem, barrier_sem = ccl_manager.get_neighbor_pad_semaphores()
        tt_v_rm = tt_v
        tt_v = ttnn.experimental.neighbor_pad_async(
            tt_v_rm,
            [2],
            [actual_pad],
            [0],
            "zeros",
            [mesh_config.sp_axis],
            [neighbor_sem],
            [barrier_sem],
            num_links=[1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        tt_v_rm.deallocate(True)
        tt_v_padded = tt_v
        tt_v = ttnn.to_layout(tt_v_padded, ttnn.TILE_LAYOUT)
        tt_v_padded.deallocate(True)

        # SDPA requires Sq == Sk for is_causal=True. Prepend actual_pad zero rows to Q
        # to match the extended K/V length. ttnn.pad doesn't support front-padding in
        # tile layout, so simulate it with concat of a zeros tensor instead.
        tt_q_orig = tt_q
        zeros_q = ttnn.zeros(
            [tt_q_orig.shape[0], tt_q_orig.shape[1], actual_pad, tt_q_orig.shape[3]],
            dtype=tt_q_orig.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q = ttnn.concat([zeros_q, tt_q_orig], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        zeros_q.deallocate(True)
        tt_q_orig.deallocate(True)

        tt_sdpa_out_padded = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            sliding_window_size=config.sliding_window,
            program_config=program_config.get_prefill_sdpa_config(mesh_device, actual_pad + seq_len),
            compute_kernel_config=sdpa_compute_config,
            attention_sink=weights.sinks,
        )
        tt_q.deallocate(True)
        tt_k.deallocate(True)
        tt_v.deallocate(True)

        # Drop the actual_pad fake output rows, keep only the real seq_len rows.
        b, nh = tt_sdpa_out_padded.shape[0], tt_sdpa_out_padded.shape[1]
        dh = tt_sdpa_out_padded.shape[3]
        tt_sdpa_out = ttnn.slice(
            tt_sdpa_out_padded,
            (0, 0, actual_pad, 0),
            (b, nh, actual_pad + seq_len, dh),
        )
        tt_sdpa_out_padded.deallocate(True)
    elif sp_factor > 1:
        # Ring attention: fuses cross-device K/V all-gather inside the kernel so
        # every token attends to the full causal context across all SP rows.
        seq_total = seq_len * sp_factor
        tt_sdpa_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            persistent_output_buffer_k=persistent_k,
            persistent_output_buffer_v=persistent_v,
            joint_strategy="rear",
            logical_n=seq_total,
            program_config=sdpa_program_config,
            compute_kernel_config=sdpa_compute_config,
            dim=2,
            multi_device_global_semaphore=ccl_manager.ring_attn_semaphores,
            num_links=ccl_manager.num_links,
            cluster_axis=mesh_config.sp_axis,
            mesh_device=mesh_device,
            topology=ccl_manager.topology,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=ccl_manager.ring_attn_ccl_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=config.scaling,
        )
        tt_q.deallocate(True)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    else:
        # SP=1: plain SDPA, no cross-device communication needed
        tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            sliding_window_size=config.sliding_window,
            program_config=sdpa_program_config,
            compute_kernel_config=sdpa_compute_config,
            attention_sink=weights.sinks,
        )
        tt_q.deallocate(True)
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    # Concat heads and apply output projection
    tt_sdpa_out_pre_concat = tt_sdpa_out
    tt_sdpa_out = concat_heads(tt_sdpa_out, is_decode_mode=False)
    tt_sdpa_out_pre_concat.deallocate(True)

    # Flatten back for output projection: [B, 1, S, H] -> [1, 1, B*S, H]
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    # Output projection + tensor-parallel allreduce.
    # When TP > 1 we use the fused matmul + reduce-scatter op; the trailing
    # all-gather + padding slice stay as separate ops. See
    # apply_output_projection_fused_rs for the per-shape tuned configs.
    if mesh_config.tp > 1 and is_shape_fused_mm_rs_supported(tt_sdpa_out):
        rs_out = apply_output_projection_fused_rs(tt_sdpa_out, weights, mesh_config, ccl_manager)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size)
    else:
        tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)
    return tt_out_result
