# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os

import ttnn

from .config import AttentionConfig, ProgramConfig
from .operations import apply_allreduce, apply_rope
from .weights import AttentionWeights

# Debug mode: Set DEBUG_DECODE_ATTENTION=1 to enable detailed logging
DEBUG_DECODE = os.getenv("DEBUG_DECODE_ATTENTION", "0") == "1"


def _debug_tensor_stats(name, tensor, user_idx=None):
    """Log tensor statistics for debugging."""
    if not DEBUG_DECODE:
        return

    import torch
    from loguru import logger

    # Get first device tensor for inspection
    device_tensors = ttnn.get_device_tensors(tensor)
    t = ttnn.to_torch(device_tensors[0]).float()

    stats = f"shape={list(t.shape)}, min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}, std={t.std().item():.4f}"

    # Check for NaN/Inf
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        stats += f" ⚠️ NaN={has_nan}, Inf={has_inf}"

    # If user_idx specified, show stats for that specific user
    if user_idx is not None and len(t.shape) >= 3:
        try:
            user_t = t[..., user_idx, :]
            user_stats = f"user_{user_idx}: min={user_t.min().item():.4f}, max={user_t.max().item():.4f}"
            stats += f" | {user_stats}"
        except Exception:
            pass

    logger.debug(f"[DECODE_ATTN] {name}: {stats}")


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

    # Debug: Log input tensor stats
    _debug_tensor_stats("hidden_states (input)", hidden_states)

    # QKV projection
    xqkv_fused = ttnn.matmul(
        hidden_states, weights.wqkv, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )
    xqkv_fused = ttnn.add(xqkv_fused, weights.wqkv_bias, output_tensor=xqkv_fused)
    _debug_tensor_stats("xqkv_fused (after QKV proj)", xqkv_fused)

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
    _debug_tensor_stats("tt_q (before RoPE)", tt_q)
    _debug_tensor_stats("tt_k (before RoPE)", tt_k)
    _debug_tensor_stats("tt_v", tt_v)

    xqkv_fused.deallocate(True)

    # Apply RoPE
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
    tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)
    _debug_tensor_stats("tt_q (after RoPE)", tt_q)
    _debug_tensor_stats("tt_k (after RoPE)", tt_k)
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Update KV cache
    k_cache, v_cache = kv_cache
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
    grid_size = mesh_device.compute_with_storage_grid_size()
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

    # Debug: Log SDPA output
    _debug_tensor_stats("tt_sdpa_tensor (SDPA output)", tt_sdpa_tensor)

    # Concat heads and apply output projection

    tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(tt_sdpa_tensor, num_heads=num_local_heads)
    tt_sdpa_tensor.deallocate(True)
    _debug_tensor_stats("tt_sdpa_out (after concat heads)", tt_sdpa_out)

    tt_out = ttnn.linear(
        tt_sdpa_out, weights.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )
    _debug_tensor_stats("tt_out (after o_proj)", tt_out)

    tt_sdpa_out.deallocate(True)
    tt_out = ttnn.add(tt_out, weights.o_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    # Keep bfloat16 for allreduce precision (was bfloat8_b causing precision loss)
    tt_out = ttnn.reshape(
        tt_out,
        (1, 1, batch_size, hidden_size),
        (1, 1, 32, hidden_size),
    )
    _debug_tensor_stats("tt_out (final, before allreduce)", tt_out)
    # tt_out = ttnn.unsqueeze(tt_out, 0)

    # Tensor parallel allreduce
    # TODO: This will need to be a reduce scatter so outputs are [1, 1, global_batch//num_rows, hidden_size//num_columns
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, batch_size, seq_len, hidden_size)

    return tt_out
