# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .config import AttentionConfig, ProgramConfig
from .operations import (
    apply_allreduce,
    apply_output_projection,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_decode,
)
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
    activation_dtype,
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
        activation_dtype: Data type for activations

    Returns:
        Attention output [batch, 1, hidden_size]
    """
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Validate decode mode
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

    # QKV projection
    xqkv_fused = apply_qkv_projection(hidden_states, weights)

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_decode(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Apply RoPE
    tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
    tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)

    # Convert to DRAM for KV cache update
    tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
    tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    tt_q.deallocate(True)

    # Concat heads and apply output projection
    tt_sdpa_out = concat_heads(tt_sdpa_tensor, is_decode_mode=True)
    tt_sdpa_tensor.deallocate(True)

    tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
    tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, hidden_size))

    # Tensor parallel allreduce
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, batch_size, seq_len, hidden_size)

    return tt_out
