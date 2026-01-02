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
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    # QKV projection
    xqkv_fused = apply_qkv_projection(hidden_states, weights)

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Apply RoPE
    tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=False)
    tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=False)

    # Fill KV cache
    k_cache, v_cache = kv_cache
    tt_k = ttnn.typecast(tt_k, k_cache.dtype)
    tt_v = ttnn.typecast(tt_v, v_cache.dtype)

    if page_table is not None:
        # Paged attention: handle potential padding
        block_size = k_cache.shape[2]
        page_len = page_table.shape[1] * block_size
        tt_k_sliced = tt_k[:, :, :page_len, :] if page_len < tt_k.shape[2] else tt_k
        tt_v_sliced = tt_v[:, :, :page_len, :] if page_len < tt_v.shape[2] else tt_v

        ttnn.experimental.paged_fill_cache(k_cache, tt_k_sliced, page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(v_cache, tt_v_sliced, page_table, batch_idx=0)
    else:
        # Non-paged attention
        ttnn.fill_cache(k_cache, tt_k, batch_idx=0)
        ttnn.fill_cache(v_cache, tt_v, batch_idx=0)

    # Scaled dot-product attention
    tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        sliding_window_size=config.sliding_window,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, seq_len),
        compute_kernel_config=program_config.get_compute_kernel_config(),
        attention_sink=weights.sinks,
    )

    # Concat heads and apply output projection
    tt_sdpa_out = concat_heads(tt_sdpa_out, is_decode_mode=False)

    tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
    tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, hidden_size))

    # Tensor parallel allreduce
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, batch_size, seq_len, hidden_size)

    return tt_out
