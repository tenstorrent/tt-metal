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
    user_id=0,
    batch_size=1,
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

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # QKV projection
    xqkv_fused = apply_qkv_projection(hidden_states, weights)

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

    # Fill KV cache
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
            # Flatten batch into seq dim, heads into last dim — single fill call, no per-user loop.
            # Paged cache just maps sequence positions to physical pages.
            k_fill = ttnn.reshape(tt_k, [1, 1, total_seq_len, -1])
            v_fill = ttnn.reshape(tt_v, [1, 1, total_seq_len, -1])
            page_table_flat = ttnn.reshape(page_table, [1, -1])
            ttnn.experimental.paged_fill_cache(k_cache, k_fill, page_table_flat, batch_idx=0)
            ttnn.experimental.paged_fill_cache(v_cache, v_fill, page_table_flat, batch_idx=0)
            k_fill.deallocate(True)
            v_fill.deallocate(True)
        else:
            tt_k_sliced = tt_k[:, :, :page_len, :] if page_len < tt_k.shape[2] else tt_k
            tt_v_sliced = tt_v[:, :, :page_len, :] if page_len < tt_v.shape[2] else tt_v
            ttnn.experimental.paged_fill_cache(k_cache, tt_k_sliced, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(v_cache, tt_v_sliced, page_table, batch_idx=user_id)

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

    tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
    # Note: apply_output_projection already deallocates its input tensor internally

    # Tensor parallel allreduce
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)

    return tt_out
