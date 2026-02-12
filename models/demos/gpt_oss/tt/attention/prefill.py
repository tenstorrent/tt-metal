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
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).
    Supports both single-user (batch_size=1) and batched (batch_size>1) prefill.

    Args:
        hidden_states: Input tensor [1, batch_size, seq_len, hidden_size]
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
        user_id: User ID (int for single user, tensor for batched)

    Returns:
        Attention output [1, batch_size, seq_len, hidden_size]
    """
    activation_dtype = ttnn.bfloat16
    _, batch_size, seq_len, hidden_size = hidden_states.shape

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # For batched prefill, flatten batch*seq for matmuls
    if batch_size > 1:
        hidden_states = ttnn.reshape(hidden_states, [1, 1, batch_size * seq_len, hidden_size])

    # QKV projection
    xqkv_fused = apply_qkv_projection(hidden_states, weights)

    # For batched prefill, reshape back to [B, 1, S, qkv_size] for head splitting
    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Apply RoPE
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=False)
    tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=False)
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
        use_batched_fill = batch_size > 1 or not isinstance(user_id, int)
        if use_batched_fill:
            # For batched prefill: flatten K/V to [1, 1, batch*seq_len, heads*head_dim]
            # This matches the llama TG batched prefill pattern
            k_fill = ttnn.reshape(tt_k, [1, 1, batch_size * seq_len, -1])
            v_fill = ttnn.reshape(tt_v, [1, 1, batch_size * seq_len, -1])
            # Ensure user_id is a tensor for batched paged fill
            if isinstance(user_id, int):
                import torch

                user_id = ttnn.from_torch(
                    torch.tensor([user_id], dtype=torch.int32),
                    device=mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            ttnn.experimental.paged_fill_cache(k_cache, k_fill, page_table, batch_idx_tensor=user_id)
            ttnn.experimental.paged_fill_cache(v_cache, v_fill, page_table, batch_idx_tensor=user_id)
        else:
            block_size = k_cache.shape[2]
            page_len = page_table.shape[1] * block_size
            tt_k_sliced = tt_k[:, :, :page_len, :] if page_len < tt_k.shape[2] else tt_k
            tt_v_sliced = tt_v[:, :, :page_len, :] if page_len < tt_v.shape[2] else tt_v
            ttnn.experimental.paged_fill_cache(k_cache, tt_k_sliced, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(v_cache, tt_v_sliced, page_table, batch_idx=user_id)
    else:
        # Non-paged attention
        ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
        ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

    # Scaled dot-product attention
    sdpa_seq_len = seq_len if batch_size == 1 else (seq_len if seq_len == 128 else batch_size * seq_len)
    tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        is_causal=True,
        sliding_window_size=config.sliding_window,
        program_config=program_config.get_prefill_sdpa_config(mesh_device, sdpa_seq_len),
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

    # For batched prefill, flatten batch*seq for output matmul
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, batch_size * seq_len, -1])

    tt_out = apply_output_projection(tt_sdpa_out, weights, activation_dtype)
    # Note: apply_output_projection already deallocates its input tensor internally

    # Reshape back to [1, B, S, H] for batched prefill
    if batch_size > 1:
        tt_out = ttnn.reshape(tt_out, [1, batch_size, seq_len, -1])

    # Tensor parallel allreduce
    tt_out = apply_allreduce(tt_out, mesh_config, ccl_manager, batch_size, seq_len, hidden_size)

    return tt_out
