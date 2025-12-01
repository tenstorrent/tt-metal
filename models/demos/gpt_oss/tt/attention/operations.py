# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights, mesh_config, ccl_manager):
    """
    Apply QKV projection with 2D sharding and all-reduce.

    With 2D weights (input across rows, output across columns):
    - Input: row-sharded (4), column-replicated (8)
    - Matmul produces partial results
    - All-reduce along rows to get full results
    - Output: row-replicated (4), column-sharded (8)

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        weights: Attention weights container (2D sharded)
        mesh_config: Mesh configuration for communication
        ccl_manager: Communication manager

    Returns:
        Fused QKV tensor [batch, seq_len, total_qkv_dim] column-sharded
    """
    # Matmul with 2D sharded weights produces partial results
    xqkv_fused = ttnn.matmul(hidden_states, weights.wqkv, dtype=ttnn.bfloat16)
    xqkv_fused = ttnn.add(xqkv_fused, weights.wqkv_bias, output_tensor=xqkv_fused)

    # All-reduce along ROWS (cluster_axis=0) to sum partial results
    # After this: row-replicated, column-sharded (ready for head splitting)
    num_rows = mesh_config.mesh_shape[0]
    if num_rows > 1:
        out = ttnn.reduce_scatter(
            xqkv_fused,
            dim=3,
            # num_links=1,
            # topology=ttnn.Topology.Ring,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.all_gather(
            out,
            dim=3,
            # num_links=1,
            # topology=ttnn.Topology.Ring,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out
    return xqkv_fused


def split_qkv_heads_decode(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split QKV into separate head tensors for decode mode.

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of Q heads
        num_kv_heads: Number of K/V heads

    Returns:
        Tuple (Q, K, V) with shapes [1, num_heads, 1, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split QKV into separate head tensors for prefill mode.

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of Q heads
        num_kv_heads: Number of K/V heads

    Returns:
        Tuple (Q, K, V) with shapes [1, num_heads, seq_len, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(tensor, rope_mats, transformation_mat, is_decode_mode: bool):
    """
    Apply rotary position embedding (RoPE).

    Args:
        tensor: Input tensor (Q or K)
        rope_mats: Tuple of (cos, sin) matrices
        transformation_mat: Transformation matrix for the mode
        is_decode_mode: Whether in decode mode

    Returns:
        Tensor with RoPE applied
    """
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
    )


def concat_heads(tensor, is_decode_mode: bool):
    """
    Concatenate attention heads back to hidden dimension.

    Args:
        tensor: Attention output tensor with separate heads
        is_decode_mode: Whether in decode mode

    Returns:
        Tensor with concatenated heads [batch, seq_len, hidden_size]
    """
    if is_decode_mode:
        tensor = ttnn.transpose(tensor, 1, 2)
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype, mesh_config, ccl_manager):
    """
    Apply output projection with 2D sharding and all-reduce.

    With 2D weights (input across columns, output across rows):
    - Input: column-sharded (8), row-replicated (4)
    - Matmul produces partial results
    - All-reduce along columns to get full results
    - Output: column-replicated (8), row-sharded (4)

    Args:
        tensor: Attention output tensor [batch, seq_len, hidden_size] column-sharded
        weights: Attention weights container (2D sharded)
        activation_dtype: Target dtype for output
        mesh_config: Mesh configuration for communication
        ccl_manager: Communication manager

    Returns:
        Output tensor [batch, seq_len, hidden_size] row-sharded
    """
    # tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
    # Matmul with 2D sharded weights produces partial results
    out = ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor.deallocate(True)
    out = ttnn.add(out, weights.o_proj_bias, output_tensor=out)

    # All-reduce along COLUMNS (cluster_axis=1) to sum partial results
    # After this: column-replicated, row-sharded (ready for residual add)
    if mesh_config.tp > 1:
        out = ttnn.reduce_scatter(
            out,
            dim=3,
            # num_links=1,
            # topology=ttnn.Topology.Ring,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.all_gather(
            out,
            dim=3,
            # num_links=1,
            # topology=ttnn.Topology.Ring,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return out
