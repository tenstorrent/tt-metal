# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Llama 3.1 8B attention primitive ops: no bias on QKV/O projections, RoPE is FULL rotary
(rotary_dim == head_dim == 128, Llama3 frequency scaling), and there is no QK-norm.
"""

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """
    Apply QKV projection (no bias).

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        weights: Attention weights container

    Returns:
        Fused QKV tensor [batch, seq_len, total_qkv_dim]
    """
    xqkv_fused = ttnn.linear(hidden_states, weights.wqkv, dtype=ttnn.bfloat16)
    return xqkv_fused


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split fused QKV into separate head tensors for prefill mode (GQA: num_heads Q, num_kv_heads K/V).

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of (local) Q heads
        num_kv_heads: Number of (local) K/V heads

    Returns:
        Tuple (Q, K, V) with shapes [1, num_heads, seq_len, head_dim] / [1, num_kv_heads, seq_len, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(
    tensor, rope_mats, transformation_mat, is_decode_mode: bool = False, kv_actual_global=None, cluster_axis=None
):
    """
    Apply rotary position embedding (RoPE) — FULL rotary for Llama 3.1 8B (rotary_dim == head_dim, 128).

    The Llama3 frequency scaling is baked into the cos/sin matrices (``rope_mats``) at build time,
    so this op is a plain full rotation of the whole head.

    Two inner ops:
      * default (``kv_actual_global`` is None): ``rotary_embedding_llama`` with a per-chunk cos/sin
        already sliced to this chunk's positions.
      * indexed (``kv_actual_global`` set): ``rotary_embedding_indexed`` — ``rope_mats`` carry the
        WHOLE-cache, block-cyclic-reordered, SP-sharded cos/sin.
    """
    if kv_actual_global is not None:
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            tensor,
            rope_mats[0],
            rope_mats[1],
            transformation_mat,
            kv_actual_global=kv_actual_global,
            cluster_axis=cluster_axis,
        )
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
    )


def concat_heads(tensor):
    """
    Concatenate attention heads back to the hidden dimension (prefill).

    Args:
        tensor: Attention output tensor with separate heads [1, n_heads, seq_len, head_dim]

    Returns:
        Tensor with concatenated heads [1, 1, seq_len, hidden_size]
    """
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype):
    """
    Apply output projection (no bias).

    Args:
        tensor: Attention output tensor [1, 1, seq_len, local_hidden]
        weights: Attention weights container
        activation_dtype: Target dtype for output

    Returns:
        Output tensor after projection
    """
    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
    out = ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor.deallocate(True)
    return out


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """
    Apply tensor-parallel allreduce if needed (TP > 1).

    For Llama with TP=4, hidden=4096, local_hidden=1024 is tile-aligned (1024/32=32), so no
    padding removal is needed after the allreduce.

    Args:
        tensor: Input tensor
        mesh_config: Mesh configuration
        ccl_manager: Communication manager
        hidden_size: Hidden size

    Returns:
        Tensor after allreduce (if TP > 1) or the original tensor
    """
    if mesh_config.tp > 1:
        tensor_allreduced = mesh_config.allreduce(tensor, ccl_manager, pad_size=0, axis=mesh_config.tp_axis)
        tensor = tensor_allreduced
    return tensor
