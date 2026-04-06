# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared attention operations for Gemma4.

Uses HF-style ttnn.experimental.rotary_embedding (not the llama variant).
No Meta-format weight conversion needed. No transformation matrices needed.

Handles:
- Per-head RMSNorm (q_norm, k_norm, v_norm) via reshape trick
- Partial RoPE for global layers (split, rotate, concat)
- K=V tying (fused Q+K+K weight, standard nlp_create_qkv_heads split)
- No bias on any projection
- scaling=1.0 (no 1/sqrt(d_k))
"""

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """Fused QKV matmul (no bias for Gemma4)."""
    return ttnn.linear(hidden_states, weights.wqkv)


def split_qkv_heads_decode(xqkv_fused, config, is_global: bool, tp: int = 1):
    """
    Split fused QKV into separate head tensors for decode mode.
    When TP > 1, uses local head counts (global / tp).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = config.num_key_value_heads // tp
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )


def split_qkv_heads_prefill(xqkv_fused, config, is_global: bool, tp: int = 1):
    """
    Split fused QKV into separate head tensors for prefill mode.
    When TP > 1, uses local head counts (global / tp).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = config.num_key_value_heads // tp
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_per_head_norm(tensor, weight, eps, with_scale=True):
    """
    Apply RMSNorm per-head on the head_dim dimension.

    Input: [1, num_heads, S, head_dim]
    Process: reshape to [1, 1, num_heads*S, head_dim] -> rms_norm -> reshape back
    """
    orig_shape = tensor.shape
    num_heads = orig_shape[1]
    seq_or_batch = orig_shape[2]
    head_dim = orig_shape[3]

    flat = ttnn.reshape(tensor, (1, 1, num_heads * seq_or_batch, head_dim))
    if with_scale and weight is not None:
        normed = ttnn.rms_norm(flat, weight=weight, epsilon=eps)
    else:
        normed = ttnn.rms_norm(flat, epsilon=eps)

    return ttnn.reshape(normed, orig_shape)


def apply_rope(tensor, cos_cache, sin_cache, token_index=None):
    """
    Apply HF-style rotary position embedding.

    Uses ttnn.experimental.rotary_embedding (not llama variant).
    No transformation matrix needed. Position slicing is internal.

    Args:
        tensor: [1, heads, S, head_dim] (prefill) or [1, batch, heads, head_dim] (decode)
        cos_cache: [1, 1, max_seq_len, head_dim] - full cos cache
        sin_cache: [1, 1, max_seq_len, head_dim] - full sin cache
        token_index: int or None. If int (decode), slices into cache at that position.
                     If None (prefill), applies to full sequence.

    Note: rotary_embedding pads dim 2 to TILE_HEIGHT (32) in decode mode.
    We reshape+slice to restore the original logical shape, following the
    tt_transformers _hf_rope_decode pattern.
    """
    orig_shape = tensor.shape
    result = ttnn.experimental.rotary_embedding(tensor, cos_cache, sin_cache, token_index)

    # In decode mode (token_index provided), dim 2 gets padded to 32.
    # Reshape to indicate logical vs padded size, then slice back.
    if token_index is not None and result.shape[2] != orig_shape[2]:
        result = ttnn.reshape(
            result,
            (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]),
            (orig_shape[0], orig_shape[1], 32, orig_shape[3]),
        )
        result = result[:, :, : orig_shape[2]]

    return result


def concat_heads(tensor, is_decode_mode: bool):
    """Concatenate attention heads back to hidden dimension."""
    if is_decode_mode:
        tensor = ttnn.transpose(tensor, 1, 2)
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights):
    """Apply output projection (no bias for Gemma4)."""
    out = ttnn.linear(tensor, weights.o_proj)
    tensor.deallocate(True)
    return out


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """Apply tensor-parallel allreduce if TP > 1."""
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    num_links = ccl_manager.num_links if ccl_manager else 1
    tensor_allreduced = ttnn.all_reduce(
        tensor,
        num_links=num_links,
        topology=ttnn.Topology.Ring,
        cluster_axis=mesh_config.tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tensor.deallocate(True)
    tensor = tensor_allreduced

    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    if padded_local_hidden != local_hidden:
        shape = tensor.shape
        tensor_sliced = ttnn.slice(tensor, [0, 0, 0, 0], [shape[0], shape[1], shape[2], hidden_size], [1, 1, 1, 1])
        tensor.deallocate(True)
        tensor = tensor_sliced

    return tensor
