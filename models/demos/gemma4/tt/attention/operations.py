# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
from models.demos.gemma4.tt.ccl import ccl_allreduce

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """Fused QKV matmul (no bias for Gemma4)."""
    return ttnn.linear(hidden_states, weights.wqkv)


def split_qkv_heads_decode(xqkv_fused, config, is_global: bool, tp: int = 1, kv_replicated: bool = False):
    """
    Split fused QKV into separate head tensors for decode mode.
    When TP > 1, uses local head counts (global / tp).
    When kv_replicated (num_kv_heads < TP), each device has 1 KV head (GQA-assigned).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    # Workaround for Blackhole bug in nlp_create_qkv_heads_decode interleaved
    # reader kernel: with DRAM input the kernel zeros odd-indexed Q rows due
    # to a NoC DRAM-read alignment-match violation (see tt-metal #16667). Move
    # the fused QKV from DRAM to L1 before the split — the L1 path uses a
    # different code path that's not affected. No-op on Wormhole (correctness
    # preserved; small extra L1 copy in exchange for arch-portable behavior).
    if xqkv_fused.memory_config().buffer_type == ttnn.BufferType.DRAM:
        xqkv_fused = ttnn.to_memory_config(xqkv_fused, ttnn.L1_MEMORY_CONFIG)
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )


def split_qkv_heads_prefill(xqkv_fused, config, is_global: bool, tp: int = 1, kv_replicated: bool = False):
    """
    Split fused QKV into separate head tensors for prefill mode.
    When TP > 1, uses local head counts (global / tp).
    When kv_replicated (num_kv_heads < TP), each device has 1 KV head (GQA-assigned).
    """
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
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
    return ccl_allreduce(tensor, mesh_config, ccl_manager)
