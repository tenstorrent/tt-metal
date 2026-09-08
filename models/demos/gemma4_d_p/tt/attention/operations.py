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

import os

import ttnn
from models.demos.gemma4_d_p.tt.ccl import ccl_allreduce

from .weights import AttentionWeights


def prefill_short_lived_memcfg() -> ttnn.MemoryConfig:
    """Choose DRAM or optional L1 storage for short-lived attention activations."""
    if os.environ.get("GEMMA4_PREFILL_L1_ACT", "0").lower() in ("1", "true", "yes"):
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def apply_qkv_projection(hidden_states, weights: AttentionWeights, memory_config=None, kv_tied: bool = False):
    """Project to QKV, or QK when kv_tied selects the narrow tied weight."""
    w_tensor = weights.wqk if kv_tied else weights.wqkv
    return ttnn.linear(hidden_states, w_tensor, memory_config=memory_config)


def qkv_projection_is_tied(weights: AttentionWeights, kv_tied: bool = False) -> bool:
    """Whether the requested narrow Q+K projection is available."""
    return kv_tied and weights.wqk is not None


def split_qkv_heads_prefill(
    xqkv_fused,
    config,
    is_global: bool,
    tp: int = 1,
    kv_replicated: bool = False,
    kv_tied: bool = False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """Split the local projection into Q, K and V head tensors.

    With kv_tied, K and V read the same projection columns but return separate tensors.
    memory_config selects storage for the resulting activations."""
    num_local_heads = config.num_attention_heads // tp
    num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=memory_config,
        kv_tied=kv_tied,
    )


def apply_per_head_norm(tensor, weight, eps, with_scale=True, memory_config=None):
    """Normalize each token and head independently along head_dim."""
    orig_shape = tensor.shape
    head_dim = orig_shape[-1]
    if len(orig_shape) == 4 and orig_shape[0] > 1:
        batch, num_heads, seq_len, _ = orig_shape
        flat = ttnn.reshape(tensor, (1, 1, batch * num_heads * seq_len, head_dim))
    else:
        num_heads = orig_shape[1]
        seq_or_batch = orig_shape[2]
        flat = ttnn.reshape(tensor, (1, 1, num_heads * seq_or_batch, head_dim))
    if with_scale and weight is not None:
        normed = ttnn.rms_norm(flat, weight=weight, epsilon=eps, memory_config=memory_config)
    else:
        normed = ttnn.rms_norm(flat, epsilon=eps, memory_config=memory_config)

    return ttnn.reshape(normed, orig_shape)


def concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Concatenate prefill attention heads into the local hidden dimension."""
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=memory_config)


def apply_output_projection(tensor, weights: AttentionWeights):
    """Apply output projection (no bias for Gemma4)."""
    out = ttnn.linear(tensor, weights.o_proj)
    tensor.deallocate(True)
    return out


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """Apply tensor-parallel allreduce if TP > 1."""
    return ccl_allreduce(tensor, mesh_config, ccl_manager)
