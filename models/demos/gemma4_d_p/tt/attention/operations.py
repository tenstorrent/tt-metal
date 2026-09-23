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
from models.demos.gemma4_d_p.tt.matmul_config import prefill_matmul_config

from .weights import AttentionWeights


def prefill_short_lived_memcfg() -> ttnn.MemoryConfig:
    """Some ops improve overall perf by leaving their activations in L1. This function returns L1 interleaved config, unless overriden to DRAM."""
    if os.environ.get("GEMMA4_ACTIVATIONS_DRAM_ONLY", "0").lower() in ("1", "true", "yes"):
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def projection_matmul_kwargs(hidden_states, weight):
    """Explicit blocking with fp32 accumulation for an attention projection, or {} for ttnn's defaults.

    Uses the device's full core grid. With this blocking, accumulating in bf16 measurably costs
    prefill KV accuracy; HiFi2 with fp32 accumulation improves on the default config.
    """
    device = hidden_states.device()
    grid = device.compute_with_storage_grid_size()
    program_config = prefill_matmul_config(hidden_states, weight, grid.x, grid.y, fp32_dest_acc=True)
    if program_config is None:
        return {}
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return {"program_config": program_config, "compute_kernel_config": compute_kernel_config}


def apply_qkv_projection(hidden_states, weights: AttentionWeights, memory_config=None, kv_tied: bool = False):
    """Project to QKV, or QK when kv_tied selects the narrow tied weight."""
    w_tensor = weights.wqk if kv_tied else weights.wqkv
    return ttnn.linear(
        hidden_states,
        w_tensor,
        memory_config=memory_config,
        **projection_matmul_kwargs(hidden_states, w_tensor),
    )


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


def apply_per_head_norm(tensor, eps, weight=None, memory_config=None):
    """Normalize each token and head independently along head_dim."""
    orig_shape = tensor.shape
    _, num_heads, seq_len, head_dim = orig_shape
    flat = ttnn.reshape(tensor, (1, 1, num_heads * seq_len, head_dim))

    # Use HiFi4 and fp32 acc for greater accuracy
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        tensor.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    normed = ttnn.rms_norm(
        flat,
        weight=weight,
        epsilon=eps,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    return ttnn.reshape(normed, orig_shape)
