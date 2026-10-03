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
from models.demos.gemma4_d_p.tt.matmul_config import (
    is_short_m,
    prefill_1d_matmul_program_config,
    prefill_matmul_program_config,
    short_m_output_memcfg,
    to_l1_width_sharded,
)

from .weights import AttentionWeights


def prefill_short_lived_memcfg() -> ttnn.MemoryConfig:
    """Some ops improve overall perf by leaving their activations in L1. This function returns L1 interleaved config, unless overriden to DRAM."""
    if os.environ.get("GEMMA4_ACTIVATIONS_DRAM_ONLY", "0").lower() in ("1", "true", "yes"):
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


# Rows per device from which the projections are FPU-bound, so one fidelity pass saves time.
_LOFI_PROJECTION_MIN_ROWS = 512


def projection_math_fidelity(rows):
    return ttnn.MathFidelity.LoFi if rows >= _LOFI_PROJECTION_MIN_ROWS else ttnn.MathFidelity.HiFi2


def projection_matmul_configs(hidden_states, weight, max_m_tiles=None):
    """(program_config, compute_kernel_config) for an attention projection: explicit blocking with fp32
    accumulation, or (None, None) for ttnn's defaults.

    Uses the device's full core grid. With this blocking, accumulating in bf16 measurably costs
    prefill KV accuracy, so it accumulates in fp32. LoFi pays from 512 rows per device; below that the
    projections stream their weights and gain nothing from it. packer_l1_acc accumulates the K-block
    partials in L1 instead of re-reading them.
    """
    device = hidden_states.device()
    grid = device.compute_with_storage_grid_size()
    program_config = prefill_1d_matmul_program_config(
        hidden_states, weight, grid, max_m_tiles=max_m_tiles
    ) or prefill_matmul_program_config(hidden_states, weight, grid.x, grid.y, fp32_dest_acc=True)
    if program_config is None:
        return None, None
    return program_config, _projection_compute_config(device, hidden_states.shape[-2])


def _projection_compute_config(device, rows):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=projection_math_fidelity(rows),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# Most tile rows for which a projection feeding the reduce-scatter writes its output width-sharded: chunk 4096 at CP8
# (16 tile rows per device).
_MAX_SHARDED_OUTPUT_M_TILES = 16


def project(hidden_states, weight, memory_config=None, into_reduce_scatter=False):
    """hidden_states @ weight for an attention projection, written interleaved (DRAM unless memory_config says
    otherwise). A short-M activation is read width-sharded from L1.

    into_reduce_scatter marks a row-parallel projection whose only consumer is the TP reduce-scatter. Up to
    _MAX_SHARDED_OUTPUT_M_TILES tile rows it then runs the 1D config on the width-sharded activation and writes a
    width-sharded L1 output, which the reduce-scatter reads as fast as an interleaved one. That skips the interleaved
    DRAM write: ~0.7 ms per chunk at 2048 and ~0.3 ms at 4096 for the attention output projection.
    """
    sharded_output = (
        into_reduce_scatter and hidden_states.padded_shape[-2] // ttnn.TILE_SIZE <= _MAX_SHARDED_OUTPUT_M_TILES
    )
    x = to_l1_width_sharded(hidden_states) if sharded_output or is_short_m(hidden_states) else hidden_states
    program_config, compute_kernel_config = projection_matmul_configs(
        x, weight, max_m_tiles=_MAX_SHARDED_OUTPUT_M_TILES if sharded_output else None
    )
    if sharded_output:
        memory_config = short_m_output_memcfg(x, weight)
    out = ttnn.linear(
        x,
        weight,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    if x is not hidden_states:
        x.deallocate(True)
    return out


def apply_qkv_projection(hidden_states, weights: AttentionWeights, memory_config=None, kv_tied: bool = False):
    """Project to QKV, or QK when kv_tied selects the narrow tied weight."""
    return project(hidden_states, weights.wqk if kv_tied else weights.wqkv, memory_config=memory_config)


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
