# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Backward-compatibility re-export layer for BGE-M3 compute/memory policy.

All tuning rules live in ``optimizations.py``. This module re-exports them
under the ``bge_m3_*`` names that existing consumers expect.
"""

from models.demos.wormhole.bge_m3.tt.optimizations import (
    _linear_activation_memory_config,
    _matmul_core_grid,
    _mlp_wi_minimal_matmul_config,
    _mlp_wi_output_memory_config,
    _mlp_wo_output_memory_config,
    attention_output_compute_kernel_config,
    attention_qkv_compute_kernel_config,
    layernorm_compute_kernel_config,
    matmul_compute_kernel_config,
    mlp_wi_compute_kernel_config,
    mlp_wo_compute_kernel_config,
    sdpa_compute_kernel_config,
    weight_dram_memory_config,
)

# Memory configs
bge_m3_weight_dram_memory_config = weight_dram_memory_config
bge_m3_linear_activation_memory_config = _linear_activation_memory_config
bge_m3_mlp_wi_output_memory_config = _mlp_wi_output_memory_config
bge_m3_mlp_wo_output_memory_config = _mlp_wo_output_memory_config

# Core grid
bge_m3_matmul_core_grid = _matmul_core_grid

# Compute kernel configs — direct delegates, no policy duplication
bge_m3_matmul_compute_kernel_config = matmul_compute_kernel_config
bge_m3_mlp_wi_compute_kernel_config = mlp_wi_compute_kernel_config
bge_m3_mlp_wo_compute_kernel_config = mlp_wo_compute_kernel_config
bge_m3_attention_output_compute_kernel_config = attention_output_compute_kernel_config
bge_m3_attention_qkv_compute_kernel_config = attention_qkv_compute_kernel_config
bge_m3_sdpa_compute_kernel_config = sdpa_compute_kernel_config
bge_m3_layernorm_compute_kernel_config = layernorm_compute_kernel_config


# Program configs
bge_m3_mlp_wi_minimal_matmul_config = _mlp_wi_minimal_matmul_config


# Sequence chunk limits
def max_qkv_mm_chunk_seq_len(_mesh_device=None):
    return 8192


def max_wo_mm_chunk_seq_len(_mesh_device=None):
    return 8192
