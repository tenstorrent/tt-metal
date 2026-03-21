# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul / linear projection helpers for the hybrid GLM-4.7-Flash.

Re-exports the agentic linear helpers which provide:
- compute_1d_prog_cfg: 1D multicast program config for decode matmuls
- mlp_linear: general-purpose linear with MLP compute kernel
- tp_row_parallel_linear: row-parallel matmul with all_reduce
- dram_sharded_linear / dram_sharded_mlp: DRAM WIDTH_SHARDED decode paths
- attn_linear: attention projection dispatching to optimal path
"""

from models.demos.glm4_moe_lite.tt.linear_helpers import (
    _DS_BATCH,
    _DS_CKC,
    attn_linear,
    compute_1d_prog_cfg,
    dram_sharded_linear,
    dram_sharded_mlp,
    mlp_linear,
    tp_row_parallel_linear,
)

__all__ = [
    "compute_1d_prog_cfg",
    "mlp_linear",
    "tp_row_parallel_linear",
    "dram_sharded_linear",
    "dram_sharded_mlp",
    "attn_linear",
    "_DS_BATCH",
    "_DS_CKC",
]
