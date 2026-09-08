# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operations for Gemma4.

GeGLU activation: gelu(gate) * up (different from GPT-OSS SwiGLU).
"""

import math

import ttnn
from models.demos.gemma4_d_p.tt.compute_config import gelu_variant


def apply_geglu(gate, up):
    """GeGLU activation: gelu(gate) * up (Accurate variant; see compute_config)."""
    activated = ttnn.gelu(gate, variant=gelu_variant())
    result = ttnn.mul(activated, up)
    return result


def _build_sparse_matmul_config(m, n, in0_block_w=1):
    """Build program config for sparse_matmul following gpt-oss pattern."""
    n_tiles = int(math.ceil(n / 32))

    # Find largest divisor of n_tiles fitting in 8×8 grid
    best_cores = 1
    best_cx, best_cy = 1, 1
    for num_cores in range(1, min(65, n_tiles + 1)):
        if n_tiles % num_cores != 0:
            continue
        for cy in range(1, 9):
            if num_cores % cy == 0:
                cx = num_cores // cy
                if cx <= 8 and num_cores > best_cores:
                    best_cores = num_cores
                    best_cx, best_cy = cx, cy
                    break

    per_core_N = n_tiles // best_cores

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_cx, best_cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_N,
        per_core_M=max(32, m) // 32,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
