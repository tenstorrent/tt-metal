# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Shared MatmulMultiCoreReuseMultiCast1DProgramConfig builder for decode (m=1) and
short-sequence prefill linears in Qwen3-TTS MLP and attention projections.

Keeping one implementation avoids drift between gate/up/down and wqkv/wo tuning.
"""

from __future__ import annotations

import math

import ttnn


def make_linear_1d_program_config(
    m: int,
    k: int,
    n: int,
    grid_x: int,
    grid_y: int,
    fp32_dest_acc_en: bool,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """
    Build a 1D multicast matmul program config for ttnn.linear.

    Args:
        m: Logical row count along the batch*sequence dimension (use 1 for decode).
        k, n: Inner and output feature dimensions for x @ W.T.
        grid_x, grid_y: Device compute grid from ``device.compute_with_storage_grid_size()``.
        fp32_dest_acc_en: Must match the paired WormholeComputeKernelConfig flag so that
            subblock divisibility matches the kernel.
    """
    tile_h = 32
    tile_w = 32
    num_cores = max(1, grid_x * grid_y)

    per_core_m = max(1, m // tile_h)
    per_core_k = max(1, math.ceil((k / tile_w) / num_cores))
    per_core_n = max(1, math.ceil((n / tile_w) / num_cores))

    subblock_limit = 4 if fp32_dest_acc_en else 8
    out_subblock_w = max(i for i in range(1, subblock_limit + 1) if per_core_n % i == 0)
    out_subblock_h = max(
        i for i in range(1, subblock_limit + 1) if per_core_m % i == 0 and i * out_subblock_w <= subblock_limit
    )

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
