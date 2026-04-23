# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuseMultiCast2D with a multi-row subblock
(out_subblock_h > 1) enabled via row_major_output=True.

Demonstrates the subblock-shape flexibility unlocked by Phase 2. Shape chosen
with small per_core_N so a 4x2 subblock exhibits the multi-row DST-amortization
benefit the non-mcast Case 2 demo captured (per prior opt PR: ~72% kernel
duration reduction). Without row_major_output=True, the matmul_device_operation
FATAL would reject this subblock shape.

Shape: M=1024 (32 tiles), K=1024 (32 tiles), N=128 (4 tiles). grid 2x8.
per_core_M=4, per_core_N=2. Subblock (4, 2) = 8 DST tiles, multi-row. Single
N-subblock per row-group.

Run:

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/mcast_2d_multi_row.py
"""

from __future__ import annotations

import os
import sys

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _perf_harness import (  # noqa: E402
    DEFAULT_MEASURE_ITERS,
    DEFAULT_WARMUP_ITERS,
    HarnessConfig,
    pick_compute_kernel_config,
    run_warmup_and_measure,
)


SCRIPT_LABEL = "mcast_2d_multi_row"


def build_inputs(device):
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    grid_x = min(grid.x, 2)
    grid_y = min(grid.y, 8)

    per_core_M = 4  # multi-row subblock needs per_core_M >= out_subblock_h
    per_core_N = 2  # small N for 4x2 subblock to fit
    in0_block_w = 4

    m_size = per_core_M * grid_y * 32
    n_size = per_core_N * grid_x * 32
    k_size = 1024

    in0_shape = [1, 1, m_size, k_size]
    in1_shape = [1, 1, k_size, n_size]

    torch_a = torch.randn(in0_shape).to(torch.bfloat16)
    torch_b = torch.randn(in1_shape).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=4,  # <-- multi-row, legacy FATAL would reject this
        out_subblock_w=per_core_N,  # in1_num_subblocks = 1
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        row_major_output=True,
    )
    return a, b, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        a, b, program_config = build_inputs(device)
        compute_config = pick_compute_kernel_config(packer_l1_acc=True)

        def run_once():
            return ttnn.matmul(
                a,
                b,
                program_config=program_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=compute_config,
            )

        run_warmup_and_measure(
            run_once,
            device=device,
            config=HarnessConfig(
                warmup_iters=DEFAULT_WARMUP_ITERS,
                measure_iters=DEFAULT_MEASURE_ITERS,
            ),
            label=SCRIPT_LABEL,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
