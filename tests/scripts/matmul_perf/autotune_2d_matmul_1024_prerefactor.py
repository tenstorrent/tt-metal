# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pre-refactor-equivalent baseline for autotune_2d_matmul_1024.py.

Replicates what the auto-config path would have emitted before the matmul
auto-tuner migration: per_core_M=per_core_N=16 (get_per_core_factor's pick
for 1024^3 DRAM, only 4 cores active on an 8x8 grid), in0_block_w=2,
subblock (1, 2) (legacy SUBBLOCK_HW_CHOICES fast-path + out_subblock_h=1
override), row_major_output=False.

Compared to autotune_2d_matmul_1024.py, the auto-tuner now picks subblock
(1, 8) (fast-path preference within the subblock_w_eq_per_core_n_required
envelope, since row_major_output falls back to False on this L1-tight
shape). Expected: modest perf win from the larger subblock.

Run:

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/autotune_2d_matmul_1024_prerefactor.py
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


SCRIPT_LABEL = "autotune_2d_matmul_1024_prerefactor"


def build_inputs(device):
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    grid_x = min(grid.x, 8)
    grid_y = min(grid.y, 8)

    per_core_M = 16
    per_core_N = 16
    in0_block_w = 2

    m_size = 1024
    n_size = 1024
    k_size = 1024

    torch_a = torch.randn([1, 1, m_size, k_size]).to(torch.bfloat16)
    torch_b = torch.randn([1, 1, k_size, n_size]).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=2,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        row_major_output=False,
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
            config=HarnessConfig(warmup_iters=DEFAULT_WARMUP_ITERS, measure_iters=DEFAULT_MEASURE_ITERS),
            label=SCRIPT_LABEL,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
