# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pre-refactor-equivalent baseline for autotune_2d_matmul_2048_dram.py.

Replicates what create_simple_matmul_program_config's all_dram_interleaved
branch would have emitted BEFORE the auto-tuner migration:
- per_core_M = per_core_N = 8 (same as post-refactor — full grid utilization)
- in0_block_w = 8 (same)
- out_subblock (4, 2) — legacy SUBBLOCK_HW_CHOICES first-fit (volume 8,
  fp32-constraint-safe, no per_core_N_eq_subblock_w constraint for
  create_simple_matmul's 2D branch since it had no constraint)
- row_major_output = False

NOTE: pre-refactor also had the hardcoded `out_subblock_h = 1` override
at line 1277 when out_subblock_w != per_core_N (which was the case here:
out_subblock_w=2 != per_core_N=8). So the actual pre-refactor pick was
(1, 2). That's what this script uses.

Run:

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/autotune_2d_matmul_2048_dram_prerefactor.py
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


SCRIPT_LABEL = "autotune_2d_matmul_2048_dram_prerefactor"


def build_inputs(device):
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    grid_x = min(grid.x, 8)
    grid_y = min(grid.y, 8)

    per_core_M = 8
    per_core_N = 8
    in0_block_w = 8

    m_size = 2048
    n_size = 2048
    k_size = 2048

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
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
