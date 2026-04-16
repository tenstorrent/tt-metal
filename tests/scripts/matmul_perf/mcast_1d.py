# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuseMultiCast1D.

Exercises the 1D multicast factory (``mcast_in0=True``). Shape derived from
the didt LM-head test but scaled down so it stays within a single-card L1
budget on the phase-2 BH P100A machine.

Current phase-1 branch does NOT emit ROW_MAJOR_OUTPUT for multicast factories
(see ``project_matmul_helpers_phase1_context.md`` and
``project_matmul_helper_followon.md``). This script captures the legacy-helper
baseline so that, when the Option-2 follow-on PR extends ROW_MAJOR_OUTPUT to
multicast, the perf delta is directly measurable on the same harness.

Run (from the repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/mcast_1d.py
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


SCRIPT_LABEL = "mcast_1d"


def build_inputs(device):
    torch.manual_seed(0)

    # Use compute grid_size.x for the 1D mcast layout.
    grid = device.compute_with_storage_grid_size()
    grid_x = min(grid.x, 7)
    grid_y = 1
    per_core_M = 1  # 32 rows per core
    per_core_N = 8  # 8x32 = 256 cols per core

    m_size = per_core_M * 32
    k_size = 512
    n_size = per_core_N * grid_x * 32

    in0_shape = [1, 1, m_size, k_size]
    in1_shape = [1, 1, k_size, n_size]

    torch_a = torch.randn(in0_shape).to(torch.bfloat16)
    torch_b = torch.randn(in1_shape).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    return a, b, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        a, b, program_config = build_inputs(device)
        compute_config = pick_compute_kernel_config(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            packer_l1_acc=True,
        )

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
