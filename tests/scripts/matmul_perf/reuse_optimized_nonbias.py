# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuseOptimized (non-bias), non-sharded.

Same program config family as ``reuse_bmm_sharded.py`` but with DRAM-interleaved
inputs rather than height-sharded, exercising the Optimized factory's non-bmm
code path. Chosen to cover the ``#ifdef ROW_MAJOR_OUTPUT`` branch in
``reader_writer_bmm_tile_layout_in1.cpp`` end-to-end with a larger tile budget.

Shape notes (justified further in the runbook):
- 2048x2048x2048 bf16 is a common production matmul size on BH/WH and has
  per-core tile budget where subblock choice matters.
- If this is too large for a given grid, phase 2 can lower per_core_M/N.

Run (from the repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/reuse_optimized_nonbias.py
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


SCRIPT_LABEL = "reuse_optimized_nonbias"


def build_inputs(device):
    torch.manual_seed(0)

    # 2048x2048x2048: one batch, interleaved DRAM on both sides, enters the
    # optimized reuse factory via explicit MatmulMultiCoreReuseProgramConfig.
    # Use a 1x1 grid for a single-core path so the Tracy measurement stays
    # clean (keeps the single-core baseline from project memory useful).
    b0, b1 = 1, 1
    m_size, k_size, n_size = 2048, 2048, 2048

    grid_y, grid_x = 1, 1
    in0_shape = [b0, b1, m_size, k_size]
    in1_shape = [b0, b1, k_size, n_size]

    torch_a = torch.randn(in0_shape).to(torch.bfloat16)
    torch_b = torch.randn(in1_shape).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m_size // 32,
        per_core_N=n_size // 32,
    )
    return a, b, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        a, b, program_config = build_inputs(device)
        compute_config = pick_compute_kernel_config()

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
