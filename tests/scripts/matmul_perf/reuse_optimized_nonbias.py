# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuseOptimized (non-bias), non-sharded.

Same program config family as ``reuse_bmm_sharded.py`` but with DRAM-interleaved
inputs rather than height-sharded, exercising the Optimized factory's non-bmm
code path. Chosen to cover the ``#ifdef ROW_MAJOR_OUTPUT`` branch in
``reader_writer_bmm_tile_layout_in1.cpp`` end-to-end with a larger tile budget.

Shape notes (justified further in the runbook):
- 1024x1024x1024 bf16 on a 4x1 grid (MatmulMultiCoreReuse requires
  per_core_N == N, so grid_x=1). Per-core output tile count is 8x32 = 256
  tiles, matching the ``reuse_fuse_bias`` precedent that fits L1 comfortably.
  A single-core 2048^3 variant overflowed L1 (9MB static CBs vs 1.5MB L1),
  so phase 2 halved the shape and spread M across 4 cores.

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

    # 1024x1024x1024 on a 4x1 grid. Grid_x must be 1 because MatmulMultiCoreReuse
    # requires per_core_N == N (it distributes only along M / batch). per_core
    # comes to 8x32 = 256 output tiles, same per-core footprint as the
    # known-working reuse_fuse_bias (512x512x512 1x1). DRAM output so L1
    # pressure stays in the CBs only — the branch's separate out_cb/partials_cb
    # reallocation doubled single-core footprint, so we go multi-core.
    b0, b1 = 1, 1
    m_size, k_size, n_size = 1024, 1024, 1024

    grid_y, grid_x = 4, 1
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
        per_core_M=m_size // (grid_y * 32),
        per_core_N=n_size // (grid_x * 32),
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
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
