# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuseMultiCast2D.

Exercises the 2D multicast factory, where rows of cores broadcast in0 and
columns broadcast in1. Shape patterned after
``test_padded_2d_matmul`` / ``test_matmul_in1_multicast`` in the TTNN suite.

As with ``mcast_1d.py``, this captures the legacy-helper baseline for the
multicast path (phase-1 does not emit ROW_MAJOR_OUTPUT for multicast factories).

Run (from the repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/mcast_2d.py
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


SCRIPT_LABEL = "mcast_2d"


def build_inputs(device):
    torch.manual_seed(0)

    # Use an 8x8 grid if available, else downsize.
    grid = device.compute_with_storage_grid_size()
    grid_x = min(grid.x, 8)
    grid_y = min(grid.y, 8)

    # Tile counts per core: choose per_core_M/N that divide evenly.
    per_core_M = 4
    per_core_N = 4
    in0_block_w = 2
    m_size = per_core_M * grid_y * 32  # 4 * 8 * 32 = 1024
    k_size = in0_block_w * 32 * 4  # 256 (must equal in0_block_w * 32 * num_blocks)
    n_size = per_core_N * grid_x * 32  # 4 * 8 * 32 = 1024

    # Ensure k_size is divisible by in0_block_w * 32.
    assert k_size % (in0_block_w * 32) == 0

    in0_shape = [1, 1, m_size, k_size]
    in1_shape = [1, 1, k_size, n_size]

    torch_a = torch.randn(in0_shape).to(torch.bfloat16)
    torch_b = torch.randn(in1_shape).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=per_core_N,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
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
