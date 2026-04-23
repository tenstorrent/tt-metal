# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Auto-tuning perf script: same shape as mcast_2d_fast_path.py (2048^3)
without a program_config. Exercises the auto-tuner's fast-path preference.

Run:

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/autotune_2d_fast_path.py
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


SCRIPT_LABEL = "autotune_2d_fast_path"


def build_inputs(device):
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    grid_x = min(grid.x, 8)
    grid_y = min(grid.y, 8)

    per_core_M = 8
    per_core_N = 8

    m_size = per_core_M * grid_y * 32
    n_size = per_core_N * grid_x * 32
    k_size = 2048

    torch_a = torch.randn([1, 1, m_size, k_size]).to(torch.bfloat16)
    torch_b = torch.randn([1, 1, k_size, n_size]).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return a, b


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        a, b = build_inputs(device)
        compute_config = pick_compute_kernel_config(packer_l1_acc=True)

        def run_once():
            return ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=compute_config)

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
