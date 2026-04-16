# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuseOptimized with FUSE_BIAS, tall-skinny.

Larger fused-bias shape that exercises the optimized reuse factory on a
multi-core grid. Shape is tall-skinny (4096x512x64) to target a common
production pattern (e.g. projection-like layers) that is distinct from the
square ``reuse_fuse_bias.py`` case.

Run (from the repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/reuse_optimized_fuse_bias.py
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


SCRIPT_LABEL = "reuse_optimized_fuse_bias"


def build_inputs(device):
    torch.manual_seed(0)

    # Tall-skinny shape. Single-core path (1x1 grid) keeps the measurement
    # comparable to the 256x512x256 baseline but with larger M to amortize
    # bias-add work across more output tiles.
    m_size, k_size, n_size = 1024, 512, 64
    grid_y, grid_x = 1, 1

    in0_shape = [1, 1, m_size, k_size]
    in1_shape = [1, 1, k_size, n_size]
    bias_shape = [1, 1, 1, n_size]

    torch_a = torch.randn(in0_shape).to(torch.bfloat16)
    torch_b = torch.randn(in1_shape).to(torch.bfloat16)
    torch_bias = torch.randn(bias_shape).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m_size // 32,
        per_core_N=n_size // 32,
    )
    return a, b, bias, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        a, b, bias, program_config = build_inputs(device)
        compute_config = pick_compute_kernel_config()

        def run_once():
            return ttnn.linear(
                a,
                b,
                bias=bias,
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
