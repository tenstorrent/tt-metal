# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Baseline pair for mcast_2d_fuse_bias.py. Same shape, legacy config:
row_major_output=False, subblock (1, 1), in0_block_w=1. Use as the main-branch
compare against the tuned variant.

Run:

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/mcast_2d_fuse_bias_baseline.py
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


SCRIPT_LABEL = "mcast_2d_fuse_bias_baseline"


def build_inputs(device):
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    grid_x = min(grid.x, 8)
    grid_y = min(grid.y, 8)

    per_core_M = 4
    per_core_N = 4
    in0_block_w = 1  # legacy: 1 K-tile per K-block

    m_size = per_core_M * grid_y * 32
    n_size = per_core_N * grid_x * 32
    k_size = 1024

    torch_a = torch.randn([1, 1, m_size, k_size]).to(torch.bfloat16)
    torch_b = torch.randn([1, 1, k_size, n_size]).to(torch.bfloat16)
    torch_bias = torch.randn([1, 1, 1, n_size]).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bias_padded = torch_bias.expand(1, 1, 32, n_size).contiguous()
    bias = ttnn.from_torch(bias_padded, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,  # legacy minimal subblock
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        # row_major_output defaults to False
    )
    return a, b, bias, program_config


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        device.enable_program_cache()
        a, b, bias, program_config = build_inputs(device)
        compute_config = pick_compute_kernel_config(packer_l1_acc=True)

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
