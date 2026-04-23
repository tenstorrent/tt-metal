# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy perf script: MatmulMultiCoreReuseMultiCast2D + FUSE_BIAS with
row_major_output=True and a 1x8 fast-path subblock.

Demonstrates the ROW_MAJOR_OUTPUT unlock on the 2D mcast factory for a bias-fused
matmul. Shape: M=K=N=1024 (32 tiles each) on an 8x8 grid. per_core_M=4,
per_core_N=4 tiles. in0_block_w=4 → num_k_blocks=8.

Configs compared (run this script on branch with row_major=False / row_major=True):
    legacy (subblk 1x1, in0_block_w=1): slow baseline.
    opt_2  (subblk 1x4, in0_block_w=4 + row_major_output=True): fast-path pack,
    4x fewer K-iterations → expect 50-75% device kernel duration reduction.

in1_num_subblocks = per_core_N / out_subblock_w = 1 — inside the known-working
envelope (helper has a latent multi-row bug when in1_num_subblocks > 2).

Run (from repo root, after ``./build_metal.sh``):

    unset TT_METAL_DPRINT_CORES
    python -m tracy -r -v tests/scripts/matmul_perf/mcast_2d_fuse_bias.py
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


SCRIPT_LABEL = "mcast_2d_fuse_bias"


def build_inputs(device):
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    grid_x = min(grid.x, 8)
    grid_y = min(grid.y, 8)

    per_core_M = 4
    per_core_N = 4
    in0_block_w = 4  # num_k_blocks = K_tiles / in0_block_w = 32/4 = 8

    m_size = per_core_M * grid_y * 32
    n_size = per_core_N * grid_x * 32
    k_size = 1024

    in0_shape = [1, 1, m_size, k_size]
    in1_shape = [1, 1, k_size, n_size]
    bias_shape = [1, 1, 1, n_size]

    torch_a = torch.randn(in0_shape).to(torch.bfloat16)
    torch_b = torch.randn(in1_shape).to(torch.bfloat16)
    torch_bias = torch.randn(bias_shape).to(torch.bfloat16)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Bias expects tile-padded first dim.
    bias_padded = torch_bias.expand(1, 1, 32, n_size).contiguous()
    bias = ttnn.from_torch(bias_padded, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=per_core_N,  # 1x4 fast-path
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        row_major_output=True,
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
