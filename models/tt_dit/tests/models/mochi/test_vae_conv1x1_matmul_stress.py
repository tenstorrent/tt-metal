# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CI probe for #58653: loop the exact Mochi VAE Conv1x1 projection (tilize -> fused-bias linear -> untilize)
that stalls intermittently on T3K, on a single Wormhole chip, to bound the per-op hang rate."""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.timeout(2400)
def test_conv1x1_matmul_stress(device):
    iterations = int(os.environ.get("MM_STRESS_ITERS", "300"))
    torch.manual_seed(0)
    # Per-device shapes of op 404616 in the failing jobs: in0 [1, 21, 240, 424, 256] bf16, in1 [256, 512], bias [1, 512].
    x = torch.randn(1, 21, 240, 424, 256, dtype=torch.bfloat16)
    w = (torch.randn(256, 512, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    b = (torch.randn(1, 512, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    x_rm = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weight = ttnn.from_torch(
        w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bias = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # Conv1x1.compute_kernel_config in models/tt_dit/models/vae/vae_mochi.py
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    t0 = time.time()
    for i in range(iterations):
        x_tile = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
        out_tile = ttnn.linear(
            x_tile,
            weight,
            bias=bias,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=device.core_grid,
        )
        ttnn.deallocate(x_tile)
        out_rm = ttnn.to_layout(out_tile, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(out_tile)
        ttnn.deallocate(out_rm)
        if (i + 1) % 25 == 0:
            ttnn.synchronize_device(device)
            logger.info(f"conv1x1 matmul stress: {i + 1}/{iterations} iterations, {time.time() - t0:.1f}s elapsed")
    ttnn.synchronize_device(device)
    logger.info(f"conv1x1 matmul stress: completed {iterations} iterations in {time.time() - t0:.1f}s")
