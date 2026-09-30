# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf research (not for merge): does the packer L1-accumulate into FP32 from a BF16 dest?

A matmul whose K is split into blocks accumulates the blocks with packer_l1_acc. With
fp32_dest_acc_en=False and a float32 output, the block partials leave a 16-bit dest and are summed in L1
in the output format. If that sum is FP32, the error vs FP64 stays near one BF16 rounding per block
partial; if it is BF16, it grows with the number of blocks like a BF16 running sum.
"""

import os

import pytest
import torch
import ttnn


@pytest.mark.skipif(os.getenv("TEST_SDPA_PERF_L1ACC") != "1", reason="Opt-in perf research")
@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("k", [1024, 16384])
def test_packer_l1acc_fp32_from_bf16_dest(device, out_dtype, k):
    torch.manual_seed(0)
    a = torch.rand(64, k).bfloat16()  # positive: a swamping BF16 sum shows up clearly
    b = torch.rand(k, 64).bfloat16()
    ref = a.double() @ b.double()
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=2,
        per_core_N=2,
    )
    out = ttnn.to_torch(ttnn.matmul(ta, tb, dtype=out_dtype, compute_kernel_config=cfg, program_config=pc)).double()
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"L1ACC k={k} out={out_dtype} blocks={k // 64} rel_l2={rel:.3e}")
