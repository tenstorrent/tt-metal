# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Repro for the WH SDXL 2D-mcast matmul hang (PR 63576): block-sharded in0 on a 5x8 grid whose
# shard holds 2 K-blocks (extract_shard_sub_blocks -> rotating Flag sender, src == dst).
import os

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

ITERS = int(os.environ.get("REPRO_ITERS", "200000"))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0, "trace_region_size": 200000}], indirect=True)
@pytest.mark.parametrize("use_trace", [True], ids=["trace"])
@pytest.mark.parametrize("in0_block_w", [4], ids=["bw4"])
def test_sdxl_mm_hang(device, use_trace, in0_block_w):
    torch.manual_seed(0)
    M, K, N = 1024, 1280, 1280
    grid = ttnn.CoreCoord(5, 8)
    a = torch.randn(1, 1, M, K).bfloat16().float()
    b = torch.randn(1, 1, K, N).bfloat16().float()

    shard_mem = ttnn.create_sharded_memory_config(
        (1, 1, M, K),
        core_grid=ttnn.CoreGrid(y=8, x=5),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard_mem)
    tt_b = ttnn.from_torch(
        b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=4,
        per_core_N=8,
        transpose_mcast=False,
        fused_activation=None,
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    def run():
        return ttnn.matmul(tt_a, tt_b, program_config=pc, memory_config=shard_mem, compute_kernel_config=ckc)

    out = run()
    ttnn.synchronize_device(device)
    expected = a @ b
    assert_with_pcc(expected, ttnn.to_torch(out), 0.99)
    out.deallocate(True)

    if use_trace:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = None
        for _ in range(20):
            if out is not None:
                out.deallocate(True)
            out = run()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for i in range(ITERS // 20):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            if i % 10 == 9:
                ttnn.synchronize_device(device)
        ttnn.synchronize_device(device)
        ttnn.release_trace(device, tid)
    else:
        for i in range(ITERS):
            out = run()
            out.deallocate(True)
            if i % 50 == 49:
                ttnn.synchronize_device(device)
        ttnn.synchronize_device(device)
