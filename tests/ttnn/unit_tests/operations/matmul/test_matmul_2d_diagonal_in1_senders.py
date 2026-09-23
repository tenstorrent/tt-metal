# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MatmulMultiCoreReuseMultiCastProgramConfig.diagonal_in1_senders: the in1 senders move from one line of cores to a
diagonal and senders / receivers share one kernel. The result must be bit-identical to the default layout on every
2D mcast code path: transpose_mcast on/off x block-sharded / interleaved in0 (the interleaved-in0 path splits the grid
into two NoC halves), including a ragged last M block owned by a diagonal sender."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _grid(gx, gy):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


# name: (M, K, N, grid (x, y), transpose_mcast, in0 block shard or None, per_core_M, per_core_N, in0_block_w, subblock w)
CASES = {
    # the SDXL full-grid FF-up: M = 1024 over 11 columns -> 3 tiles per core, the last column ragged (33 >= 32)
    "t11x10_sharded_ragged": (1024, 1280, 5120, (11, 10), True, [96, 128], 3, 16, 4, 8),
    "t11x10_in0_il_ragged": (1024, 1280, 5120, (11, 10), True, None, 3, 16, 4, 8),
    "rm10x8_sharded": (1024, 1280, 5120, (10, 8), False, [128, 128], 4, 16, 4, 8),
    "rm10x8_in0_il": (1024, 1280, 5120, (10, 8), False, None, 4, 16, 4, 8),
}


def _run(device, case, diagonal):
    M, K, N, (gx, gy), transpose, shard, pcm, pcn, ibw, sbw = CASES[case]
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K)
    b = torch.randn(1, 1, K, N) / K**0.5
    if shard is None:
        in0_mc = ttnn.L1_MEMORY_CONFIG
    else:
        orient = ttnn.ShardOrientation.COL_MAJOR if transpose else ttnn.ShardOrientation.ROW_MAJOR
        in0_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(_grid(gx, gy), shard, orient)
        )
    ta = ttnn.from_torch(a, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mc)
    tb = ttnn.from_torch(
        b, ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sbw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=transpose,
        fused_activation=None,
        diagonal_in1_senders=diagonal,
    )
    out = ttnn.linear(
        ta,
        tb,
        program_config=program_config,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG if shard is not None else ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        ),
    )
    result = ttnn.to_torch(out).float()
    for t in (ta, tb, out):
        ttnn.deallocate(t)
    return a @ b, result


@pytest.mark.parametrize("case", list(CASES))
def test_diagonal_in1_senders(device, case):
    gx, gy = CASES[case][3]
    grid = device.compute_with_storage_grid_size()
    if grid.x < gx or grid.y < gy:
        pytest.skip(f"needs a {gx}x{gy} worker grid, device has {grid.x}x{grid.y}")
    ref, line = _run(device, case, diagonal=False)
    _, diag = _run(device, case, diagonal=True)
    assert_with_pcc(ref, diag, 0.999)
    assert torch.equal(line, diag), "diagonal in1 senders changed the result"


def test_diagonal_in1_senders_repr():
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    assert cfg.diagonal_in1_senders is False
    cfg.diagonal_in1_senders = True
    assert "diagonal_in1_senders=1" in repr(cfg).lower()
