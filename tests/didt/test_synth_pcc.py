# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC guard for the reconfig-hoist fix: same block-sharded l1_acc matmul as test_synth_mm, but
# compares ttnn output to a torch golden. Verifies the in_place path (SBM and TRM subblocks) still
# produces correct values after moving pack_reconfig_data_format / llk_pack_reconfig_l1_acc from
# per-subblock to per-K-block.  env: SYN_PCN, SYN_K, SYN_SBH, SYN_SBW, SYN_TRM
import os
import pytest
import torch
import ttnn
from tests.didt.op_test_base import get_mesh_grid_size
from models.common.utility_functions import comp_pcc, is_blackhole


def _e(n, d, c=str):
    v = os.environ.get(n)
    return c(v) if v not in (None, "") else d


@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chips")], indirect=["mesh_device"])
def test_synth_pcc(mesh_device):
    per_core_M = 4
    per_core_N = _e("SYN_PCN", 22, int)
    in0_block_w = _e("SYN_K", 2, int)
    sbh, sbw, trm = _e("SYN_SBH", 4, int), _e("SYN_SBW", 2, int), _e("SYN_TRM", "1") == "1"
    grid = get_mesh_grid_size(mesh_device)
    cr = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))
    in0_shard = ttnn.ShardSpec(ttnn.CoreRangeSet({cr}), [32 * per_core_M, 32 * 18], ttnn.ShardOrientation.ROW_MAJOR)
    in0_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_shard)
    in1_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )
    if trm:
        pc.tile_pack_row_major = True
    CC = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    cc = CC(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)

    M, K, N = 32 * per_core_M * grid.y, 576 * grid.x, 32 * per_core_N * grid.x
    a = torch.randn(1, 1, M, K)
    b = torch.randn(1, 1, K, N)
    at = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in0_mc, device=mesh_device)
    bt = ttnn.from_torch(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=in1_mc, device=mesh_device)
    out = ttnn.matmul(at, bt, program_config=pc, memory_config=out_mc, dtype=ttnn.bfloat16, compute_kernel_config=cc)
    golden = ttnn.to_torch(at).float() @ ttnn.to_torch(bt).float()
    ok, msg = comp_pcc(golden, ttnn.to_torch(out), 0.99)
    print(f"SYNPCC sbh={sbh} sbw={sbw} trm={trm} l1acc=1: {msg} -> {'PASS' if ok else 'FAIL'}", flush=True)
    assert ok, f"PCC fail: {msg}"
