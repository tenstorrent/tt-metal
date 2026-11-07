# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Interleaved block-float operands whose tile is not a multiple of the NOC DRAM read alignment, through the Quasar
copy of the 2D, 1D and Reuse matmul factories (#46657 on the mainline op).

A bfp8 32x16 or 16x32 tile is 544 B. Blackhole reads DRAM at 64 B alignment, so the allocator pads each tile to a
576 B page and the readers must lay tiles out in L1 at that stride (get_interleaved_tile_stride); Wormhole reads at
32 B and keeps the 544 B stride. Quasar has no block-float formats, so the module is skipped there (its MX formats
take the same path in test_matmul_mx.py)."""

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc

pytestmark = pytest.mark.use_module_device

qsr = ttnn._ttnn.operations.experimental.quasar


@pytest.fixture(autouse=True)
def _block_float_arch(device):
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar has no block-float formats")


def _program_config(family, Mt, Kt, Nt):
    if family == "2d":
        grid = (2, 2)
        return qsr.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
            in0_block_w=Kt // grid[0],
            out_subblock_h=1,
            out_subblock_w=2,
            out_block_h=Mt // grid[1],
            out_block_w=Nt // grid[0],
            per_core_M=Mt // grid[1],
            per_core_N=Nt // grid[0],
            transpose_mcast=False,
            fused_activation=None,
        )
    if family in ("1d_in0", "1d_in1"):
        num_cores = 4
        mcast_in0 = family == "1d_in0"
        per_core_M = Mt if mcast_in0 else Mt // num_cores
        per_core_N = Nt // num_cores if mcast_in0 else Nt
        return qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(num_cores, 1),
            in0_block_w=Kt // 2,
            out_subblock_h=1,
            out_subblock_w=2,
            out_block_h=per_core_M,
            out_block_w=per_core_N,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=mcast_in0,
        )
    assert family == "reuse"
    return qsr.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 1),
        in0_block_w=Kt // 2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=Mt // 4,
        per_core_N=Nt,
    )


@pytest.mark.parametrize("family", ["2d", "1d_in0", "1d_in1", "reuse"])
@pytest.mark.parametrize(
    "in0_tile, in0_dtype, in1_tile, in1_dtype",
    [
        ((32, 32), ttnn.bfloat16, (32, 16), ttnn.bfloat8_b),  # in1 tile 544 B
        ((16, 32), ttnn.bfloat8_b, (32, 32), ttnn.bfloat8_b),  # in0 tile 544 B
        ((16, 32), ttnn.bfloat8_b, (32, 16), ttnn.bfloat4_b),  # both unaligned: 544 B and 288 B
    ],
    ids=["in1_32x16_bfp8", "in0_16x32_bfp8", "in0_16x32_bfp8_in1_32x16_bfp4"],
)
def test_quasar_matmul_tiny_tile_block_float(device, family, in0_tile, in0_dtype, in1_tile, in1_dtype):
    torch.manual_seed(0)
    out_tile = (in0_tile[0], in1_tile[1])
    M, K, N = 4 * in0_tile[0], 8 * 32, 8 * in1_tile[1]
    Mt, Kt, Nt = M // in0_tile[0], K // 32, N // in1_tile[1]
    if family == "reuse" and in0_tile[0] == 16:
        pytest.skip("reuse factory keeps a 32-row in0 tile")

    a = torch.randn(1, 1, M, K)
    b = torch.randn(1, 1, K, N)
    a_t = ttnn.from_torch(a, tile=ttnn.Tile(in0_tile), dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b_t = ttnn.from_torch(b, tile=ttnn.Tile(in1_tile), dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    # Golden from the values the device actually holds after block-float rounding.
    golden = ttnn.to_torch(a_t).float() @ ttnn.to_torch(b_t).float()

    out = ttnn.experimental.quasar.matmul(
        a_t,
        b_t,
        program_config=_program_config(family, Mt, Kt, Nt),
        dtype=ttnn.bfloat16,
        output_tile=ttnn.Tile(out_tile),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=True
        ),
    )
    passed, pcc = comp_pcc(golden, ttnn.to_torch(out).float(), 0.999)
    logger.info(f"{family} in0 {in0_tile} {in0_dtype} in1 {in1_tile} {in1_dtype}: {pcc}")
    assert passed, pcc
