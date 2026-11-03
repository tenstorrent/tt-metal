# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""#58177 in the Quasar copy of the 2D, 1D in1-multicast and Reuse factories: with packer_l1_acc requested, two K
blocks must keep partial sums in the accumulation format like more blocks do, instead of rounding them through the
block-float output format. The same matmul split into two and into four K blocks must be equally accurate.

#61018: with fp32_dest_acc_en the partials DFB is Float32 and the Metal 2.0 validator rejected the program for lack of
an unpack_modes entry. The Quasar factories now accumulate fp32 partials in L1 (packer) and keep the partials on the
src path, so every fp32-accumulation config builds and matches an fp32 torch golden."""

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import comp_pcc

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device

qsr = ttnn._ttnn.operations.experimental.quasar


# Per (family, grid): per_core_M, per_core_N, out_subblock_h, out_subblock_w, with out_block == per_core. The 8x1 and
# 4x4 grids are the ones the #58177 / #61018 repros were written for; the 2x1 twins (two cores along x) fit the small
# compute grids of the emulator images. M = N = 8 tiles in every case, so each config covers the whole output.
_BLOCKING = {
    ("reuse", (8, 1)): (1, 8, 1, 4),
    ("reuse", (2, 1)): (4, 8, 1, 4),
    ("1d_in1", (8, 1)): (1, 8, 1, 4),
    ("1d_in1", (2, 1)): (4, 8, 1, 4),
    ("1d_in0", (8, 1)): (8, 1, 4, 1),
    ("1d_in0", (2, 1)): (8, 4, 4, 1),
    ("2d", (4, 4)): (2, 2, 1, 2),
    ("2d", (2, 1)): (8, 4, 1, 4),
}
_FULL_GRID = {"reuse": (8, 1), "1d_in1": (8, 1), "1d_in0": (8, 1), "2d": (4, 4)}


def _family_grid_params():
    return [
        pytest.param(family, grid, id=f"{family}-{grid[0]}x{grid[1]}", marks=pytest.mark.requires_grid_size(grid))
        for family, grid in _BLOCKING
    ]


def _two_vs_four_k_blocks_config(family, in0_block_w, grid=None):
    grid = grid or _FULL_GRID[family]
    per_core_M, per_core_N, out_subblock_h, out_subblock_w = _BLOCKING[(family, grid)]
    grid_size = ttnn.CoreCoord(*grid)
    if family == "2d":
        return qsr.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=per_core_M,
            out_block_w=per_core_N,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
        )
    if family in ("1d_in1", "1d_in0"):  # 1d_in0: M stays on every core, N is split
        return qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=per_core_M,
            out_block_w=per_core_N,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=family == "1d_in0",
        )
    return qsr.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


@pytest.mark.parametrize("family", ["2d", "1d_in1", "reuse"])
def test_quasar_matmul_packer_l1_acc_two_k_blocks(device, family):
    dtype = ttnn.bfloat4_b
    torch.manual_seed(0)
    M, K, N = 256, 512, 256
    Kt = K // 32

    torch_input_a = torch.randn(1, 1, M, K)
    torch_input_b = torch.randn(1, 1, K, N)
    torch_output = torch_input_a @ torch_input_b
    ttnn_input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    ttnn_input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    pcc = {}
    for num_k_blocks in (2, 4):
        output = ttnn.experimental.quasar.matmul(
            ttnn_input_a,
            ttnn_input_b,
            program_config=_two_vs_four_k_blocks_config(family, Kt // num_k_blocks),
            dtype=dtype,
            compute_kernel_config=compute_kernel_config,
        )
        _, pcc[num_k_blocks] = comp_pcc(torch_output, ttnn.to_torch(output).float())
    logger.info(f"{family}: PCC with 2 K blocks {pcc[2]:.5f}, with 4 K blocks {pcc[4]:.5f}")
    assert 1 - pcc[2] <= 1.1 * (1 - pcc[4]), f"{family}: PCC with 2 K blocks {pcc[2]:.5f}, with 4 K blocks {pcc[4]:.5f}"


@pytest.mark.parametrize("family, grid", _family_grid_params())
@pytest.mark.parametrize("num_k_blocks, packer_l1_acc", [(4, False), (4, True), (1, False)])
def test_fp32_dest_acc_reuse_1d_2d(device, family, grid, num_k_blocks, packer_l1_acc):
    """The #61018 repro (256x512 @ 512x256 bf16, HiFi4, fp32_dest_acc_en) through every factory, with and without
    packer_l1_acc requested, and with a single K block (the partials DFB is consumed even then), on the full grid of
    the repro and on a two-core grid (skipped where the device grid is too small)."""
    torch.manual_seed(0)
    M, N = 256, 256
    in0_block_w = 4
    K = 32 * in0_block_w * num_k_blocks

    torch_input_a = torch.randn(1, 1, M, K).bfloat16()
    torch_input_b = torch.randn(1, 1, K, N).bfloat16()
    golden = torch_input_a.float() @ torch_input_b.float()
    ttnn_input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    ttnn_input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=packer_l1_acc
    )

    output = ttnn.experimental.quasar.matmul(
        ttnn_input_a,
        ttnn_input_b,
        program_config=_two_vs_four_k_blocks_config(family, in0_block_w, grid),
        compute_kernel_config=compute_kernel_config,
    )
    output = ttnn.to_torch(output).float()
    atol = 0.02 * golden.abs().max().item()
    max_err = (output - golden).abs().max().item()
    logger.info(
        f"{family} on {grid[0]}x{grid[1]}, {num_k_blocks} K blocks, packer_l1_acc={packer_l1_acc}: "
        f"max abs err {max_err:.4g} vs atol {atol:.4g}"
    )
    assert torch.allclose(
        output, golden, rtol=0.02, atol=atol
    ), f"{family}: max abs err {max_err:.4g} vs atol {atol:.4g}"
