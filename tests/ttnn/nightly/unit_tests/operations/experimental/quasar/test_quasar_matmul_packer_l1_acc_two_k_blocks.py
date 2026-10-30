# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""#58177 in the Quasar copy of the 2D, 1D in1-multicast and Reuse factories: with packer_l1_acc requested, two K
blocks must keep partial sums in the accumulation format like more blocks do, instead of rounding them through the
block-float output format. The same matmul split into two and into four K blocks must be equally accurate."""

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


def _two_vs_four_k_blocks_config(family, in0_block_w):
    if family == "2d":
        return qsr.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=2,
            out_block_h=2,
            out_block_w=2,
            per_core_M=2,
            per_core_N=2,
            transpose_mcast=False,
            fused_activation=None,
        )
    if family == "1d_in1":
        return qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=4,
            out_block_h=1,
            out_block_w=8,
            per_core_M=1,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
    return qsr.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=8,
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
