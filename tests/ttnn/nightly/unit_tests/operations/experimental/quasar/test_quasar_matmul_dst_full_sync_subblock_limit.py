# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""#59689 in the Quasar matmul validation: dst_full_sync_en doesn't raise the subblock limit of the 2D, 1D and Reuse
factories. Subblocks above 8 tiles (4 with fp32 accumulation) are rejected instead of computing wrong values, and an
allowed subblock computes correctly."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device

qsr = ttnn._ttnn.operations.experimental.quasar


def _dst_full_sync_program_config(family, sbh, sbw):
    grid = ttnn.CoreCoord(8, 8)
    if family == "1d":  # M 1 tile, per core 1x10
        return qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=2,
            out_subblock_h=sbh,
            out_subblock_w=sbw,
            out_block_h=1,
            out_block_w=10,
            per_core_M=1,
            per_core_N=10,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    if family == "2d":  # M 1 tile, per core 1x16
        return qsr.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=2,
            out_subblock_h=sbh,
            out_subblock_w=sbw,
            out_block_h=1,
            out_block_w=16,
            per_core_M=1,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
        )
    return qsr.MatmulMultiCoreReuseProgramConfig(  # batch 8, per core 8x8 (8 blocks, one per core)
        compute_with_storage_grid_size=grid,
        in0_block_w=1,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        per_core_M=8,
        per_core_N=8,
    )


@pytest.mark.parametrize(
    "family, a_shape, b_shape",
    [
        ("1d", [32, 128], [128, 20480]),
        ("2d", [32, 64], [64, 4096]),
        ("reuse", [8, 256, 96], [8, 96, 256]),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_quasar_matmul_dst_full_sync_subblock_limit(device, expect_error, family, a_shape, b_shape, fp32_dest_acc_en):
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 8:
        pytest.skip("device does not have 8x8 grid")
    torch.manual_seed(0)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )
    # (an allowed subblock, a subblock over the half-sync limit but within the full-sync one), within each
    # family's per-core block
    allowed, over = {
        ("1d", False): ((1, 5), (1, 10)),
        ("1d", True): ((1, 2), (1, 5)),
        ("2d", False): ((1, 8), (1, 16)),
        ("2d", True): ((1, 4), (1, 8)),
        ("reuse", False): ((1, 8), (2, 8)),
        ("reuse", True): ((1, 4), (1, 8)),
    }[(family, fp32_dest_acc_en)]
    torch_a = torch.randn(a_shape).bfloat16()
    torch_b = torch.randn(b_shape).bfloat16()
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    with expect_error(RuntimeError, "to fit in hardware"):
        ttnn.experimental.quasar.matmul(
            a,
            b,
            program_config=_dst_full_sync_program_config(family, *over),
            compute_kernel_config=compute_kernel_config,
        )

    if fp32_dest_acc_en:
        # The Quasar copies of these factories can't build fp32-accumulation programs with partials yet (Metal 2.0
        # validation: the FP32 partials DFB has no unpack_modes entry), so the fp32 cases only check the rejection.
        return
    output = ttnn.experimental.quasar.matmul(
        a,
        b,
        program_config=_dst_full_sync_program_config(family, *allowed),
        compute_kernel_config=compute_kernel_config,
    )
    assert_with_pcc(torch_a.float() @ torch_b.float(), ttnn.to_torch(output).float(), 0.999)
