# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""WH gather_in0 DRAM-sharded bank lookup must TT_FATAL on a y-map miss, not deref end()."""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.unit_tests.operations.matmul.test_ring_matmul import GRID

qsr = ttnn._ttnn.operations.experimental.quasar


@pytest.mark.skipif(not is_wormhole_b0(), reason="WH-only sparse y-to-DRAM-bank maps")
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_quasar_gather_in0_unknown_worker_y_is_fatal(device, expect_error):
    """Ring GRID first-col workers (1,3) and (2,3) have y=3, not in WH map {0,4,5,9}."""
    worker_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in GRID])
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})
    in0_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(worker_grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [1280, 320], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(worker_grid, [32, 160], ttnn.ShardOrientation.ROW_MAJOR),
    )
    program_config = qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(6, 4),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=1,
        per_core_N=5,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ttnn.CoreRangeSet([]),
        untilize_out=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    torch.manual_seed(12345)
    in0 = torch.rand([1, 1, 32, 1280], dtype=torch.bfloat16)
    in1 = torch.rand([1, 1, 1280, 3200], dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in0_mc)
    in1_t = ttnn.from_torch(in1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, device=device, memory_config=in1_mc)

    with expect_error(RuntimeError, "NOT FOUND in first-col map"):
        ttnn.experimental.quasar.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=out_mc,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
