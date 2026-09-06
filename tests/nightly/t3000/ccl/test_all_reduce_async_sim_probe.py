# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
craq-sim verification probe for ttnn.experimental.all_reduce_async on an 8-chip Blackhole line.

WHY THIS EXISTS: every shipped all_reduce_async test gates on `mesh_device.get_num_devices() == 32`
(TG / Galaxy), but the multichip simulator's reliable Blackhole configuration is the 8-chip all-MMIO
`blackhole_8xP150` descriptor. This probe drives the SAME op through the SAME reusable impl the TG
tests use (`run_all_reduce_with_mesh_tensor_along_row`), just on a (1, 8) line, so the
all_reduce_async dataflow path — the writer kernel migrated onto the CCL dataflow helper's duplex /
fused / multicast channels — is actually executed and PCC-verified rather than only compiled.

This is a verification vehicle, not a CI test: it is gated on the simulator being active so it
skips on real hardware and in normal CI collection.
"""

import os

import pytest
import ttnn

from tests.nightly.tg.ccl.test_all_reduce_async import run_all_reduce_with_mesh_tensor_along_row


pytestmark = pytest.mark.skipif(
    not os.environ.get("TT_METAL_SIMULATOR"),
    reason="craq-sim multichip verification probe; requires TT_METAL_SIMULATOR",
)


@pytest.mark.parametrize("num_devices_per_line", [8])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("per_chip_output_shape", [([1, 1, 32, 256])])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("num_iters", [2])  # >1 so the program-cache-hit path is exercised too
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8_line")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_all_reduce_async_sim_line(
    mesh_device,
    num_devices_per_line,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    buffer_type,
    num_iters,
    function_level_defaults,
):
    run_all_reduce_with_mesh_tensor_along_row(
        mesh_device,
        num_devices_per_line,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        buffer_type,
        function_level_defaults,
        num_all_reduce_instances=1,
        num_iters=num_iters,
        cluster_axis=1,
    )
