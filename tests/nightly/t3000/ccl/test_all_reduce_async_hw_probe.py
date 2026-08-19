# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Hardware probe: ttnn.experimental.all_reduce_async on a (1, 4) line.

Mirrors the sim probe (test_all_reduce_async_sim_probe.py) at (1, 4): every shipped
all_reduce_async test gates on 32 devices (TG/Galaxy), so this drives the same op through the same
reusable impl on any 4+-device system — real-silicon coverage of the worker_writer migrated onto the
CCL dataflow helper's duplex Cast::Multicast stream (arm_write + arm_fused_write_inc).
"""

import pytest
import ttnn

from tests.nightly.tg.ccl.test_all_reduce_async import run_all_reduce_with_mesh_tensor_along_row


@pytest.mark.parametrize("num_devices_per_line", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("per_chip_output_shape", [([1, 1, 32, 256]), ([1, 1, 64, 512])])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.DRAM])
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("num_iters", [2])  # >1 so the program-cache-hit path is exercised too
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4_line")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_all_reduce_async_hw_line(
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
