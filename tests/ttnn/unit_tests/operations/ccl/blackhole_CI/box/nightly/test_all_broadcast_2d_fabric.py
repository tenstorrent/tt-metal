# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_new_all_broadcast import run_all_broadcast_impl

import pytest
import ttnn
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_dev, skip_for_n_or_less_dev


@skip_for_n_or_less_dev(1)
@skip_for_wormhole_b0()
@skip_for_n_dev(8)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("output_shape", [[1, 1, 1, 32, 1024]])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("mem_config", [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)])
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_all_broadcast_2d_fabric(
    bh_1d_mesh_device,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    num_devices = bh_1d_mesh_device.shape[0]
    cluster_axis = 0

    topology = ttnn.Topology.Linear
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, cluster_axis)

    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    run_all_broadcast_impl(
        bh_1d_mesh_device,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_broadcast_topology=topology,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
    )
