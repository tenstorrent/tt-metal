# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_to_all import run_all_to_all_impl
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_dev


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_dev(8)
@pytest.mark.parametrize(
    "num_devices, num_links, logical_shape, in_dim, out_dim, layout, input_dtype, mem_config, enable_trace",
    [
        (
            2,
            1,
            [1, 1, 128, 256],
            2,
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            False,
        ),
        (
            4,
            1,
            [1, 1, 256, 512],
            2,
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
            False,
        ),
    ],
    ids=["2dev_2to3", "4dev_2to3"],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 90112, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_all_to_all_2d_fabric(
    bh_2d_mesh_device,
    num_devices,
    logical_shape,
    in_dim,
    out_dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
    enable_trace,
):
    # all_to_all requires Ring topology, but FABRIC_2D only supports Linear topology
    # FABRIC_1D_RING is used for Ring topology, but there's no FABRIC_2D_RING
    pytest.skip("all_to_all requires Ring topology which is not supported with FABRIC_2D - use FABRIC_1D_RING instead")

    if bh_2d_mesh_device.shape[0] != 1 and bh_2d_mesh_device.shape[1] != 1:
        pytest.skip("2D dynamic requires one dimension to be 1")

    # all_to_all requires Ring topology
    topology = ttnn.Topology.Ring

    # Determine which axis has enough devices - for Ring, need full row/column
    cluster_axis = 0 if bh_2d_mesh_device.shape[0] == num_devices else 1
    mesh_shape = (num_devices, 1) if cluster_axis == 0 else (1, num_devices)

    # Ring topology requires the entire row or column
    if bh_2d_mesh_device.shape[cluster_axis] != num_devices:
        pytest.skip(
            f"Ring topology requires full dimension: need {num_devices} devices in axis {cluster_axis}, have {bh_2d_mesh_device.shape[cluster_axis]}"
        )

    validate_test(num_devices, topology, bh_2d_mesh_device.shape, cluster_axis)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))

    run_all_to_all_impl(
        submesh_device,
        num_devices,
        logical_shape,
        in_dim,
        out_dim,
        num_links,
        input_dtype,
        layout,
        topology=topology,
        num_iters=num_iters,
        mem_config=mem_config,
        do_check=True,
        trace_mode=enable_trace,
        reuse_inputs=False,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
