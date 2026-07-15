# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("ag_output_shape", [[1, 1, 128, 128]])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("ag_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
    ],
)
@pytest.mark.parametrize("enable_trace", [False])
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize("use_semaphore_free_all_gather_impl", [True])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0_row", "axis_1_col"])
def test_all_gather_2d_fabric_turning(
    bh_2d_mesh_device,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
    use_semaphore_free_all_gather_impl,
    cluster_axis,
):
    """
    Turning test that exercises FABRIC_2D routing along both cluster_axis 0 (row) and 1 (column).
    This tests the 2D fabric's ability to route data in both dimensions, requiring "turning corners"
    in the routing network - behavior that is unique to FABRIC_2D and cannot be tested with FABRIC_1D.
    """
    # Determine number of devices and create appropriate submesh based on cluster_axis
    mesh_shape = bh_2d_mesh_device.shape

    # For axis 0: use devices along first dimension (row)
    # For axis 1: use devices along second dimension (column)
    if cluster_axis == 0:
        # On bh-llmbox (4,1 mesh), use 2 devices to avoid fabric routing issues
        if mesh_shape == ttnn.MeshShape(4, 1):
            num_devices = 2
        else:
            # On other machines, use all devices in first dimension
            num_devices = mesh_shape[0]
        submesh_shape = ttnn.MeshShape(num_devices, 1)
    else:  # cluster_axis == 1
        num_devices = mesh_shape[1]
        submesh_shape = ttnn.MeshShape(1, num_devices)

    validate_test(num_devices, ttnn.Topology.Linear, mesh_shape, cluster_axis)
    submesh_device = bh_2d_mesh_device.create_submesh(submesh_shape)

    run_all_gather_impl(
        submesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        allowed_pcc=0.9999,
        use_semaphore_free_all_gather_impl=use_semaphore_free_all_gather_impl,
    )
