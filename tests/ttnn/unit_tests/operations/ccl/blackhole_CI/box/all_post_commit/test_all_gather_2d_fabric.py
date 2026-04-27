# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


@skip_for_wormhole_b0()
@skip_for_n_dev(8)
@pytest.mark.parametrize("num_devices", [2])
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
@pytest.mark.parametrize("num_iters", [10])
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
def test_all_gather_2d_fabric(
    bh_2d_mesh_device,
    num_devices,
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
    function_level_defaults,
):
    if bh_2d_mesh_device.shape[0] != 1 and bh_2d_mesh_device.shape[1] != 1:
        pytest.skip("2D dynamic requires one dimension to be 1")

    # Determine which axis has enough devices
    cluster_axis_actual = 0 if bh_2d_mesh_device.shape[0] >= num_devices else 1
    mesh_shape = (num_devices, 1) if cluster_axis_actual == 0 else (1, num_devices)

    validate_test(num_devices, all_gather_topology, bh_2d_mesh_device.shape, cluster_axis_actual)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))
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
        cluster_axis=cluster_axis_actual,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        allowed_pcc=0.9999,
        use_semaphore_free_all_gather_impl=use_semaphore_free_all_gather_impl,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
