# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
        (2, [1, 1, 128, 128], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device_test", "2_device_test"],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        "uint_32",
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=["dram_only", "l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_linear", "fabric_ring"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
def test_all_gather_nightly(
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
):
    if (2 == num_devices) and (all_gather_topology == ttnn.Topology.Ring):
        pytest.skip("Ring configuration requires more than 2 devices")
    if (bh_2d_mesh_device.shape[0] != num_devices) and (all_gather_topology == ttnn.Topology.Ring):
        pytest.skip("Ring configuration requires the entire row or column so it loops around")
    if bh_2d_mesh_device.shape[0] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
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
    )
    ttnn.ReadDeviceProfiler(submesh_device)
