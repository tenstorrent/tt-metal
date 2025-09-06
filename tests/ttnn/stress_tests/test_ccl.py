# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("num_links", [1, 2], ids=["1_link", "2_links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, all_gather_topology",
    [
        (4, [1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Linear),
        (4, [1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Ring),
        (2, [1, 1, 2048, 2048], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Linear),
    ],
    ids=["4 device line", "4_device_ring", "2_device_line"],
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
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=[
        "DRAM_ONLY",
        "L1_TO_DRAM",
        "DRAM_TO_L1",
        "L1_ONLY",
    ],
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
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize(
    "cluster_axis",
    [
        0,
        1,
    ],
    ids=["row", "column"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
def test_ccl_smoke_test(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    cluster_axis,
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
    if (8 == ttnn.get_num_devices()) and (all_gather_topology == ttnn.Topology.Ring):
        pytest.skip("Rackbox is a mesh not a torus so ring wouldn't work")
    if bh_2d_mesh_device.shape[cluster_axis] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform in this axis")
    if (bh_2d_mesh_device.shape[cluster_axis] != num_devices) and (all_gather_topology == ttnn.Topology.Ring):
        pytest.skip("Ring configuration requires the entire row or column so it loops around")
    for i in range(bh_2d_mesh_device.shape[(cluster_axis - 1) % 2]):
        if cluster_axis == 0:
            submesh_device = bh_2d_mesh_device.create_submesh(
                ttnn.MeshShape((num_devices, 1)), offset=ttnn.MeshCoordinate(0, i)
            )
        else:
            submesh_device = bh_2d_mesh_device.create_submesh(
                ttnn.MeshShape((1, num_devices)), offset=ttnn.MeshCoordinate(i, 0)
            )
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
