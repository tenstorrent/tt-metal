# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from models.utility_functions import skip_for_blackhole, skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.nightly.test_all_gather_nightly import validate_test


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("num_links", [1, 2], ids=["1_link", "2_links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout",
    [
        (4, [1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
        (4, [1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT),
        (2, [1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
        (2, [1, 1, 32, 768], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device_128_2048", "4_device_32_4096", "2_device_32_1024", "2_device_32_768"],
)
@pytest.mark.parametrize(
    "rs_input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.uint32, #Bad PCC
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        # "uint_32", #Bad PCC
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
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
    ids=["dram_only", "l1_to_dram", "dram_to_l1", "l1_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 10),
    ],
    ids=[
        "trace",
        "non-trace",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=[
        "fabric_linear",
        "fabric_ring",
    ],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
def test_rs_row_nightly(
    bh_1d_mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    rs_topology,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    cluster_axis = 0
    validate_test(num_devices, rs_topology, bh_1d_mesh_device.shape, cluster_axis)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    run_reduce_scatter_impl(
        submesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
