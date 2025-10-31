# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype",
    [
        (8, [1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "sd35_spatial",
        "sd35_prompt",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize(
    "all_gather_function",
    [ttnn.experimental.all_gather_async, ttnn.experimental.all_gather_async_reversed],
    ids=["normal", "reversed"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_gather_async(
    mesh_device,
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
    all_gather_function,
):
    if num_devices < 8:
        submesh_shape = (1, num_devices)
        cluster_axis = 1
    else:
        submesh_shape = (num_devices, 1)
        cluster_axis = 0
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))
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
        all_gather_function=all_gather_function,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype",
    [
        (4, [1, 1, 32, 896], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 32, 256], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 8, 32, 576], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "deepseek_1",
        "deepseek_2",
        "deepseek_3",
        "deepseek_4",
        "deepseek_5",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize(
    "all_gather_function",
    [ttnn.experimental.all_gather_async],
    ids=["normal"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_gather_deepseek(
    mesh_device,
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
    all_gather_function,
):
    if num_devices < 8:
        submesh_shape = (1, num_devices)
        cluster_axis = 1
    else:
        submesh_shape = (num_devices, 1)
        cluster_axis = 0
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))
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
        all_gather_function=all_gather_function,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("num_links", [1], ids=["1links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype",
    [
        (4, [1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "sd35_spatial",
        "sd35_prompt",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
def test_all_gather_async_blackhole(
    mesh_device,
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
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
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
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@pytest.mark.parametrize("num_links", [1], ids=["1links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype",
    [
        (8, [8, 1, 1, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "gather_dim_3",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 190112,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
def test_all_gather_async_big_mesh(
    mesh_device,
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
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
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
        cluster_axis=None,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@pytest.mark.parametrize("num_links", [1], ids=["1links"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype",
    [
        ([16, 1, 1, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "gather_dim_3",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 190112,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("mesh_device", [(8, 16)], indirect=True)
def test_all_gather_async_quad_host_mesh(
    mesh_device,
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
):
    cluster_axis = 1
    shape = (1, mesh_device.shape[cluster_axis]) if cluster_axis == 1 else (mesh_device.shape[cluster_axis], 1)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(shape))
    run_all_gather_impl(
        submesh_device,
        submesh_device.shape[cluster_axis],
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
    )
    ttnn.ReadDeviceProfiler(submesh_device)


# AllGather	1	Linear	bfloat16	2	[1, 1, 1024, 8192]	[1, 1, 8192, 8192]	DRAM / DRAM	8
# AllGather	1	Linear	bfloat16	3	[1, 1, 8192, 1024]	[1, 1, 8192, 8192]	DRAM / DRAM	8
# AllGather	1	Linear	bfloat16	3	[1, 1, 8192, 128]	[1, 1, 8192, 1024]	DRAM / DRAM	8
# AllGather	1	Linear	bfloat16	1	[1, 1, 8192, 64]	[1, 8, 8192, 64]	DRAM / DRAM	8
# AllGather	1	Linear	bfloat16	1	[1, 8, 8192, 64]	[1, 64, 8192, 64]	DRAM / DRAM	8
# AllGather	1	Linear	bfloat16	3	[1, 1, 8191, 4000]	[1, 1, 8191, 32000]	DRAM / DRAM	8


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [1, 2, 4], ids=["1link", "2link", "4link"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype",
    [
        (8, [1, 1, 8192, 8192], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 8192, 8192], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 8192, 1024], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 8, 8192, 64], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 64, 8192, 64], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 8191, 32000], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "llama_1",
        "llama_2",
        "llama_3",
        "llama_4",
        "llama_5",
        "llama_6",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 6),
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        # 1D_Ring
        (
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "1D_Ring",
    ],
)
@pytest.mark.parametrize("num_workers_per_link", [1, 2, 4], ids=["1worker", "2worker", "4worker"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_all_gather_llama(
    mesh_device,
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
    num_workers_per_link,
):
    submesh_shape = (2, 4)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))
    submesh_device.reshape(ttnn.MeshShape((1, num_devices)))
    ttnn.visualize_mesh_device(mesh_device)
    ttnn.visualize_mesh_device(submesh_device)

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
        cluster_axis=None,
        use_barrier=True,
        use_persistent_buffers=True,
        chunks_per_sync=None,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=None,
        skip_check=True,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
