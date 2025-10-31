# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype",
    [
        (8, [8, 1, 512, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [4, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, [1, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, [1, 1, 333, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [2, 1, 2048, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # from CSV
        (8, [1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # from CSV
    ],
    ids=[
        "batch_8",
        "batch_4",
        "batch_1_sd35_spatial",
        "batch_1_sd35_prompt",
        "batch_2",
        "batch_1",
        "deepseek_1",
        "deepseek_2",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
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
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_reduce_scatter_async(
    mesh_device,
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
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
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


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [1], ids=["1links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype",
    [
        (8, [1, 1, 8, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
    ],
    ids=[
        "deepseek_like",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
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
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
def test_reduce_scatter_async_big_mesh(
    mesh_device,
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
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    cluster_axis = None
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
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [1], ids=["1links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype",
    [
        (16, [1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
    ],
    ids=[
        "deepseek_4host",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
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
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("mesh_device", [(8, 16)], indirect=True)
def test_reduce_scatter_async_quad_host_mesh(
    mesh_device,
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
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    cluster_axis = None
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
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [1, 2, 4], ids=["1link", "2link", "4link"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype",
    [
        (8, [1, 1, 8192, 8192], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 8192, 1024], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 8192, 16], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, [1, 1, 1, 1], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "llama_1",
        "llama_2",
        "llama_3",
        "llama_4",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
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
    ],
    ids=["perf"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("num_workers_per_link", [1, 2, 4], ids=["1worker", "2worker", "4worker"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_reduce_scatter_async_llama(
    mesh_device,
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
    num_workers_per_link,
):
    submesh_shape = (1, 8)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))
    submesh_device.reshape(ttnn.MeshShape((1, num_devices)))
    ttnn.visualize_mesh_device(mesh_device)
    ttnn.visualize_mesh_device(submesh_device)
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
        cluster_axis=None,
        use_barrier=True,
        use_persistent_buffers=True,
        chunks_per_sync=None,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=None,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
