# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0
from models.common.utility_functions import comp_allclose


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


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [1])
def test_reduce_scatter_all_gather_async(mesh_device, cluster_axis):
    """
    Test reduce_scatter_minimal_async followed by all_gather_async.
    Starting with a DRAM interleaved tensor of shape [1, 1, 32, 1536] replicated on entire mesh.
    """
    import torch
    from loguru import logger

    # Input shape: [1, 1, 32, 1536]
    shape = [1, 1, 32, 1536]
    logger.info(f"mesh shape: {mesh_device.shape}")

    # Create random input tensor
    torch_input = torch.ones(shape, dtype=torch.bfloat16)

    # Create DRAM interleaved tensor replicated on entire mesh
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

    # Create global semaphores for reduce_scatter
    rs_multi_device_semaphores = [ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)]

    rs_barrier_semaphore = ttnn.create_global_semaphore(mesh_device, core_range_set, 0)

    # Create global semaphores for all_gather
    ag_multi_device_semaphores = [ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)]

    # Use same barrier semaphore as reduce_scatter
    ag_barrier_semaphore = ttnn.create_global_semaphore(mesh_device, core_range_set, 0)
    ttnn.synchronize_device(mesh_device)

    # Configure reduce_scatter_minimal_async
    rs_config = {
        "dim": 3,
        "multi_device_global_semaphore": rs_multi_device_semaphores,
        "num_links": 1,
        "barrier_semaphore": rs_barrier_semaphore,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "topology": ttnn.Topology.Linear,
        "cluster_axis": cluster_axis,
    }

    # Configure all_gather_async
    ag_config = {
        "dim": 3,
        "cluster_axis": cluster_axis,
        "mesh_device": mesh_device,
        "topology": ttnn.Topology.Linear,
        "multi_device_global_semaphore": ag_multi_device_semaphores,
        "num_links": 1,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "barrier_semaphore": ag_barrier_semaphore,
    }

    # Apply reduce_scatter
    logger.info(f"Applying reduce_scatter_minimal_async")
    tt_output_rs = ttnn.experimental.reduce_scatter_minimal_async(tt_input, **rs_config)

    # Apply all_gather
    logger.info(f"Applying all_gather_async")
    tt_output = ttnn.experimental.all_gather_async(tt_output_rs, **ag_config)
    logger.info(f"Done applying all_gather_async")

    # Convert output to torch
    torch_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # shape should be [8, 1, 32, 1536]
    assert torch_output.shape == (32, 1, 32, 1536)
    # input was all ones, after reduce scatter along cluster_axis = 1 which has 8 devices, the output should be all 8
    eq, output = comp_allclose(torch_output, torch.ones(shape) * 8)
    assert eq, f"Reduce scatter async failed: {output}"
