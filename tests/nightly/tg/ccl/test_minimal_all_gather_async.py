# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters",
    [
        # Perf variant (with tracing)
        (8, [1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),
        # Check variant (without tracing)
        (8, [1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),
    ],
    ids=[
        "sd35_spatial-perf",
        "sd35_prompt_check",
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
    ids=["DRAM_memconfig"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,
        ),
        # (
        #    {
        #        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        #        "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        #        "trace_region_size": 90112,
        #    },
        #    ttnn.Topology.Linear,
        # ),  # test removed due to issue 35320
    ],
    indirect=["device_params"],
    ids=[
        "fabric_linear",
        # "fabric_manager_enabled_linear" # test removed due to issue 35320
    ],
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
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters",
    [
        # Perf variants (with tracing)
        (4, [1, 1, 32, 896], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),
        (8, [1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),
        (8, [1, 8, 32, 576], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),
        # Check variants (without tracing)
        (8, [1, 1, 32, 256], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),
        (8, [1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),
    ],
    ids=[
        "deepseek_1-perf",
        "deepseek_3-perf",
        "deepseek_4-perf",
        "deepseek_2-check",
        "deepseek_5-check",
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


@pytest.mark.parametrize("num_links", [1], ids=["1links"])
@pytest.mark.parametrize(
    "input_shape, gather_dim, cluster_axis,layout, ag_input_dtype",
    [
        ([1, 10, 640, 128], 2, None, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "wan_all_gather_1",
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
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 190112,
            },
            ttnn.Topology.Linear,
        ),
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
    ids=["fabric_2d_linear", "fabric_linear"],
)
@pytest.mark.parametrize("mesh_device", [(4, 32)], indirect=True)
def test_all_gather_4x32_sanity(
    mesh_device,
    input_shape,
    gather_dim,
    cluster_axis,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
):
    from loguru import logger

    torch.manual_seed(2005)
    devices = mesh_device.get_num_devices()
    input_shape[gather_dim] *= devices

    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    use_submesh = False
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 32))) if use_submesh else mesh_device
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ag_input_dtype,
        memory_config=mem_config_input,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh_device, dim=gather_dim),
        device=submesh_device,
    )

    logger.info(f"Starting all-gather")
    tt_output = ttnn.all_gather(
        tt_input,
        dim=gather_dim,
        cluster_axis=cluster_axis,
        topology=all_gather_topology,
        num_links=num_links,
        memory_config=mem_config_ag,
    )
    logger.info(f"All-gather completed")
    # logger.info(f"tt_output.shape = {tt_output.shape}")
    # torch_output = ttnn.to_torch(
    #     tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    # )
    # logger.info(f"torch_output.shape = {torch_output.shape}")
    # torch_reference = torch_input.repeat([devices, 1, 1, 1])
    # eq, output = comp_equal(torch_output, torch_reference)
    # assert eq, f"Output mismatch between torch and ttnn all-gather: {output}"


@pytest.mark.parametrize("num_links", [1, 3, 4], ids=["1links", "3links", "4links"])
@pytest.mark.parametrize(
    "ag_output_shape, gather_dim, cluster_axis,layout, ag_input_dtype",
    [
        ([1, 1, 2560, 5120], 3, 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ([1, 10, 81920, 128], 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
        "wan_all_gather_0",
        "wan_all_gather_1",
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
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 190112,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_linear", "fabric_ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 32)], indirect=True)
def test_all_gather_async_wan_galaxy_4x32(
    mesh_device,
    ag_output_shape,
    gather_dim,
    cluster_axis,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
):
    torch.manual_seed(2005)
    from loguru import logger

    devices = mesh_device.get_num_devices()
    input_shape = ag_output_shape

    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ag_input_dtype,
        memory_config=mem_config_input,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=gather_dim),
        device=mesh_device,
    )

    logger.info(f"tt_input.shape = {tt_input.shape}")
    tt_output = ttnn.all_gather(tt_input, dim=gather_dim, cluster_axis=cluster_axis, topology=all_gather_topology)

    torch_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    print("Warning: No PCC check for this test")
    # torch_reference = torch_input.repeat([devices, 1, 1, 1])
    # eq, output = comp_equal(torch_output, torch_reference)
    # assert eq, f"Output mismatch between torch and ttnn all-gather: {output}"
