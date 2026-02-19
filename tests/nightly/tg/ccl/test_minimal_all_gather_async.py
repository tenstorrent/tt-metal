# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import math

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.perf.benchmarking_utils import BenchmarkProfiler
from tracy import signpost


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


# Modified from tests/ttnn/multidevice_perf_tests/sweep_all_gather_hyperparameters_t3000.py
def get_max_chunks_per_sync(num_devices, ag_output_shape, num_links, packet_size, dtype_size):
    packet_elems = packet_size // dtype_size
    total_elems = math.prod(ag_output_shape)
    return (total_elems // packet_elems) // (num_devices * num_links)


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters",
    [
        # Perf variant (with tracing)
        (8, [1, 1, 128, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),
        # Check variant (without tracing)
        (8, [1, 1, 128, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),
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
@pytest.mark.parametrize("chunks_per_sync", [1])
@pytest.mark.parametrize("num_workers_per_link", [1])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize(
    "all_gather_function",
    [ttnn.experimental.all_gather_async],
    ids=["normal"],
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


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "ag_output_shape, dim, cluster_axis, ag_input_dtype, layout, mem_config_input, mem_config_ag",
    [
        ([1, 1, 9472, 5120], 3, 0, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 9472, 256], 3, 0, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 9472, 128], 3, 0, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 118, 128], 3, 0, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=[
        "spatial_activation",
        "layernorm_stats",
        "rmsnorm_stats_spatial",
        "rmsnorm_stats_prompt",
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology, max_payload_size",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(size),
                "trace_region_size": 1000000,
            },
            ttnn.Topology.Ring,
            size,
        )
        for size in [9216]
    ],
    indirect=["device_params"],
    ids=[f"fabric_ring_{size}B" for size in [9216]],
)
@pytest.mark.parametrize("num_links", [2], ids=lambda v: f"{v}links")
@pytest.mark.parametrize("chunks_per_sync", [320, "MAXby8"], ids=lambda v: f"{v}chunks")
@pytest.mark.parametrize("num_workers_per_link", [3], ids=lambda v: f"{v}workers")
@pytest.mark.parametrize("num_buffers_per_channel", [4], ids=lambda v: f"{v}buffers")
@pytest.mark.parametrize("num_iters, warmup_iters", [(75, 10)])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_all_gather_wan(
    mesh_device,
    ag_output_shape,
    dim,
    cluster_axis,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    num_links,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
    all_gather_topology,
    max_payload_size,
    num_iters,
    warmup_iters,
):
    from loguru import logger

    # Create input tensor
    mesh_shape = tuple(mesh_device.shape)
    input_shape = ag_output_shape
    num_devices = mesh_shape[cluster_axis]

    torch.manual_seed(2005)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    shard_dims = (None, dim) if cluster_axis == 1 else (dim, None)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=ag_input_dtype,
        memory_config=mem_config_input,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        device=mesh_device,
    )

    # AllGather config
    if isinstance(chunks_per_sync, str) and chunks_per_sync.startswith("MAXby"):
        divisor = int(chunks_per_sync[5:])  # extract int after "MAXby"
        max_chunks_per_sync = get_max_chunks_per_sync(num_devices, ag_output_shape, num_links, max_payload_size, 2)
        chunks_per_sync_val = max_chunks_per_sync // divisor
    else:
        chunks_per_sync_val = chunks_per_sync
    if chunks_per_sync_val < 1:
        # pytest.skip(f"Chunks per sync value is too small: {chunks_per_sync_val}")
        chunks_per_sync_val = 1

    # Compile Run
    logger.info("Compiling op")
    tt_output = ttnn.all_gather(
        tt_input,
        dim=dim,
        cluster_axis=cluster_axis,
        topology=all_gather_topology,
        num_links=num_links,
        memory_config=mem_config_ag,
        chunks_per_sync=chunks_per_sync_val,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.synchronize_device(mesh_device)

    # Check output
    errors = []
    for dev, tt_out in enumerate(ttnn.get_device_tensors(tt_output)):
        eq, mess = comp_pcc(torch_input, ttnn.to_torch(tt_out))
        if not eq:
            errors.append(f"Device {dev}: {mess}")
    assert not errors, f"PCC check failed on {len(errors)} device(s):\n" + "\n".join(errors)

    ################## TRACE RUN #######################

    # Capture trace
    logger.info("Capturing trace")

    def capture_trace(n_iters):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(n_iters):
            _ = ttnn.all_gather(
                tt_input,
                dim=dim,
                cluster_axis=cluster_axis,
                topology=all_gather_topology,
                num_links=num_links,
                memory_config=mem_config_ag,
                chunks_per_sync=chunks_per_sync_val,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        return trace_id

    if warmup_iters > 0:
        trace_id_warmup = capture_trace(warmup_iters)
    trace_id = capture_trace(num_iters)

    # Run the op
    logger.info("Starting Trace perf test...")
    profiler = BenchmarkProfiler()
    profiler.start("all-gather-async-trace-warmup")
    if warmup_iters > 0:
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
    profiler.end("all-gather-async-trace-warmup")

    profiler.start("all-gather-async-trace")
    signpost("start")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    profiler.end("all-gather-async-trace")
    time_taken = profiler.get_duration("all-gather-async-trace")
    logger.info(f"Time taken e2e: {time_taken} s")
    logger.info(f"Time per iter e2e: {time_taken / num_iters} s")
    logger.info(f"Time per iter e2e: {time_taken / num_iters * 1e6} us")
