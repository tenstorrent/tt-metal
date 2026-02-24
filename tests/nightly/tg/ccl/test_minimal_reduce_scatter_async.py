# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from models.common.utility_functions import skip_for_blackhole, skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.perf.benchmarking_utils import BenchmarkProfiler
from tracy import signpost


# TODO import this from the correct file after PR is merged
def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [4], ids=["4links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype, enable_trace, num_iters",
    [
        # Perf variants (with tracing)
        (8, [8, 1, 512, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # use batching when fused
        (8, [4, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # use batching when fused
        (8, [1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # use batching when fused
        (8, [1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # from CSV
        # Check variants (without tracing)
        (4, [1, 1, 333, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # use batching when fused
        (8, [2, 1, 2048, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # use batching when fused
        (8, [1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # use batching when fused
        (8, [1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # from CSV
    ],
    ids=[
        "batch_8-perf",
        "batch_4-perf",
        "batch_1-perf",
        "deepseek_1-perf",
        "batch_1_sd35_prompt-check",
        "batch_2-check",
        "batch_1-check",
        "deepseek_2-check",
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
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
        ),
        # (
        #    {
        #        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        #        "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        #        "trace_region_size": 90112,
        #    },
        #    ttnn.Topology.Linear,
        # ),
    ],
    indirect=["device_params"],
    ids=[
        "fabric_ring",
        # "fabric_manager_enabled_linear" # test removed due to issue 35320
    ],
)
@pytest.mark.parametrize("chunks_per_sync", [1])
@pytest.mark.parametrize("num_workers_per_link", [1])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
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
    "rs_input_shape, dim, cluster_axis, rs_input_dtype, layout, mem_config_input, mem_config_rs",
    [
        ([1, 1, 9472, 5120], 3, 0, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=[
        "matmul_rs",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology, max_payload_size",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(size),
                "trace_region_size": 1100000,
            },
            ttnn.Topology.Ring,
            size,
        )
        for size in [2048, 3072, 4096, 5120, 6144, 7168, 8704, 8192, 9216, 15232]
    ],
    indirect=["device_params"],
    ids=[f"fabric_ring_{size}B" for size in [2048, 3072, 4096, 5120, 6144, 7168, 8704, 8192, 9216, 15232]],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=lambda v: f"{v}links")
@pytest.mark.parametrize("chunks_per_sync", [1, 10, 160, 320, 1000, 2000, 3000], ids=lambda v: f"{v}chunks")
@pytest.mark.parametrize("num_workers_per_link", [1, 2, 3, 4, 8], ids=lambda v: f"{v}workers")
@pytest.mark.parametrize("num_buffers_per_channel", [1, 2, 4, 8], ids=lambda v: f"{v}buffers")
@pytest.mark.parametrize("num_iters, warmup_iters", [(75, 10)])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_reduce_scatter_async_wan(
    mesh_device,
    rs_input_shape,
    dim,
    cluster_axis,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    num_links,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
    rs_topology,
    max_payload_size,
    num_iters,
    warmup_iters,
):
    from loguru import logger

    # Create input tensor
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[cluster_axis]

    torch.manual_seed(2005)
    torch_input_shape = list(rs_input_shape)
    torch_input_shape[0] *= num_devices
    torch_input = torch.rand(torch_input_shape, dtype=torch.bfloat16)  # [num_devices * a, b, c, d]

    # reference output
    torch_ref = torch_input.reshape(num_devices, *rs_input_shape)  # [num_devices, a, b, c, d]
    torch_ref = torch_ref.sum(dim=0)  # [a, b, c, d]
    torch_ref_list = []  # num_devices tensors of [a, b, c, d // num_devices]
    for dev in range(num_devices):
        N = rs_input_shape[dim] // num_devices
        indices = [slice(None)] * torch_ref.ndim  # [slice(None)] is equivalent to ':'
        indices[dim] = slice(dev * N, (dev + 1) * N)  # extract device slice along `dim`
        torch_ref_list.append(torch_ref[tuple(indices)])

    shard_dims = (None, 0) if cluster_axis == 1 else (0, None)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=rs_input_dtype,
        memory_config=mem_config_input,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        device=mesh_device,
    )

    # Compile Run
    logger.info("Compiling op")
    tt_output = ttnn.reduce_scatter(
        tt_input,
        dim,
        cluster_axis=cluster_axis,
        memory_config=mem_config_rs,
        intermediate_memory_config=None,  # TODO constrain this?
        topology=rs_topology,
        num_links=num_links,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.synchronize_device(mesh_device)

    # Check output
    errors = []
    d = 0
    tt_outs = ttnn.get_device_tensors(tt_output)
    for i in range(mesh_shape[0]):
        for j in range(mesh_shape[1]):
            repl_dim, shard_dim = (i, j) if cluster_axis == 1 else (j, i)
            eq, mess = comp_pcc(torch_ref_list[shard_dim], ttnn.to_torch(tt_outs[d]))
            if not eq:
                errors.append(f"Device {dev}: {mess}")
            d += 1
    assert not errors, f"PCC check failed on {len(errors)} device(s):\n" + "\n".join(errors)

    ################## TRACE RUN #######################

    # Capture trace
    logger.info("Capturing trace")

    def capture_trace(n_iters):
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(n_iters):
            tt_output = ttnn.reduce_scatter(
                tt_input,
                dim,
                cluster_axis=cluster_axis,
                memory_config=mem_config_rs,
                intermediate_memory_config=None,  # TODO constrain this?
                topology=rs_topology,
                num_links=num_links,
                chunks_per_sync=chunks_per_sync,
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
    profiler.start("reduce-scatter-async-trace-warmup")
    if warmup_iters > 0:
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
    profiler.end("reduce-scatter-async-trace-warmup")

    profiler.start("reduce-scatter-async-trace")
    signpost("start")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    profiler.end("reduce-scatter-async-trace")
    time_taken = profiler.get_duration("reduce-scatter-async-trace") - profiler.get_duration(
        "reduce-scatter-async-trace-warmup"
    )
    effective_iter = num_iters - warmup_iters
    logger.info(f"Time taken e2e: {time_taken} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter} s")
    logger.info(f"Time per iter e2e: {time_taken / effective_iter * 1e6} us")

    # TODO ttnn.ReadDeviceProfiler(mesh_device)
