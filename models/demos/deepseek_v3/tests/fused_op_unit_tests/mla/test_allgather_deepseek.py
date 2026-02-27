# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def run_allgather_deepseek_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor_mesh,
    dim,
    cluster_axis,
    num_links,
    output_mem_config,
    ccl_semaphore_handles,
    barrier_semaphore,
    num_iter=100,
    warmup_iters=10,
    subdevice_id=None,
    profiler=BenchmarkProfiler(),
):
    """Run all-gather with trace mode for performance measurement."""
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_gather_async(
        input_tensor_mesh,
        dim,
        cluster_axis,
        mesh_device,
        all_gather_topology,
        [ccl_semaphore_handles[0], ccl_semaphore_handles[1]],
        num_links=num_links,
        memory_config=output_mem_config,
        use_broadcast=True,
        barrier_semaphore=barrier_semaphore,
        subdevice_id=subdevice_id,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(warmup_iters):
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            cluster_axis,
            mesh_device,
            all_gather_topology,
            [ccl_semaphore_handles[2 * i], ccl_semaphore_handles[2 * i + 1]],
            num_links=num_links,
            memory_config=output_mem_config,
            use_broadcast=True,
            barrier_semaphore=barrier_semaphore,
            subdevice_id=subdevice_id,
        )
        tt_out_tensor.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iter} iterations")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            cluster_axis,
            mesh_device,
            all_gather_topology,
            [ccl_semaphore_handles[2 * i], ccl_semaphore_handles[2 * i + 1]],
            num_links=num_links,
            memory_config=output_mem_config,
            use_broadcast=True,
            barrier_semaphore=barrier_semaphore,
            subdevice_id=subdevice_id,
        )
        if i != num_iter - 1:
            tt_out_tensor.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler.start("all-gather-trace-warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    profiler.end("all-gather-trace-warmup")

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("all-gather-trace")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    profiler.end("all-gather-trace")
    signpost("stop")

    return tt_out_tensor


@pytest.mark.parametrize(
    "op_name, num_devices, input_shape, output_shape, dim, layout, cluster_axis",
    [
        (
            "height_sharded_ag",
            32,  # 32 devices (4x8 grid)
            [1, 1, 4, 16384],  # Input per device: HEIGHT_SHARDED
            [1, 1, 16, 16384],  # Output shape after all-gather along dim 2 (4*4=16)
            2,  # Gather along dim 2
            ttnn.ROW_MAJOR_LAYOUT,
            1,  # Cluster axis 1
        ),
    ],
    ids=["height_sharded_ag"],
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear])
@pytest.mark.parametrize("output_sharding", ["interleaved", "width_sharded"])
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 100)])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 901200,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_all_gather_trace_mode(
    mesh_device,
    op_name,
    num_devices,
    input_shape,
    output_shape,
    dim,
    num_links,
    trace_mode,
    input_dtype,
    layout,
    cluster_axis,
    topology,
    output_sharding,
    warmup_iters,
    num_iters,
    function_level_defaults,
):
    """
    Test all-gather operations with HEIGHT_SHARDED input and trace mode for performance measurement.

    This test captures traces of all-gather operations and executes them multiple times to measure performance.
    Uses signposts for Tracy profiling integration.

    Configuration:
    - Input shape: [1, 1, 4, 16384] per device (HEIGHT_SHARDED)
    - Output shape: [1, 1, 16, 16384] (gather along dim=2, 4*4=16)
    - All-gather on dim=2 (across cluster_axis=1 with 4x8 grid)
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Topology: Ring and Linear (parameterized)
    - Input: HEIGHT_SHARDED L1 memory with shard_spec grid [0:0-1:1], shape [1, 16384]
    - Output: INTERLEAVED or WIDTH_SHARDED L1 memory (parameterized)
      - INTERLEAVED: no sharding
      - WIDTH_SHARDED: shard_spec grid [0:0-7:1], shape [16, 1024], ROW_MAJOR
    - Mesh device: 4x8 grid (32 devices)
    - num_links: 4
    """
    torch.manual_seed(0)

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    # Set up sub-devices and semaphores for async operation
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # Create global semaphore handles (need 2 per iteration)
    ccl_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters * 2)
    ]
    # Create barrier semaphore
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    # Memory config - HEIGHT_SHARDED input, INTERLEAVED or WIDTH_SHARDED output
    # Input: shape=[1, 16384], shard_spec grid=[{x:0,y:0}-{x:1,y:1}], ROW_MAJOR orientation
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        [1, 16384],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )

    # Configure output memory config based on sharding type
    if output_sharding == "width_sharded":
        # WIDTH_SHARDED output: grid=[{x:0,y:0}-{x:7,y:1}], shape=[16, 1024], ROW_MAJOR
        output_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            [16, 1024],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            output_shard_spec,
        )
    else:
        # INTERLEAVED output
        output_mem_config = ttnn.L1_MEMORY_CONFIG

    # Create golden output tensor and input
    logger.info(f"Running all-gather test: {op_name}")
    logger.info(f"Running on {mesh_device.get_num_devices()} devices")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output shape: {output_shape}, dim: {dim}, num_devices: {num_devices}")

    output_tensor = torch.rand(output_shape, dtype=torch.bfloat16)

    # Create mesh tensor using MeshMapperConfig with PlacementShard on cluster_axis
    # This will automatically shard the tensor along the specified dimension
    # For 4x8 mesh with cluster_axis=1, we shard along columns (8 devices)
    # The dim=2 is the tensor dimension where all-gather happens
    input_tensor_mesh = ttnn.from_torch(
        output_tensor,
        device=mesh_device,
        layout=layout,
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(4, 8)),
        ),
    )

    all_gather_topology = topology
    profiler = BenchmarkProfiler()

    try:
        if trace_mode:
            # Run all-gather with trace
            tt_out_tensor = run_allgather_deepseek_with_trace(
                mesh_device,
                all_gather_topology,
                input_tensor_mesh,
                dim,
                cluster_axis,
                num_links,
                output_mem_config,
                ccl_semaphore_handles=ccl_semaphore_handles,
                barrier_semaphore=barrier_semaphore,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                subdevice_id=worker_sub_device_id,
                profiler=profiler,
            )
            tt_out_tensor_list = [tt_out_tensor]
        else:
            # Run without trace (for debugging)
            tt_out_tensor = ttnn.experimental.all_gather_async(
                input_tensor_mesh,
                dim,
                cluster_axis,
                mesh_device,
                all_gather_topology,
                [ccl_semaphore_handles[0], ccl_semaphore_handles[1]],
                num_links=num_links,
                memory_config=output_mem_config,
                use_broadcast=True,
                barrier_semaphore=barrier_semaphore,
                subdevice_id=worker_sub_device_id,
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_out_tensor_list = [tt_out_tensor]

        # Verify correctness
        logger.info("Verifying correctness")
        passed = True
        for tensor_index in range(len(tt_out_tensor_list)):
            tt_out_tensor = tt_out_tensor_list[tensor_index]
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                logger.info(f"Checking for device {t.device().id()}")

                eq, output = comp_equal(tt_output_tensor, output_tensor)
                if not eq:
                    logger.error(f"output mismatch for tensor {i}")
                    passed = False

        assert (
            mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
        ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

        assert passed, f"Output verification failed"

        logger.info("✓ Trace mode all-gather test passed with correct output")

    finally:
        # Clean up sub-device configuration
        mesh_device.reset_sub_device_stall_group()
