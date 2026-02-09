# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test

from tracy import signpost
from models.perf.benchmarking_utils import BenchmarkProfiler


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def run_with_trace(
    mesh_device,
    input_tensor_mesh,
    intermediate_tensor,
    cluster_axis,
    num_links,
    topology,
    num_iter=20,
    num_warmup_iter=15,
    subdevice_id=None,
    persistent_output_tensor=None,
):
    """Run the all-reduce operation with trace capture for performance testing."""
    profiler = BenchmarkProfiler()

    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.deepseek_minimal_all_reduce(
        input_tensor_mesh,
        cluster_axis=cluster_axis,
        intermediate_tensor=intermediate_tensor,
        persistent_output_tensor=persistent_output_tensor,
        num_links=num_links,
        topology=topology,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info("Capturing warmup trace")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_warmup_iter):
        tt_out_tensor = ttnn.experimental.deepseek_minimal_all_reduce(
            input_tensor_mesh,
            cluster_axis=cluster_axis,
            intermediate_tensor=intermediate_tensor,
            persistent_output_tensor=persistent_output_tensor,
            num_links=num_links,
            topology=topology,
        )
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.deepseek_minimal_all_reduce(
            input_tensor_mesh,
            cluster_axis=cluster_axis,
            intermediate_tensor=intermediate_tensor,
            persistent_output_tensor=persistent_output_tensor,
            num_links=num_links,
            topology=topology,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace...")
    profiler.start("deepseek-all-reduce-warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    ttnn.synchronize_device(mesh_device)
    profiler.end("deepseek-all-reduce-warmup")

    # Execute main trace with signposts for profiling
    logger.info("Starting Trace perf test...")
    signpost("start")
    profiler.start("deepseek-all-reduce-trace")

    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    profiler.end("deepseek-all-reduce-trace")
    signpost("stop")

    return tt_out_tensor


def run_with_trace_residual(
    mesh_device,
    input_tensor_mesh,
    intermediate_tensor,
    residual_tensor_mesh,
    cluster_axis,
    num_links,
    topology,
    num_iter=20,
    num_warmup_iter=15,
    subdevice_id=None,
    persistent_output_tensor=None,
):
    """Run the all-reduce operation with fused residual add and trace capture for performance testing."""
    profiler = BenchmarkProfiler()

    # Compile Run
    logger.info("Compiling model with residual")
    tt_out_tensor = ttnn.experimental.deepseek_minimal_all_reduce(
        input_tensor_mesh,
        cluster_axis=cluster_axis,
        intermediate_tensor=intermediate_tensor,
        residual_tensor=residual_tensor_mesh,
        persistent_output_tensor=persistent_output_tensor,
        num_links=num_links,
        topology=topology,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info("Capturing warmup trace with residual")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_warmup_iter):
        tt_out_tensor = ttnn.experimental.deepseek_minimal_all_reduce(
            input_tensor_mesh,
            cluster_axis=cluster_axis,
            intermediate_tensor=intermediate_tensor,
            residual_tensor=residual_tensor_mesh,
            persistent_output_tensor=persistent_output_tensor,
            num_links=num_links,
            topology=topology,
        )
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info("Capturing trace with residual")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.deepseek_minimal_all_reduce(
            input_tensor_mesh,
            cluster_axis=cluster_axis,
            intermediate_tensor=intermediate_tensor,
            residual_tensor=residual_tensor_mesh,
            persistent_output_tensor=persistent_output_tensor,
            num_links=num_links,
            topology=topology,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace with residual...")
    profiler.start("deepseek-all-reduce-residual-warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    ttnn.synchronize_device(mesh_device)
    profiler.end("deepseek-all-reduce-residual-warmup")

    # Execute main trace with signposts for profiling
    logger.info("Starting Trace perf test with residual...")
    signpost("start")
    profiler.start("deepseek-all-reduce-residual-trace")

    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    profiler.end("deepseek-all-reduce-residual-trace")
    signpost("stop")

    return tt_out_tensor


def run_deepseek_minimal_all_reduce_impl(
    mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    function_level_defaults,
    num_iters=1,
    trace_mode=False,
    rand_tensor=True,
    input_shard_shape=None,
    input_shard_grid=None,
    tensor_mem_layout=None,
    cluster_axis=None,
    mesh_mapper_config=None,
    topology=ttnn.Topology.Linear,
    input_tile_shape=(1, 32),
    intermediate_shape=None,
    intermediate_tile_shape=(32, 32),
    intermediate_shard_shape=None,
    use_persistent_buffers=True,
):
    if mesh_mapper_config is None:
        # Mesh shape is (num_devices, 1), so we shard along dimension 0 (batch) across the first mesh dimension
        mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], mesh_device.shape
        )
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    logger.info(f"Output shape: {output_shape}")
    logger.info(f"input_shard_shape: {input_shard_shape}")
    logger.info(f"input_shard_grid: {input_shard_grid}")
    logger.info(f"cluster_axis: {cluster_axis}")
    logger.info(f"num_devices: {num_devices}")

    # For sharded all reduce
    if not (bool(input_shard_shape) == bool(input_shard_grid) == bool(tensor_mem_layout)):
        pytest.fail(
            "Both input_shard_shape, shard_grid, and tensor_mem_layout must be provided together or all must be None"
        )

    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)

    input_tensor_mesh_list = []
    intermediate_tensor_list = []
    golden_output_list = []

    logger.info(f"Mesh shape: {mesh_device.shape}")

    for i in range(num_iters):
        # Create input tensors for each device in the reduction group
        device_tensors = []
        for device_idx in range(num_devices):
            if rand_tensor:
                tensor = torch.rand(output_shape, dtype=torch.bfloat16)
            else:
                tensor = torch.arange(
                    1 + device_idx * torch.prod(torch.tensor(output_shape)),
                    1 + (device_idx + 1) * torch.prod(torch.tensor(output_shape)),
                    dtype=torch.bfloat16,
                ).reshape(output_shape)
            device_tensors.append(tensor)

        # Golden output is the sum of all input tensors
        golden_output = torch.sum(torch.stack(device_tensors), dim=0)
        golden_output_list.append(golden_output)

        # Concatenate along dim 0 (batch) to form the mesh tensor
        mesh_tensor_torch = torch.cat(device_tensors, dim=0)

        input_tensor_mesh = ttnn.from_torch(
            mesh_tensor_torch,
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile(input_tile_shape),
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                mesh_mapper_config,
            ),
        )

        # Create intermediate tensor with standard 32x32 tiles
        # Use provided intermediate_shape or derive from output_shape
        actual_intermediate_shape = intermediate_shape if intermediate_shape else [32, 224]
        actual_intermediate_shard_shape = (
            intermediate_shard_shape if intermediate_shard_shape else tuple(actual_intermediate_shape)
        )
        intermediate_tensor_torch = torch.zeros(actual_intermediate_shape, dtype=torch.bfloat16)

        intermediate_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            actual_intermediate_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        intermediate_mem_config_32x32 = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=intermediate_shard_spec
        )

        intermediate_tensor = ttnn.from_torch(
            intermediate_tensor_torch,
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile(intermediate_tile_shape),
            dtype=input_dtype,
            memory_config=intermediate_mem_config_32x32,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                mesh_mapper_config,
            ),
        )
        intermediate_tensor_list.append(intermediate_tensor)

        # Create persistent output tensor with same shape and memory config as input
        # Must concatenate num_devices copies to match the mesh mapper config (PlacementShard(0))
        if use_persistent_buffers:
            output_tensor_per_device = torch.zeros(output_shape, dtype=torch.bfloat16)
            output_tensor_torch = torch.cat([output_tensor_per_device] * num_devices, dim=0)
            persistent_output_tensor = ttnn.from_torch(
                output_tensor_torch,
                device=mesh_device,
                layout=layout,
                tile=ttnn.Tile(input_tile_shape),
                dtype=input_dtype,
                memory_config=input_mem_config,
                mesh_mapper=ttnn.create_mesh_mapper(
                    mesh_device,
                    mesh_mapper_config,
                ),
            )
        else:
            persistent_output_tensor = None
        input_tensor_mesh_list.append((input_tensor_mesh, persistent_output_tensor))

    tt_out_tensor_list = []
    if trace_mode:
        input_tensor, persistent_output = input_tensor_mesh_list[0]
        tt_out_tensor = run_with_trace(
            mesh_device,
            input_tensor,
            intermediate_tensor_list[0],
            cluster_axis,
            num_links,
            topology,
            num_iter=num_iters,
            subdevice_id=worker_sub_device_id,
            persistent_output_tensor=persistent_output,
        )
        tt_out_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            logger.info(f"Running iteration {i}")
            input_tensor, persistent_output = input_tensor_mesh_list[i]
            tt_out_tensor = ttnn.experimental.deepseek_minimal_all_reduce(
                input_tensor,
                cluster_axis=cluster_axis,
                intermediate_tensor=intermediate_tensor_list[i],
                persistent_output_tensor=persistent_output,
                num_links=num_links,
                topology=topology,
            )
            tt_out_tensor_list.append(tt_out_tensor)

        logger.info("Waiting for op to complete")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info("Op completed")

    # Compare tensors
    for iter_idx in range(len(tt_out_tensor_list)):
        output_tensor_torch = ttnn.to_torch(
            tt_out_tensor_list[iter_idx],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        golden_output = golden_output_list[iter_idx]

        logger.info(f"Comparing iteration {iter_idx}")
        slice_size = output_shape[0]  # Batch dimension
        mismatch = False
        for device_idx in range(num_devices):
            start = device_idx * slice_size
            end = start + slice_size
            received = output_tensor_torch[start:end, :]
            logger.info(f"Device {device_idx} output shape: {received.shape}")
            logger.info(f"Golden output shape: {golden_output.shape}")

            eq, output = comp_pcc(received, golden_output)

            if not eq:
                logger.error(f"Output mismatch for device {device_idx}")
                mismatch = True
            else:
                logger.info(f"Output match for device {device_idx}")

        assert not mismatch, f"Iteration {iter_idx} FAILED: {output}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, input_shard_shape, input_shard_grid, tensor_mem_layout",
    [
        (
            2,
            2,
            [1, 7168],
            (1, 7168),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),  # single core
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("input_tile_shape", [(1, 32)])
@pytest.mark.parametrize("intermediate_shape", [[32, 224]])
@pytest.mark.parametrize("intermediate_tile_shape", [(32, 32)])
@pytest.mark.parametrize("intermediate_shard_shape", [(32, 224)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("use_persistent_buffers", [True, False], ids=["persistent_buffers", "no_persistent_buffers"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        },
    ],
    indirect=["device_params"],
    ids=["fabric_1d_trace"],
)
def test_deepseek_minimal_all_reduce_trace(
    bh_2d_mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    num_iters,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    tensor_mem_layout,
    cluster_axis,
    topology,
    input_tile_shape,
    intermediate_shape,
    intermediate_tile_shape,
    intermediate_shard_shape,
    use_persistent_buffers,
):
    """Trace-enabled version of the all-reduce test for performance testing."""
    # Validate we have the right mesh configuration
    validate_test(num_devices, topology, bh_2d_mesh_device.shape, cluster_axis)

    # Create a 2x1 submesh
    mesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(num_devices, 1))

    run_deepseek_minimal_all_reduce_impl(
        mesh_device,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        num_iters=num_iters,
        trace_mode=True,
        rand_tensor=True,
        input_shard_shape=input_shard_shape,
        input_shard_grid=input_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
        cluster_axis=cluster_axis,
        topology=topology,
        input_tile_shape=input_tile_shape,
        intermediate_shape=intermediate_shape,
        intermediate_tile_shape=intermediate_tile_shape,
        intermediate_shard_shape=intermediate_shard_shape,
        use_persistent_buffers=use_persistent_buffers,
    )


def run_deepseek_minimal_all_reduce_with_residual_impl(
    mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    function_level_defaults,
    num_iters=1,
    trace_mode=False,
    rand_tensor=True,
    input_shard_shape=None,
    input_shard_grid=None,
    tensor_mem_layout=None,
    cluster_axis=None,
    mesh_mapper_config=None,
    topology=ttnn.Topology.Linear,
    input_tile_shape=(1, 32),
    intermediate_shape=None,
    intermediate_tile_shape=(32, 32),
    intermediate_shard_shape=None,
    use_persistent_buffers=True,
):
    """Test implementation with fused residual add."""
    if mesh_mapper_config is None:
        mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], mesh_device.shape
        )
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    logger.info(f"Output shape: {output_shape}")
    logger.info(f"input_shard_shape: {input_shard_shape}")
    logger.info(f"cluster_axis: {cluster_axis}")
    logger.info(f"num_devices: {num_devices}")

    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)

    for i in range(num_iters):
        # Create input tensors for each device
        device_tensors = []
        residual_tensors = []
        for device_idx in range(num_devices):
            if rand_tensor:
                tensor = torch.rand(output_shape, dtype=torch.bfloat16)
                residual = torch.rand(output_shape, dtype=torch.bfloat16)
            else:
                tensor = torch.full(output_shape, float(device_idx + 1), dtype=torch.bfloat16)
                residual = torch.full(output_shape, float(device_idx + 10), dtype=torch.bfloat16)
            device_tensors.append(tensor)
            residual_tensors.append(residual)

        # Golden output: (sum of all input tensors) + residual for each device
        all_reduce_result = torch.sum(torch.stack(device_tensors), dim=0)

        # Concatenate tensors for mesh
        mesh_tensor_torch = torch.cat(device_tensors, dim=0)
        residual_mesh_torch = torch.cat(residual_tensors, dim=0)

        input_tensor_mesh = ttnn.from_torch(
            mesh_tensor_torch,
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile(input_tile_shape),
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
        )

        residual_tensor_mesh = ttnn.from_torch(
            residual_mesh_torch,
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile(input_tile_shape),
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
        )

        # Create intermediate tensor with standard 32x32 tiles
        actual_intermediate_shape = intermediate_shape if intermediate_shape else [32, 224]
        actual_intermediate_shard_shape = (
            intermediate_shard_shape if intermediate_shard_shape else tuple(actual_intermediate_shape)
        )
        intermediate_tensor_torch = torch.zeros(actual_intermediate_shape, dtype=torch.bfloat16)

        intermediate_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            actual_intermediate_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        intermediate_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=intermediate_shard_spec
        )

        intermediate_tensor = ttnn.from_torch(
            intermediate_tensor_torch,
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile(intermediate_tile_shape),
            dtype=input_dtype,
            memory_config=intermediate_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
        )

        if use_persistent_buffers:
            output_tensor_per_device = torch.zeros(output_shape, dtype=torch.bfloat16)
            output_tensor_torch = torch.cat([output_tensor_per_device] * num_devices, dim=0)
            persistent_output_tensor = ttnn.from_torch(
                output_tensor_torch,
                device=mesh_device,
                layout=layout,
                tile=ttnn.Tile(input_tile_shape),
                dtype=input_dtype,
                memory_config=input_mem_config,
                mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper_config),
            )
        else:
            persistent_output_tensor = None

        if trace_mode and i == 0:
            # For trace mode, run with trace on first iteration only
            tt_out_tensor = run_with_trace_residual(
                mesh_device,
                input_tensor_mesh,
                intermediate_tensor,
                residual_tensor_mesh,
                cluster_axis,
                num_links,
                topology,
                num_iter=num_iters,
                subdevice_id=worker_sub_device_id,
                persistent_output_tensor=persistent_output_tensor,
            )
        elif not trace_mode:
            logger.info(f"Running iteration {i} with residual add")
            tt_out_tensor = ttnn.experimental.deepseek_minimal_all_reduce(
                input_tensor_mesh,
                cluster_axis=cluster_axis,
                intermediate_tensor=intermediate_tensor,
                residual_tensor=residual_tensor_mesh,
                persistent_output_tensor=persistent_output_tensor,
                num_links=num_links,
                topology=topology,
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        else:
            # Skip remaining iterations in trace mode (trace handles them)
            break

        # Compare output
        output_tensor_torch = ttnn.to_torch(
            tt_out_tensor,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )

        slice_size = output_shape[0]
        mismatch = False
        for device_idx in range(num_devices):
            start = device_idx * slice_size
            end = start + slice_size
            received = output_tensor_torch[start:end, :]
            # Golden: all_reduce_result + residual for this device
            golden = all_reduce_result + residual_tensors[device_idx]

            logger.info(f"Device {device_idx} output shape: {received.shape}")
            logger.info(f"Golden output shape: {golden.shape}")

            eq, output = comp_pcc(received, golden)

            if not eq:
                logger.error(f"Output mismatch for device {device_idx}")
                mismatch = True
            else:
                logger.info(f"Output match for device {device_idx}")

        assert not mismatch, f"Iteration {i} FAILED: {output}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, input_shard_shape, input_shard_grid, tensor_mem_layout",
    [
        (
            2,
            2,
            [1, 7168],
            (1, 7168),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("input_tile_shape", [(1, 32)])
@pytest.mark.parametrize("intermediate_shape", [[32, 224]])
@pytest.mark.parametrize("intermediate_tile_shape", [(32, 32)])
@pytest.mark.parametrize("intermediate_shard_shape", [(32, 224)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("use_persistent_buffers", [True, False], ids=["persistent_buffers", "no_persistent_buffers"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        },
    ],
    indirect=["device_params"],
    ids=["fabric_2d_trace"],
)
def test_deepseek_minimal_all_reduce_with_residual_trace(
    bh_2d_mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    num_iters,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    tensor_mem_layout,
    cluster_axis,
    topology,
    input_tile_shape,
    intermediate_shape,
    intermediate_tile_shape,
    intermediate_shard_shape,
    use_persistent_buffers,
):
    """Trace-enabled test for all-reduce with fused residual add."""
    validate_test(num_devices, topology, bh_2d_mesh_device.shape, cluster_axis)

    mesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(num_devices, 1))

    run_deepseek_minimal_all_reduce_with_residual_impl(
        mesh_device,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        num_iters=num_iters,
        trace_mode=True,
        rand_tensor=True,
        input_shard_shape=input_shard_shape,
        input_shard_grid=input_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
        cluster_axis=cluster_axis,
        topology=topology,
        input_tile_shape=input_tile_shape,
        intermediate_shape=intermediate_shape,
        intermediate_tile_shape=intermediate_tile_shape,
        intermediate_shard_shape=intermediate_shard_shape,
        use_persistent_buffers=use_persistent_buffers,
    )
