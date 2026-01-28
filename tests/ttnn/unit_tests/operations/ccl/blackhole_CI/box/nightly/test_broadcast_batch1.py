# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def run_with_trace(
    mesh_device,
    sender_coord,
    broadcast_topology,
    input_tensor_mesh,
    num_links,
    output_mem_config,
    num_iter=20,
    subdevice_id=None,
    cluster_axis=None,
    secondary_cluster_axis=None,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.deepseek_minimal_broadcast(
        input_tensor_mesh,
        sender_coord=sender_coord,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=broadcast_topology,
        subdevice_id=subdevice_id,
        cluster_axis=cluster_axis,
        secondary_cluster_axis=secondary_cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.deepseek_minimal_broadcast(
            input_tensor_mesh,
            sender_coord=sender_coord,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=broadcast_topology,
            subdevice_id=subdevice_id,
            cluster_axis=cluster_axis,
            secondary_cluster_axis=secondary_cluster_axis,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return tt_out_tensor


def run_broadcast_impl(
    mesh_device,
    sender_coord,
    sender_coord_tuple,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    function_level_defaults,
    broadcast_topology,
    num_iters=1,
    trace_mode=True,
    rand_tensor=True,
    mem_config=None,
    input_shard_shape=None,
    input_shard_grid=None,
    output_shard_shape=None,
    output_shard_grid=None,
    tensor_mem_layout=None,
    cluster_axis=None,
    mesh_mapper_config=None,
    secondary_cluster_axis=None,
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

    ### For sharded all broadcast only
    if not (bool(input_shard_shape) == bool(input_shard_grid) == bool(tensor_mem_layout)):
        pytest.fail(
            "Both input_shard_shape, shard_grid, and tensor_mem_layout must be provided together or all must be None"
        )
    if input_shard_shape and input_shard_grid:
        input_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec
        )
        if output_shard_shape is None:
            assert (
                output_shard_grid is None
            ), "output_shard_grid must not be provided if output_shard_shape is not provided"
            output_shard_shape = input_shard_shape
            output_shard_spec = ttnn.ShardSpec(
                input_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
        else:
            assert output_shard_grid is not None, "output_shard_grid must be provided if output_shard_shape is provided"
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
    else:
        assert mem_config is not None
        input_mem_config = mem_config
        output_mem_config = mem_config
    ###

    input_tensor_mesh_list = []
    sender_tensor_list = []

    for i in range(num_iters):
        # Create sender tensor
        if rand_tensor:
            sender_tensor = torch.rand(output_shape, dtype=torch.bfloat16)
        else:
            sender_tensor = torch.arange(1, 1 + torch.prod(torch.tensor(output_shape)), dtype=torch.bfloat16).reshape(
                output_shape
            )
        sender_tensor_list.append(sender_tensor)

        # Create mesh tensor with sender's tensor at sender_coord, zeros elsewhere
        device_tensors = []
        for device_idx in range(num_devices):
            if device_idx == sender_coord_tuple[0]:
                device_tensors.append(sender_tensor)
            else:
                device_tensors.append(torch.zeros_like(sender_tensor))
        # Concatenate along dim 0 (batch) to form the mesh tensor
        mesh_tensor_torch = torch.cat(device_tensors, dim=0)
        input_tensor_mesh = ttnn.from_torch(
            mesh_tensor_torch,
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile((1, 32)),
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                mesh_mapper_config,
            ),
        )
        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor = run_with_trace(
            mesh_device,
            sender_coord,
            broadcast_topology,
            input_tensor_mesh_list[0],
            num_links,
            output_mem_config,
            num_iter=num_iters,
            subdevice_id=worker_sub_device_id,
            cluster_axis=cluster_axis,
            secondary_cluster_axis=secondary_cluster_axis,
        )
        tt_out_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            tt_out_tensors = ttnn.experimental.deepseek_minimal_broadcast(
                input_tensor_mesh_list[i],
                sender_coord=sender_coord,
                num_links=num_links,
                memory_config=output_mem_config,
                topology=broadcast_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                secondary_cluster_axis=secondary_cluster_axis,
            )
            tt_out_tensor_list.append(tt_out_tensors)

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    # compare tensors
    for iter_idx in range(len(tt_out_tensor_list)):
        output_tensor_torch = ttnn.to_torch(
            tt_out_tensor_list[iter_idx],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),  # Concatenate along batch dimension
        )
        sender_tensor = sender_tensor_list[iter_idx]
        slice_size = output_shape[0]  # Batch dimension
        for i in range(num_devices):
            start = i * slice_size
            end = start + slice_size
            # Slice along dimension 0 (batch)
            received = output_tensor_torch[start:end, :]
            assert (
                received.shape == sender_tensor.shape
            ), f"Shape mismatch: received {received.shape}, expected {sender_tensor.shape}"
            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(received, sender_tensor)
            else:
                eq, output = comp_pcc(received, sender_tensor)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
                assert eq, f"{i} FAILED: {output}"

    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_or_less_dev(3)
@pytest.mark.parametrize(
    "num_devices, num_links, sender_idx, output_shape, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        (
            4,
            1,
            1,
            [1, 7168],
            (1, 7168),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            None,
            None,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(15232),
                "trace_region_size": 573440,
            }
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_1d_linear_trace"],
)
def test_broadcast_batch1(
    bh_2d_mesh_device,
    num_devices,
    sender_idx,
    output_shape,
    num_links,
    input_dtype,
    layout,
    num_iters,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    tensor_mem_layout,
    topology,
):
    validate_test(num_devices, topology, bh_2d_mesh_device.shape, 0)
    mesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    sender_coord_tuple = (sender_idx, 0)
    sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)

    run_broadcast_impl(
        mesh_device,
        sender_coord,
        sender_coord_tuple,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        cluster_axis=0,
        broadcast_topology=topology,
        num_iters=num_iters,
        rand_tensor=True,
        input_shard_shape=input_shard_shape,
        input_shard_grid=input_shard_grid,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
    )


@skip_for_wormhole_b0("This test is for blackhole")
@skip_for_n_or_less_dev(7)  # Need 8 devices for 4x2 mesh
@pytest.mark.parametrize(
    "mesh_rows, mesh_cols, num_links, sender_row, sender_col, output_shape, input_shard_shape, input_shard_grid, tensor_mem_layout",
    [
        (
            4,
            2,
            1,
            1,
            0,
            [1, 7168],
            (1, 7168),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize("use_persistent_buffer", [False, True], ids=["no_persistent", "with_persistent"])
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(15232),
                "trace_region_size": 573440,
            }
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_2d_dual_axis_trace"],
)
def test_broadcast_batch1_dual_axis(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    sender_row,
    sender_col,
    output_shape,
    num_links,
    input_dtype,
    layout,
    num_iters,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    tensor_mem_layout,
    topology,
    use_persistent_buffer,
):
    """
    Test dual-axis broadcast on a 2D mesh.
    Sender at (sender_row, sender_col) broadcasts:
    1. First across secondary_cluster_axis (axis 1) to the device at same row, different column
    2. Then both sender and secondary sender broadcast along cluster_axis (axis 0) to all devices in their columns

    When use_persistent_buffer=True, a pre-allocated output buffer is used and barrier
    synchronization is skipped.
    """
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")
    mesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    sender_coord_tuple = (sender_row, sender_col)
    sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)

    # For 2D mesh, use a mesh mapper that shards along row (axis 0) and replicates along column (axis 1)
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], mesh_device.shape)

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

    # Set up sharded memory config
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_mem_config = input_mem_config

    sender_tensor = torch.rand(output_shape, dtype=torch.bfloat16)

    device_tensors = []
    for row in range(mesh_rows):
        if row == sender_row:
            device_tensors.append(sender_tensor)
        else:
            device_tensors.append(torch.zeros_like(sender_tensor))

    # Concatenate along dim 0 (batch) - this gives [mesh_rows, 7168]
    mesh_tensor_torch = torch.cat(device_tensors, dim=0)
    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=mesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            mesh_mapper_config,
        ),
    )

    # Create persistent output buffer if requested
    persistent_output_buffer = None
    if use_persistent_buffer:
        logger.info("Creating persistent output buffer...")
        persistent_output_buffer = ttnn.from_torch(
            torch.zeros(output_shape, dtype=torch.bfloat16),
            device=mesh_device,
            layout=layout,
            tile=ttnn.Tile((1, 32)),
            dtype=input_dtype,
            memory_config=output_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Run once to compile
    logger.info(f"Running dual-axis broadcast (compiling)... persistent_buffer={use_persistent_buffer}")
    tt_out_tensor = ttnn.experimental.deepseek_minimal_broadcast(
        input_tensor_mesh,
        sender_coord=sender_coord,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=topology,
        subdevice_id=worker_sub_device_id,
        cluster_axis=0,
        secondary_cluster_axis=1,
        persistent_output_buffer=persistent_output_buffer,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iters):
        tt_out_tensor = ttnn.experimental.deepseek_minimal_broadcast(
            input_tensor_mesh,
            sender_coord=sender_coord,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=topology,
            subdevice_id=worker_sub_device_id,
            cluster_axis=0,
            secondary_cluster_axis=1,
            persistent_output_buffer=persistent_output_buffer,
        )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute trace
    logger.info("Executing trace...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    logger.info("Done trace execution")

    # Verify that the output tensor uses the persistent buffer address when applicable
    if use_persistent_buffer:
        persistent_output_tensors = ttnn.get_device_tensors(persistent_output_buffer)
        output_tensors = ttnn.get_device_tensors(tt_out_tensor)
        for persistent_tensor, output_tensor in zip(persistent_output_tensors, output_tensors):
            assert (
                persistent_tensor.buffer_address() == output_tensor.buffer_address()
            ), "Persistent tensor address mismatch - output should reuse persistent buffer"

    # Compare tensors - all devices should have the sender's data
    # ConcatMeshToTensor concatenates all num_devices tensors along dim 0
    output_tensor_torch = ttnn.to_torch(
        tt_out_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    slice_size = output_shape[0]  # Batch dimension
    # Validate ALL devices (mesh_rows * mesh_cols)
    for device_idx in range(num_devices):
        start = device_idx * slice_size
        end = start + slice_size
        received = output_tensor_torch[start:end, :]
        assert (
            received.shape == sender_tensor.shape
        ), f"Shape mismatch at device {device_idx}: received {received.shape}, expected {sender_tensor.shape}"
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(received, sender_tensor)
        else:
            eq, output = comp_pcc(received, sender_tensor)
        if not eq:
            logger.error(f"output mismatch for device {device_idx}")
            assert eq, f"Device {device_idx} FAILED: {output}"

    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
