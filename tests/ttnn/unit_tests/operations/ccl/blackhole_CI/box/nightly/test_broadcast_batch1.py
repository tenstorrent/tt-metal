# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


def run_with_trace(
    mesh_device,
    sender_coord,
    broadcast_topology,
    input_tensor_mesh,
    num_links,
    output_mem_config,
    num_iter=20,
    subdevice_id=None,
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
            [1, 1536],
            (1, 1536),
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
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 217872}),
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
):
    validate_test(num_devices, ttnn.Topology.Linear, bh_2d_mesh_device.shape, 0)
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
        broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        input_shard_shape=input_shard_shape,
        input_shard_grid=input_shard_grid,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
    )
