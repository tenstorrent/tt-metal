# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


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
    tt_out_tensor = ttnn.broadcast(
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
        tt_out_tensor = ttnn.broadcast(
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
    trace_mode=False,
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
        mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)], ttnn.MeshShape(1, num_devices)
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
    if bool(input_shard_shape) != bool(input_shard_grid) and bool(tensor_mem_layout) != bool(input_shard_grid):
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
            if device_idx == sender_coord_tuple[cluster_axis]:
                device_tensors.append(sender_tensor)
            else:
                device_tensors.append(torch.zeros_like(sender_tensor))
        # Concatenate along cluster_axis to form the mesh tensor
        mesh_tensor_torch = torch.cat(device_tensors, dim=-1)
        input_tensor_mesh = ttnn.from_torch(
            mesh_tensor_torch,
            device=mesh_device,
            layout=layout,
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
            tt_out_tensors = ttnn.broadcast(
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

    passed = True
    # compare tensors
    for iter_idx in range(len(tt_out_tensor_list)):
        output_tensor_torch = ttnn.to_torch(
            tt_out_tensor_list[iter_idx],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=cluster_axis),
        )
        sender_tensor = sender_tensor_list[iter_idx]
        slice_size = output_shape[cluster_axis]
        for i in range(num_devices):
            start = i * slice_size
            end = start + slice_size
            # Build slice for all dimensions
            slices = [slice(None)] * output_tensor_torch.dim()
            slices[cluster_axis] = slice(start, end)
            received = output_tensor_torch[tuple(slices)]
            assert (
                received.shape == sender_tensor.shape
            ), f"Shape mismatch: received {received.shape}, expected {sender_tensor.shape}"
            if input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(received, sender_tensor)
            else:
                eq, output = comp_pcc(received, sender_tensor)
            if not eq:
                logger.error(f"output mismatch for tensor {i}")
                passed = False
                assert eq, f"{i} FAILED: {output}"

    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    if not passed:
        assert eq, f"{i} FAILED: {output}"


# Enumerate the post-commit cases explicitly
@pytest.mark.parametrize(
    "num_devices, sender_idx, num_links, output_shape, layout, input_dtype, mem_config",
    [
        (2, 0, 1, [32, 32], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        (
            2,
            1,
            1,
            [3, 121, 2042],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ),
        (
            4,
            2,
            1,
            [1, 1, 32, 1024],
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ),
        (4, 3, 1, [2, 64, 512], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)),
        (8, 4, 1, [256, 3328], ttnn.TILE_LAYOUT, ttnn.bfloat8_b, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        (
            4,
            0,
            1,
            [1, 69, 4000],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ),
        (
            8,
            5,
            1,
            [10, 8320],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ),
        (
            2,
            1,
            1,
            [1, 10, 32784],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ),
        (
            4,
            1,
            1,
            [1, 2, 16300],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ),
    ],
    ids=[
        "2-dev-DRAM",
        "2-dev-L1",
        "4-dev-DRAM",
        "4-dev-L1",
        "8-dev-DRAM",
        "4-dev-L1-2",
        "8-dev-DRAM-2",
        "2-dev-L1-2",
        "4-dev-DRAM-3",
    ],
)
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_broadcast(
    t3k_mesh_device,
    output_shape,
    num_devices,
    sender_idx,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    mesh_device = t3k_mesh_device
    mesh_shape = tuple(mesh_device.shape)
    sender_coord_tuple = (0, sender_idx)
    sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)

    run_broadcast_impl(
        t3k_mesh_device,
        sender_coord,
        sender_coord_tuple,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
        cluster_axis=1,
    )


# Enumerate the post-commit cases explicitly
@pytest.mark.parametrize(
    "num_devices, sender_idx, num_links, output_shape, layout, input_dtype",
    [
        (4, 2, 1, [256, 3328], ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (2, 0, 1, [1, 69, 4000], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 10000}], indirect=True
)
def test_broadcast_trace(
    t3k_mesh_device,
    sender_idx,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    mesh_device = t3k_mesh_device
    mesh_shape = tuple(mesh_device.shape)
    sender_coord_tuple = (0, sender_idx)
    sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)

    run_broadcast_impl(
        t3k_mesh_device,
        sender_coord,
        sender_coord_tuple,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        cluster_axis=1,
        broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
        trace_mode=True,
    )


@pytest.mark.parametrize(
    "num_devices, sender_idx, output_shape, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        (
            4,
            2,
            [2, 32, 256],
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            None,
            None,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            4,
            3,
            [192, 64],
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 1))}),
            None,
            None,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            2,
            0,
            [2, 3, 64, 1024],
            (384, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 1)),
                }
            ),
            None,
            None,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            8,
            1,
            [768, 32],
            (32, 32),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 1), ttnn.CoreCoord(6, 2)),
                }
            ),
            None,
            None,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            4,
            0,
            [2, 4, 32, 256],
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            None,
            None,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
        (
            4,
            2,
            [1, 1, 32, 32768],
            (32, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))}),
            None,
            None,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_broadcast_sharded(
    t3k_mesh_device,
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
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")
    mesh_device = t3k_mesh_device
    mesh_shape = tuple(mesh_device.shape)
    sender_coord_tuple = (0, sender_idx)
    sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)

    run_broadcast_impl(
        t3k_mesh_device,
        sender_coord,
        sender_coord_tuple,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        cluster_axis=1,
        broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        input_shard_shape=input_shard_shape,
        input_shard_grid=input_shard_grid,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
    )


@pytest.mark.parametrize(
    "num_devices, sender_idx, output_shape, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        (
            4,
            1,
            [2, 4, 32, 256],
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            None,
            None,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ),
    ],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_broadcast_sharded_2x4(
    mesh_device,
    sender_idx,
    num_devices,
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
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    mesh_shape = tuple(submesh_device.shape)
    sender_coord_tuple = (0, sender_idx)
    sender_coord = ttnn.MeshCoordinate(sender_coord_tuple)
    run_broadcast_impl(
        submesh_device,
        sender_coord,
        sender_coord_tuple,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        cluster_axis=1,
        broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        input_shard_shape=input_shard_shape,
        input_shard_grid=input_shard_grid,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
    )


# Enumerate the post-commit cases explicitly
@pytest.mark.parametrize(
    "num_devices, sender_idx, num_links, output_shape, layout, input_dtype, mem_config",
    [
        (4, 0, 1, [64, 32], ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)),
        (
            2,
            1,
            1,
            [2, 90, 1042],
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ),
    ],
    ids=["4-dev-DRAM", "2-dev-L1"],
)
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
def test_broadcast_2d(
    mesh_device,
    mesh_shape,
    output_shape,
    num_devices,
    sender_idx,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        pytest.skip("bfloat8_b not supported for row-major")

    sender_coord_tuple = (0, sender_idx)
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
        broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
        cluster_axis=1,
    )
