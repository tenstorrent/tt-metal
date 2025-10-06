# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from models.common.utility_functions import skip_for_blackhole


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(3)]
    return ccl_semaphore_handles


def run_reduce_scatter_impl(
    mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    num_iters=1,
    enable_trace=True,
    ones_tensor=False,
    mem_config_intermediate=None,
    cluster_axis=None,
    use_barrier=False,
    use_persistent_buffers=True,
    chunks_per_sync=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    verify_output=True,
    use_new=False,
):
    torch.manual_seed(0)

    tile = (32, 32)

    ##### Fabric setup #####
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

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    barrier_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    intermediate_shape = rs_input_shape[:]
    if rs_topology == ttnn.Topology.Linear:
        # Line RS requires double-sized input for forward/backward
        intermediate_shape.insert(0, 2)
    if use_persistent_buffers:
        persistent_intermediate_buffers = [
            ttnn.from_torch(
                torch.zeros(intermediate_shape),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=rs_input_dtype,
                memory_config=mem_config_intermediate,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(num_iters)
        ]
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[dim] //= num_devices
    if use_persistent_buffers:
        persistent_output_buffers = [
            ttnn.from_torch(
                torch.zeros(rs_output_shape),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=rs_input_dtype,
                memory_config=mem_config_rs,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(num_iters)
        ]

    logger.info("Done creating persistent buffers")

    ##### All gather input setup #####
    logger.info(f"Reduce scatter shape: {rs_input_shape}")
    logger.info(f"Reduce scatter dim: {dim}")
    logger.info(f"input mem config: {mem_config_input}")
    logger.info(f"Reduce input mem config: {mem_config_rs}")
    logger.info(f"intermediate mem config: {mem_config_intermediate}")
    logger.info(f"topology: {rs_topology}")

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        rs_global_input_shape = rs_input_shape[:]
        rs_global_input_shape[dim] *= num_devices
        if ones_tensor:
            rs_input_tensor = torch.ones(rs_global_input_shape).bfloat16()
        else:
            rs_input_tensor = torch.rand(rs_global_input_shape).bfloat16()
        input_tensors = torch.chunk(rs_input_tensor, num_devices, dim)
        torch_input_tensor_list.append(input_tensors)

        input_tensor_mesh = ttnn.from_torch(
            rs_input_tensor,
            device=mesh_device,
            layout=layout,
            dtype=rs_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, num_devices)
                ),
            ),
        )

        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform torch ops #####
    torch_reduce_scatter_output_list = []
    for i in range(num_iters):
        reduce_output = torch.sum(torch.stack(torch_input_tensor_list[i]), dim=0)
        scatter_output = torch.chunk(reduce_output, num_devices, dim)
        torch_reduce_scatter_output_list.append(scatter_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []

    def run_op(i):
        if use_new:
            logger.info(f"Using new reduce scatter")
            tt_reduce_scatter_output_tensor = ttnn.reduce_scatter(
                tt_input_tensor_mesh_list[i],
                dim=dim,
                num_links=num_links,
                memory_config=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
            )
        else:
            logger.info(f"Using experimental reduce scatter")
            tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                tt_input_tensor_mesh_list[i],
                persistent_output_buffers=[persistent_intermediate_buffers[i], persistent_output_buffers[i]]
                if use_persistent_buffers
                else None,
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                num_links=num_links,
                memory_config=mem_config_rs,
                intermediate_memory_config=mem_config_intermediate,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )

        return tt_reduce_scatter_output_tensor

    if enable_trace:
        # Compile the op
        tt_reduce_scatter_output_trace_list = []
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_trace_list.append(tt_reduce_scatter_output_tensor)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        for tt_tensor in tt_reduce_scatter_output_trace_list:
            tt_rs_out = ttnn.from_device(tt_tensor)
            tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
            tt_tensor.deallocate(True)
            tt_reduce_scatter_output_list.append(tt_rs_out)
    else:
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_rs_out = ttnn.from_device(tt_reduce_scatter_output_tensor)
            tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
            tt_reduce_scatter_output_tensor.deallocate(True)
            tt_reduce_scatter_output_list.append(tt_rs_out)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    if verify_output:
        for i in range(num_iters):
            tt_rs_out = tt_reduce_scatter_output_list[i]
            torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

            torch_rs_out = torch.cat(torch_rs_out_tensor, dim)

            if ones_tensor:
                eq, output = comp_equal(tt_rs_out, torch_rs_out)
            else:
                eq, output = comp_pcc(tt_rs_out, torch_rs_out)

            logger.info(f"{output}, iteration {i}")
            assert eq, f"{i} FAILED ag: {output}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout, rs_input_dtype, use_new",
    [
        # Dim 1 tests
        ([2, 24, 256, 256], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        ([2, 16, 56, 56], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        ([2, 8, 512, 512], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),  # use batching when fused
        # Dim 2 tests
        ([2, 4, 1024, 1024], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        ([4, 1, 1024, 340], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),  # use batching when fused
        ([1, 1, 512, 512], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        # Dim 3 tests
        ([2, 4, 1024, 1024], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        ([1, 1, 13, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),  # use batching when fused
        ([3, 1, 41, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        ([8, 1, 512, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),  # use batching when fused
        ([4, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        ([1, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),  # use batching when fused
        ([1, 1, 352, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        ([2, 1, 2048, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),  # use batching when fused
        ([1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        # Composite-RS tests
        ([1, 1, 1, 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),
        ([2, 32, 2048, 64], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
        ([1, 1, 1, 16], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, False),
        ([1, 1, 29, 32], 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, True),
    ],
    ids=[
        "scatter_dim_1_test_one",
        "scatter_dim_1_test_two",
        "scatter_dim_1_test_three",
        "scatter_dim_2_test_one",
        "scatter_dim_2_test_two",
        "scatter_dim_2_test_three",
        "non_zero_dim_1",
        "padded_dim_2_test_one",
        "padded_dim_2_test_two",
        "batch_8",
        "batch_4",
        "batch_1_sd35_spatial",
        "batch_1_sd35_prompt",
        "batch_2",
        "batch_1",
        "composite_rs_test_one",
        "composite_rs_test_two",
        "composite_rs_test_three",
        "composite_rs_test_four",
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
    "ones_tensor",
    [
        True,
        False,
    ],
    ids=["ones", "random"],
)
@pytest.mark.parametrize(
    "use_barrier, use_persistent_buffers",
    [
        (True, True),
        (True, False),
        (False, True),
    ],
    ids=["barrier_with_persistent_buffers", "barrier_without_persistent_buffers", "no_barrier_with_persistent_buffers"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_reduce_scatter_async(
    mesh_device,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    ones_tensor,
    use_barrier,
    use_persistent_buffers,
    rs_topology,
    use_new,
):
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        ones_tensor=ones_tensor,
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
        use_new=use_new,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout, rs_input_dtype, use_new",
    [
        # Scatter on dim 0
        ([16, 1, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
        ([16, 16, 128, 128], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
        ([8, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),
        # Scatter on dim 1
        ([1, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
        ([16, 16, 128, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
        ([16, 8, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),
        # Scatter on dim 2
        ([1, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
        ([16, 1, 512, 128], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
        ([16, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),
        # Scatter on dim 3
        ([1, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
        ([16, 1, 128, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),
        ([16, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),
    ],
    ids=[
        "tt_training_test_one",
        "tt_training_test_two",
        "tt_training_test_three",
        "tt_training_test_four",
        "tt_training_test_five",
        "tt_training_test_six",
        "tt_training_test_seven",
        "tt_training_test_eight",
        "tt_training_test_nine",
        "tt_training_test_ten",
        "tt_training_test_eleven",
        "tt_training_test_twelve",
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
    "enable_trace,num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "ones_tensor",
    [
        True,
        False,
    ],
    ids=["ones", "random"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_reduce_scatter_async_training_shapes(
    mesh_device,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    rs_topology,
    num_iters,
    ones_tensor,
    use_new,
):
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        ones_tensor=ones_tensor,
        use_barrier=True,
        use_persistent_buffers=False,
        use_new=use_new,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "layout, rs_input_dtype",
    [
        (ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, intermediate_shard_shape, intermediate_shard_grid, intermediate_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout, buffer_type, use_new",
    [
        (
            [1, 1, 32, 3072],
            3,
            [32, 512],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [32, 512],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            True,
        ),
        (
            [4, 1, 384, 1024],
            3,
            [256, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [256, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [256, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            False,
        ),
        (
            [4, 1, 384, 3072],
            3,
            [256, 3072],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [1536, 512],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [1536, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            False,
        ),
        # Composite RS
        (
            [1, 1, 384, 240],
            3,
            [64, 256],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [64, 256],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [64, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            False,
        ),
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_reduce_scatter_async_sharded_to_sharded(
    mesh_device,
    num_links,
    rs_input_dtype,
    layout,
    rs_input_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    intermediate_shard_shape,
    intermediate_shard_grid,
    intermediate_mem_layout,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    buffer_type,
    enable_trace,
    num_iters,
    rs_topology,
    use_new,
):
    adjusted_intermediate_shard_shape = intermediate_shard_shape[:]
    if rs_topology == ttnn.Topology.Linear:
        adjusted_intermediate_shard_shape[0] *= 2

    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    intermediate_shard_spec = ttnn.ShardSpec(
        intermediate_shard_grid,
        adjusted_intermediate_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(input_mem_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    mem_config_intermediate = ttnn.MemoryConfig(
        intermediate_mem_layout, buffer_type=buffer_type, shard_spec=intermediate_shard_spec
    )
    mem_config_rs = ttnn.MemoryConfig(output_mem_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)

    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        mem_config_intermediate=mem_config_intermediate,
        use_new=use_new,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "layout, rs_input_dtype",
    [
        (ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, intermediate_shard_shape, intermediate_shard_grid, intermediate_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout, buffer_type, use_new",
    [
        (
            [4, 1, 256, 3072],
            3,
            [1024, 512],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [1024, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            False,
        ),
        (
            [4, 1, 384, 1024],
            3,
            [256, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [256, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            False,
        ),
        # Composite RS
        (
            [1, 1, 384, 240],
            3,
            [64, 256],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [64, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            True,
        ),
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_reduce_scatter_async_interleaved_to_sharded(
    mesh_device,
    num_links,
    rs_input_dtype,
    layout,
    rs_input_shape,
    dim,
    intermediate_shard_shape,
    intermediate_shard_grid,
    intermediate_mem_layout,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    buffer_type,
    enable_trace,
    num_iters,
    rs_topology,
    use_new,
):
    adjusted_intermediate_shard_shape = intermediate_shard_shape[:]
    if rs_topology == ttnn.Topology.Linear:
        adjusted_intermediate_shard_shape[0] *= 2

    intermediate_shard_spec = ttnn.ShardSpec(
        intermediate_shard_grid,
        adjusted_intermediate_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
    mem_config_intermediate = ttnn.MemoryConfig(
        intermediate_mem_layout, buffer_type=buffer_type, shard_spec=intermediate_shard_spec
    )
    mem_config_rs = ttnn.MemoryConfig(output_mem_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)

    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        mem_config_intermediate=mem_config_intermediate,
        use_new=use_new,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "layout, rs_input_dtype",
    [
        (ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, buffer_type, use_new",
    [
        (
            [4, 1, 256, 3072],
            3,
            [1024, 512],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            False,
        ),
        (
            [4, 1, 384, 1024],
            3,
            [256, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            True,
        ),
        # Composite RS
        (
            [1, 1, 384, 240],
            3,
            [64, 256],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            False,
        ),
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_reduce_scatter_async_sharded_to_interleaved(
    mesh_device,
    num_links,
    rs_input_dtype,
    layout,
    rs_input_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    buffer_type,
    enable_trace,
    num_iters,
    rs_topology,
    use_new,
):
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(input_mem_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    mem_config_intermediate = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
    mem_config_rs = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        mem_config_intermediate=mem_config_intermediate,
        use_new=use_new,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout, rs_input_dtype",
    [
        ([1, 1, 8, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
    ],
    ids=[
        "deepseek_rs",
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
        (False, 3),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "use_barrier, use_persistent_buffers",
    [
        (True, False),
    ],
    ids=["barrier_with_no_persistent_buffers"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("cluster_axis", [1], ids=["cluster_axis_1"])
def test_reduce_scatter_async_2x4(
    mesh_device,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    use_barrier,
    use_persistent_buffers,
    rs_topology,
    cluster_axis,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 4)))
    run_reduce_scatter_impl(
        submesh_device,
        submesh_device.get_num_devices(),
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
        ones_tensor=False,
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
        cluster_axis=cluster_axis,
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("input_shard_grid", [(7, 8)])
@pytest.mark.parametrize("input_shard_shape", [(32, 128)])
@pytest.mark.parametrize("output_shard_grid", [(4, 4)])
@pytest.mark.parametrize("output_shard_shape", [(32, 128)])
@pytest.mark.parametrize("input_mem_layout", [ttnn.TensorMemoryLayout.WIDTH_SHARDED])
@pytest.mark.parametrize("output_mem_layout", [ttnn.TensorMemoryLayout.WIDTH_SHARDED])
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1])
@pytest.mark.parametrize("rs_input_shape", [[1, 1, 32, 7168]])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("rs_input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("cluster_axis", [1])
def test_reduce_scatter_minimal_async_linear_sharded(
    mesh_device,
    input_shard_grid,
    input_shard_shape,
    output_shard_grid,
    output_shard_shape,
    input_mem_layout,
    output_mem_layout,
    buffer_type,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    cluster_axis,
):
    num_devices = tuple(mesh_device.shape)[cluster_axis]

    input_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(input_shard_grid[0] - 1, input_shard_grid[1] - 1))}
    )
    input_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_shard_grid[0] - 1, output_shard_grid[1] - 1))}
    )
    output_shard_spec = ttnn.ShardSpec(
        output_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(input_mem_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    mem_config_rs = ttnn.MemoryConfig(output_mem_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)

    run_reduce_scatter_impl(
        mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=ttnn.Topology.Linear,  # <- Behavior we are testing occurs with linear topology
        num_iters=1,
        enable_trace=False,
        ones_tensor=False,
        cluster_axis=1,
        use_barrier=False,
        use_persistent_buffers=False,
        chunks_per_sync=None,
        num_workers_per_link=None,
        num_buffers_per_channel=None,
        verify_output=True,
    )
