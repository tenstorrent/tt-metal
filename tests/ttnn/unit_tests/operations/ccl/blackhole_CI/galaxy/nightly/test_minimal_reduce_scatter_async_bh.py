# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from tt_py_test_utils_common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)


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
):
    torch.manual_seed(0)

    # Set the default config
    if mem_config_intermediate is None:
        mem_config_intermediate = mem_config_rs

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


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices,cluster_axis, rs_input_shape, dim, layout, rs_input_dtype",
    [
        (4, 1, [1, 1, 13, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, 1, [3, 1, 41, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, 1, [8, 1, 512, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, 1, [4, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, 1, [1, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, 1, [1, 1, 352, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, 1, [2, 1, 2048, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (4, 1, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        # Composite-RS tests
        (4, 1, [1, 1, 1, 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, 1, [1, 1, 1, 16], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (4, 1, [1, 1, 32, 32], 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ],
    ids=[
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
        (True, 10),
    ],
)
@pytest.mark.parametrize(
    "ones_tensor",
    [
        False,
    ],
    ids=["random"],
)
@pytest.mark.parametrize(
    "use_barrier, use_persistent_buffers",
    [
        (True, False),
    ],
    ids=["barrier_without_persistent_buffers"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1171456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async_ring(
    bh_2d_mesh_device,
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
    ones_tensor,
    use_barrier,
    use_persistent_buffers,
    cluster_axis,
    rs_topology,
):
    pytest.skip("Skipping due to hang")

    run_reduce_scatter_impl(
        bh_2d_mesh_device,
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
        ones_tensor=ones_tensor,
        use_barrier=use_barrier,
        cluster_axis=cluster_axis,
        use_persistent_buffers=use_persistent_buffers,
    )


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype, is_training_shape",
    [
        (4, [1, 1, 13, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        (4, [3, 1, 41, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        (4, [8, 1, 512, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        (4, [4, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        (4, [1, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        (4, [1, 1, 352, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        (4, [2, 1, 2048, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
        (4, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False),  # use batching when fused
    ],
    ids=[
        "padded_dim_2_test_one",
        "padded_dim_2_test_two",
        "batch_8",
        "batch_4",
        "batch_1_sd35_spatial",
        "batch_1_sd35_prompt",
        "batch_2",
        "batch_1",
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
        False,
    ],
    ids=["random"],
)
@pytest.mark.parametrize(
    "use_barrier, use_persistent_buffers",
    [
        (True, False),
    ],
    ids=["barrier_without_persistent_buffers"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async_line(
    bh_2d_mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    is_training_shape,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    ones_tensor,
    use_barrier,
    use_persistent_buffers,
    rs_topology,
):
    run_reduce_scatter_impl(
        bh_2d_mesh_device,
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
        ones_tensor=ones_tensor,
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
    )


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype",
    [
        # Scatter on dim 0
        (4, [16, 1, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [16, 16, 128, 128], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [8, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # Scatter on dim 1
        (4, [1, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [16, 16, 128, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [16, 8, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # Scatter on dim 2
        (4, [1, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [16, 1, 512, 128], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [16, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # # Scatter on dim 3
        (4, [1, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [16, 1, 128, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, [16, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
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
        False,
    ],
    ids=["random"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async_training_shapes(
    bh_2d_mesh_device,
    num_devices,
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
):
    if dim != 3:
        pytest.skip("#26572: Can only operate on dim 3")
    run_reduce_scatter_impl(
        bh_2d_mesh_device,
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
        ones_tensor=ones_tensor,
        use_barrier=True,
        use_persistent_buffers=False,
    )


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, layout, rs_input_dtype",
    [
        (4, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, intermediate_shard_shape, intermediate_shard_grid, intermediate_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout",
    [
        (
            [1, 1, 32, 3072],
            3,
            [32, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [32, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [32, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [4, 1, 384, 1024],
            3,
            [512, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [512, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [512, 256],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            [4, 1, 384, 3072],
            3,
            [512, 3072],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [1536, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [1536, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
    "ones_tensor",
    [
        False,
    ],
    ids=["random"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 113664}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 113664}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async_sharded_to_sharded(
    bh_2d_mesh_device,
    num_links,
    num_devices,
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
    enable_trace,
    num_iters,
    ones_tensor,
    rs_topology,
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

    mem_config_input = ttnn.MemoryConfig(
        input_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=input_shard_spec
    )
    mem_config_intermediate = ttnn.MemoryConfig(
        intermediate_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=intermediate_shard_spec
    )
    mem_config_rs = ttnn.MemoryConfig(output_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=output_shard_spec)

    run_reduce_scatter_impl(
        bh_2d_mesh_device,
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
        ones_tensor=ones_tensor,
        mem_config_intermediate=mem_config_intermediate,
    )


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, layout, rs_input_dtype",
    [
        (4, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, intermediate_shard_shape, intermediate_shard_grid, intermediate_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout",
    [
        (
            [4, 1, 256, 3072],
            3,
            [1024, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            [1024, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [4, 1, 384, 1024],
            3,
            [512, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            [256, 256],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
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
    "ones_tensor",
    [
        False,
    ],
    ids=["random"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 113664}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 113664}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async_interleaved_to_sharded(
    bh_2d_mesh_device,
    num_links,
    num_devices,
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
    enable_trace,
    num_iters,
    ones_tensor,
    rs_topology,
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

    mem_config_input = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_intermediate = ttnn.MemoryConfig(
        intermediate_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=intermediate_shard_spec
    )
    mem_config_rs = ttnn.MemoryConfig(output_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=output_shard_spec)

    run_reduce_scatter_impl(
        bh_2d_mesh_device,
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
        ones_tensor=ones_tensor,
        mem_config_intermediate=mem_config_intermediate,
    )


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, layout, rs_input_dtype",
    [
        (4, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "rs_input_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout",
    [
        (
            [4, 1, 256, 3072],
            3,
            [1024, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            [4, 1, 384, 1024],
            3,
            [512, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
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
    "ones_tensor",
    [
        False,
    ],
    ids=["random"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 113664}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 113664}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async_sharded_to_interleaved(
    bh_2d_mesh_device,
    num_links,
    num_devices,
    rs_input_dtype,
    layout,
    rs_input_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    enable_trace,
    num_iters,
    ones_tensor,
    rs_topology,
):
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(
        input_mem_layout, buffer_type=ttnn.BufferType.DRAM, shard_spec=input_shard_spec
    )
    mem_config_intermediate = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_rs = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    run_reduce_scatter_impl(
        bh_2d_mesh_device,
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
        ones_tensor=ones_tensor,
        mem_config_intermediate=mem_config_intermediate,
    )


# ---- row-free line reduce-scatter ----

# Per-device tile rows H3 serves on a 4x8: 5 s, 2.5 s, 10 s, 15 s, capacity.
H3_TILE_ROWS = [148, 74, 287, 427, 470]
H3_FF2_WIDTH = 5376


def _line_reduce_scatter_h3_case(mesh_device, tile_rows, width):
    """One [1, 1, rows, width] bf16 plane per device, uploaded before the flat-cache window, and the expected result."""
    rows = tile_rows * 32
    tp, sp = tuple(mesh_device.shape)
    torch_input = torch.randn(tp, sp, rows, width, dtype=torch.bfloat16)
    host_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(tp, sp), dims=(0, 1)),
    )
    tt_input = ttnn.to_device(host_input, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    summed = torch_input.float().sum(dim=0)  # [8, rows, width]: the TP reduction of each column's planes
    slice_width = width // tp
    expected = torch.stack([summed[:, :, i * slice_width : (i + 1) * slice_width] for i in range(tp)], dim=0)
    return {"rows": rows, "width": width, "input": tt_input, "expected": expected}


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}],
    indirect=True,
)
def test_reduce_scatter_line_h3_rows_flat_cache(mesh_device):
    """One program serves every H3 row count of the pinned dim-3 line reduce-scatter; a new width compiles its own."""
    from tests.ttnn.utils_for_testing import assert_program_cache_flat

    compute_grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)]
    barrier = ttnn.create_global_semaphore(mesh_device, cores, 0)
    cases = [_line_reduce_scatter_h3_case(mesh_device, rows, H3_FF2_WIDTH) for rows in H3_TILE_ROWS]
    cases.append(_line_reduce_scatter_h3_case(mesh_device, H3_TILE_ROWS[0], 2688))

    def run(case):
        tt_output = ttnn.experimental.reduce_scatter_minimal_async(
            case["input"],
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=semaphores,
            barrier_semaphore=barrier,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=0,
            chunks_per_sync=2,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        ttnn.synchronize_device(mesh_device)
        output = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1)),
        ).float()
        assert output.shape == case["expected"].shape, (output.shape, case["expected"].shape)
        torch.testing.assert_close(output, case["expected"], rtol=1.6e-2, atol=1e-1)

    entries = assert_program_cache_flat(mesh_device, run, cases, miss_indices=(len(cases) - 1,))
    assert entries == 2, f"expected one program per width, got {entries} entries"
