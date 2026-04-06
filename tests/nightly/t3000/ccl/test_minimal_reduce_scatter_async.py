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
    use_sub_devices = False
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

    if use_sub_devices:
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
                intermediate_memory_config=mem_config_intermediate,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
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
    if use_sub_devices:
        mesh_device.clear_loaded_sub_device_manager()


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout, rs_input_dtype, use_new, enable_trace, num_iters, use_barrier, use_persistent_buffers, chunks_per_sync, num_workers_per_link, num_buffers_per_channel,",
    [
        # Dim 0 tests
        (
            [16, 2, 128, 128],
            0,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            True,
            10,
            True,
            True,
            None,
            None,
            None,
        ),  # perf, barrier_with_persistent
        (
            [8, 2, 128, 128],
            0,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            False,
            1,
            True,
            False,
            None,
            None,
            None,
        ),  # check, barrier_without_persistent
        # Dim 1 tests
        (
            [2, 24, 256, 256],
            1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            True,
            10,
            True,
            True,
            None,
            None,
            None,
        ),  # perf, barrier_with_persistent
        (
            [2, 16, 56, 56],
            1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            False,
            1,
            True,
            False,
            None,
            None,
            None,
        ),  # check, barrier_without_persistent
        (
            [2, 8, 512, 512],
            1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            True,
            10,
            False,
            True,
            None,
            None,
            None,
        ),  # perf, no_barrier_with_persistent
        # Dim 2 tests
        (
            [2, 4, 1024, 1024],
            2,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            False,
            1,
            True,
            True,
            None,
            None,
            None,
        ),  # check, barrier_with_persistent
        (
            [4, 1, 1024, 340],
            2,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            True,
            10,
            True,
            False,
            None,
            None,
            None,
        ),  # perf, barrier_without_persistent
        (
            [1, 1, 512, 512],
            2,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            False,
            1,
            False,
            True,
            None,
            None,
            None,
        ),  # check, no_barrier_with_persistent
        # Dim 3 tests
        (
            [2, 4, 1024, 1024],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            True,
            10,
            True,
            True,
            None,
            None,
            None,
        ),  # perf, barrier_with_persistent
        (
            [1, 1, 13, 512],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            False,
            1,
            True,
            False,
            None,
            None,
            None,
        ),  # check, barrier_without_persistent
        (
            [3, 1, 41, 512],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            True,
            10,
            False,
            True,
            None,
            None,
            None,
        ),  # perf, no_barrier_with_persistent
        (
            [8, 1, 512, 2560],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            False,
            1,
            True,
            True,
            None,
            None,
            None,
        ),  # check, barrier_with_persistent
        (
            [4, 1, 1024, 2560],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            True,
            10,
            True,
            False,
            None,
            None,
            None,
        ),  # perf, barrier_without_persistent
        (
            [1, 1, 1024, 2560],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            False,
            1,
            False,
            True,
            None,
            None,
            None,
        ),  # check, no_barrier_with_persistent
        (
            [1, 1, 352, 2560],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            True,
            10,
            True,
            True,
            None,
            None,
            None,
        ),  # perf, barrier_with_persistent
        (
            [2, 1, 2048, 2560],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            False,
            1,
            True,
            False,
            2,
            2,
            8,
        ),  # check, barrier_without_persistent_with_hyperparams
        (
            [1, 1, 4096, 2560],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            True,
            10,
            False,
            True,
            2,
            2,
            8,
        ),  # perf, no_barrier_with_persistent_with_hyperparams
        # Composite-RS tests
        (
            [1, 1, 1, 8],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            False,
            1,
            True,
            True,
            None,
            None,
            None,
        ),  # check, barrier_with_persistent
        (
            [2, 32, 2048, 64],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            True,
            10,
            True,
            False,
            None,
            None,
            None,
        ),  # perf, barrier_without_persistent
        (
            [1, 1, 1, 16],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
            False,
            False,
            1,
            False,
            True,
            2,
            2,
            8,
        ),  # check, no_barrier_with_persistent_with_hyperparams
        (
            [1, 1, 29, 32],
            3,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            True,
            True,
            10,
            True,
            True,
            2,
            2,
            8,
        ),  # perf, barrier_with_persistent_with_hyperparams
    ],
    ids=[
        "scatter_dim_0_test_one-perf-barrier_with_persistent",
        "scatter_dim_0_test_two-check-barrier_without_persistent",
        "scatter_dim_1_test_one-perf-barrier_with_persistent",
        "scatter_dim_1_test_two-check-barrier_without_persistent",
        "scatter_dim_1_test_three-perf-no_barrier_with_persistent",
        "scatter_dim_2_test_one-check-barrier_with_persistent",
        "scatter_dim_2_test_two-perf-barrier_without_persistent",
        "scatter_dim_2_test_three-check-no_barrier_with_persistent",
        "non_zero_dim_1-perf-barrier_with_persistent",
        "padded_dim_2_test_one-check-barrier_without_persistent",
        "padded_dim_2_test_two-perf-no_barrier_with_persistent",
        "batch_8-check-barrier_with_persistent",
        "batch_4-perf-barrier_without_persistent",
        "batch_1_sd35_spatial-check-no_barrier_with_persistent",
        "batch_1_sd35_prompt-perf-barrier_with_persistent",
        "batch_2-check-barrier_without_persistent_with_hyperparams",
        "batch_1-perf-no_barrier_with_persistent_with_hyperparams",
        "composite_rs_test_one-check-barrier_with_persistent",
        "composite_rs_test_two-perf-barrier_without_persistent",
        "composite_rs_test_three-check-no_barrier_with_persistent_with_hyperparams",
        "composite_rs_test_four-perf-barrier_with_persistent_with_hyperparams",
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
    "ones_tensor",
    [
        False,
    ],
    ids=["random"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1540000}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1540000}, ttnn.Topology.Linear),
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
    use_new,
    enable_trace,
    num_iters,
    use_barrier,
    use_persistent_buffers,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
    mem_config_input,
    mem_config_rs,
    ones_tensor,
    rs_topology,
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
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        use_new=use_new,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout, rs_input_dtype, use_new, enable_trace, num_iters",
    [
        # Scatter on dim 0
        ([16, 1, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, False, 1),  # check
        ([16, 16, 128, 128], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([8, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        # Scatter on dim 1
        ([1, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([16, 16, 128, 128], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, False, 1),  # check
        ([16, 8, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, True, 10),  # perf
        # Scatter on dim 2
        ([1, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, False, 1),  # check
        ([16, 1, 512, 128], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([16, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        # Scatter on dim 3
        ([1, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([16, 1, 128, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        ([16, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
    ],
    ids=[
        "tt_training_test_one-check",
        "tt_training_test_two-perf",
        "tt_training_test_three-check",
        "tt_training_test_four-perf",
        "tt_training_test_five-check",
        "tt_training_test_six-perf",
        "tt_training_test_seven-check",
        "tt_training_test_eight-perf",
        "tt_training_test_nine-check",
        "tt_training_test_ten-perf",
        "tt_training_test_eleven-check",
        "tt_training_test_twelve-perf",
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
    "ones_tensor",
    [
        False,
    ],
    ids=["random"],
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
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    use_new,
    enable_trace,
    num_iters,
    mem_config_input,
    mem_config_rs,
    ones_tensor,
    rs_topology,
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
    "rs_input_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, intermediate_shard_shape, intermediate_shard_grid, intermediate_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout, buffer_type, use_new, enable_trace, num_iters",
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
            False,
            True,
            10,  # perf
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
            True,
            False,
            1,  # check
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
            True,
            10,  # perf
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
            True,
            False,
            1,  # check
        ),
    ],
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
    use_new,
    enable_trace,
    num_iters,
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
    "rs_input_shape, dim, intermediate_shard_shape, intermediate_shard_grid, intermediate_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout, buffer_type, use_new, enable_trace, num_iters",
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
            False,
            1,  # check
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
            True,
            True,
            10,  # perf
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
            False,
            False,
            1,  # check
        ),
    ],
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
    use_new,
    enable_trace,
    num_iters,
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
    "rs_input_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, buffer_type, use_new, enable_trace, num_iters",
    [
        (
            [4, 1, 256, 3072],
            3,
            [1024, 512],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            False,
            True,
            10,  # perf
        ),
        (
            [4, 1, 384, 1024],
            3,
            [256, 1024],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            True,
            False,
            1,  # check
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
            True,
            10,  # perf
        ),
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1271456}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1271456}, ttnn.Topology.Linear),
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
    use_new,
    enable_trace,
    num_iters,
    rs_topology,
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


MESH_SHAPE = (1, 8)
LAYOUT = ttnn.TILE_LAYOUT

NUM_ITERS = 1


def _valid_cluster_div(input_shape, dim, cluster_axis, mesh_shape, **kwargs):
    return input_shape[dim] % (prod(mesh_shape) if cluster_axis is None else mesh_shape[cluster_axis]) == 0


def _get_tensors(
    input_shape,
    mesh_shape,
    dim,
    cluster_axis,
    dtype,
    memory_config,
    layout,
    device,
    math_op=ttnn.ReduceType.Sum,
):
    assert _valid_cluster_div(input_shape, dim, cluster_axis, mesh_shape)

    num_devices = math.prod(mesh_shape)

    elems = math.prod(input_shape)

    torch_inputs = [
        torch.linspace(i * elems, (i + 1) * elems, elems).reshape(input_shape).bfloat16() for i in range(num_devices)
    ]
    torch_input = torch.concat(torch_inputs, dim=0)

    torch_reference = torch.reshape(torch_input, tuple(list(mesh_shape) + input_shape))
    torch_reference = torch.sum(torch_reference, dim=cluster_axis)

    dim_per_device = input_shape[dim] // mesh_shape[cluster_axis]

    torch_reference_slices = []
    for x in range(mesh_shape[0]):
        for y in range(mesh_shape[1]):
            i, j = (x, y) if cluster_axis == 1 else (y, x)

            torch_reference_slice = torch_reference[i].split(dim_per_device, dim=dim)[j]
            torch_reference_slices.append(torch_reference_slice)

    torch_reference = torch.concat(torch_reference_slices, dim=0)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
        device=device,
    )

    return tt_input, torch_reference


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "input_shape", [[256, 256], [8, 8, 256, 256], [8, 256, 256], [8, 8, 8, 8, 256, 256], [8, 8, 8, 16, 16]]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear])
def test_nd(mesh_device, input_shape, dim, cluster_axis, dtype, memory_config, topology):
    if dim >= len(input_shape):
        pytest.skip("Invalid scatter dim")

    tt_input, torch_reference = _get_tensors(
        input_shape,
        tuple(mesh_device.shape),
        dim,
        cluster_axis,
        dtype,
        memory_config,
        ttnn.TILE_LAYOUT,
        mesh_device,
    )

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(3)]

    for i in range(NUM_ITERS):
        tt_out_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            tt_input,
            dim=dim,
            multi_device_global_semaphore=semaphores,
            cluster_axis=cluster_axis,
            topology=topology,
        )

        tt_output_tensor = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(tt_out_tensor)])
        eq, mess = comp_pcc(torch_reference, tt_output_tensor)
        assert eq, mess


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}, {"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
    ids=["fabric_linear", "fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("input_shape", [[2, 2, 32, 32]])
def test_reduce_scatter_async_2x4_non_flat_mesh(mesh_device, input_shape):
    torch.manual_seed(520)
    devices = mesh_device.get_num_devices()
    input_shape[-1] *= devices

    torch_inputs_per_device = [torch.rand(input_shape, dtype=torch.bfloat16) for _ in range(devices)]

    torch_reference = torch.zeros_like(torch_inputs_per_device[0])
    for i in range(devices):
        torch_reference += torch_inputs_per_device[i]

    tt_input = ttnn.from_torch(
        torch.cat(torch_inputs_per_device, dim=0),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        device=mesh_device,
    )  # [2, 2, 32, 32*devices] per device

    tt_output = ttnn.reduce_scatter(tt_input, dim=3)  # [2, 2, 32, 32] per device
    torch_output = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)
    )  # [2, 2, 32, 32*devices]

    assert torch.allclose(
        torch_reference, torch_output, atol=1e-1, rtol=1e-2
    ), "Output mismatch between torch and ttnn reduce-scatter"
