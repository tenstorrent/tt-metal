# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull


def run_with_trace(
    mesh_device,
    all_broadcast_topology,
    input_tensor_mesh,
    num_links,
    output_mem_config,
    multi_device_global_semaphore,
    num_iter=20,
    subdevice_id=None,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_broadcast_async(
        input_tensor_mesh,
        multi_device_global_semaphore=multi_device_global_semaphore,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=all_broadcast_topology,
        subdevice_id=subdevice_id,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_broadcast_async(
            input_tensor_mesh,
            multi_device_global_semaphore=multi_device_global_semaphore,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_broadcast_topology,
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


def run_all_broadcast_impl(
    mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    function_level_defaults,
    all_broadcast_topology,
    num_iters=1,
    trace_mode=False,
    rand_tensor=True,
    mem_config=None,
    input_shard_shape=None,
    input_shard_grid=None,
    output_shard_shape=None,
    output_shard_grid=None,
    tensor_mem_layout=None,
    use_cluster_axis_api=False,
    cluster_axis=None,
):
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
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

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
    output_tensor_goldens_list = []

    for i in range(num_iters):
        output_tensors = []
        for k in range(num_devices):
            if rand_tensor:
                output_tensor = torch.rand(output_shape).bfloat16()
            else:
                output_tensor = torch.zeros(output_shape)
                row_id = 1
                # Fix indices for tensors with ranks 2 or 3
                for w in range(output_shape[0]):
                    for z in range(output_shape[1]):
                        for y in range(0, output_shape[2], 32):
                            for x in range(0, output_shape[3], 32):
                                output_tensor[w, z, y, :] = row_id
                                row_id += 1
            output_tensors.append(output_tensor)

        output_tensor_goldens_list.append(output_tensors)
        temp_output_tensor = torch.cat(output_tensors, -1)

        input_tensor_mesh = ttnn.from_torch(
            temp_output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)], ttnn.MeshShape(1, num_devices)
                ),
            ),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor = run_with_trace(
            mesh_device,
            all_broadcast_topology,
            input_tensor_mesh_list[0],
            num_links,
            output_mem_config,
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            num_iter=num_iters,
            subdevice_id=worker_sub_device_id,
        )
        tt_out_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            tt_out_tensors = ttnn.experimental.all_broadcast_async(
                input_tensor_mesh_list[i],
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=output_mem_config,
                topology=all_broadcast_topology,
                subdevice_id=worker_sub_device_id,
            )
            tt_out_tensor_list.append(tt_out_tensors)

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    passed = True
    for tensor_index in range(len(tt_out_tensor_list)):
        tt_out_tensors = tt_out_tensor_list[tensor_index]
        output_tensors = output_tensor_goldens_list[tensor_index]
        for k in range(num_devices):
            output_tensor = output_tensors[k]
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensors[k])):
                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                logger.info(f"Checking for device {t.device().id()}")
                if input_dtype == ttnn.bfloat16:
                    eq, output = comp_equal(tt_output_tensor, output_tensor)
                else:
                    eq, output = comp_pcc(tt_output_tensor, output_tensor)
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
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, layout, input_dtype",
    [
        (2, 1, [2, 30], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        (2, 1, [3, 122, 2042], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        (4, 1, [1, 1, 32, 1024], ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (4, 1, [2, 64, 512], ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [256, 3328], ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (4, 1, [1, 69, 4000], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        (8, 1, [10, 8320], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
    ],
)
@pytest.mark.parametrize("num_iters", [3])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_all_broadcast(
    t3k_mesh_device,
    # pcie_mesh_device,
    num_devices,
    output_shape,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    function_level_defaults,
):
    run_all_broadcast_impl(
        t3k_mesh_device,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        mem_config=mem_config,
    )


@pytest.mark.parametrize(
    "num_devices, output_shape, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        (
            4,
            [2, 32, 256],
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            None,
            None,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ),
        (
            4,
            [192, 64],
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 1))}),
            None,
            None,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ),
        (
            2,
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
def test_all_broadcast_sharded(
    t3k_mesh_device,
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
    run_all_broadcast_impl(
        t3k_mesh_device,
        num_devices,
        output_shape,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        all_broadcast_topology=ttnn.Topology.Linear,
        num_iters=num_iters,
        rand_tensor=True,
        input_shard_shape=input_shard_shape,
        input_shard_grid=input_shard_grid,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
    )
