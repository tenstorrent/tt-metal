# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from dataclasses import dataclass, astuple
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from models.common.utility_functions import skip_for_blackhole


@dataclass
class ReduceScatterTestConfig:
    """Test configuration for reduce scatter operations.

    Using a dataclass ensures parameters can't be provided in wrong order
    and makes test cases self-documenting.
    """

    rs_input_shape: list
    dim: int
    layout: object  # ttnn.Layout
    rs_input_dtype: object  # ttnn.DataType
    use_new: bool
    enable_trace: bool
    num_iters: int
    use_barrier: bool
    use_persistent_buffers: bool
    use_strided: bool
    verify_output_shape: bool
    verify_output_pcc: bool
    small_random_ints: bool
    mm_cores_y: object = None  # Optional[int]
    mm_block_ht: object = None  # Optional[int]
    mm_block_wt: object = None  # Optional[int]
    mm_N_full_block_wt: object = None  # Optional[int]
    chunk_width_in_mm_blocks: object = None  # Optional[int]
    num_workers_per_link: object = None  # Optional[int]
    num_buffers_per_channel: object = None  # Optional[int]


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
    small_random_ints=False,
    mem_config_intermediate=None,
    cluster_axis=None,
    use_barrier=False,
    use_persistent_buffers=True,
    chunks_per_sync=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    verify_output=True,
    use_new=False,
    use_strided=False,
    verify_output_shape=True,
    verify_output_pcc=True,
    mm_cores_y=None,
    mm_block_ht=None,
    mm_block_wt=None,
    mm_N_full_block_wt=None,
    chunk_width_in_mm_blocks=None,
):
    torch.manual_seed(0)

    ##### Fabric setup #####
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

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
    else:
        # Ring RS intermediate holds only one batch element at a time
        intermediate_shape[0] = 1
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

    ##### Reduce scatter input setup #####
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
        elif small_random_ints:
            # Random integers from {0, 1, 2, 3} for easier debugging
            rs_input_tensor = torch.randint(0, 4, rs_global_input_shape).bfloat16()
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

    ##### Verify Correct Reference Output #####
    torch_reduce_scatter_output_list = []
    for i in range(num_iters):
        reduce_output = torch.sum(torch.stack(torch_input_tensor_list[i]), dim=0)
        scatter_output = torch.chunk(reduce_output, num_devices, dim)
        torch_reduce_scatter_output_list.append(scatter_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []

    def run_op(i):
        if use_strided:
            logger.info(f"Using strided reduce scatter")
            tt_reduce_scatter_output_tensor = ttnn.experimental.strided_reduce_scatter_async(
                tt_input_tensor_mesh_list[i],
                persistent_output_buffers=[persistent_intermediate_buffers[i], persistent_output_buffers[i]]
                if use_persistent_buffers
                else None,
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                num_links=num_links,
                memory_config=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
                mm_cores_y=mm_cores_y,
                mm_block_ht=mm_block_ht,
                mm_block_wt=mm_block_wt,
                mm_N_full_block_wt=mm_N_full_block_wt,
                chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
            )
        elif use_new:
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
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )
        return tt_reduce_scatter_output_tensor

    if enable_trace:
        for i in range(num_iters):
            run_op(i)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info("Done compiling ops")

        logger.info(f"Capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_reduce_scatter_output_trace_list = []
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
        implementation_name = "strided" if use_strided else ("new" if use_new else "experimental")
        logger.info(f"Verifying {implementation_name} reduce scatter output")

        for i in range(num_iters):
            tt_rs_out = tt_reduce_scatter_output_list[i]
            torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

            # Split the concatenated output back into per-device chunks
            tt_output_chunks = torch.chunk(tt_rs_out, num_devices, dim=dim)
            for device_id in range(num_devices):
                tt_output_tensor = tt_output_chunks[device_id]
                torch_output_tensor = torch_rs_out_tensor[device_id]

                if small_random_ints:
                    eq, output = comp_equal(tt_output_tensor, torch_output_tensor)
                else:
                    eq, output = comp_pcc(tt_output_tensor, torch_output_tensor)
                logger.info(f"{output}, device {device_id}, iteration {i}")

                if verify_output_shape:
                    assert (
                        tt_output_tensor.shape == torch_output_tensor.shape
                    ), f"Shape mismatch: {tt_output_tensor.shape} != {torch_output_tensor.shape}"
                if verify_output_pcc:
                    assert eq, f"{i} FAILED reduce scatter: {output}"

    mesh_device.reset_sub_device_stall_group()
    logger.info("Done")


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[8, 1, 512, 2560],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=True,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=False,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=False,
            ),
            id="new_standard_implementation",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[8, 1, 512, 2560],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=False,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=False,
            ),
            id="experimental_standard_implementation",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 2 x 2 tiles with a single mm block from an assumed 1 x 8 mm core grid
                rs_input_shape=[8, 1, 64, 512],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_minimal_2x2",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 4 x 2 tiles with two mm blocks from an assumed 1 x 8 mm core grid
                rs_input_shape=[8, 1, 128, 512],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_toy_4x2",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 2 x 4 tiles with two mm blocks from an assumed 1 x 8 mm core grid
                rs_input_shape=[8, 1, 64, 1024],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=4,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_toy_2x4",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 4 x 4 tiles with four 2 x 2 mm blocks from an assumed 1 x 8 mm core grid
                rs_input_shape=[4, 1, 128, 1024],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=4,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_toy_4x4",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 4 x 4 tiles with four 2 x 2 mm blocks from an assumed 2 x 8 mm core grid
                rs_input_shape=[4, 1, 128, 1024],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_4x4_2x8_mm_grid",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 16 x 8 tiles with eight 4 x 4 mm blocks from an assumed 2 x 8 mm core grid
                # chunk is 8 tiles wide
                rs_input_shape=[4, 1, 512, 2048],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=3,  # multi-iter test
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=4,
                mm_block_wt=4,
                mm_N_full_block_wt=8,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_16x8_2x8_mm_grid_chunk_2_mm_blocks_wide_multi_iter",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 16 x 8 tiles with eight 4 x 4 mm blocks from an assumed 2 x 8 mm core grid
                # chunk is 8 tiles wide
                rs_input_shape=[4, 1, 512, 2048],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=4,
                mm_block_wt=4,
                mm_N_full_block_wt=8,
                chunk_width_in_mm_blocks=2,
                num_workers_per_link=3,
            ),
            id="experimental_strided_4x4_2x8_mm_grid_chunk_2_mm_blocks_wide_3_workers_per_link",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 16 x 8 tiles with asymmetric tall-narrow 8x2 mm blocks
                # from an assumed 2 x 8 mm core grid; chunk width is 4 tiles (2 chunks per N-block)
                rs_input_shape=[4, 1, 512, 2048],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=8,
                mm_block_wt=2,
                mm_N_full_block_wt=8,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_16x8_asymmetric_tall_narrow_8x2_blocks",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 16 x 8 tiles with asymmetric short-wide 2x8 mm blocks
                # from an assumed 4 x 8 mm core grid; single chunk covers entire N-block
                rs_input_shape=[4, 1, 512, 2048],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=4,
                mm_block_ht=2,
                mm_block_wt=8,
                mm_N_full_block_wt=8,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_16x8_asymmetric_short_wide_2x8_blocks",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 16 x 10 tiles (non-power-of-2 slice width from 2560-wide input)
                # from an assumed 2 x 8 mm core grid with 4x4 mm blocks
                # chunk width is 8 tiles, partial last chunk (effective width 2)
                rs_input_shape=[4, 1, 512, 2560],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=4,
                mm_block_wt=4,
                mm_N_full_block_wt=10,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_16x10_partial_last_chunk_4x4_blocks",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 24 x 12 tiles (from 768x3072 input)
                # from an assumed 2 x 8 mm core grid with non-power-of-2 3x3 mm blocks
                # key property: 2 N-blocks of width 6 AND 2 chunks per N-block (chunk width 3)
                # -- tests the combination of multiple N-blocks with multiple chunks within each
                rs_input_shape=[4, 1, 768, 3072],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=3,
                mm_block_wt=3,
                mm_N_full_block_wt=6,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_24x12_3x3_blocks_2_N_blocks_chunk_1_mm_blocks",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 24 x 30 tiles (from 768x7680 input)
                # from an assumed 2 x 8 mm core grid with non-power-of-2 3x3 mm blocks
                # key property: 2 N-blocks of width 15 AND 3 chunks per N-block (chunk width 6)
                # with partial last chunk (pattern: 6, 6, 3) -- tests the combination of
                # multiple N-blocks, multiple chunks, AND a partial final chunk
                rs_input_shape=[4, 1, 768, 7680],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=3,
                mm_block_wt=3,
                mm_N_full_block_wt=15,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_24x30_3x3_blocks_2_N_blocks_partial_last_chunk",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 128 x 16 tiles with 4x4 mm blocks
                # from an assumed 8 x 8 mm core grid; 2 N-blocks per slice
                # explicit 6 workers per link
                rs_input_shape=[4, 1, 4096, 4096],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=8,
                mm_block_ht=4,
                mm_block_wt=4,
                mm_N_full_block_wt=8,
                chunk_width_in_mm_blocks=1,
                num_workers_per_link=6,
            ),
            id="experimental_strided_128x16_8_cores_2_N_blocks_6_workers",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 128 x 16 tiles with 8x8 mm blocks
                # from an assumed 4 x 8 mm core grid; single N-block spanning full width
                # chunk width is 16 tiles (single chunk covers entire N-block)
                rs_input_shape=[4, 1, 4096, 4096],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=4,
                mm_block_ht=8,
                mm_block_wt=8,
                mm_N_full_block_wt=16,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_128x16_4_cores_single_N_block_wide_chunk",
        ),
        # Partial last M-block cases: slice_Ht_per_core is not a multiple of mm_block_ht.
        # These are deliberately tiny to isolate the partial-block code path.
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 3 x 2 tiles; 1 core, mm_block_ht=2
                # slice_Ht_per_core=3 → 1 full block (2 rows) + 1 partial block (1 row)
                rs_input_shape=[8, 1, 96, 512],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_partial_M_block_1core_ht3_block2",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 6 x 2 tiles; 2 cores → slice_Ht_per_core=3, mm_block_ht=2
                # same partial pattern as above but split across 2 MM cores
                rs_input_shape=[8, 1, 192, 512],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_partial_M_block_2cores_ht3_block2",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 6 x 2 tiles; 1 core, mm_block_ht=4
                # slice_Ht_per_core=6 → 1 full block (4 rows) + 1 partial block (2 rows)
                rs_input_shape=[8, 1, 192, 512],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=4,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_partial_M_block_1core_ht6_block4",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[2, 1, 10240, 5120],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=8,
                mm_block_ht=8,
                mm_block_wt=8,
                mm_N_full_block_wt=20,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_large_input_shape_8_cores_20_N_blocks_1_chunk",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[2, 1, 9472, 5120],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=8,
                mm_block_ht=8,
                mm_block_wt=8,
                mm_N_full_block_wt=20,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_large_input_shape_8_cores_8_N_blocks_1_chunk",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # slice_Ht=296 is not divisible by mm_cores_y=7.
                # padded_slice_Ht=301, slice_Ht_per_core=43.
                # 43 % mm_block_ht=8 = 3, so the last M-block per core is partial (3 rows).
                # Core 6: rows 258-295 real (38 rows) + rows 296-300 ghost (5 rows);
                # ghost rows bleed across M-block boundaries within the last core.
                rs_input_shape=[2, 1, 9472, 5120],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=7,
                mm_block_ht=8,
                mm_block_wt=8,
                mm_N_full_block_wt=20,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_large_input_shape_7_cores_non_divisible_Ht_and_block",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # slice_Ht=296 is not divisible by mm_cores_y=7 (same padding as above).
                # num_workers_per_link=4 → effective_advance_by_tiles=8, which increases the
                # chance that tile1 lands in-bounds while tile2 lands out-of-bounds, exercising
                # the tile1_in_bounds && can_use_two_tiles_in_packet && !tile2_in_bounds writer path.
                rs_input_shape=[2, 1, 9472, 5120],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=7,
                mm_block_ht=8,
                mm_block_wt=8,
                mm_N_full_block_wt=22,
                chunk_width_in_mm_blocks=1,
                num_workers_per_link=4,
            ),
            id="experimental_strided_large_7_cores_non_divisible_Ht_and_Wt_multi_worker",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # slice_Ht=3 tiles is not divisible by mm_cores_y=2.
                # padded_slice_Ht=4, slice_Ht_per_core=2.
                # Core 0: rows 0-1 (real). Core 1: row 2 (real) + row 3 (ghost).
                rs_input_shape=[8, 1, 96, 512],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_non_divisible_slice_Ht_partial_last_core",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # slice_Ht=6 tiles is not divisible by mm_cores_y=4.
                # padded_slice_Ht=8, slice_Ht_per_core=2.
                # Cores 0-2: 2 real rows each. Core 3: 0 real rows, 2 ghost rows (all skipped).
                rs_input_shape=[8, 1, 192, 512],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=4,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_non_divisible_slice_Ht_all_ghost_last_core",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # slice_Ht=7 is not divisible by mm_cores_y=2, and after padding
                # slice_Ht_per_core=4 is not divisible by mm_block_ht=3.
                # padded_slice_Ht=8, slice_Ht_per_core=4, mm_M_unit_blocks_per_core=div_up(4,3)=2.
                # Core 0: rows 0-3 real → M-block 0 full (3 rows), M-block 1 partial (1 real row).
                # Core 1: rows 4-6 real + row 7 ghost →
                #   M-block 0 partial (3 real rows), M-block 1 entirely ghost (1 ghost row).
                rs_input_shape=[8, 1, 224, 512],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=3,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_non_divisible_slice_Ht_and_slice_Ht_per_core",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # Non-div Wt with wide mm blocks and single worker.
                # mm_block_wt=3 → chunk_width_in_tiles=3; effective_advance_by_tiles=2 (1 worker).
                # 2 % 3 != 0 → consecutive tiles in a 2-tile packet have different chunk_col values,
                # which can straddle the left-bound validity threshold.
                # chip 1: skip_cols_left = (1*7) % 9 = 7; in the last chunk of N-block 0
                # (chunk_idx=2, cols 6..8), packet slot 0 has slice_col=6 (invalid, <7) and
                # slot 1 has slice_col=8 (valid, >=7) — exercises the CB-packing fix.
                rs_input_shape=[8, 1, 128, 1792],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=4,
                mm_block_wt=3,
                mm_N_full_block_wt=9,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_non_div_Wt_wide_mm_block_cross_col_single_worker",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # Same shape as above but with 2 workers (effective_advance=4, 4%3=1≠0).
                # chip k: skip = (k*7)%9 = {0,7,5,3,1,8,6,4} — diverse skips across chips.
                # chip 1 (skip=7): in chunk_idx=2 of N-block 0, advance of 4 within a width-3
                # subchunk causes col sequence (0,1,2,0,...); a packet landing on (col=2, col=0)
                # wraps to (invalid, valid) — same root cause as the large_7_cores test.
                rs_input_shape=[8, 1, 128, 1792],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=4,
                mm_block_wt=3,
                mm_N_full_block_wt=9,
                chunk_width_in_mm_blocks=1,
                num_workers_per_link=2,
            ),
            id="experimental_strided_non_div_Wt_wide_mm_block_cross_col_two_workers",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # Non-div Wt with partial last chunk and multi-worker: smaller version of the
                # large_7_cores test pattern. mm_N_full_block_wt=14, mm_block_wt=8 → chunk_width=8,
                # chunks_per=2; partial last chunk has effective_width=6.
                # effective_advance=4 (2 workers); 4 % 6 = 4 ≠ 0 → cross-column in partial chunk.
                # chip 1 (skip=9%14=9): partial chunk cols 8..13; packet (col=0,col=4) →
                # slice_col=8 invalid (8<9), slice_col=12 valid — exercises the partial-chunk path.
                rs_input_shape=[4, 1, 256, 2304],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=4,
                mm_block_wt=8,
                mm_N_full_block_wt=14,
                chunk_width_in_mm_blocks=1,
                num_workers_per_link=2,
            ),
            id="experimental_strided_non_div_Wt_partial_last_chunk_cross_col_two_workers",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # All 8 chips have distinct skip values: slice_Wt=9, mm_N_full_block_wt=11 →
                # chip k: skip = (k*9) % 11 = {0,9,7,5,3,1,10,8} — all 8 values are different,
                # exercising every skip magnitude from 0 to 10 across one test run.
                # mm_block_wt=3, chunk_width_in_mm_blocks=1 → chunk_width=3, chunks_per=4
                # (last chunk partial: effective_width=2); effective_advance=4 (2 workers).
                # 4 % 3 = 1 ≠ 0 in full chunks → cross-column packets.
                # Also combines non-div Ht (slice_Ht=11, mm_cores_y=3: padded=12, per_core=4,
                # ghost rows=1 on last core; mm_block_ht=3: partial last M-block of 1 row).
                rs_input_shape=[8, 1, 352, 2304],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=3,
                mm_block_ht=3,
                mm_block_wt=3,
                mm_N_full_block_wt=11,
                chunk_width_in_mm_blocks=1,
                num_workers_per_link=2,
            ),
            id="experimental_strided_non_div_Wt_all_unique_skips_non_div_Ht_multi_worker",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # slice_Wt=3 is not divisible by mm_N_full_block_wt=2 (3 % 2 = 1).
                # Fallback: mm_N_full_block_wt_val overridden to slice_Wt=3, 1 chunk covering all
                # 3 tiles; mm_blocks_sem_override = div_up(2,1) = 2.
                # slice_Ht=8, mm_cores_y=1, single M-block covers all rows.
                rs_input_shape=[8, 1, 256, 768],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=8,
                mm_block_wt=1,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_non_divisible_slice_Wt",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # slice_Wt=3 is not divisible by mm_N_full_block_wt=2 (3 % 2 = 1) AND
                # slice_Ht=7 is not divisible by mm_cores_y=2; padded_slice_Ht=8,
                # slice_Ht_per_core=4, mm_block_ht=3 → partial last M-block on both cores.
                # Both the non-div Wt fallback and the ghost-row path are active together.
                rs_input_shape=[8, 1, 224, 768],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=3,
                mm_block_wt=1,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_non_divisible_slice_Wt_and_slice_Ht",
        ),
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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_reduce_scatter_async(
    mesh_device,
    num_links,
    test_config,
    mem_config_input,
    mem_config_rs,
    ones_tensor,
    rs_topology,
):
    # Unpack test configuration
    (
        rs_input_shape,
        dim,
        layout,
        rs_input_dtype,
        use_new,
        enable_trace,
        num_iters,
        use_barrier,
        use_persistent_buffers,
        use_strided,
        verify_output_shape,
        verify_output_pcc,
        small_random_ints,
        mm_cores_y,
        mm_block_ht,
        mm_block_wt,
        mm_N_full_block_wt,
        chunk_width_in_mm_blocks,
        num_workers_per_link,
        num_buffers_per_channel,
    ) = astuple(test_config)

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
        small_random_ints=small_random_ints,
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
        use_new=use_new,
        use_strided=use_strided,
        verify_output_shape=verify_output_shape,
        verify_output_pcc=verify_output_pcc,
        mm_cores_y=mm_cores_y,
        mm_block_ht=mm_block_ht,
        mm_block_wt=mm_block_wt,
        mm_N_full_block_wt=mm_N_full_block_wt,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}],
    indirect=True,
)
@pytest.mark.parametrize(
    # Shape [4,1,512,2048] dim=3 with ring_size=8 gives:
    #   slice_Ht = 16 tiles, slice_Wt = 8 tiles
    #
    # Constraints:
    #   slice_Wt (8) % mm_N_full_block_wt == 0
    #   NOTE: slice_Ht does NOT need to be divisible by mm_cores_y — ghost rows are skipped.
    #         slice_Ht_per_core does NOT need to be divisible by mm_block_ht either.
    #
    # Derived values:
    #   padded_slice_Ht = round_up(16, mm_cores_y)
    #   slice_Ht_per_core = padded_slice_Ht / mm_cores_y
    #   mm_M_unit_blocks_per_core = ceil(slice_Ht_per_core / mm_block_ht)
    #   N_blocks_per_slice = 8 / mm_N_full_block_wt
    #   chunk_width_in_tiles = chunk_width_in_mm_blocks * mm_block_wt
    #   chunks_per_N_block = ceil(mm_N_full_block_wt / chunk_width_in_tiles)
    "mm_cores_y, mm_block_ht, mm_block_wt, mm_N_full_block_wt, chunk_width_in_mm_blocks",
    [
        # Finest granularity: M=16, N=8, chunk=1 tile each
        (1, 1, 1, 1, 1),
        # Coarsest: single M-block, single N-block, single chunk covers all
        (1, 16, 8, 8, 1),
        # Max cores (8), finest M-block, single N-block, many tiny chunks
        (8, 2, 1, 8, 1),
        # Max cores, finest M-block, wide chunks covering entire N-block
        (8, 1, 4, 8, 2),
        # Multiple N-blocks (4), chunk exactly fills each N-block
        (4, 2, 2, 2, 1),
        # Two N-blocks, many chunks per N-block (4 chunks of width 1)
        (2, 4, 1, 4, 1),
        # Single N-block, max chunks (8 chunks of 1 tile)
        (4, 4, 1, 8, 1),
        # Large chunk_width_in_mm_blocks (8), single chunk covers N-block
        (1, 8, 1, 8, 8),
        # Partial last chunk: chunk_w=6, N_block=8, chunks_per=2 (last chunk effective_w=2)
        (2, 8, 2, 8, 3),
        # Non-power-of-2 block width: chunk_w=3, N_block=8, chunks_per=3 (last chunk effective_w=2)
        (1, 4, 3, 8, 1),
        # Chunk wider than N-block: chunk_w=16 > N_block=8, clamped to single chunk of effective_w=8
        (1, 8, 2, 8, 8),
        # Partial last M-block: 1 core, slice_Ht_per_core=16, mm_block_ht=3 → 5 full + 1 partial (1 row)
        (1, 3, 2, 4, 1),
        # Partial last M-block: 2 cores, slice_Ht_per_core=8, mm_block_ht=3 → 2 full + 1 partial (2 rows)
        (2, 3, 2, 4, 1),
        # Partial last M-block: 4 cores, slice_Ht_per_core=4, mm_block_ht=3 → 1 full + 1 partial (1 row)
        (4, 3, 2, 4, 1),
        # Non-divisible slice_Ht/mm_cores_y: padded=18, per_core=6, ghost=2;
        # core 2: 4 real + 2 ghost rows; 6%4=2 → partial last M-block
        (3, 4, 2, 4, 1),
        # Non-divisible: padded=20, per_core=4; core 4: all 4 rows ghost; 4%3=1 → partial M-block
        (5, 3, 2, 4, 1),
        # Non-divisible: padded=18, per_core=3; core 5: 1 real + 2 ghost; exact M-block (3%3=0)
        (6, 3, 2, 4, 1),
        # Non-div Wt: slice_Wt=8, mm_N_full_block_wt=3 (8%3=2≠0).
        # mm_block_wt=2 → chunk_width=2 = effective_advance; same column per packet advance.
        (1, 4, 2, 3, 1),
        # Non-div Wt + non-div Ht: slice_Wt=8, mm_N_full_block_wt=5 (8%5=3≠0) AND
        # slice_Ht=16, mm_cores_y=4; padded_Ht=16 (exact).
        # mm_block_wt=2 → chunk_width=2 = effective_advance; same column per packet advance.
        (4, 4, 2, 5, 1),
        # Non-div Wt: mm_N_full_block_wt=3, but now mm_block_wt=4 → chunk_width=4 > effective_advance=2.
        # Consecutive packet slots advance by 2 within a width-4 subchunk → different columns.
        # chip k: skip = (k*8)%3 = {0,2,1,0,...}; chip 2 (skip=1): packet (col=0,col=2) →
        # col0 invalid, col2 valid — exercises the CB-packing fix path.
        (1, 4, 4, 3, 1),
        # Non-div Wt + non-div Ht: mm_N_full_block_wt=5 (8%5=3), mm_block_wt=4, mm_cores_y=2.
        # Both cross-column packets (chunk_width=4 > effective_advance=2) and ghost rows.
        (2, 4, 4, 5, 1),
    ],
    ids=[
        "finest_granularity",
        "coarsest_single_block",
        "max_cores_tiny_chunks",
        "max_cores_wide_chunks",
        "four_N_blocks",
        "two_N_blocks_many_chunks",
        "single_N_block_max_chunks",
        "large_chunk_width_in_mm_blocks",
        "partial_last_chunk",
        "non_power_of_2_block_wt",
        "chunk_wider_than_N_block",
        "partial_M_block_1core",
        "partial_M_block_2cores",
        "partial_M_block_4cores",
        "non_div_Ht_3cores_partial_Mblock",
        "non_div_Ht_5cores_all_ghost_last",
        "non_div_Ht_6cores_exact_Mblock_ghost_tail",
        "non_div_Wt_N3_into_8",
        "non_div_Wt_N5_into_8_multi_cores",
        "non_div_Wt_N3_into_8_wide_block_cross_col",
        "non_div_Wt_N5_into_8_wide_block_cross_col_non_div_Ht",
    ],
)
def test_strided_reduce_scatter_blocking_sweep(
    mesh_device,
    mm_cores_y,
    mm_block_ht,
    mm_block_wt,
    mm_N_full_block_wt,
    chunk_width_in_mm_blocks,
):
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        [4, 1, 512, 2048],
        3,  # dim
        1,  # num_links
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        rs_topology=ttnn.Topology.Ring,
        enable_trace=False,
        num_iters=1,
        small_random_ints=True,
        use_barrier=True,
        use_persistent_buffers=True,
        use_strided=True,
        verify_output_shape=True,
        verify_output_pcc=True,
        mm_cores_y=mm_cores_y,
        mm_block_ht=mm_block_ht,
        mm_block_wt=mm_block_wt,
        mm_N_full_block_wt=mm_N_full_block_wt,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}],
    indirect=True,
)
@pytest.mark.parametrize(
    # Shape [4,1,4096,4096] dim=3 with ring_size=8 gives:
    #   slice_Ht = 128 tiles, slice_Wt = 16 tiles
    #
    # NOTE: slice_Wt does NOT need to be divisible by mm_N_full_block_wt — kernels handle
    #       non-divisible slices dynamically.
    # NOTE: slice_Ht does NOT need to be divisible by mm_cores_y — ghost rows are skipped.
    #         slice_Ht_per_core does NOT need to be divisible by mm_block_ht either.
    #
    # Derived values:
    #   padded_slice_Ht = round_up(128, mm_cores_y)
    #   slice_Ht_per_core = padded_slice_Ht / mm_cores_y
    "mm_cores_y, mm_block_ht, mm_block_wt, mm_N_full_block_wt, chunk_width_in_mm_blocks",
    [
        # Coarsest: single M-block (128 rows), single N-block, single chunk
        (1, 128, 16, 16, 1),
        # Max cores, many M-blocks, tiny 1-tile chunks
        (8, 4, 1, 16, 1),
        # Max cores, wide chunks covering entire N-block
        (8, 2, 8, 16, 2),
        # 8 N-blocks, chunk exactly fills each N-block
        (4, 8, 2, 2, 1),
        # 4 N-blocks, 4 chunks per N-block
        (2, 8, 1, 4, 1),
        # Single N-block, max chunks (16 chunks of 1 tile)
        (4, 8, 1, 16, 1),
        # Large chunk_width_in_mm_blocks (16), single chunk covers N-block
        (1, 16, 1, 16, 16),
        # Partial last chunk: chunk_w=6, N_block=16, chunks=3, pattern: 6,6,4
        (2, 16, 2, 16, 3),
        # Non-power-of-2 block width: chunk_w=3, chunks=6, pattern: 3,3,3,3,3,1
        (4, 8, 3, 16, 1),
        # Asymmetric tall-narrow: mm_block_ht=16 tall, mm_block_wt=2, many chunks
        (8, 16, 2, 16, 1),
        # Chunk wider than N-block: chunk_w=32 > N_block=16, clamped to single chunk of effective_w=16
        (1, 16, 2, 16, 16),
        # Partial last M-block: 8 cores, slice_Ht_per_core=16, mm_block_ht=6 → 2 full + 1 partial (4 rows)
        (8, 6, 2, 8, 1),
        # Partial last M-block: 4 cores, slice_Ht_per_core=32, mm_block_ht=5 → 6 full + 1 partial (2 rows)
        (4, 5, 2, 4, 1),
        # Partial last M-block: 1 core, slice_Ht_per_core=128, mm_block_ht=6 → 21 full + 1 partial (2 rows)
        (1, 6, 2, 8, 1),
        # Non-divisible: padded=130, per_core=26, last core has 24 real + 2 ghost; 26%6=2 → partial M-block
        (5, 6, 2, 8, 1),
        # Non-divisible: padded=129, per_core=43, last core has 42 real + 1 ghost; 43%5=3 → partial M-block
        (3, 5, 2, 8, 1),
        # Non-divisible: padded=132, per_core=22, last core has 18 real + 4 ghost; 22%5=2 → partial M-block
        (6, 5, 2, 8, 1),
        # Non-div Wt: slice_Wt=16, mm_N_full_block_wt=5 (16%5=1≠0).
        # mm_block_wt=3 → chunk_width=3 > effective_advance=2; cross-column packets hit the
        # CB-packing fix. chip k: skip=(k*16)%5={0,1,2,3,4,0,...} — 5 distinct skip values.
        (1, 8, 3, 5, 1),
        # Non-div Wt: slice_Wt=16, mm_N_full_block_wt=3 (16%3=1≠0), mm_block_wt=4, 4 cores.
        # chunk_width=4 > effective_advance=2; cross-column. chip k skip=(k*16)%3={0,1,2,0,...}.
        (4, 4, 4, 3, 1),
    ],
    ids=[
        "coarsest_single_block",
        "max_cores_many_M_tiny_chunks",
        "max_cores_wide_chunks",
        "eight_N_blocks",
        "four_N_blocks_many_chunks",
        "single_N_block_max_chunks",
        "large_chunk_width_in_mm_blocks",
        "partial_last_chunk",
        "non_power_of_2_block_wt",
        "asymmetric_tall_narrow",
        "chunk_wider_than_N_block",
        "partial_M_block_8cores",
        "partial_M_block_4cores",
        "partial_M_block_1core",
        "non_div_Ht_5cores_partial_Mblock",
        "non_div_Ht_3cores_partial_Mblock",
        "non_div_Ht_6cores_partial_Mblock",
        "non_div_Wt_N5_into_16_wide_block_cross_col",
        "non_div_Wt_N3_into_16_wide_block_cross_col_multi_cores",
    ],
)
@pytest.mark.skip(reason="Sweep test, can take a long time to run, run manually")
def test_strided_reduce_scatter_blocking_sweep_large(
    mesh_device,
    mm_cores_y,
    mm_block_ht,
    mm_block_wt,
    mm_N_full_block_wt,
    chunk_width_in_mm_blocks,
):
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        [4, 1, 4096, 4096],
        3,  # dim
        1,  # num_links
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        rs_topology=ttnn.Topology.Ring,
        enable_trace=False,
        num_iters=1,
        small_random_ints=False,
        use_barrier=True,
        use_persistent_buffers=True,
        use_strided=True,
        verify_output_shape=True,
        verify_output_pcc=True,
        mm_cores_y=mm_cores_y,
        mm_block_ht=mm_block_ht,
        mm_block_wt=mm_block_wt,
        mm_N_full_block_wt=mm_N_full_block_wt,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}],
    indirect=True,
)
@pytest.mark.parametrize(
    # Shape [4,1,416,2048] dim=3 with ring_size=8 gives:
    #   slice_Ht = 13 tiles (prime — not divisible by 2, 3, 4, 5, 6, 7, or 8)
    #   slice_Wt = 8 tiles
    #
    # Every mm_cores_y > 1 case exercises the ghost-row path:
    #   padded_slice_Ht = round_up(13, mm_cores_y)
    #   slice_Ht_per_core = padded_slice_Ht / mm_cores_y
    #   ghost rows on last core(s) = padded_slice_Ht - 13
    #
    # NOTE: slice_Wt does NOT need to be divisible by mm_N_full_block_wt — kernels handle
    #       non-divisible slices dynamically.
    "mm_cores_y, mm_block_ht, mm_block_wt, mm_N_full_block_wt, chunk_width_in_mm_blocks",
    [
        # padded=14, per_core=7, ghost=1; 7%3=1 → partial last M-block on every core
        # core 1: rows 7-12 real (6) + row 13 ghost
        (2, 3, 2, 4, 1),
        # padded=16, per_core=4, ghost=3; core 3: only row 12 real + rows 13-15 ghost
        # M-block 1 on core 3 is entirely ghost
        (4, 3, 2, 4, 1),
        # padded=14, per_core=2, ghost=1; core 6: row 12 real + row 13 ghost (mixed M-block)
        (7, 2, 2, 8, 1),
        # padded=15, per_core=5, ghost=2; core 2: rows 10-12 real (3) + rows 13-14 ghost
        # 5%4=1 → partial last M-block; ghost rows trail into last M-block
        (3, 4, 2, 4, 1),
        # padded=16, per_core=2, ghost=3; core 6: row 12 real + row 13 ghost;
        # core 7: rows 14-15 all ghost
        (8, 2, 2, 4, 1),
        # padded=14, per_core=7, block_ht=7 → 1 M-block per core (exact fit);
        # core 1: 6 real rows + 1 ghost — ghost hides within the single M-block
        (2, 7, 2, 4, 1),
        # padded=16, per_core=4, block_ht=2 → 2 M-blocks per core;
        # core 3: row 12 real + rows 13-15 ghost → M-block 0 mixed, M-block 1 all ghost
        (4, 2, 2, 4, 1),
        # Non-div Ht + non-div Wt: slice_Ht=13, mm_cores_y=2 (padded=14, per_core=7, ghost=1);
        # slice_Wt=8, mm_N_full_block_wt=3 (8%3=2≠0).
        # mm_block_wt=2 → chunk_width=2 = effective_advance; same column per packet advance.
        (2, 3, 2, 3, 1),
        # Non-div Ht + non-div Wt: padded=16, per_core=4, ghost=3; core 3 mostly ghost;
        # slice_Wt=8, mm_N_full_block_wt=5 (8%5=3≠0).
        # mm_block_wt=2 → chunk_width=2 = effective_advance; same column per packet advance.
        (4, 3, 2, 5, 1),
        # Non-div Ht + non-div Wt + cross-column: same as above but mm_block_wt=4 →
        # chunk_width=4 > effective_advance=2; consecutive packet slots visit different columns.
        # chip k: skip=(k*8)%3={0,2,1,0,...}; chip 2 (skip=1): packet (col=0,col=2) → invalid,valid.
        (2, 3, 4, 3, 1),
        # Non-div Ht + non-div Wt + cross-column, more ghost rows: padded=16, per_core=4, ghost=3.
        # mm_N_full_block_wt=5 (8%5≠0), mm_block_wt=4 → chunk_width=4 > effective_advance=2.
        (4, 3, 4, 5, 1),
    ],
    ids=[
        "2cores_partial_last_Mblock",
        "4cores_all_ghost_Mblock",
        "7cores_prime_divisor",
        "3cores_partial_last_Mblock",
        "8cores_ghost_bleed_across_cores",
        "2cores_exact_block_ghost_tail",
        "4cores_ghost_spans_Mblocks",
        "non_div_Ht_and_Wt_N3_into_8",
        "non_div_Ht_and_Wt_N5_into_8",
        "non_div_Ht_and_Wt_N3_wide_block_cross_col",
        "non_div_Ht_and_Wt_N5_wide_block_cross_col",
    ],
)
def test_strided_reduce_scatter_blocking_sweep_non_divisible_Ht(
    mesh_device,
    mm_cores_y,
    mm_block_ht,
    mm_block_wt,
    mm_N_full_block_wt,
    chunk_width_in_mm_blocks,
):
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        [4, 1, 416, 2048],
        3,  # dim
        1,  # num_links
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        rs_topology=ttnn.Topology.Ring,
        enable_trace=False,
        num_iters=1,
        small_random_ints=True,
        use_barrier=True,
        use_persistent_buffers=True,
        use_strided=True,
        verify_output_shape=True,
        verify_output_pcc=True,
        mm_cores_y=mm_cores_y,
        mm_block_ht=mm_block_ht,
        mm_block_wt=mm_block_wt,
        mm_N_full_block_wt=mm_N_full_block_wt,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
    )
