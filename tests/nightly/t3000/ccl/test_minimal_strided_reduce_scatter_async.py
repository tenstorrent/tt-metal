# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
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
    mm_N_block_wt: object = None  # Optional[int]
    chunk_width_in_mm_blocks: object = None  # Optional[int]
    num_workers_per_link: object = None  # Optional[int]


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
    mm_N_block_wt=None,
    chunk_width_in_mm_blocks=None,
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
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
                mm_cores_y=mm_cores_y,
                mm_block_ht=mm_block_ht,
                mm_block_wt=mm_block_wt,
                mm_N_block_wt=mm_N_block_wt,
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

                eq, output = comp_pcc(tt_output_tensor, torch_output_tensor)
                logger.info(f"{output}, device {device_id}, iteration {i}")

                if verify_output_shape:
                    assert (
                        tt_output_tensor.shape == torch_output_tensor.shape
                    ), f"Shape mismatch: {tt_output_tensor.shape} != {torch_output_tensor.shape}"
                if verify_output_pcc:
                    assert eq, f"{i} FAILED reduce scatter: {output}"

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
                mm_N_block_wt=2,
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
                mm_N_block_wt=2,
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
                mm_N_block_wt=4,
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
                mm_N_block_wt=4,
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
                mm_N_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_4x4_2x8_mm_grid",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 16 x 8 tiles with eight 4 x 4 mm blocks from an assumed 2 x 8 mm core grid
                # chunk is 8 tileswide
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
                mm_N_block_wt=8,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_16x8_2x8_mm_grid_chunk_2_mm_blocks_wide_multi_iter",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 4 x 4 tiles with four 2 x 2 mm blocks from an assumed 2 x 8 mm core grid
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
                mm_N_block_wt=8,
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
                mm_N_block_wt=8,
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
                mm_N_block_wt=8,
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
                mm_N_block_wt=10,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_16x10_partial_last_chunk_4x4_blocks",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 16 x 10 tiles (non-power-of-2 slice width from 2560-wide input)
                # from an assumed 2 x 8 mm core grid with smaller 2x2 mm blocks
                # larger chunk_width_in_mm_blocks=4 (chunk width 8), partial last chunk (effective width 2)
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
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_block_wt=10,
                chunk_width_in_mm_blocks=4,
            ),
            id="experimental_strided_16x10_partial_last_chunk_2x2_blocks_large_chunk_mm",
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
                mm_N_block_wt=8,
                chunk_width_in_mm_blocks=1,
                num_workers_per_link=6,
            ),
            id="experimental_strided_128x16_8_cores_2_N_blocks_6_workers",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 128 x 16 tiles with balanced 8x8 mm blocks
                # from an assumed 4 x 8 mm core grid;
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
                mm_N_block_wt=8,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_128x16_4_cores_balanced_8x8_blocks_2_N_blocks",
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
                mm_N_block_wt=16,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_128x16_4_cores_single_N_block_wide_chunk",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                # each device has 128 x 16 tiles with 8x8 mm blocks
                # from an assumed 8 x 8 mm core grid; single N-block spanning full width
                # same as above but 8 core rows (2 M-blocks/core vs 4)
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
                mm_block_ht=8,
                mm_block_wt=8,
                mm_N_block_wt=16,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_128x16_8_cores_single_N_block_wide_chunk",
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
        mm_N_block_wt,
        chunk_width_in_mm_blocks,
        num_workers_per_link,
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
        mm_N_block_wt=mm_N_block_wt,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        num_workers_per_link=num_workers_per_link,
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
    #   input_Ht = 16 tiles, slice_Wt = 8 tiles
    #
    # Constraints:
    #   slice_Ht (16) % mm_cores_y == 0
    #   (slice_Ht / mm_cores_y) % mm_block_ht == 0
    #   slice_Wt (8) % mm_N_block_wt == 0
    #
    # Derived values:
    #   M_blocks_per_core = (16 / mm_cores_y) / mm_block_ht
    #   N_blocks_per_slice = 8 / mm_N_block_wt
    #   chunk_width_in_tiles = chunk_width_in_mm_blocks * mm_block_wt
    #   chunks_per_N_block = ceil(mm_N_block_wt / chunk_width_in_tiles)
    "mm_cores_y, mm_block_ht, mm_block_wt, mm_N_block_wt, chunk_width_in_mm_blocks",
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
    ],
)
def test_strided_reduce_scatter_blocking_sweep(
    mesh_device,
    mm_cores_y,
    mm_block_ht,
    mm_block_wt,
    mm_N_block_wt,
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
        mm_N_block_wt=mm_N_block_wt,
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
    #   input_Ht = 128 tiles, slice_Wt = 16 tiles
    #
    # Constraints:
    #   slice_Ht (128) % mm_cores_y == 0
    #   (slice_Ht / mm_cores_y) % mm_block_ht == 0
    #   slice_Wt (16) % mm_N_block_wt == 0
    "mm_cores_y, mm_block_ht, mm_block_wt, mm_N_block_wt, chunk_width_in_mm_blocks",
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
    ],
)
# @pytest.mark.skip(reason="Sweep test, can take a long time to run, run manually")
def test_strided_reduce_scatter_blocking_sweep_large(
    mesh_device,
    mm_cores_y,
    mm_block_ht,
    mm_block_wt,
    mm_N_block_wt,
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
        mm_N_block_wt=mm_N_block_wt,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
    )
