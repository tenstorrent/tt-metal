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

                # print(f"iteration {i}, device {device_id}")
                # print(f"tt_output_tensor: {tt_output_tensor}")
                # print(f"torch_output_tensor: {torch_output_tensor}")
                # print(f"--------------------------------")

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
                rs_input_shape=[1, 1, 64, 512],
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
                small_random_ints=False,
                mm_cores_y=1,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="experimental_strided_minimal_correctness_check_1",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[1, 1, 64, 512],
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
            id="experimental_strided_minimal_correctness_check_2",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_minimal_correctness_check_3",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_1_correctness_check",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_2_correctness_check",
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[8, 1, 128, 1024],
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
            id="experimental_strided_toy_3_correctness_check_1",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_3_correctness_check_2",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_3_correctness_check_3",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
                mm_cores_y=1,
                mm_block_ht=4,
                mm_block_wt=4,
                mm_N_block_wt=8,
                chunk_width_in_mm_blocks=2,
            ),
            id="experimental_strided_toy_4_correctness_check_1",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            ),
            id="experimental_strided_toy_4_correctness_check_2",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_4_correctness_check_2",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_4_correctness_check_3",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_4_correctness_check_3",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_5_correctness_check",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_6_correctness_check",
        ),
        pytest.param(
            ReduceScatterTestConfig(
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
            id="experimental_strided_toy_7_correctness_check",
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
    )
