# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import copy
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import is_unsupported_case
from models.common.utility_functions import skip_for_blackhole

from tracy import signpost


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def run_strided_all_gather_impl(
    mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    other_dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
    mm_cores_y,
    mm_block_h,
    mm_block_w,
    num_iters=1,
    enable_trace=True,
    cluster_axis=1,
    tiles_per_chunk=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    allowed_pcc=1,
    skip_check=False,
    num_l1_banks=64,
):
    torch.manual_seed(0)

    tile = (32, 32)

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape,
        dim,
        mem_config_ag,
        num_devices,
        num_links,
        ag_input_dtype,
        layout,
        tile,
        num_l1_banks,
        mem_config_input,
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    ##### All gather setup #####
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, num_devices, all_cores, 0) for _ in range(num_iters)]

    ##### All gather input setup #####
    logger.info(f"All gather output shape: {ag_output_shape}")
    logger.info(f"All gather dim: {dim}")

    input_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []
    _, _, _, hidden_dim = ag_output_shape

    shard_dims = [other_dim, dim]
    for i in range(num_iters):
        ag_output_tensor = torch.rand(ag_output_shape).bfloat16()
        ag_output_tensor_goldens_list.append(ag_output_tensor)

        input_tensor_mesh = ttnn.from_torch(
            ag_output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=ag_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    persistent_buffer_shape = copy.deepcopy(ag_output_shape)
    persistent_buffer_shape[other_dim] = persistent_buffer_shape[other_dim] // mesh_device.shape[0]
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(persistent_buffer_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ag_input_dtype,
            memory_config=mem_config_ag,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(num_iters)
    ]
    logger.info("Done creating persistent buffers")

    ##### Perform the TT ops #####
    tt_all_gather_out_tensor_list = []

    def run_op(i):
        tt_all_gather_out_tensor = ttnn.experimental.strided_all_gather_async(
            input_tensor_mesh_list[i],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=dim,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            num_links=num_links,
            memory_config=mem_config_ag,
            topology=all_gather_topology,
            cluster_axis=cluster_axis,
            tiles_per_chunk=tiles_per_chunk,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=num_buffers_per_channel,
            mm_cores_y=mm_cores_y,
            mm_block_ht=mm_block_h // 32,
            mm_block_wt=mm_block_w // 32,
        )

        return tt_all_gather_out_tensor

    if enable_trace:
        # Compile the op
        run_op(0)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_all_gather_out_tensor = run_op(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Done capturing trace")

        # Execute trace
        signpost("start")
        for i in range(num_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
        logger.info(f"Done executing trace")
        signpost("stop")
    else:
        for i in range(num_iters):
            ttnn.synchronize_device(mesh_device)
            tt_all_gather_out_tensor = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    if not skip_check:
        for i in range(num_iters):
            tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
            torch_ag_out_tensor = ag_output_tensor_goldens_list[i if not enable_trace else 0]
            expected_tensor = torch_ag_out_tensor

            tt_ag_out = ttnn.from_device(tt_ag_out_tensor)
            tt_ag_out = ttnn.to_torch(
                tt_ag_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims
                ),
            )

            tt_ag_out = tt_ag_out[:, :, :, 0 : expected_tensor.shape[3]]
            eq, output = comp_pcc(tt_ag_out, expected_tensor, allowed_pcc)
            logger.info(f"{output}, iteration {i}")
            assert eq, f"{i} FAILED ag: {output}"


# tiles_per_chunk needs to be divisible by num_workers_per_link
# mm_cores_y is the number of in0 first col cores
# mm_block_h and mm_block_w is the mm_block of a single mm_core_y
# so the result of one chunk transfer will be mm_cores_y * mm_block_h * mm_block_w, which will be tiles_per_chunk.  tiles_per_chunk % num_workers_per_link must equal 0
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, other_dim, num_workers_per_link, tiles_per_chunk, layout, ag_input_dtype, mm_cores_y, mm_block_h, mm_block_w",
    [
        # ([1, 1, 32, 256], 3, 2, 1, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 32),
        # ([1, 1, 32, 512], 3, 2, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 64),
        # ([1, 1, 32, 512], 3, 2, 1, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 32),
        # ([1, 1, 32, 512], 3, 2, 1, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 64),
        # ([1, 1, 32, 768], 3, 2, 1, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 32),
        # ([1, 1, 32, 1024], 3, 2, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 64),
        # # 2 row tests
        # ([1, 1, 64, 256], 3, 2, 1, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 64, 32),
        # ([1, 1, 64, 256], 3, 2, 1, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 32),
        # ([1, 1, 64, 512], 3, 2, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 64),
        # # 4 row tests
        # ([1, 1, 128, 256], 3, 2, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 64, 32),
        # # Multiple y core tests
        # ([1, 1, 128, 256], 3, 2, 1, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 2, 32, 32),
        # ([1, 1, 128, 256], 3, 2, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 2, 32, 32),
        # Full tests
        ([1, 1, 4096, 2560], 3, 2, 2, 1024, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 4096, 320),
    ],
    ids=[
        # "1tile1chunk1worker1row",
        # "1tile1chunk2worker1row",
        # "1tile2chunk1worker1row",
        # "2tile1chunk1worker1row",
        # "1tile3chunk1worker1row",
        # "2tile2chunk2worker1row",
        # # 2 row tests
        # "2tile1chunk1worker2row",
        # "1tile2chunk1worker2row",
        # "2tile2chunk2worker2row",
        # # 4 row tests
        # "2tile2chunk2worker4row",
        # # Multiple y core tests
        # "2tile2chunk1worker4row2ycores",
        # "2tile2chunk2worker4row2ycores",
        # Full tests
        "4k4k",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
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
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_all_gather_async(
    mesh_device,
    ag_output_shape,
    dim,
    other_dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    num_workers_per_link,
    tiles_per_chunk,
    mm_cores_y,
    mm_block_h,
    mm_block_w,
):
    run_strided_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        ag_output_shape,
        dim,
        other_dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        num_workers_per_link=num_workers_per_link,
        tiles_per_chunk=tiles_per_chunk,
        mm_cores_y=mm_cores_y,
        mm_block_h=mm_block_h,
        mm_block_w=mm_block_w,
    )
