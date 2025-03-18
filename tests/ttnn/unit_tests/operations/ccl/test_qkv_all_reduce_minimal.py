# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from time import time
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)

from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.tt_eager.python_api_testing.unit_testing.misc.test_matmul_1d_gather_in0 import (
    num_cores_to_rectangle_grid,
    round_up,
)


def check_mesh_tensor_alloc(tensor):
    device_tensors = ttnn.get_device_tensors(tensor)
    buffer_addr = device_tensors[0].buffer_address()

    if len(device_tensors) > 1:
        for i in range(1, len(device_tensors)):
            addr = device_tensors[i].buffer_address()
            if not addr == buffer_addr:
                return False
    return True


def run_all_reduce_impl(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    enable_async=False,
):
    cluster_shape = (8, 4)

    create_persistent_fabric = True
    teardown_persistent_fabric = True
    enable_persistent_fabric = True

    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    ##################################
    ##### Set up fabric stuff
    ##################################
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
    if create_persistent_fabric:
        mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
            mesh_device, [worker_sub_device], 0, 0, enable_persistent_fabric
        )
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0)]

    logger.info(f"Output shape: {output_shape}")

    try:
        ##################################
        ##### Set up input tensors/configs
        ##################################

        ##### FF2 Case #####
        M, N = output_shape[2:]
        N_per_shard = round_up(math.ceil(N / input_num_cores), ttnn.TILE_SIZE)
        output_N_per_shard = round_up(math.ceil(N / output_num_cores), ttnn.TILE_SIZE)
        input_shape = [*cluster_shape, M, N]  # [8, 4, 32, 1280]
        intermediate_shape = [*input_shape[:-1], N * cluster_shape[cluster_axis]]

        CORE_RANGE = [(x, y) for y in range(compute_grid_size.y) for x in range(compute_grid_size.x)]
        core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in CORE_RANGE[:input_num_cores]
            ]
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_set,
                [M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_core_range_set = ttnn.num_cores_to_corerangeset(output_num_cores, ttnn.CoreCoord(8, 5), row_wise=True)
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                output_core_range_set,
                [M, output_N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        intermediate_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                output_core_range_set,
                [M, output_N_per_shard * cluster_shape[cluster_axis]],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        logger.info(f"Input shape: {input_shape[2:]}, Padded shape: {[M, N_per_shard * input_num_cores]}")
        input_tensor = torch.randn(input_shape)
        # input_tensor torch.Size([8, 4, 32, 1280])
        tt_qkv = ttnn.from_torch(
            input_tensor,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=input_dtype,
            memory_config=input_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )  # [1, 1, 32, 1280]
        check_mesh_tensor_alloc(tt_qkv)

        intermediate_tensor = torch.zeros(intermediate_shape)
        tt_intermediate_tensors = [
            ttnn.from_torch(
                intermediate_tensor,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=input_dtype,
                memory_config=intermediate_mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
        ]

        # Validate that the tensor is allocated in same location across devices
        check_mesh_tensor_alloc(tt_intermediate_tensors[0])

        # Select batch_offset with create_qkv_heads_decode instead of selection matmul
        batch_offset = [0, 8, 16, 24]
        batch_offset_tt_tensor = ttnn.as_tensor(
            torch.tensor(batch_offset, dtype=torch.int32).reshape(4, 1),
            dtype=ttnn.int32,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=mesh_device, dims=(None, 0), mesh_shape=list(mesh_device.shape)
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ##################################
        ##### Run the op
        ##################################
        # tt_qkv Shape([1, 1, 32, 1280])
        # tt_qkv_reduced Shape([1, 1, 32, 1280])
        # MemoryConfig(memory_layout=TensorMemoryLayout::WIDTH_SHARDED,buffer_type=BufferType::L1,shard_spec=ShardSpec(grid={[(x=0,y=0) - (x=7,y=4)]},shape={32, 32},orientation=ShardOrientation::ROW_MAJOR,mode=ShardMode::PHYSICAL,physical_shard_shape=std::nullopt))
        tt_qkv_reduced, *_ = ttnn.experimental.all_reduce_create_qkv_heads(
            tt_qkv,
            tt_intermediate_tensors[0],
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            queue_id=0,
            num_heads=8,
            memory_config=output_mem_config,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            subdevice_id=worker_sub_device_id,
            num_kv_heads=1,
            final_memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            overlap_qk_coregrid=False,
            batch_offset=batch_offset_tt_tensor,
            slice_size=8,
        )  # [1, 1, 32, 1280]
        # breakpoint()
        # Batch Slicing
        # 32 BS is split into 8 Mini BS across 4 devices
        ttnn.synchronize_device(mesh_device)

        (
            q_heads_pre_rot_1BQD,  # Shape([1, 8, 8, 128])
            k_heads_pre_rot_1BKD,  # Shape([1, 8, 1, 128])
            v_heads_1BKD,  # Shape([1, 8, 1, 128])
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            tt_qkv_reduced,
            num_heads=8,
            num_kv_heads=1,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            overlap_qk_coregrid=False,
            batch_offset=batch_offset_tt_tensor,
            slice_size=8,
        )  # [1, 8, 8[32], 128], [1, 8, 1[32], 128], [1, 8, 1[32], 128]
        # q MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED,buffer_type=BufferType::L1,shard_spec=ShardSpec(grid={[(x=0,y=0) - (x=7,y=0)]},shape={32, 128},orientation=ShardOrientation::ROW_MAJOR,mode=ShardMode::PHYSICAL,physical_shard_shape=std::nullopt))
        # k MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED,buffer_type=BufferType::L1,shard_spec=ShardSpec(grid={[(x=0,y=1) - (x=7,y=1)]},shape={32, 128},orientation=ShardOrientation::ROW_MAJOR,mode=ShardMode::PHYSICAL,physical_shard_shape=std::nullopt))
        # v MemoryConfig(memory_layout=TensorMemoryLayout::HEIGHT_SHARDED,buffer_type=BufferType::L1,shard_spec=ShardSpec(grid={[(x=0,y=0) - (x=7,y=0)]},shape={32, 128},orientation=ShardOrientation::ROW_MAJOR,mode=ShardMode::PHYSICAL,physical_shard_shape=std::nullopt))
        # breakpoint()
        # After ConcatMesh2dToTensor
        # [1, 8, 8[32], 128] -> [8, 32, 8[32], 128]

        # Get non-distributed tensors
        q_non_distributed = ttnn.to_torch(
            q_heads_pre_rot_1BQD,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                dims=(0, 1),
                mesh_shape=cluster_shape,
            ),
        )
        # [1, 8, 1[32], 128] -> [8, 32, 1[32], 128]
        k_non_distributed = ttnn.to_torch(
            k_heads_pre_rot_1BKD,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                dims=(0, 1),
                mesh_shape=cluster_shape,
            ),
        )
        # [1, 8, 1[32], 128] -> [8, 32, 1[32], 128]
        v_non_distributed = ttnn.to_torch(
            v_heads_1BKD,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                dims=(0, 1),
                mesh_shape=cluster_shape,
            ),
        )
        # breakpoint()

        # input = [8, 4, 32, 1280]

        # reduced_input = [8, 32, 1280]
        reduced_input_tensor = input_tensor.sum(dim=1)

        # reduced_input_reshaped = [8, 32, 10, 128]
        reduced_input_tensor_reshaped = reduced_input_tensor.reshape(8, 32, 10, 128)

        # q_output = [8, 32, 8, 128]
        q_output_tensor = reduced_input_tensor_reshaped[:, :, :8, :]

        # k_output = [8, 32, 1, 128]
        k_output_tensor = reduced_input_tensor_reshaped[:, :, 8:9, :]

        # v_output = [8, 32, 1, 128]
        v_output_tensor = reduced_input_tensor_reshaped[:, :, 9:10, :]

        # Compare results
        assert_with_pcc(q_output_tensor, q_non_distributed, 0.9999)
        assert_with_pcc(k_output_tensor, k_non_distributed, 0.9999)
        assert_with_pcc(v_output_tensor, v_non_distributed, 0.9999)

    finally:
        if enable_persistent_fabric and teardown_persistent_fabric:
            mesh_device.reset_sub_device_stall_group()
            t1 = time()
            teardown_fabric_interface(mesh_device)
            t2 = time()
            logger.info(f"Teardown time: {t2 - t1}")


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "output_shape, cluster_axis, num_links, input_num_cores, output_num_cores",
    [
        ([1, 1, 32, 1280], 1, 3, 24, 40),  # QKV all reduce
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize("enable_async", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_all_reduce(
    mesh_device,
    output_shape,
    cluster_axis,
    input_dtype,
    num_links,
    input_num_cores,
    output_num_cores,
    enable_async,
    use_program_cache,
):
    if len(mesh_device.get_devices()) != 32:
        pytest.skip("Not TG!")

    run_all_reduce_impl(
        mesh_device,
        output_shape,
        cluster_axis,
        input_dtype,
        num_links,
        input_num_cores,
        output_num_cores,
        enable_async=enable_async,
    )
