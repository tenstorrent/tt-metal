import pytest

import ttnn
import torch
from loguru import logger
from models.demos.llama3_70b_galaxy.tt.model_config import (
    get_core_ranges,
)
from models.utility_functions import torch_random


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_qwen3_tg_rs_create_heads(
    mesh_device,
):
    """
    Test rs_create_heads operation for Qwen3 model.
    Generates a random xqkv tensor and runs rs_create_heads to compute q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD.
    """
    torch.manual_seed(0)
    galaxy_type = "6U" if ttnn.GetNumPCIeDevices() == 32 else "4U"

    sub_core_grids = ttnn.CoreRangeSet(  # 50 cores total
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    qkv_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            sub_core_grids,
            [32, 128],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    _, _, _, pf_receiver_cores_list, _, _, _, _ = get_core_ranges(12, 2, False)
    pf_mm_out_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in pf_receiver_cores_list
        ]
    )

    SHARDED_QKV_OUT_RING_MEMCFG = ttnn.create_sharded_memory_config(
        shape=(32, 12288 // 8 // 24),  # (32, 64)
        core_grid=pf_mm_out_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create random input tensors
    torch_xqkv_tensor = torch_random([1, 1, 32, 1280], -1, 1, dtype=torch.bfloat16)
    # Convert to ttnn tensors with appropriate memory configuration
    xqkv_tensor = ttnn.from_torch(
        torch_xqkv_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=SHARDED_QKV_OUT_RING_MEMCFG,
    )

    # Set up persistent buffers to mimic tt_ccl settings
    RS_CREATE_HEADS_PACKET_WORKER_CRS = ttnn.CoreRangeSet(
        [  # 5 cores total
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
        ]
    )
    RS_CREATE_HEADS_INTERIM_MEMCFG = ttnn.create_sharded_memory_config(
        shape=(32, 512),
        core_grid=RS_CREATE_HEADS_PACKET_WORKER_CRS,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    cluster_shape = (8, 4)
    num_pages_per_packet = 4
    shard_height = 32

    # (*cluster_shape, shard_height, cluster_shape[1] * num_pages_per_packet * 32 * 5)
    persistent_buffers = ttnn.from_torch(
        torch.zeros((*cluster_shape, 32, 512 * 5)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=RS_CREATE_HEADS_INTERIM_MEMCFG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(8, 4)),
    )

    global_semaphore = ttnn.create_global_semaphore(mesh_device, sub_core_grids, 0)

    logger.info(f"Starting rs_create_heads")

    # Perform rs_create_heads using ttnn
    (
        q_heads_pre_rot_1BQD,
        k_heads_pre_rot_1BKD,
        v_heads_1BKD,
    ) = ttnn.experimental.llama_rs_create_heads(
        input_tensor=xqkv_tensor,
        intermediate_packet_buffer=persistent_buffers,
        dim=3,
        cross_device_semaphore=global_semaphore,
        subdevice_id=ttnn.SubDeviceId(0),
        cluster_axis=1,
        mesh_device=mesh_device,
        topology={"6U": ttnn.Topology.Ring, "4U": ttnn.Topology.Linear}.get(galaxy_type),
        num_links=1,
        num_heads=8,
        num_kv_heads=1,
        memory_config=qkv_memory_config,
        qkv_memory_config=qkv_memory_config,
        use_noc1_only=False,
        use_optimal_ccl_for_llama=False,
    )

    logger.info(f"Finished rs_create_heads")

    breakpoint()
