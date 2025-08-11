import pytest

import ttnn
import torch
from loguru import logger
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_qwen_mlp_rs(mesh_device):
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=1, dummy_weights=False, max_seq_len=128)
    model_config = model_args.get_model_config()

    cores = model_args.sub_core_grids
    global_semaphores = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)]

    PACKET_WORKER_CRS = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
        ]
    )
    REDUCE_SCATTER_INTERIM_MEMCFG = ttnn.create_sharded_memory_config(
        shape=(32, 512),
        core_grid=PACKET_WORKER_CRS,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Input tensor shape: [1, 1, 32, 3840]

    # 512 = 4 devices * 4 pages per packet * 32 tile_width
    persistent_interim_buffers = ttnn.from_torch(  # We need to be able to fit 2*3840 into the second dimension
        torch.zeros((*(8, 4), 32, 3200)),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=REDUCE_SCATTER_INTERIM_MEMCFG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(8, 4)),
    )

    persistent_output_buffers = ttnn.from_torch(
        torch.zeros((*(8, 4), 32, 3200 // 4)),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=model_config["REDUCE_SCATTER_OUT_MEMCFG"],
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(8, 4)),
    )

    w1_out = ttnn.from_torch(
        torch.randn(
            (1, 1, 32, 3200),
        ),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
    )

    logger.info(f"Starting RS")

    # w1_out = ttnn.to_memory_config(w1_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w1_out_reduced = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor=w1_out,  # [1, 1, 32, 3200]
        persistent_output_buffers=[persistent_interim_buffers, persistent_output_buffers],
        dim=3,
        multi_device_global_semaphore=global_semaphores,
        num_links=1,
        memory_config=model_config["REDUCE_SCATTER_OUT_MEMCFG"],
        topology=model_config["CCL_TOPOLOGY"],
        cluster_axis=1,
    )

    logger.info(f"Finished RS")

    # w1_out_reduced = ttnn.to_memory_config(w1_out_reduced, memory_config=model_config["REDUCE_SCATTER_OUT_MEMCFG"])

    logger.info(f"w1_out_reduced: {w1_out_reduced}")
    breakpoint()
