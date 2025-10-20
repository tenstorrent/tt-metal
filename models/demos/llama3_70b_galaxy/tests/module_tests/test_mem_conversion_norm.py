import torch
import ttnn


def test_mem_conversion_norm(device):
    original_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(8, 128),
        core_grid=ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]
        ),  # [0, 0], [1, 0], ..., [7, 0]
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tensor = ttnn.from_torch(torch.randn((1, 8, 8, 128)), device=device, memory_config=original_mem_cfg)

    reshape_output_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(64, 128),
        core_grid=ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
        ),  # resharding tensor to 1 core
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    reshaped = ttnn.to_memory_config(tensor, memory_config=reshape_output_mem_cfg)
    reshaped = ttnn.reshape(reshaped, [1, 1, 64, 128])
    reshaped = ttnn.to_layout(reshaped, ttnn.TILE_LAYOUT)
