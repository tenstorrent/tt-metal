import ttnn
import torch


def test_qwen_typecast_repro(mesh_device):
    core_grid_ln, grid_offset = (10, 2), ttnn.CoreCoord(1, 0)
    core_range = ttnn.CoreRange(
        grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
    )
    num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
    gather_in_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, 1280 // num_cores_ln),  # [1, 1, 32, 64]
        core_grid=ttnn.CoreRangeSet(
            {
                core_range,
            }
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )

    inp = ttnn.from_torch(
        torch.randn((1, 1, 32, 1280)), device=mesh_device, dtype=ttnn.bfloat8_b, memory_config=gather_in_mem_cfg
    )
    inp = ttnn.typecast(inp, ttnn.bfloat16)
    assert inp.dtype == ttnn.bfloat16
    assert inp.memory_config() == gather_in_mem_cfg
