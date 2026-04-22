import torch
import pytest
import ttnn


@pytest.fixture
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def make_block_sharded(grid_r: int, grid_c: int, shard_h: int, shard_w: int) -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_c - 1, grid_r - 1))]),
            [shard_h, shard_w],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def test_reshape_block_sharded_dispatches_twice(device):
    """
    Shape: [8, 197, 768] -> [1576, 768]  (ViT-Base QKV-projection reshape)
    Grid: 8x8 cores, shard [224, 96] per core.

    Expected: 1 x ReshapeViewDeviceOperation per ttnn.reshape call.
    Actual  : 2 x ReshapeViewDeviceOperation for block-sharded output; 1 for interleaved.
    """
    B, S, H = 8, 197, 768
    flat = B * S  # 1576 (padded to 1600)

    input_mem = make_block_sharded(8, 8, shard_h=224, shard_w=96)
    output_mem = make_block_sharded(8, 8, shard_h=224, shard_w=96)

    t = torch.randn(B, S, H, dtype=torch.bfloat16)
    t_tt = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    t_sharded = ttnn.to_memory_config(t_tt, input_mem)

    # Triggers the double dispatch (2x ReshapeViewDeviceOperation in tracy)
    result_sharded = ttnn.reshape(t_sharded, [flat, H], memory_config=output_mem)

    result_cpu = ttnn.to_torch(result_sharded)
    assert torch.allclose(result_cpu[:flat], t.reshape(flat, H), atol=1e-1, rtol=1e-1)

    # Contrast: force interleaved output -> single dispatch (1x ReshapeViewDeviceOperation)
    t_sharded2 = ttnn.to_memory_config(t_tt, input_mem)
    _ = ttnn.reshape(t_sharded2, [flat, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn.deallocate(t_tt)
    ttnn.deallocate(t_sharded)
    ttnn.deallocate(result_sharded)
    ttnn.deallocate(t_sharded2)
