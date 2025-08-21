import torch
import ttnn
import pytest


# (1, 2, 112, 448),
# ttnn.bfloat16,
# ttnn.TILE_LAYOUT,
# ttnn.ShardStrategy.BLOCK,
# ttnn.ShardOrientation.COL_MAJOR,
# hw_as_shard_shape False,
# coregridx=7,
# coregridyy=7
@pytest.mark.parametrize(
    "input_shapes",
    (([1, 2, 112, 448]),),
)
@pytest.mark.parametrize(
    "ttnn_fn, atol",
    [
        (ttnn.neg, 1e-10),
        (ttnn.identity, 1e-10),
    ],
)
def test_ops_sharded1(device, input_shapes, ttnn_fn, atol):
    torch.manual_seed(0)
    high = 1000
    low = -1000
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (7, 7)),
        }
    )
    shard_shape = [32, 64]  # N*C*H // 7 and W // 7
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardMode.PHYSICAL)
    shard_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=shard_config,
    )

    output_tensor = ttnn_fn(
        input_tensor1,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_function(in_data1)

    assert torch.equal(golden_tensor, output_tensor)

    # (4, 48, 1024),
    # ttnn.bfloat16,
    # ttnn.TILE_LAYOUT,
    # ttnn.ShardStrategy.BLOCK,
    # ttnn.ShardOrientation.COL_MAJOR,
    # hw_as_shard_shape False,
    # coregridx=6,
    # coregridy=2,


@pytest.mark.parametrize(
    "input_shapes",
    (([1, 4, 48, 1024]),),
)
@pytest.mark.parametrize(
    "ttnn_fn, atol",
    [
        (ttnn.neg, 1e-10),
        (ttnn.identity, 1e-10),
    ],
)
def test_ops_sharded2(device, input_shapes, ttnn_fn, atol):
    torch.manual_seed(0)
    high = 1000
    low = -1000
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (7, 7)),
        }
    )
    shard_shape = [32, 512]  # N*C*H // 6 and W // 2
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardMode.PHYSICAL)
    shard_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=shard_config,
    )

    output_tensor = ttnn_fn(
        input_tensor1,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_function(in_data1)

    assert torch.equal(golden_tensor, output_tensor)


# (416, 32),
#      ttnn.bfloat16,
#      ttnn.TILE_LAYOUT,
#       ttnn.ShardStrategy.BLOCK,
#        ttnn.ShardOrientation.COL_MAJOR,
#        hw_as_shard_shape True,
#        coregridx=3,
#        coregridy=3,
@pytest.mark.parametrize(
    "input_shapes",
    (([1, 1, 416, 32]),),
)
@pytest.mark.parametrize(
    "ttnn_fn, atol",
    [
        (ttnn.neg, 1e-10),
        (ttnn.identity, 1e-10),
    ],
)
def test_ops_sharded3(device, input_shapes, ttnn_fn, atol):
    torch.manual_seed(0)
    high = 1000
    low = -1000
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (7, 7)),
        }
    )
    shard_shape = [104, 32]  # what is the correct shard shape here ?
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardMode.LOGICAL)
    shard_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec)

    input_tensor1 = ttnn.from_torch(
        in_data1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=shard_config,
    )

    output_tensor = ttnn_fn(
        input_tensor1,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_function(in_data1)

    assert torch.equal(golden_tensor, output_tensor)
