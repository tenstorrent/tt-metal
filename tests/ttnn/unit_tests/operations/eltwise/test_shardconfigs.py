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
        (ttnn.abs, 1e-10),
    ],
)
def test_ops_sharded1(device, input_shapes, ttnn_fn, atol):
    torch.manual_seed(0)
    high = 1000
    low = -1000
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (6, 6)),
        }
    )

    shard_config = ttnn.create_sharded_memory_config(
        (224, 448),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.COL_MAJOR,  # passes for both ROW and COL MAJOR orientation
        use_height_and_width_as_shard_shape=True,
        # when False RuntimeError: height and width must be shard shape with CoreRangeSet
    )

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
        (ttnn.abs, 1e-10),
    ],
)
def test_ops_sharded2(device, input_shapes, ttnn_fn, atol):
    torch.manual_seed(0)
    high = 1000
    low = -1000
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (1, 5)),
        }
    )

    shard_config = ttnn.create_sharded_memory_config(
        (192, 1024),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
        # when False: RuntimeError: height and width must be shard shape with CoreRangeSet
    )

    # core_grid=ttnn.CoreGrid(y=2, x=6),
    # num_shards_along_width <= shard_grid.y

    #  (0, 0), (5, 1) => ( y, x) ?
    # E       RuntimeError: TT_FATAL @ /home/ubuntu/Repo/tt-metal/ttnn/core/tensor/tensor_spec.cpp:86: num_shards_along_width <= shard_grid.y
    # E       info:
    # E       Number of shards along width 6 must not exceed number of rows 2 for column major orientation!
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
        (ttnn.abs, 1e-10),
    ],
)
def test_ops_sharded3(device, input_shapes, ttnn_fn, atol):
    torch.manual_seed(0)
    high = 1000
    low = -1000
    in_data1 = torch.rand((input_shapes), dtype=torch.bfloat16) * (high - low) + low

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange((0, 0), (2, 2)),
        }
    )

    shard_config = ttnn.create_sharded_memory_config(
        (416, 32),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,  # only COL MAJOR orientation fails
        use_height_and_width_as_shard_shape=True,
    )
    # Error:  num_shards_along_height <= shard_grid.x with col_major
    # E       RuntimeError: TT_FATAL @ /home/ubuntu/Repo/tt-metal/ttnn/core/tensor/tensor_spec.cpp:81: num_shards_along_height <= shard_grid.x
    # E       info:
    # E       Number of shards along height 13 must not exceed number of columns 3 for column major orientation!

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
