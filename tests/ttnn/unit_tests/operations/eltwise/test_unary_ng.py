# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

height_sharded_memory_config = ttnn.create_sharded_memory_config(
    [128, 160],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6)), ttnn.CoreRange((1, 0), (1, 6))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

width_sharded_memory_config = ttnn.create_sharded_memory_config(
    [1792, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3)), ttnn.CoreRange((1, 0), (1, 3))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [256, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([4, 7, 64, 128]),
    ],
)
@pytest.mark.parametrize(
    "input_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize(
    "out_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize(
    "ttnn_op, dtype",
    [
        (ttnn.abs, ttnn.bfloat16),
        (ttnn.neg, ttnn.bfloat16),
    ],
)
def test_unary_sharded_interleaved(input_shape, input_config, out_config, ttnn_op, dtype, device):
    """Test unary_ng ops with all combinations of sharded/interleaved input and output."""
    torch.manual_seed(42)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(input_shape, dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, device=device)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_config,
    )

    ttnn_output = ttnn.to_torch(ttnn_op(ttnn_input, memory_config=out_config))
    assert torch.equal(ttnn_output, torch_output)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 1, 32, 64]),
        torch.Size([1, 1, 64, 128]),
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.abs, ttnn.neg])
def test_unary_ng_row_major(input_shape, ttnn_op, device):
    """Test unary_ng ops with ROW_MAJOR layout on interleaved tensors."""
    torch.manual_seed(42)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, device=device)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.to_torch(ttnn_op(ttnn_input))
    assert torch.equal(ttnn_output, torch_output)


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        (
            torch.Size([1, 2, 32, 960]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            torch.Size([1, 7, 32, 96]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.abs, ttnn.neg])
def test_unary_ng_sub_core_grids(shape, sub_core_grid, ttnn_op, device):
    """Test unary_ng ops with sub_core_grids on interleaved tensors."""
    torch.manual_seed(0)
    torch_input = torch.empty(shape, dtype=torch.bfloat16).uniform_(-100, 100)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.to_torch(ttnn_op(ttnn_input, sub_core_grids=sub_core_grid))
    assert torch.equal(ttnn_output, golden_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([4, 7, 64, 128]),
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.abs, ttnn.neg])
def test_unary_ng_dram_sharded_fallback(input_shape, ttnn_op, device):
    """Test that DRAM-backed sharded tensors fall back to interleaved path gracefully.

    When either input or output uses DRAM sharding, is_native_L1_sharding returns false
    and unary_ng falls back to the TensorAccessor (interleaved) path.
    """
    torch.manual_seed(42)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, device=device)

    dram_sharded_config = ttnn.create_sharded_memory_config(
        [128, 160],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6)), ttnn.CoreRange((1, 0), (1, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.to_torch(ttnn_op(ttnn_input, memory_config=dram_sharded_config))
    assert torch.equal(ttnn_output, torch_output)


@pytest.mark.parametrize("ttnn_op", [ttnn.abs, ttnn.neg])
def test_unary_ng_uneven_sharding_fallback(ttnn_op, device):
    """Test that uneven sharding falls back to interleaved path gracefully.

    When the tensor dimensions don't divide evenly into the shard shape, is_uneven()
    returns true and unary_ng falls back to the TensorAccessor (interleaved) path
    instead of using the native sharded path.
    """
    torch.manual_seed(42)
    input_shape = torch.Size([1, 1, 160, 96])
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, device=device)

    # Shard height 64 does not divide evenly into tensor height 160 (160 / 64 = 2.5),
    # so the last core gets a partial shard → is_uneven returns true → fallback.
    uneven_shard_config = ttnn.create_sharded_memory_config(
        [64, 96],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 2))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=uneven_shard_config,
    )

    ttnn_output = ttnn.to_torch(ttnn_op(ttnn_input, memory_config=uneven_shard_config))
    assert torch.equal(ttnn_output, torch_output)
