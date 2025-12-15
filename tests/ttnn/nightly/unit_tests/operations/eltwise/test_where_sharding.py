# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def make_condition_tensor(shape, dtype, condition, stride=8):
    C = torch.ones(shape, dtype=dtype) * condition
    C_flat = C.flatten()
    C_flat[::stride] = 1 - condition
    return C_flat.reshape(shape)


# TTS/TST no_bcast sharded cases
@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_where_with_height_sharding(
    device, input_a_sharded, condition, input_b_sharded, out_sharded, shard_orientation, tor_dtype, ttnn_dtype
):
    torch.manual_seed(0)
    shape = (1, 1, 512, 512)

    torch_input_tensor_a = make_condition_tensor(shape, tor_dtype, condition, stride=8)
    torch_input_tensor_b = torch.rand(shape, dtype=tor_dtype)
    scalar = 15.5

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (512 // 8, 512)
    else:
        shard_shape = (512, 512 // 8)

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_input_tensor_a.bool(), torch_input_tensor_b, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, height_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(input_tensor_a, input_tensor_b, scalar, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_where_with_width_sharding(
    device, input_a_sharded, input_b_sharded, out_sharded, condition, shard_orientation, tor_dtype, ttnn_dtype
):
    torch.manual_seed(0)
    shape = (1, 1, 512, 512)
    torch_input_tensor_a = make_condition_tensor(shape, tor_dtype, condition, stride=8)
    torch_input_tensor_b = torch.rand(shape, dtype=tor_dtype)
    scalar = 15.5

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (512, 512 // 8)
    else:
        shard_shape = (512 // 8, 512)

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_input_tensor_a.bool(), torch_input_tensor_b, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, width_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(input_tensor_a, input_tensor_b, scalar, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_where_with_block_sharding(
    device, input_a_sharded, input_b_sharded, out_sharded, condition, shard_orientation, tor_dtype, ttnn_dtype
):
    torch.manual_seed(0)
    shape = (1, 1, 512, 512)

    torch_input_tensor_a = make_condition_tensor(shape, tor_dtype, condition, stride=8)
    torch_input_tensor_b = torch.rand(shape, dtype=tor_dtype)
    scalar = 15.5

    shard_shape = (512 // 2, 512 // 4)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_input_tensor_a.bool(), torch_input_tensor_b, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, block_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(input_tensor_a, input_tensor_b, scalar, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(torch_output_tensor, output_tensor)


# TTS/TST bcast sharded cases
@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("scalar", [15.5, -0.25])
def test_where_sharded_bcast_hw_mixed_width(device, dtype_pt, dtype_tt, condition, scalar):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 1, 1, 7 * 32])
    b_shape = torch.Size([1, 1, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 1 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config),
    )

    for src_config, dst_config in input_combinations:
        a_pt = make_condition_tensor(a_shape, dtype_pt, condition, stride=8)
        b_pt = torch.rand(b_shape, dtype=dtype_pt) * 100 - 50

        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=src_config,
        )
        b_tt = ttnn.from_torch(
            b_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dst_config,
        )

        out_pt_tts = torch.where(a_pt.bool(), b_pt, scalar)
        out_tt_tts = ttnn.where(a_tt, b_tt, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out_tt_tts_shard = ttnn.to_torch(out_tt_tts)
        assert torch.equal(out_pt_tts, out_tt_tts_shard)

        out_pt_tst = torch.where(a_pt.bool(), scalar, b_pt)
        out_tt_tst = ttnn.where(a_tt, scalar, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out_tt_tst_shard = ttnn.to_torch(out_tt_tst)
        assert torch.equal(out_pt_tst, out_tt_tst_shard)

        out_tt_tts = ttnn.where(a_tt, b_tt, scalar)
        out_tt_tts_shard = ttnn.to_torch(out_tt_tts)
        assert torch.equal(out_pt_tts, out_tt_tts_shard)

        out_tt_tst = ttnn.where(a_tt, scalar, b_tt)
        out_tt_tst_shard = ttnn.to_torch(out_tt_tst)
        assert torch.equal(out_pt_tst, out_tt_tst_shard)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("scalar", [15.5, -0.25])
def test_where_sharded_bcast_w_width(device, dtype_pt, dtype_tt, condition, scalar):
    torch.manual_seed(0)
    a_shape = torch.Size([1, 2, 2 * 32, 40 * 32])
    b_shape = torch.Size([1, 1, 2 * 32, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2 * 2, 10 * 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [32 * 2 * 1, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = make_condition_tensor(a_shape, dtype_pt, condition, stride=8)
        b_pt = torch.rand(b_shape, dtype=dtype_pt) * 100 - 50

        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=a_config,
        )
        b_tt = ttnn.from_torch(
            b_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=b_config,
        )

        out_pt_tts = torch.where(a_pt.bool(), b_pt, scalar)
        out_tt_sharded = ttnn.where(a_tt, b_tt, scalar, memory_config=out_config)
        out_tt_sharded_tts = ttnn.to_torch(out_tt_sharded)
        assert torch.equal(out_tt_sharded_tts, out_pt_tts)

        out_pt_tst = torch.where(a_pt.bool(), scalar, b_pt)
        out_tt_sharded = ttnn.where(a_tt, scalar, b_tt, memory_config=out_config)
        out_tt_sharded_tst = ttnn.to_torch(out_tt_sharded)
        assert torch.equal(out_tt_sharded_tst, out_pt_tst)

        out_tt_tts = ttnn.where(a_tt, b_tt, scalar)
        out_tt_tts_shard = ttnn.to_torch(out_tt_tts)
        assert torch.equal(out_pt_tts, out_tt_tts_shard)

        out_tt_tst = ttnn.where(a_tt, scalar, b_tt)
        out_tt_tst_shard = ttnn.to_torch(out_tt_tst)
        assert torch.equal(out_pt_tst, out_tt_tst_shard)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("scalar", [15.5, -0.25])
def test_where_sharded_bcast_h_width(device, dtype_pt, dtype_tt, condition, scalar):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 1, 64, 7 * 32])
    b_shape = torch.Size([1, 1, 1, 7 * 32])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 1 * 64, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = make_condition_tensor(a_shape, dtype_pt, condition, stride=8)
        b_pt = torch.rand(b_shape, dtype=dtype_pt) * 100 - 50
        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=a_config,
        )
        b_tt = ttnn.from_torch(
            b_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=b_config,
        )

        out_pt_tts = torch.where(a_pt.bool(), b_pt, scalar)
        out_tt_sharded = ttnn.where(a_tt, b_tt, scalar, memory_config=out_config)
        out_tt_sharded_tts = ttnn.to_torch(out_tt_sharded)
        assert torch.equal(out_tt_sharded_tts, out_pt_tts)

        out_pt_tst = torch.where(a_pt.bool(), scalar, b_pt)
        out_tt_sharded = ttnn.where(a_tt, scalar, b_tt, memory_config=out_config)
        out_tt_sharded_tst = ttnn.to_torch(out_tt_sharded)
        assert torch.equal(out_tt_sharded_tst, out_pt_tst)

        out_tt_tts = ttnn.where(a_tt, b_tt, scalar)
        out_tt_tts_shard = ttnn.to_torch(out_tt_tts)
        assert torch.equal(out_pt_tts, out_tt_tts_shard)

        out_tt_tst = ttnn.where(a_tt, scalar, b_tt)
        out_tt_tst_shard = ttnn.to_torch(out_tt_tst)
        assert torch.equal(out_pt_tst, out_tt_tst_shard)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("scalar", [15.5, -0.25])
def test_where_sharded_bcast_scalar_width(device, dtype_pt, dtype_tt, condition, scalar):
    torch.manual_seed(0)
    a_shape = torch.Size([2, 1, 64, 7 * 32])
    b_shape = torch.Size([1, 1, 1, 1])

    a_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 1 * 64, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    b_sharded_config = ttnn.create_sharded_memory_config(
        [1 * 32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_combinations = (
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, b_sharded_config, a_sharded_config),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, ttnn.DRAM_MEMORY_CONFIG, a_sharded_config),
        (a_sharded_config, b_sharded_config, ttnn.DRAM_MEMORY_CONFIG),
        (a_sharded_config, b_sharded_config, a_sharded_config),
    )

    for a_config, b_config, out_config in input_combinations:
        a_pt = make_condition_tensor(a_shape, dtype_pt, condition, stride=8)
        b_pt = torch.rand(b_shape, dtype=dtype_pt) * 100 - 50

        a_tt = ttnn.from_torch(
            a_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=a_config,
        )
        b_tt = ttnn.from_torch(
            b_pt,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=b_config,
        )

        out_pt_tts = torch.where(a_pt.bool(), b_pt, scalar)
        out_tt_sharded = ttnn.where(a_tt, b_tt, scalar, memory_config=out_config)
        out_tt_sharded_tts = ttnn.to_torch(out_tt_sharded)
        assert torch.equal(out_tt_sharded_tts, out_pt_tts)

        out_pt_tst = torch.where(a_pt.bool(), scalar, b_pt)
        out_tt_sharded = ttnn.where(a_tt, scalar, b_tt, memory_config=out_config)
        out_tt_sharded_tst = ttnn.to_torch(out_tt_sharded)
        assert torch.equal(out_tt_sharded_tst, out_pt_tst)

        out_tt_tts = ttnn.where(a_tt, b_tt, scalar)
        out_tt_tts_shard = ttnn.to_torch(out_tt_tts)
        assert torch.equal(out_pt_tts, out_tt_tts_shard)

        out_tt_tst = ttnn.where(a_tt, scalar, b_tt)
        out_tt_tst_shard = ttnn.to_torch(out_tt_tst)
        assert torch.equal(out_pt_tst, out_tt_tst_shard)


@pytest.mark.parametrize(
    "test_shapes",
    ([[1, 1, 2304, 1792], [1, 1, 2112, 1792]],),
)
# HEIGHT SHARDING test - tests program cache with different shapes
def test_where_height_sharded_different_shapes(test_shapes, device):
    grid_size = device.compute_with_storage_grid_size()

    if grid_size.x < 5 or grid_size.y < 4:
        pytest.skip(
            f"This test is intended to run on devices with at least 5x4 core grid. Core grid: {grid_size.x}x{grid_size.y}"
        )

    import math

    for iteration, shape in enumerate(test_shapes):
        # Generate random tensors
        torch_predicate = torch.randint(0, 2, shape, dtype=torch.bfloat16)
        torch_true = torch.rand(shape, dtype=torch.bfloat16)
        torch_false = torch.rand(shape, dtype=torch.bfloat16)

        # Calculate shard dimensions
        total_rows = math.prod(shape[:-1])  # 2304 or 2112
        total_cols = shape[-1]  # 1792

        num_cores = 20

        shard_height = math.ceil(total_rows / num_cores / 32) * 32
        shard_width = math.ceil(total_cols / 32) * 32

        # Create HEIGHT_SHARDED memory config
        sharded_memory_config = ttnn.create_sharded_memory_config(
            shape=(shard_height, shard_width),
            core_grid=ttnn.CoreGrid(y=4, x=5),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Create tensors on device as HEIGHT_SHARDED
        predicate_tensor = ttnn.from_torch(
            torch_predicate,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=sharded_memory_config,
        )
        true_tensor = ttnn.from_torch(
            torch_true,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=sharded_memory_config,
        )
        false_tensor = ttnn.from_torch(
            torch_false,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=sharded_memory_config,
        )

        ttnn.where(predicate_tensor, true_tensor, false_tensor, output_tensor=predicate_tensor)
        output_tensor = ttnn.to_torch(predicate_tensor)

        # Validate results
        golden_fn = ttnn.get_golden_function(ttnn.where)
        torch_output_tensor = golden_fn(torch_predicate.bool(), torch_true, torch_false)
        assert torch.equal(torch_output_tensor, output_tensor)

        # Cleanup before next iteration
        ttnn.deallocate(predicate_tensor)
        ttnn.deallocate(true_tensor)
        ttnn.deallocate(false_tensor)
