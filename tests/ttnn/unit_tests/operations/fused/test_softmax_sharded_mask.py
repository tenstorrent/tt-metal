# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Sharded scale_mask_softmax_in_place when the input shards and the mask do not line up the simple way:
# shards that start mid-batch or span several batches, a one-batch mask shared by every batch, a sharded
# mask without is_causal_mask, and the mask layouts and shapes the sharded readers cannot walk.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

TILE = 32
SCALE = 0.125


def _height_sharded(num_cores, shard_h, width, orientation):
    if orientation == ttnn.ShardOrientation.ROW_MAJOR:
        end = ttnn.CoreCoord(num_cores - 1, 0)
        grid_size = (num_cores, 1)
    else:
        end = ttnn.CoreCoord(0, num_cores - 1)
        grid_size = (1, num_cores)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), end)})
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_h, width], orientation),
    )
    return memory_config, grid_size


def _program_config(grid_size, block_h, width):
    return ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=1,
        block_h=block_h,
        block_w=width // TILE,
    )


def _random_input(shape):
    # Values exact in bfloat16, so the reference sees the same input as the device.
    return (torch.randint(-16, 17, shape) / 8).to(torch.bfloat16)


def _random_mask(shape):
    # Each row keeps at least its first column, so no softmax row is fully masked.
    mask = torch.where(torch.rand(shape) > 0.5, torch.tensor(-1e4), torch.tensor(0.0))
    mask[..., 0] = 0.0
    return mask.to(torch.bfloat16)


def _run_and_check(device, torch_input, torch_mask_full, ttnn_mask, memory_config, program_config, is_causal_mask):
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    expected = torch.softmax(torch_input.float() * SCALE + torch_mask_full.float(), dim=-1)
    output = ttnn.scale_mask_softmax_in_place(
        ttnn_input, SCALE, ttnn_mask, program_config=program_config, is_causal_mask=is_causal_mask
    )
    output = ttnn.to_torch(output).float()

    # Check every tile row on its own: a row that got another batch's mask stays wrong even if the rest is right.
    for row in range(expected.shape[-2] // TILE):
        rows = slice(row * TILE, (row + 1) * TILE)
        assert_numeric_metrics(
            expected[..., rows, :],
            output[..., rows, :],
            pcc_threshold=0.999,
            rtol=0.05,
            atol=0.01,
            frobenius_threshold=0.05,
        )


@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "input_shape, mask_batch, num_cores, block_h",
    [
        # 3 tile rows per batch, 2 per shard: the middle shard holds the last row of batch 0 and the first of batch 1
        ((2, 1, 96, 128), 2, 3, 2),
        # 1 tile row per batch, 2 per shard: every shard holds two batches
        ((4, 1, 32, 128), 4, 2, 2),
        # one mask shared by both batches
        ((2, 1, 32, 128), 1, 2, 1),
        # one mask shared by every batch, shards spanning batches
        ((2, 1, 96, 128), 1, 3, 2),
        # control: every shard inside one batch
        ((2, 2, 64, 128), 2, 4, 2),
    ],
)
def test_sharded_softmax_broadcast_mask_across_batches(
    device, input_shape, mask_batch, num_cores, block_h, orientation
):
    torch.manual_seed(0)
    batch, heads, height, width = input_shape
    torch_input = _random_input(input_shape)
    # A broadcast mask is one row per batch, added to every row of that batch.
    torch_mask_row = _random_mask((mask_batch, 1, 1, width))
    torch_mask_tile = torch_mask_row.expand(mask_batch, 1, TILE, width).contiguous()
    ttnn_mask = ttnn.from_torch(torch_mask_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    memory_config, grid_size = _height_sharded(num_cores, block_h * TILE, width, orientation)
    _run_and_check(
        device,
        torch_input,
        torch_mask_row,
        ttnn_mask,
        memory_config,
        _program_config(grid_size, block_h, width),
        is_causal_mask=False,
    )


@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "input_shape, num_cores, block_h",
    [
        # 3 tile rows per batch, 2 per shard: the middle shard holds rows of both batches
        ((2, 1, 96, 128), 3, 2),
        # 1 tile row per batch, 2 per shard: every shard holds two batches
        ((4, 1, 32, 128), 2, 2),
    ],
)
def test_sharded_softmax_row_major_mask_across_batches(device, input_shape, num_cores, block_h, orientation):
    torch.manual_seed(0)
    batch, heads, height, width = input_shape
    torch_input = _random_input(input_shape)
    torch_mask_row = _random_mask((batch, 1, 1, width))
    # The documented ROW_MAJOR form: [B, 1, W / 32, 32].
    ttnn_mask = ttnn.from_torch(
        torch_mask_row.reshape(batch, 1, width // TILE, TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    memory_config, grid_size = _height_sharded(num_cores, block_h * TILE, width, orientation)
    _run_and_check(
        device,
        torch_input,
        torch_mask_row,
        ttnn_mask,
        memory_config,
        _program_config(grid_size, block_h, width),
        is_causal_mask=False,
    )


@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "input_shape, num_cores, block_h",
    [
        # one mask head per shard, as the existing tests use
        ((2, 2, 64, 128), 4, 2),
        # 3 tile rows per batch, 1 mask row per head, shards spanning batches
        ((2, 3, 32, 128), 3, 2),
    ],
)
def test_sharded_softmax_causal_mask_across_batches(device, input_shape, num_cores, block_h, orientation):
    torch.manual_seed(0)
    batch, heads, height, width = input_shape
    torch_input = _random_input(input_shape)
    # A full mask per batch, [B, 1, H, W], repeated for every head.
    torch_mask = _random_mask((batch, 1, height, width))
    ttnn_mask = ttnn.from_torch(torch_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    memory_config, grid_size = _height_sharded(num_cores, block_h * TILE, width, orientation)
    _run_and_check(
        device,
        torch_input,
        torch_mask,
        ttnn_mask,
        memory_config,
        _program_config(grid_size, block_h, width),
        is_causal_mask=True,
    )


@pytest.mark.parametrize("is_causal_mask", [False, True])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_sharded_softmax_sharded_mask(device, is_causal_mask, orientation):
    # A sharded mask has the input's shape, so it is one mask row per input row whether or not is_causal_mask is set.
    torch.manual_seed(0)
    input_shape = (1, 4, 64, 128)
    width = input_shape[-1]
    torch_input = _random_input(input_shape)
    torch_mask = _random_mask(input_shape)

    memory_config, grid_size = _height_sharded(4, 2 * TILE, width, orientation)
    ttnn_mask = ttnn.from_torch(
        torch_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    _run_and_check(
        device,
        torch_input,
        torch_mask,
        ttnn_mask,
        memory_config,
        _program_config(grid_size, 2, width),
        is_causal_mask=is_causal_mask,
    )


def test_sharded_softmax_rejects_row_major_causal_mask(device, expect_error):
    # The row-major reader produces one mask row per batch; a causal mask is consumed row by row.
    width = 128
    memory_config, grid_size = _height_sharded(1, 2 * TILE, width, ttnn.ShardOrientation.ROW_MAJOR)
    ttnn_input = ttnn.from_torch(
        _random_input((1, 1, 64, width)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    ttnn_mask = ttnn.from_torch(
        _random_mask((1, 1, width // TILE, TILE)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "A causal mask must have TILE layout"):
        ttnn.scale_mask_softmax_in_place(
            ttnn_input, SCALE, ttnn_mask, program_config=_program_config(grid_size, 2, width), is_causal_mask=True
        )


def _height_sharded_2x2(shard_h, width, orientation):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_h, width], orientation),
    )


@pytest.mark.parametrize("mask_layout", ["other_orientation", "fewer_cores"])
def test_sharded_softmax_rejects_mismatched_sharded_mask(device, expect_error, mask_layout):
    input_shape = (1, 4, 64, 128)
    width = input_shape[-1]
    # 2x2 grid: in row-major order core (1, 0) holds shard 1, in column-major order it holds shard 2.
    input_memory_config = _height_sharded_2x2(2 * TILE, width, ttnn.ShardOrientation.ROW_MAJOR)
    grid_size = (2, 2)
    if mask_layout == "other_orientation":
        mask_memory_config = _height_sharded_2x2(2 * TILE, width, ttnn.ShardOrientation.COL_MAJOR)
    else:
        # The whole mask on 2 cores, with shards twice as tall.
        mask_memory_config, _ = _height_sharded(2, 4 * TILE, width, ttnn.ShardOrientation.ROW_MAJOR)

    ttnn_input = ttnn.from_torch(
        _random_input(input_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    ttnn_mask = ttnn.from_torch(
        _random_mask(input_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mask_memory_config,
    )
    with expect_error(RuntimeError, "A sharded mask must be sharded like the input"):
        ttnn.scale_mask_softmax_in_place(
            ttnn_input, SCALE, ttnn_mask, program_config=_program_config(grid_size, 2, width), is_causal_mask=True
        )


@pytest.mark.parametrize("mask_batch", [2, 3])
def test_sharded_softmax_rejects_mask_batch_mismatch(device, expect_error, mask_batch):
    # A mask must have one batch, shared by every input batch, or one per input batch.
    input_shape = (4, 1, 32, 128)
    width = input_shape[-1]
    memory_config, grid_size = _height_sharded(4, TILE, width, ttnn.ShardOrientation.ROW_MAJOR)
    ttnn_input = ttnn.from_torch(
        _random_input(input_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    ttnn_mask = ttnn.from_torch(
        _random_mask((mask_batch, 1, TILE, width)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "Mask batch size must be 1 or match the input batch size"):
        ttnn.scale_mask_softmax_in_place(
            ttnn_input, SCALE, ttnn_mask, program_config=_program_config(grid_size, 1, width), is_causal_mask=False
        )


@pytest.mark.parametrize("program", ["default", "sharded"])
def test_softmax_in_place_rejects_mask_width_mismatch(device, expect_error, program):
    # The readers fetch one input row's worth of mask tiles per mask row, so a narrower mask would be read past its end.
    input_shape = (1, 1, 32, 256)
    width = input_shape[-1]
    if program == "sharded":
        memory_config, grid_size = _height_sharded(1, TILE, width, ttnn.ShardOrientation.ROW_MAJOR)
        program_config = _program_config(grid_size, 1, width)
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        program_config = ttnn.SoftmaxDefaultProgramConfig()
    ttnn_input = ttnn.from_torch(
        _random_input(input_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    ttnn_mask = ttnn.from_torch(
        _random_mask((1, 1, TILE, 64)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "Mask width must match the input width"):
        ttnn.scale_mask_softmax_in_place(ttnn_input, SCALE, ttnn_mask, program_config=program_config)
