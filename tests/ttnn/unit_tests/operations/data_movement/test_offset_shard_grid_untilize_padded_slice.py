# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shard grids that do not start at core (0, 0) must still index shards relative to the grid origin.

Both block-sharded factories below derived a shard index straight from a core's absolute coordinates.
That is only correct for an origin-rooted grid: with an origin at (1, 0), (0, 1) or (1, 1) the first
core's absolute x/y was used as its shard column/row, so every shard was read from and written to the
wrong offset and the result came back almost entirely wrong while validation still accepted the grid.
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

TILE = 32
GRID = (2, 2)

# The origin (0, 0) entry is the control: a grid already rooted at zero must stay bit-identical.
ORIGINS = [(0, 0), (1, 0), (0, 1), (1, 1)]

ORIENTATIONS = [
    ("row_major", ttnn.ShardOrientation.ROW_MAJOR),
    ("col_major", ttnn.ShardOrientation.COL_MAJOR),
]

# exact pass-through, both dims unaligned, then a width-only and a height-only trim, so row and
# column shard indices are each exercised separately as well as together.
UNTILIZE_ENDS = [(128, 128), (100, 100), (128, 96), (96, 128)]


def _shard_grid(origin):
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(origin[0], origin[1]),
                ttnn.CoreCoord(origin[0] + GRID[0] - 1, origin[1] + GRID[1] - 1),
            )
        }
    )


def _block_sharded(origin, shard_shape, orientation):
    spec = ttnn.ShardSpec(_shard_grid(origin), list(shard_shape), orientation)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def _seed(origin, tag):
    # A distinct seed per (origin, tag) so a misplaced shard cannot coincidentally carry the right
    # contents: neighbouring origins never share data.
    return 1000 * origin[0] + 100 * origin[1] + sum((i + 1) * ord(c) for i, c in enumerate(tag))


def _skip_if_grid_too_small(device, origin):
    grid = device.compute_with_storage_grid_size()
    if origin[0] + GRID[0] > grid.x or origin[1] + GRID[1] > grid.y:
        pytest.skip(
            f"needs a {origin[0] + GRID[0]}x{origin[1] + GRID[1]} worker grid, the device has {grid.x}x{grid.y}"
        )


def _assert_same(got, expected):
    got = (got if isinstance(got, torch.Tensor) else ttnn.to_torch(got)).float()
    expected = expected.float()
    assert got.shape == expected.shape, (got.shape, expected.shape)
    wrong = got != expected
    assert wrong.sum() == 0, f"{int(wrong.sum())} of {got.numel()} values wrong"


@pytest.mark.parametrize("end_hw", UNTILIZE_ENDS)
@pytest.mark.parametrize("orientation_name, orientation", ORIENTATIONS, ids=[name for name, _ in ORIENTATIONS])
@pytest.mark.parametrize("origin", ORIGINS)
def test_untilize_with_unpadding_block_sharded_offset_grid(device, origin, orientation_name, orientation, end_hw):
    """Block-sharded input on an offset grid: the writer's row offsets are grid-relative."""
    _skip_if_grid_too_small(device, origin)
    end_h, end_w = end_hw
    torch.manual_seed(_seed(origin, f"untilize_{orientation_name}_{end_h}x{end_w}"))
    src = torch.randn((1, 1, 128, 128), dtype=torch.bfloat16)

    sharded = ttnn.from_torch(
        src,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_block_sharded(origin, (64, 64), orientation),
    )
    output = ttnn.untilize_with_unpadding(
        sharded,
        [0, 0, end_h - 1, end_w - 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _assert_same(output, src[..., :end_h, :end_w])


@pytest.mark.parametrize("orientation_name, orientation", ORIENTATIONS, ids=[name for name, _ in ORIENTATIONS])
@pytest.mark.parametrize("origin", ORIGINS)
def test_padded_slice_block_sharded_offset_grid(device, origin, orientation_name, orientation):
    """Block-sharded output on an offset grid: the reader and writer shard indices are grid-relative."""
    _skip_if_grid_too_small(device, origin)
    start = [0, 0, 0, 0]
    end = [1, 1, 64, 128]
    step = [1, 1, 1, 1]
    torch.manual_seed(_seed(origin, f"padded_slice_{orientation_name}"))
    # 64 columns past the slice end so the sliced-out region exists to be dropped.
    src = torch.randn((1, 1, 64, 192), dtype=torch.bfloat16)

    interleaved = ttnn.from_torch(
        src,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = ttnn.experimental.padded_slice(
        interleaved,
        start,
        end,
        step,
        memory_config=_block_sharded(origin, (32, 64), orientation),
    )
    assert list(output.shape) == end
    _assert_same(output, src[..., : end[3]])
