# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1: interleaved_2d_split (host rectangle selector) + in-kernel pad fill."""

import math

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


def _pad_expected(x, padded_shape, value):
    if x.dim() < len(padded_shape):
        x = x.reshape((1,) * (len(padded_shape) - x.dim()) + tuple(x.shape))
    pads = tuple(j for i in reversed(range(x.dim())) for j in (0, padded_shape[i] - x.shape[i]))
    return torch.nn.functional.pad(x, pads, value=value)


@pytest.mark.parametrize(
    "rows, cols",
    [(1, 64), (64, 64), (2, 128), (1, 2), (3, 257)],
    ids=["short_wide_canonical", "square_large", "short_wide_two_rows", "tiny", "ragged_cols"],
)
def test_2d_split_fills_grid(device, rows, cols):
    """The rectangle split reaches min(R*C, grid) cores worth of work balance and covers every tile once."""
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    _, assignment, cores_c = pd._rect_assignment(device, rows, cols, 1)
    covered = set()
    for _, r0, nr, c0, nc in assignment:
        for r in range(r0, r0 + nr):
            for c in range(c0, c0 + nc):
                assert (r, c) not in covered
                covered.add((r, c))
    assert len(covered) == rows * cols
    assert len(assignment) <= num_cores
    busiest = max(nr * nc for _, _, nr, _, nc in assignment)
    # Never worse than the row split, and short_wide spreads over the columns.
    assert busiest <= math.ceil(rows / min(rows, num_cores)) * cols
    if rows < num_cores and rows * cols >= num_cores:
        assert len(assignment) >= num_cores // 2, f"only {len(assignment)} cores on {rows}x{cols}"


@pytest.mark.parametrize(
    "shape, padded, pad_value, tile_h, low_l1",
    [
        ((1, 1, 40, 1000), None, -2.5, 32, False),  # 2-D split, W tail on a ragged column group
        ((1, 1, 1, 9000), None, 1.0, 32, False),  # single stick, many column blocks per core
        ((1, 1, 1, 9000), None, 1.0, 32, True),  # same, low_l1 blocking
        ((2, 1, 33, 700), None, 0.0, 16, False),  # tiny tile + H tail across leading dims
        ((1, 1, 50, 50), (2, 3, 64, 96), 5.0, 32, False),  # leading-dim padding (whole pad images)
        ((3, 50), (2, 4, 64, 128), -7.0, 32, False),  # rank grows + leading pad
        ((1, 1, 32, 31), (1, 1, 32, 64), 7.0, 32, False),  # odd row bytes, whole pad column
    ],
    ids=["ragged_w_tail", "single_stick", "single_stick_low_l1", "tiny_tile_h_tail", "lead_pad", "rank_grow", "odd_w"],
)
def test_tilize_padding_geometry(device, shape, padded, pad_value, tile_h, low_l1):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kwargs = dict(pad_value=pad_value, low_l1=low_l1)
    if padded is not None:
        kwargs["output_padded_shape"] = list(padded)
    if tile_h != 32:
        kwargs["tile"] = ttnn.Tile([tile_h, 32])
    out = tilize(tt_in, **kwargs)
    if padded is None:
        padded = list(shape[:-2]) + [-(-shape[-2] // tile_h) * tile_h, -(-shape[-1] // 32) * 32]
    assert list(out.shape) == list(shape)
    assert list(out.padded_shape) == list(padded)
    assert torch.equal(ttnn.to_torch(out), x)
    assert torch.equal(out.cpu().to_torch_with_padded_shape(), _pad_expected(x, list(padded), pad_value))
