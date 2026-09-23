# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1: sharded / L1 placement and rank widening for tilize.

Pins each sub-case of the `sharded_resident` / `sharded_accessor` regimes
(tilize_program_descriptor._core_assignment):
  * both sides resident (same spec: no NoC traffic at all),
  * input resident + streamed output, streamed input + resident output,
  * cross-spec (height in, width out: output resident, input read remotely),
  * a width-sharded Layout::ROW_MAJOR input read through TensorAccessor
    (page = shard-width stick, so each segment splits at page boundaries),
  * DRAM-sharded sides and ND specs without a 2-D equivalent (streamed only),
  * BLOCK shard -> core maps for both orientations, and ragged final shards.
Every case is an identity check (tilize is a bit-exact re-lay at bf16).
"""
import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import _core_assignment, _tile_grid

L1 = ttnn.BufferType.L1
DRAM = ttnn.BufferType.DRAM
ROW = ttnn.ShardOrientation.ROW_MAJOR
COL = ttnn.ShardOrientation.COL_MAJOR
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def _crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _legacy(scheme, grid, shard, orientation=ROW, buffer=L1):
    return ttnn.MemoryConfig(scheme, buffer, ttnn.ShardSpec(grid, shard, orientation))


def _nd(grid, shard, orientation=ROW):
    return ttnn.MemoryConfig(L1, ttnn.NdShardSpec(ttnn.Shape(shard), grid, orientation))


def _run(device, shape, in_mc, out_mc):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    out = tilize(t, memory_config=out_mc)
    assert out.layout == ttnn.TILE_LAYOUT
    y = ttnn.to_torch(out)
    assert list(y.shape) == list(shape)
    assert torch.equal(x, y), f"max diff {(x.float() - y.float()).abs().max()}"
    return t, out


def _residency(t_in, t_out):
    shape = list(t_in.shape)
    R, _ = _tile_grid(shape, 32)
    assignment, in_res, out_res, _ = _core_assignment(
        t_in, t_out, rows_total=R * 32, width=int(shape[-1]), tile_h=32, max_block_width=64
    )
    return assignment, in_res, out_res


CASES = [
    # name, shape, in memory config, out memory config, expected (input_resident, output_resident)
    (
        "height_same_spec",
        [1, 1, 512, 64],
        _legacy(HEIGHT, _crs(0, 0, 3, 0), (128, 64)),
        _legacy(HEIGHT, _crs(0, 0, 3, 0), (128, 64)),
        (True, True),
    ),
    (
        "width_same_spec",
        [1, 1, 64, 512],
        _legacy(WIDTH, _crs(0, 0, 3, 0), (64, 128)),
        _legacy(WIDTH, _crs(0, 0, 3, 0), (64, 128)),
        (True, True),
    ),
    (
        "block_col_major_same_spec",
        [1, 1, 128, 128],
        _legacy(BLOCK, _crs(0, 0, 1, 1), (64, 64), COL),
        _legacy(BLOCK, _crs(0, 0, 1, 1), (64, 64), COL),
        (True, True),
    ),
    # Streamed input, resident BLOCK output: the reader addresses exactly the rectangle the
    # host maps to each core, so a wrong shard -> core map shows up as wrong data.
    (
        "interleaved_to_block_row_major",
        [1, 1, 128, 256],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(BLOCK, _crs(0, 0, 3, 1), (64, 64), ROW),
        (False, True),
    ),
    (
        "interleaved_to_block_col_major",
        [1, 1, 128, 256],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(BLOCK, _crs(0, 0, 1, 3), (64, 64), COL),
        (False, True),
    ),
    (
        "interleaved_to_height_2d_grid_col_major",
        [1, 1, 512, 64],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(HEIGHT, _crs(0, 0, 3, 1), (64, 64), COL),
        (False, True),
    ),
    (
        "height_to_interleaved",
        [1, 1, 256, 128],
        _legacy(HEIGHT, _crs(0, 0, 3, 0), (64, 128)),
        ttnn.DRAM_MEMORY_CONFIG,
        (True, False),
    ),
    (
        "width_to_l1_interleaved",
        [1, 1, 64, 512],
        _legacy(WIDTH, _crs(0, 0, 3, 0), (64, 128)),
        ttnn.L1_MEMORY_CONFIG,
        (True, False),
    ),
    (
        "cross_spec_height_in_width_out",
        [1, 1, 128, 128],
        _legacy(HEIGHT, _crs(0, 0, 1, 0), (64, 128)),
        _legacy(WIDTH, _crs(0, 0, 1, 0), (128, 64)),
        (False, True),
    ),
    # Width-sharded Layout::ROW_MAJOR input streamed: 128-element pages, segments split per page.
    (
        "width_in_height_out_paged",
        [1, 1, 64, 512],
        _legacy(WIDTH, _crs(0, 0, 3, 0), (64, 128)),
        _legacy(HEIGHT, _crs(0, 0, 1, 0), (32, 512)),
        (False, True),
    ),
    (
        "ragged_height_same_spec",
        [1, 1, 160, 64],
        _legacy(HEIGHT, _crs(0, 0, 2, 0), (64, 64)),
        _legacy(HEIGHT, _crs(0, 0, 2, 0), (64, 64)),
        (True, True),
    ),
    (
        "ragged_height_to_interleaved",
        [1, 1, 160, 64],
        _legacy(HEIGHT, _crs(0, 0, 2, 0), (64, 64)),
        ttnn.DRAM_MEMORY_CONFIG,
        (True, False),
    ),
    (
        "interleaved_to_ragged_block",
        [1, 1, 96, 160],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(BLOCK, _crs(0, 0, 2, 1), (64, 64), ROW),
        (False, True),
    ),
    (
        "dram_height_sharded_out",
        [1, 1, 128, 64],
        _legacy(HEIGHT, _crs(0, 0, 1, 0), (64, 64)),
        _legacy(HEIGHT, _crs(0, 0, 1, 0), (64, 64), buffer=DRAM),
        (True, False),
    ),
    (
        "dram_width_sharded_both",
        [1, 1, 64, 512],
        _legacy(WIDTH, _crs(0, 0, 3, 0), (64, 128), buffer=DRAM),
        _legacy(WIDTH, _crs(0, 0, 3, 0), (64, 128), buffer=DRAM),
        (False, False),
    ),
    (
        "nd_same_spec_2d_equivalent",
        [1, 1, 128, 64],
        _nd(_crs(0, 0, 1, 0), [1, 1, 64, 64]),
        _nd(_crs(0, 0, 1, 0), [1, 1, 64, 64]),
        (True, True),
    ),
    (
        "nd_3d_to_interleaved",
        [4, 128, 128],
        _nd(_crs(0, 0, 1, 1), [2, 64, 64]),
        ttnn.DRAM_MEMORY_CONFIG,
        (False, False),
    ),
    (
        "nd_3d_to_nd_other_spec",
        [3, 160, 160],
        _nd(_crs(0, 0, 1, 1), [2, 64, 64]),
        _nd(_crs(0, 0, 1, 1), [1, 64, 96]),
        (False, False),
    ),
    (
        "interleaved_to_nd",
        [1, 1, 128, 64],
        ttnn.DRAM_MEMORY_CONFIG,
        _nd(_crs(0, 0, 1, 0), [1, 1, 64, 64]),
        (False, True),
    ),
]


@pytest.mark.parametrize("name,shape,in_mc,out_mc,residency", CASES, ids=[c[0] for c in CASES])
def test_tilize_sharded(device, name, shape, in_mc, out_mc, residency):
    t_in, t_out = _run(device, shape, in_mc, out_mc)
    _, in_res, out_res = _residency(t_in, t_out)
    assert (in_res, out_res) == residency


def test_same_spec_moves_no_bytes(device):
    """Same spec both sides: every assigned core owns its whole rectangle on both sides."""
    mc = _legacy(HEIGHT, _crs(0, 0, 3, 0), (128, 64))
    t_in, t_out = _run(device, [1, 1, 512, 64], mc, mc)
    assignment, in_res, out_res = _residency(t_in, t_out)
    assert in_res and out_res
    assert [(c.x, c.y) for c, *_ in assignment] == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert [(rs, rows, cs, cols) for _, rs, rows, cs, cols in assignment] == [(4 * k, 4, 0, 2) for k in range(4)]


@pytest.mark.parametrize(
    "shape,mc",
    [
        pytest.param([64, 128], ttnn.DRAM_MEMORY_CONFIG, id="rank2"),
        pytest.param([2, 32, 64], ttnn.DRAM_MEMORY_CONFIG, id="rank3"),
        pytest.param([1, 2, 1, 32, 64], ttnn.DRAM_MEMORY_CONFIG, id="rank5"),
        pytest.param([2, 1, 1, 1, 32, 64], ttnn.DRAM_MEMORY_CONFIG, id="rank6"),
        pytest.param([1, 1, 64, 128], ttnn.L1_MEMORY_CONFIG, id="l1_to_l1"),
    ],
)
def test_tilize_rank_and_l1(device, shape, mc):
    _run(device, shape, mc, mc)


def test_tilize_l1_crossovers(device):
    _run(device, [1, 1, 64, 128], ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
    _run(device, [1, 1, 64, 64], ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
