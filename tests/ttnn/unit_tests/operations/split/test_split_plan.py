# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design asks for beyond the acceptance test (op_design.md, Regimes and Work Distribution).

- which regime each named Phase 0 entry selects (the host's selection function, called directly);
- tile_row_shift shapes whose per-core ranges start mid-column (halo) and span several blocks of
  BLOCK_H_TILES, with chunk boundaries inside a block, and blocks that span several short tile columns;
- every output page written: exact comparison of every chunk against torch.split on random values;
- chunk placement (R5): each chunk in the input's buffer, interleaved;
- chunk counts 1, 3, 4 and a smaller last chunk through the public entry point (SUPPORTED claims them since
  Refinement 2), and a fifth chunk refused by validate() (the writer binds at most MAX_CHUNKS = 4 chunks);
- the op in one device session after another program and before a tt-metal program (target page, Ring protocols).
"""

import pytest
import torch

import ttnn
from ttnn.operations.split import split
from ttnn.operations.split.split_program_artifacts import (
    BLOCK_H_TILES,
    PAGE_REMAP,
    RM_SUBSTICK,
    TILE_COL_SHIFT,
    TILE_ROW_SHIFT,
    derive_geometry,
    select_regime,
    split_units,
)

MEMORY = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}


def _put(device, t, layout, memory="DRAM"):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=MEMORY[memory])


def _check_chunks(chunks, torch_input, split_size, dim, memory="DRAM"):
    expected = torch.split(torch_input, split_size, dim=dim)
    assert isinstance(chunks, list) and len(chunks) == len(expected)
    for k, (chunk, want) in enumerate(zip(chunks, expected)):
        assert list(chunk.shape) == list(want.shape), (k, chunk.shape, want.shape)
        assert chunk.memory_config().buffer_type == MEMORY[memory].buffer_type, k
        assert chunk.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, k
        got = ttnn.to_torch(chunk)
        assert torch.equal(got, want), f"chunk {k}: max diff {(got.float() - want.float()).abs().max()}"


# --- regime selection --------------------------------------------------------------------------------------

REGIME_CASES = [
    # page_remap / TILE: the cut at a multiple of the tile height
    ((1, 64, 416), 32, ttnn.TILE_LAYOUT, PAGE_REMAP),
    ((4, 128, 512), 64, ttnn.TILE_LAYOUT, PAGE_REMAP),
    ((1, 1024, 64), 512, ttnn.TILE_LAYOUT, PAGE_REMAP),
    # tile_row_shift: the cut inside a tile
    ((1, 4, 400), 2, ttnn.TILE_LAYOUT, TILE_ROW_SHIFT),
    ((1, 2, 64), 1, ttnn.TILE_LAYOUT, TILE_ROW_SHIFT),
    ((1, 32, 50), 16, ttnn.TILE_LAYOUT, TILE_ROW_SHIFT),
    ((2, 6, 512), 3, ttnn.TILE_LAYOUT, TILE_ROW_SHIFT),
    ((1, 34, 4096), 17, ttnn.TILE_LAYOUT, TILE_ROW_SHIFT),
    ((1, 4, 8400), 2, ttnn.TILE_LAYOUT, TILE_ROW_SHIFT),
] + [
    # page_remap / ROW_MAJOR: every Phase 0 entry
    (shape, s, ttnn.ROW_MAJOR_LAYOUT, PAGE_REMAP)
    for shape, s in [((1, 4, 8400), 2), ((1, 4, 400), 2), ((1, 2, 64), 1), ((1, 32, 50), 16), ((2, 6, 512), 3),
                     ((1, 64, 416), 32), ((4, 128, 512), 64), ((1, 34, 4096), 17), ((1, 1024, 64), 512)]
]


@pytest.mark.parametrize("shape,split_size,layout,regime", REGIME_CASES, ids=lambda v: str(v))
def test_regime_selection(shape, split_size, layout, regime):
    assert select_regime(layout, -2, split_size) == regime


def test_cut_w_regimes_are_named():
    assert select_regime(ttnn.ROW_MAJOR_LAYOUT, -1, 16) == RM_SUBSTICK == "rm_substick"
    assert select_regime(ttnn.TILE_LAYOUT, -1, 16) == TILE_COL_SHIFT == "tile_col_shift"
    assert select_regime(ttnn.TILE_LAYOUT, -1, 64) == PAGE_REMAP
    assert select_regime(ttnn.TILE_LAYOUT, -3, 1) == PAGE_REMAP


# --- tile_row_shift: halo and several blocks per core ------------------------------------------------------

# (shape, split_size): every one is a Phase 0 cell (rank 3, dim -2, two equal chunks).
WINDOW_CASES = [
    ((1, 100, 64), 50),  # 2 columns x 4 tiles on 8 cores: every range but the first starts mid-column
    ((4, 1000, 96), 500),  # 12 columns x 32 tiles = 384 on 32 cores: 12 per core, blocks of 8 + 4, mid-column starts
    ((1, 600, 32), 300),  # one tall column, the chunk boundary at row 300 (tile 9, row 12)
    ((3, 70, 40), 35),  # W and H non-aligned, chunk boundary inside a tile, ragged last tiles
    # short columns (2 output tiles each, 2 input tiles) over 512 columns: 32 output tiles per core, blocks span
    # 4 columns, ~32 window reads per core, so the window's read sequence wraps the 2*BLOCK_H_TILES slots twice
    ((2, 40, 8192), 20),
]


@pytest.mark.parametrize("memory", ["DRAM", "L1"])
@pytest.mark.parametrize("shape,split_size", WINDOW_CASES, ids=lambda v: str(v))
def test_tile_row_shift_window(device, shape, split_size, memory):
    regime, n_chunks, num_units, geometry = derive_geometry(
        ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT), split_size, -2
    )
    assert regime == TILE_ROW_SHIFT and n_chunks == 2
    torch.manual_seed(7)
    t = torch.randn(shape, dtype=torch.bfloat16)
    _check_chunks(split(_put(device, t, ttnn.TILE_LAYOUT, memory), split_size, 1), t, split_size, 1, memory)


def test_window_case_engages_several_blocks_and_halos(device):
    """The structural counts the window cases are chosen for: blocks per core > 1 and ranges starting mid-column."""
    shape, split_size = (4, 1000, 96), 500
    tensor = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT)
    _, _, num_units, geometry = derive_geometry(tensor, split_size, -2)
    _, counts, starts = split_units(num_units, device.compute_with_storage_grid_size())
    assert len(counts) == device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    assert max(counts.values()) > BLOCK_H_TILES
    assert any(start % geometry["line_units"] != 0 for start in starts.values())


# --- page_remap: larger ranges, both layouts ---------------------------------------------------------------


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
@pytest.mark.parametrize("shape,split_size", [((4, 128, 512), 64), ((3, 96, 100), 48)], ids=str)
def test_page_remap(device, shape, split_size, layout):
    torch.manual_seed(3)
    t = torch.randn(shape, dtype=torch.bfloat16)
    _check_chunks(split(_put(device, t, layout), split_size, -2), t, split_size, -2)


# --- chunk counts 1 to 4 and a smaller last chunk (claimed by Refinement 2) --------------------------------


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
@pytest.mark.parametrize(
    "shape,split_size",
    [((1, 50, 64), 64), ((1, 17, 128), 6), ((1, 8, 96), 2), ((1, 128, 64), 32), ((1, 100, 64), 40)],
    ids=["1chunk", "3chunks_smaller", "4chunks", "4chunks_aligned", "3chunks_smaller_multitile"],
)
def test_chunk_counts(device, shape, split_size, layout):
    torch.manual_seed(11)
    t = torch.randn(shape, dtype=torch.bfloat16)
    chunks = split(_put(device, t, layout), split_size, -2)
    _check_chunks(chunks, t, split_size, -2)


def test_fifth_chunk_is_refused(device):
    """TARGET's `chunks` ends at 4 (the writer's MAX_CHUNKS): a call that would return 5 is refused by validate()."""
    from ttnn.operations._op_contract import UnsupportedAxisValue

    t = torch.zeros((1, 10, 64), dtype=torch.bfloat16)
    with pytest.raises(UnsupportedAxisValue, match="chunks"):
        split(_put(device, t, ttnn.ROW_MAJOR_LAYOUT), 2, -2)


# --- one device session: before and after other programs --------------------------------------------------


def test_after_another_program_and_before_tt_metal(device):
    """A tt-metal program, then split (both regimes, both layouts), then tt-metal's own to_layout on every TILE
    chunk, all on one open device. The kernels hand the sticky transaction tag back as their last statement."""
    torch.manual_seed(5)
    t = torch.randn((1, 4, 400), dtype=torch.bfloat16)
    other = ttnn.to_layout(_put(device, torch.randn((1, 64, 64), dtype=torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT),
                           ttnn.TILE_LAYOUT)
    assert list(other.shape) == [1, 64, 64]

    for layout, src, split_size in [
        (ttnn.TILE_LAYOUT, t, 2),  # tile_row_shift
        (ttnn.TILE_LAYOUT, torch.randn((1, 64, 416), dtype=torch.bfloat16), 32),  # page_remap / TILE
        (ttnn.ROW_MAJOR_LAYOUT, t, 2),  # page_remap / RM
    ]:
        chunks = split(_put(device, src, layout, "L1"), split_size, 1)
        expected = torch.split(src, split_size, dim=1)
        for chunk, want in zip(chunks, expected):
            if layout == ttnn.TILE_LAYOUT:
                chunk = ttnn.to_layout(chunk, ttnn.ROW_MAJOR_LAYOUT)
            assert torch.equal(ttnn.to_torch(ttnn.from_device(chunk)), want)

    # dim = -1 in each Refinement 1 regime (explicit-producer readers; the sized explicit writer in the last).
    w = torch.randn((1, 64, 50), dtype=torch.bfloat16)
    for layout, split_size in [
        (ttnn.TILE_LAYOUT, 20),  # tile_col_shift
        (ttnn.ROW_MAJOR_LAYOUT, 32),  # rm_substick, aligned segments
        (ttnn.ROW_MAJOR_LAYOUT, 17),  # rm_substick, odd shifts, shorter last-chunk page in L1 (sized writes)
    ]:
        assert select_regime(layout, -1, split_size) in (TILE_COL_SHIFT, RM_SUBSTICK)
        chunks = split(_put(device, w, layout, "L1"), split_size, -1)
        expected = torch.split(w, split_size, dim=-1)
        for chunk, want in zip(chunks, expected):
            if layout == ttnn.TILE_LAYOUT:
                chunk = ttnn.to_layout(chunk, ttnn.ROW_MAJOR_LAYOUT)
            assert torch.equal(ttnn.to_torch(ttnn.from_device(chunk)), want)
    # A tt-metal program after the explicit (sized) writer: tilize the last row-major chunk on device.
    tiled = ttnn.to_layout(chunks[-1], ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(tiled)), expected[-1])


def test_split_back_to_back(device):
    """Explicit-producer (tile_row_shift) and implicit (page_remap) programs alternating in one session."""
    torch.manual_seed(9)
    for step in range(4):
        layout = ttnn.TILE_LAYOUT if step % 2 == 0 else ttnn.ROW_MAJOR_LAYOUT
        split_size = 17 if step < 2 else 32
        t = torch.randn((2, 2 * split_size, 160), dtype=torch.bfloat16)
        _check_chunks(split(_put(device, t, layout), split_size, -2), t, split_size, -2)


# --- R6: a sharded input is refused by name (the generic runner builds interleaved tensors only) -------------


@pytest.mark.parametrize("strategy", ["HEIGHT", "WIDTH"])
def test_sharded_input_is_refused_by_name(device, strategy):
    from eval.memory import memory_axes
    from eval.sharding import auto_shard_config
    from ttnn.operations._op_contract import UnsupportedAxisValue

    shape = (1, 64, 256)
    memory_config = auto_shard_config(
        shape, getattr(ttnn.ShardStrategy, strategy), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    sharded = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    assert memory_axes(sharded)["sharding"] == strategy.lower()
    with pytest.raises(UnsupportedAxisValue, match="sharding"):
        split(sharded, 32, 1)
