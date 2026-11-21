# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design requires beyond the acceptance test (op_design.md: Regime-pinned tests, Work Distribution
"Tests the plan needs"): each regime pinned by shape, forced multi-block schedules, the engaged-cluster count read
off the built ProgramSpec, the thread plan's striding, and the op in one device session with other programs."""

import math

import pytest
import torch

import ttnn
from ttnn.operations.reshape import reshape
from ttnn.operations.reshape.reshape import _reshape, resolve_shape
from ttnn.operations.reshape.reshape_program_artifacts import (
    REGIME_PAGE_IDENTITY,
    REGIME_TILE_ROW_REGROUP,
    create_program_artifacts,
    select_regime,
)

TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
A, B = REGIME_PAGE_IDENTITY, REGIME_TILE_ROW_REGROUP


def _input(device, shape, layout, mem, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)


def _check(out, t, new_shape, layout, mem):
    expected = torch.reshape(t, tuple(resolve_shape(tuple(t.shape), new_shape)))
    assert tuple(out.shape) == tuple(expected.shape)
    assert out.layout == layout
    assert out.memory_config().buffer_type == mem.buffer_type  # R7
    got = ttnn.to_torch(out)
    assert torch.equal(got, expected), (got.float() - expected.float()).abs().max()


REGIME_CASES = [
    # (shape, new shape, layout, memory, regime)
    ((1, 64, 8400), (1, 4, 16, 8400), RM, DRAM, A),  # wide page
    ((4, 128, 47), (2, 2, 128, 47), RM, L1, A),  # 94 B page, 96 B entry
    ((512,), (1, 1, 1, 512), TILE, L1, A),  # rank 1 tile
    ((1, 64, 128), (1, 2, 32, 128), TILE, DRAM, A),  # both H multiples of 32
    ((1, 64, 400), (1, 4, 16, 400), TILE, L1, B),  # traced call, H_out < 32
    ((1, 40, 40, 512), (1, 1, 1600, 512), TILE, DRAM, B),  # H_in non-aligned, rows cross planes
    ((1, 64, 4, 64), (1, 8, 32, 64), TILE, DRAM, B),  # H_in < 32: many input tile-rows per output tile-row
    ((2, 3, 5, 64), (2, 1, 15, 64), TILE, DRAM, B),
    ((2, 1, 100, 47), (2, 4, 25, 47), TILE, L1, B),  # W non-aligned, ragged
    ((1, 1, 50, 128), (1, 5, 10, 128), TILE, DRAM, B),  # row groups re-read boundary tile-rows
]


@pytest.mark.parametrize(
    "shape,new_shape,layout,mem,regime",
    REGIME_CASES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v).split(".")[-1].split("_")[0],
)
def test_regime_pinned(device, shape, new_shape, layout, mem, regime):
    assert select_regime(shape, resolve_shape(shape, new_shape), layout) == regime
    t, x = _input(device, shape, layout, mem)
    _check(reshape(x, new_shape), t, new_shape, layout, mem)


@pytest.mark.parametrize(
    "knobs",
    [
        {"block_tile_cols": 2},  # several column blocks per core: window depth 2
        {"block_out_tile_rows": 1},  # several row blocks per core: overlapping windows
        {"block_tile_cols": 3, "block_out_tile_rows": 1},  # both, with ragged column blocks
        {"row_groups": 1},  # column-only split (minimum traffic)
    ],
    ids=lambda k: "-".join(f"{a}{b}" for a, b in k.items()),
)
def test_regroup_forced_blocks(device, knobs):
    shape, new_shape = (1, 64, 8400), (1, 4, 16, 8400)
    t, x = _input(device, shape, TILE, DRAM)
    in_out = (tuple(x.shape), new_shape)
    out = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(new_shape, x.dtype, TILE, x.memory_config()), device)
    _, plan, *_ = create_program_artifacts(x, out, *in_out, **knobs)
    if "block_tile_cols" in knobs or "block_out_tile_rows" in knobs:
        assert plan.max_blocks_per_core > 1 and plan.input_window_depth == 2
    _check(_reshape(x, new_shape, **knobs), t, new_shape, TILE, DRAM)


def test_regroup_small_planes_row_blocks(device):
    """H_in < 32 with one-output-tile-row blocks: every block's window starts mid-way through the previous one."""
    t, x = _input(device, (4, 12, 5, 64), TILE, DRAM)
    new_shape = (4, 2, 30, 64)
    _check(_reshape(x, new_shape, block_out_tile_rows=1, row_groups=1), t, new_shape, TILE, DRAM)


@pytest.mark.parametrize("block_bytes_target", [2048, 4096 * 3])
def test_page_identity_many_blocks(device, block_bytes_target):
    """Several blocks per core, the last one ragged, in regime A."""
    shape, new_shape = (1, 40, 40, 512), (1, 1, 1600, 512)
    t, x = _input(device, shape, RM, DRAM)
    _check(_reshape(x, new_shape, block_bytes_target=block_bytes_target), t, new_shape, RM, DRAM)


@pytest.mark.parametrize(
    "shape,new_shape,layout",
    [((1, 40, 40, 512), (1, 1, 1600, 512), RM), ((1, 40, 40, 512), (1, 1, 1600, 512), TILE)],
    ids=["page_identity", "tile_row_regroup"],
)
def test_engaged_clusters(device, shape, new_shape, layout):
    """The cores the built ProgramSpec targets are the split's: a correct result alone does not show that every
    cluster ran. Both regimes fill the grid on this shape."""
    t, x = _input(device, shape, layout, DRAM)
    out = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(new_shape, x.dtype, layout, x.memory_config()), device)
    regime, plan, spec, *_ = create_program_artifacts(x, out, tuple(x.shape), new_shape)
    grid = device.compute_with_storage_grid_size()
    target = spec.work_units[0].target_nodes
    assert target.num_cores() == len(plan.cores) == grid.x * grid.y
    if regime == A:
        assert sum(plan.core_pages.values()) == x.buffer_num_pages()
    else:
        assert plan.col_groups * plan.row_groups == len(plan.cores)
        assert sum(plan.col_count[c] * plan.row_count[c] for c in plan.cores) == out.buffer_num_pages()
        # Data-movement budget: input read once on this shape (row-group boundaries fall on tile-row boundaries).
        assert plan.input_tiles_read() == x.buffer_num_pages()


@pytest.mark.parametrize("threads", [(2, 1, 2), (3, 1, 1), (1, 1, 3)], ids=lambda p: "R%dM%dW%d" % p)
def test_thread_plan_page_identity(device, threads):
    """Regime A's reader and writer stride their loops by the plan's counts: with more threads, a thread that did
    not run would leave pages unwritten."""
    plan = dict(zip(("reader_threads", "relayout_threads", "writer_threads"), threads))
    shape, new_shape = (1, 40, 40, 512), (1, 1, 1600, 512)
    t, x = _input(device, shape, RM, DRAM)
    _check(_reshape(x, new_shape, plan=plan), t, new_shape, RM, DRAM)


@pytest.mark.parametrize("threads", [(1, 1, 2)], ids=lambda p: "R%dM%dW%d" % p)
def test_thread_plan_regroup_writer(device, threads):
    plan = dict(zip(("reader_threads", "relayout_threads", "writer_threads"), threads))
    shape, new_shape = (1, 40, 40, 512), (1, 1, 1600, 512)
    t, x = _input(device, shape, TILE, DRAM)
    _check(_reshape(x, new_shape, plan=plan), t, new_shape, TILE, DRAM)


def test_session_after_other_programs(device):
    """In one device session: reshape after a tt-metal program and after the other regime."""
    # A tt-metal program first (ttnn.add is a Metal 1 op, refused on this part; to_layout runs here).
    t, x = _input(device, (1, 64, 400), TILE, L1, seed=1)
    y = ttnn.to_layout(x, RM)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(y)), t)
    steps = [
        ((1, 64, 400), (1, 4, 16, 400), TILE, L1),  # B
        ((1, 40, 40, 256), (1, 1, 1600, 256), RM, DRAM),  # A
        ((1, 40, 40, 256), (1, 1, 1600, 256), TILE, DRAM),  # B
        ((1, 1, 400, 512), (1, 20, 20, 512), RM, L1),  # A
        ((2, 1, 100, 47), (2, 4, 25, 47), TILE, L1),  # B
    ]
    for seed, (shape, new_shape, layout, mem) in enumerate(steps):
        t, x = _input(device, shape, layout, mem, seed)
        _check(reshape(x, new_shape), t, new_shape, layout, mem)


@pytest.mark.parametrize(
    "shape,new_shape,layout,to",
    [
        ((1, 64, 400), (1, 4, 16, 400), TILE, RM),  # B, then untilize
        ((1, 1, 2048, 256), (1, 4, 512, 256), TILE, RM),  # A (tile), then untilize
        ((1, 40, 40, 512), (1, 1, 1600, 512), RM, TILE),  # A (row-major), then tilize
    ],
    ids=["regroup_then_untilize", "tile_identity_then_untilize", "rm_identity_then_tilize"],
)
def test_a_tt_metal_program_follows(device, shape, new_shape, layout, to):
    """A tt-metal program after ours in the same session: it issues untagged transfers, which a sticky ring tag
    left by our harts would have counted for a ring."""
    t, x = _input(device, shape, layout, DRAM)
    out = ttnn.to_layout(reshape(x, new_shape), to)
    got = ttnn.to_torch(ttnn.from_device(out))
    assert torch.equal(got, torch.reshape(t, new_shape))


def test_rejections_and_shape_forms(device):
    t, x = _input(device, (1, 64, 400), TILE, L1)
    with pytest.raises(ValueError):
        reshape(x, (1, 4, 16, 401))  # X1
    with pytest.raises(ValueError):
        reshape(x, (-1, -1, 400))  # X2
    with pytest.raises(ValueError):
        reshape(x, (-1, 3, 7))  # X3
    with pytest.raises(NotImplementedError):
        reshape(x, (1, 400, 64))  # changes = w: outside PHASE0
    _check(reshape(x, [1, 4, -1, 400]), t, (1, 4, 16, 400), TILE, L1)  # list with -1 (R2, R3)
    assert math.prod(resolve_shape((2, 4, 17, 50), (-1, 17, 50))) == 2 * 4 * 17 * 50
