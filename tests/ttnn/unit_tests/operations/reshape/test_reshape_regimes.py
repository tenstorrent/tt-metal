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
    REGIME_W_REGROUP_RM,
    REGIME_W_REGROUP_TILE,
    create_program_artifacts,
    select_regime,
)

TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
A, B, W, WT = REGIME_PAGE_IDENTITY, REGIME_TILE_ROW_REGROUP, REGIME_W_REGROUP_RM, REGIME_W_REGROUP_TILE


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
    """A tt-metal program after ours in the same session: it issues untagged transfers."""
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
    _check(reshape(x, (1, 400, 64)), t, (1, 400, 64), TILE, L1)  # changes = w in tile layout (Refinement 2)
    _check(reshape(x, [1, 4, -1, 400]), t, (1, 4, 16, 400), TILE, L1)  # list with -1 (R2, R3)
    assert math.prod(resolve_shape((2, 4, 17, 50), (-1, 17, 50))) == 2 * 4 * 17 * 50


@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH], ids=["height", "width"])
def test_sharded_input_refused_by_name(device, strategy):
    """R8: the op derives `memory_config` / `sharding` with eval.memory.memory_axes and gates on them, so a sharded
    input is refused by name (the generic runner builds interleaved tensors only, so the op carries this test)."""
    shape = (1, 1, 256, 256)
    grid = ttnn.CoreGrid(y=1, x=8)
    mc = ttnn.create_sharded_memory_config(shape, core_grid=grid, strategy=strategy)
    x = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=TILE, device=device,
                        memory_config=mc)
    with pytest.raises(NotImplementedError, match="sharding"):
        reshape(x, (1, 4, 64, 256))


# ---------------------------------------------------------------------------------------------------------------
# Refinement 0: uint32 tensors and rank 2 (Llama, Qwen)
# ---------------------------------------------------------------------------------------------------------------


def _uint32_input(device, shape, layout, mem, seed=0):
    """uint32 values over the whole 32-bit range: a value that passed through a narrower or float dtype would not
    come back (rule R1)."""
    torch.manual_seed(seed)
    t = torch.randint(0, 2**32, shape, dtype=torch.int64)
    x = ttnn.from_torch(t, dtype=ttnn.uint32, layout=layout, device=device, memory_config=mem)
    return ttnn.to_torch(ttnn.from_device(x)), x


def _check_exact(out, t, new_shape, dtype, layout, mem):
    expected = torch.reshape(t, tuple(resolve_shape(tuple(t.shape), new_shape)))
    assert tuple(out.shape) == tuple(expected.shape)
    assert out.dtype == dtype and out.layout == layout
    assert out.memory_config().buffer_type == mem.buffer_type  # R7
    got = ttnn.to_torch(ttnn.from_device(out))
    assert torch.equal(got, expected)


@pytest.mark.parametrize(
    "shape,new_shape,dtype,layout,mem",
    [
        ((32,), (1, 1, 1, 32), ttnn.uint32, RM, DRAM),  # Llama, Qwen: token indices to rank 4
        ((1,), (1, 1), ttnn.uint32, RM, DRAM),  # Qwen: one token index
        ((128, 512), (1, 128, 512), ttnn.bfloat16, TILE, DRAM),  # Qwen: a leading 1 on a rank-2 tensor
    ],
    ids=["llama_qwen_token_indices", "qwen_one_token", "qwen_rank2_unsqueeze"],
)
def test_model_calls(device, shape, new_shape, dtype, layout, mem):
    """The three Llama / Qwen calls of Refinement 0, made as the models make them."""
    if dtype == ttnn.uint32:
        t, x = _uint32_input(device, shape, layout, mem)
    else:
        t, x = _input(device, shape, layout, mem)
    _check_exact(reshape(x, new_shape), t, new_shape, dtype, layout, mem)


UINT32_CASES = [
    # (shape, new shape, layout, memory, regime): every regime and placement, rank 2 in and out
    ((1, 64, 8400), (1, 4, 16, 8400), RM, DRAM, A),  # widest page: 33,600 B
    ((4, 128, 47), (2, 2, 128, 47), RM, L1, A),  # 188 B page, 192 B entry
    ((1,), (1, 1), TILE, L1, A),  # one element, one tile
    ((128, 512), (4, 32, 512), TILE, L1, A),  # rank 2 in
    ((1, 64, 400), (1, 4, 16, 400), TILE, L1, B),  # 64 B face rows
    ((2, 1, 100, 47), (2, -1, 25, 47), TILE, DRAM, B),  # W non-aligned, ragged, -1
    ((4, 128, 512), (-1, 512), TILE, DRAM, A),  # rank 2 out
]


@pytest.mark.parametrize(
    "shape,new_shape,layout,mem,regime",
    UINT32_CASES,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v).split(".")[-1].split("_")[0],
)
def test_uint32_regimes(device, shape, new_shape, layout, mem, regime):
    assert select_regime(shape, resolve_shape(shape, new_shape), layout) == regime
    t, x = _uint32_input(device, shape, layout, mem)
    _check_exact(reshape(x, new_shape), t, new_shape, ttnn.uint32, layout, mem)


def test_uint32_session_after_other_programs(device):
    """uint32 in one device session, after a tt-metal program and between bfloat16 calls of both regimes."""
    t, x = _input(device, (1, 64, 400), TILE, L1, seed=1)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(x, RM))), t)
    for seed, (shape, new_shape, layout, mem) in enumerate(
        [((1, 64, 400), (1, 4, 16, 400), TILE, L1), ((32,), (1, 1, 1, 32), RM, DRAM)]
    ):
        t, x = _input(device, shape, layout, mem, seed)
        _check(reshape(x, new_shape), t, new_shape, layout, mem)
        t, x = _uint32_input(device, shape, layout, mem, seed)
        _check_exact(reshape(x, new_shape), t, new_shape, ttnn.uint32, layout, mem)


# ---------------------------------------------------------------------------------------------------------------
# Refinement 1: changes = w in row-major layout (regime w_regroup_rm)
# ---------------------------------------------------------------------------------------------------------------

W_CASES = [
    # (shape, new shape, memory): output row j is bytes [j * page_out, (j + 1) * page_out) of the input's byte stream
    ((1, 1, 17, 50), (1, 1, 25, 34), DRAM),  # W non-aligned in and out (100 B -> 68 B rows)
    ((4, 128, 47), (4, 47, 128), L1),  # 94 B rows: runs 2 bytes off the word phase (shift-merge copy)
    ((3, 5, 7), (7, 15), DRAM),  # 14 B -> 30 B rows: every run boundary off the word phase; rank 2 out
    ((1, 32, 128), (4096,), DRAM),  # flattened: one output page gathered from 32 input pages
    ((2048,), (32, 64), L1),  # rank 1 in: one input page cut into 32 output pages
    ((1, 1, 32, 8192), (1, 8, 32, 1024), DRAM),  # wide: 16 KiB input pages
    ((1, 1, 64, 128), (1, 1, 128, 64), L1),  # ResNet-50's call, in row-major
]


def _ids(v):
    return "x".join(map(str, v)) if isinstance(v, tuple) else str(v).split(".")[-1].split("_")[0]


def _w_plan(device, x, new_shape, **knobs):
    out = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(new_shape, x.dtype, RM, x.memory_config()), device)
    regime, plan, spec, *_ = create_program_artifacts(x, out, tuple(x.shape), new_shape, **knobs)
    assert regime == W
    return plan, spec


@pytest.mark.parametrize("shape,new_shape,mem", W_CASES, ids=_ids)
def test_w_regroup_rm_pinned(device, shape, new_shape, mem):
    assert select_regime(shape, resolve_shape(shape, new_shape), RM) == W
    t, x = _input(device, shape, RM, mem)
    _check(reshape(x, new_shape), t, new_shape, RM, mem)


@pytest.mark.parametrize(
    "shape,new_shape,knobs",
    [
        ((1, 1, 32, 8192), (1, 8, 32, 1024), {"block_out_rows": 3}),  # ragged row blocks, window depth 2
        ((4, 128, 47), (4, 47, 128), {"row_groups": 1, "block_out_rows": 5}),  # one cluster, many blocks
        ((1, 1, 17, 50), (1, 1, 25, 34), {"row_groups": 7, "block_out_rows": 2}),  # uneven ranges, mid-page starts
    ],
    ids=["ragged_blocks", "one_cluster_many_blocks", "uneven_ranges"],
)
def test_w_regroup_rm_forced_blocks(device, shape, new_shape, knobs):
    t, x = _input(device, shape, RM, DRAM)
    plan, _ = _w_plan(device, x, new_shape, **knobs)
    assert plan.max_blocks_per_core > 1 and plan.input_window_depth == 2
    _check(_reshape(x, new_shape, **knobs), t, new_shape, RM, DRAM)


@pytest.mark.parametrize(
    "shape,new_shape", [((1, 1, 32, 8192), (1, 8, 32, 1024)), ((4, 8, 32, 256), (4, 8, 16, 512))], ids=_ids
)
def test_w_regroup_rm_split(device, shape, new_shape):
    """Structural counts read off the built plan: the grid is filled up to the input page count (the cap that keeps
    boundary re-reads at one input page per range boundary), every output page belongs to one cluster, each
    cluster's range is one block, and input reads stay below in_pages + row_groups - 1."""
    t, x = _input(device, shape, RM, DRAM)
    plan, spec = _w_plan(device, x, new_shape)
    grid = device.compute_with_storage_grid_size()
    assert spec.work_units[0].target_nodes.num_cores() == len(plan.cores) == min(
        grid.x * grid.y, plan.in_pages, plan.out_pages
    )
    assert sum(plan.row_count.values()) == plan.out_pages
    assert plan.max_blocks_per_core == 1
    assert plan.in_pages <= plan.input_pages_read() <= plan.in_pages + plan.row_groups - 1


@pytest.mark.parametrize(
    "shape,new_shape,mem",
    [((4, 128, 47), (4, 47, 128), L1), ((1, 1, 17, 50), (1, 1, 25, 34), DRAM), ((2048,), (32, 64), DRAM)],
    ids=_ids,
)
def test_w_regroup_rm_uint32(device, shape, new_shape, mem):
    """4-byte elements: every byte count derives from the page size, so uint32 runs the same walker bit for bit."""
    t, x = _uint32_input(device, shape, RM, mem)
    _check_exact(reshape(x, new_shape), t, new_shape, ttnn.uint32, RM, mem)


def test_w_regroup_rm_session(device):
    """One device session: a tt-metal program, then the W regime between the other regimes, then a tt-metal
    program (tilize) on a W-regime output."""
    t, x = _input(device, (1, 64, 400), TILE, L1, seed=1)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(x, RM))), t)
    steps = [
        ((4, 128, 47), (4, 47, 128), RM, L1),  # W
        ((1, 64, 400), (1, 4, 16, 400), TILE, L1),  # B
        ((1, 1, 17, 50), (1, 1, 25, 34), RM, DRAM),  # W
        ((1, 40, 40, 256), (1, 1, 1600, 256), RM, DRAM),  # A
        ((2048,), (32, 64), RM, DRAM),  # W
    ]
    for seed, (shape, new_shape, layout, mem) in enumerate(steps):
        t, x = _input(device, shape, layout, mem, seed)
        _check(reshape(x, new_shape), t, new_shape, layout, mem)
    t, x = _input(device, (4, 8, 32, 256), RM, DRAM, seed=7)
    out = ttnn.to_layout(reshape(x, (4, 8, 16, 512)), TILE)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(out)), torch.reshape(t, (4, 8, 16, 512)))


# ---------------------------------------------------------------------------------------------------------------
# Refinement 2: changes = w in tile layout (regime w_regroup_tile)
# ---------------------------------------------------------------------------------------------------------------

W_TILE_CASES = [
    # (shape, new shape, memory): element runs move between tiles; output tiles padded for the new shape (R4)
    ((1, 1, 32, 64), (1, 1, 64, 32), L1),  # H and W exchanged (not a transpose)
    ((1, 1, 17, 50), (1, 1, 25, 34), DRAM),  # non-aligned in and out: runs cut at input row ends mid-face
    ((4, 128, 47), (4, 47, 128), L1),  # W non-aligned -> aligned, H_out non-aligned
    ((1, 1, 32, 8192), (1, 8, 32, 1024), DRAM),  # wide: a 256-tile input tile-row; the window must fit
    ((1, 1, 64, 128), (1, 1, 128, 64), L1),  # ResNet-50's call
    ((1, 32, 128), (4096,), DRAM),  # rank 1 out: 128 output tiles with one valid row each
    ((2048,), (32, 64), L1),  # rank 1 in: one valid input row per tile
    ((2, 3, 5, 24), (4, 9, 20), DRAM),  # H_in < 32: an output tile-row gathers from several input planes
    ((3, 5, 7), (7, 15), DRAM),  # 7 -> 15 element rows: every run off the word phase
]


def _w_tile_plan(device, x, new_shape, **knobs):
    out = ttnn.allocate_tensor_on_device(ttnn.TensorSpec(new_shape, x.dtype, TILE, x.memory_config()), device)
    regime, plan, spec, *_ = create_program_artifacts(x, out, tuple(x.shape), new_shape, **knobs)
    assert regime == WT
    return plan, spec, out


@pytest.mark.parametrize("shape,new_shape,mem", W_TILE_CASES, ids=_ids)
def test_w_regroup_tile_pinned(device, shape, new_shape, mem):
    assert select_regime(shape, resolve_shape(shape, new_shape), TILE) == WT
    t, x = _input(device, shape, TILE, mem)
    _check(reshape(x, new_shape), t, new_shape, TILE, mem)


@pytest.mark.parametrize(
    "shape,new_shape,knobs",
    [
        ((4, 128, 47), (4, 47, 128), {"row_groups": 1, "block_out_tile_rows": 3, "col_groups": 3}),
        ((1, 1, 32, 8192), (1, 8, 32, 1024), {"row_groups": 3, "col_groups": 5, "block_out_tile_rows": 2}),
        ((2, 3, 5, 24), (4, 9, 20), {"row_groups": 1, "block_out_tile_rows": 1}),
    ],
    ids=["one_cluster_ragged_blocks", "column_windows_mid_row", "small_planes_row_blocks"],
)
def test_w_regroup_tile_forced_blocks(device, shape, new_shape, knobs):
    """Several blocks per cluster (window depth 2, ragged last block) and column groups whose input column window
    starts mid-row (win_col0 > 0), forced through the host knobs."""
    t, x = _input(device, shape, TILE, DRAM)
    plan, _, _ = _w_tile_plan(device, x, new_shape, **knobs)
    assert plan.max_blocks_per_core > 1 and plan.input_window_depth == 2
    _check(_reshape(x, new_shape, **knobs), t, new_shape, TILE, DRAM)


@pytest.mark.parametrize(
    "shape,new_shape",
    [((4, 8, 32, 256), (4, 8, 16, 512)), ((4, 128, 47), (4, 47, 128)), ((1, 1, 32, 8192), (1, 8, 32, 1024))],
    ids=_ids,
)
def test_w_regroup_tile_split(device, shape, new_shape):
    """Structural counts read off the built plan: the cores the ProgramSpec targets are the split's, every output tile
    belongs to one cluster, each cluster's range is one block, and the input reads stay within the column-split
    bound max(2 x minimum, the column-unsplit reads) -- the minimum itself where row ranges meet input tile-rows."""
    t, x = _input(device, shape, TILE, DRAM)
    plan, spec, out = _w_tile_plan(device, x, new_shape)
    assert spec.work_units[0].target_nodes.num_cores() == len(plan.cores) == plan.row_groups * plan.col_groups
    grid = device.compute_with_storage_grid_size()
    assert plan.row_groups == min(plan.tensor_out_tile_rows, plan.tensor_in_tile_rows, grid.x * grid.y)
    assert sum(plan.col_count[c] * plan.row_count[c] for c in plan.cores) == out.buffer_num_pages()
    assert plan.max_blocks_per_core == 1
    unsplit, _, _ = _w_tile_plan(device, x, new_shape, col_groups=1)
    assert plan.input_tiles_read() <= max(2 * plan.minimum_input_tiles(), unsplit.input_tiles_read())
    if shape == (4, 8, 32, 256):  # one input plane -> one output tile-row: read once, the whole grid engaged
        assert len(plan.cores) == grid.x * grid.y
        assert plan.input_tiles_read() == plan.minimum_input_tiles() == x.buffer_num_pages()


@pytest.mark.parametrize(
    "shape,new_shape,mem",
    [((1, 1, 17, 50), (1, 1, 25, 34), L1), ((4, 128, 47), (4, 47, 128), DRAM), ((1, 1, 32, 8192), (1, 8, 32, 1024), DRAM)],
    ids=_ids,
)
def test_w_regroup_tile_uint32(device, shape, new_shape, mem):
    """4-byte elements: run lengths and offsets derive from the page size, so uint32 runs the same walker bit for bit.
    The wide shape's 4 KiB tiles make a whole input tile-row (1 MiB) too big for one ring: the fit search splits the
    output columns instead of refusing."""
    t, x = _uint32_input(device, shape, TILE, mem)
    if shape == (1, 1, 32, 8192):
        plan, _, _ = _w_tile_plan(device, x, new_shape)
        assert plan.col_groups > 1 and plan.window_entries < plan.in_w_tiles
    _check_exact(reshape(x, new_shape), t, new_shape, ttnn.uint32, TILE, mem)


def test_w_regroup_tile_session(device):
    """One device session: a tt-metal program, then the tile W regime between the other regimes, then a tt-metal
    program (untilize) on a tile W-regime output."""
    t, x = _input(device, (1, 64, 400), TILE, L1, seed=1)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(x, RM))), t)
    steps = [
        ((1, 1, 64, 128), (1, 1, 128, 64), TILE, L1),  # WT
        ((1, 64, 400), (1, 4, 16, 400), TILE, L1),  # B
        ((1, 1, 17, 50), (1, 1, 25, 34), TILE, DRAM),  # WT
        ((4, 128, 47), (4, 47, 128), RM, L1),  # W
        ((1, 40, 40, 256), (1, 1, 1600, 256), RM, DRAM),  # A
        ((4, 128, 47), (4, 47, 128), TILE, DRAM),  # WT
    ]
    for seed, (shape, new_shape, layout, mem) in enumerate(steps):
        t, x = _input(device, shape, layout, mem, seed)
        _check(reshape(x, new_shape), t, new_shape, layout, mem)
    t, x = _input(device, (1, 1, 17, 50), TILE, DRAM, seed=7)
    out = ttnn.to_layout(reshape(x, (1, 1, 25, 34)), RM)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(out)), torch.reshape(t, (1, 1, 25, 34)))
