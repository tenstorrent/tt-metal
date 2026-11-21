# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""permute: the tests the design's Work Distribution requires beyond the golden suite.

(1) every thread of each regime's plan ran: the output is pre-filled with a sentinel and the kernels size their
    loops from the plan's compile-time counts, so a thread that did not run leaves sentinel pages behind;
(2) device sessions: another program before permute, and a tt-metal program (to_layout) after it;
(3) regime pinning: select_regime() sends each case to the regime named, and the result is exact;
(4) the block knobs are live: other values of each knob give the same exact result.
"""

import pytest
import torch

import ttnn
from ttnn.operations.permute import permute
from ttnn.operations.permute.permute import _output_spec, _permute_into, normalize_dims
from ttnn.operations.permute.permute_program_artifacts import (
    GATHER_THREADS,
    _ceil,
    _prod,
    canonical,
    plan_staged_gather,
    select_regime,
    split_work,
)

SENTINEL = -12345.0  # exactly representable in bfloat16; randn never draws it


def _input(device, shape, layout, mem=ttnn.DRAM_MEMORY_CONFIG, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.bfloat16)
    return x, ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)


def _sentinel_output(device, t, norm):
    shape = [int(t.shape[d]) for d in norm]
    return ttnn.from_torch(
        torch.full(shape, SENTINEL, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=t.layout,
        device=device,
        memory_config=t.memory_config(),
    )


def _check(got, x, dims):
    ref = x.permute(dims)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert not (got == SENTINEL).any(), f"{int((got == SENTINEL).sum())} elements never written"
    assert torch.equal(got, ref), (got.float() - ref.float()).abs().max()


def _run_into(device, shape, dims, layout, mem=ttnn.DRAM_MEMORY_CONFIG, **knobs):
    x, t = _input(device, shape, layout, mem)
    norm = normalize_dims(dims, len(shape))
    out = _sentinel_output(device, t, norm)
    _check(ttnn.to_torch(_permute_into(t, norm, out, **knobs)), x, dims)
    return t


# ---------------------------------------------------------------------------
# (1) every thread of the plan ran
# ---------------------------------------------------------------------------


def test_every_gather_thread_ran(device):
    """staged_gather: 192 boxes over 32 clusters x 6 harts, so every hart of every cluster owns one box."""
    shape, dims = (2, 512, 1024), (0, 2, 1)
    x, t = _input(device, shape, ttnn.ROW_MAJOR_LAYOUT)
    s, d = canonical(shape, dims)
    assert select_regime(t.layout, d) == "staged_gather"
    out = _sentinel_output(device, t, list(dims))
    _, e, threads = plan_staged_gather(t, out, s, d)
    n_blocks = _prod(_ceil(s[j], e[j]) for j in range(4))
    grid = device.compute_with_storage_grid_size()
    _, counts, _ = split_work(n_blocks, grid)
    assert threads == GATHER_THREADS and len(counts) == grid.x * grid.y
    assert min(counts.values()) >= threads, (e, n_blocks, counts)
    _check(ttnn.to_torch(_permute_into(t, list(dims), out)), x, dims)


@pytest.mark.parametrize(
    "shape,dims,layout,regime",
    [
        ((4, 8, 32, 256), (1, 0, 2, 3), ttnn.TILE_LAYOUT, "page_gather"),  # 256 tiles: 8 per cluster
        ((2, 4, 64, 128), (0, 2, 1, 3), ttnn.ROW_MAJOR_LAYOUT, "page_gather"),  # 512 rows
        ((4, 128, 512), (0, 2, 1), ttnn.TILE_LAYOUT, "tile_transpose"),  # 256 tiles
        ((2, 4, 64, 128), (1, 0, 3, 2), ttnn.TILE_LAYOUT, "tile_transpose"),  # leading axes move too
    ],
)
def test_every_ring_thread_ran(device, shape, dims, layout, regime):
    s, d = canonical(shape, normalize_dims(dims, len(shape)))
    assert select_regime(layout, d) == regime
    _run_into(device, shape, dims, layout)


# ---------------------------------------------------------------------------
# (2) device sessions
# ---------------------------------------------------------------------------

SESSION = [
    ((1, 4, 16, 400), (0, 1, 3, 2), ttnn.TILE_LAYOUT),  # tile_transpose
    ((1, 16, 20, 20), (0, 2, 3, 1), ttnn.ROW_MAJOR_LAYOUT),  # staged_gather RM
    ((1, 4, 16, 20), (0, 2, 1, 3), ttnn.ROW_MAJOR_LAYOUT),  # page_gather RM
    ((1, 4, 400, 1), (0, 3, 1, 2), ttnn.TILE_LAYOUT),  # staged_gather TILE
    ((1, 2, 64, 64), (1, 0, 2, 3), ttnn.TILE_LAYOUT),  # page_gather TILE
    ((1, 400, 144), (0, 2, 1), ttnn.TILE_LAYOUT),  # tile_transpose
]


def test_regimes_share_a_device_session(device):
    """Each regime's program after every other one, in one open device session (implicit-sync rings, then the
    tagged explicit-sync gather, then rings again)."""
    for seed, (shape, dims, layout) in enumerate(SESSION + SESSION[::-1]):
        torch.manual_seed(seed)
        x = torch.randn(shape, dtype=torch.bfloat16)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
        assert torch.equal(ttnn.to_torch(permute(t, dims)), x.permute(dims))


@pytest.mark.parametrize("shape,dims,layout", SESSION, ids=lambda v: str(v))
def test_a_tt_metal_program_follows(device, shape, dims, layout):
    """A tt-metal program after ours in the same session: to_layout issues untagged transfers."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    y = permute(t, dims)
    other = ttnn.ROW_MAJOR_LAYOUT if layout == ttnn.TILE_LAYOUT else ttnn.TILE_LAYOUT
    z = ttnn.to_layout(y, other)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(z)), x.permute(dims))
    # and permute again after tt-metal's program
    assert torch.equal(ttnn.to_torch(permute(t, dims)), x.permute(dims))


# ---------------------------------------------------------------------------
# (3) regime pinning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,dims,layout,regime",
    [
        ((1, 1, 64, 128), (0, 1, 2, 3), ttnn.TILE_LAYOUT, "page_gather"),  # identity
        ((2, 4, 17, 50), (1, 0, -2, -1), ttnn.ROW_MAJOR_LAYOUT, "page_gather"),
        ((2, 4, 17, 50), (1, 0, -2, -1), ttnn.TILE_LAYOUT, "page_gather"),
        ((2, 1, 100, 47), (0, 1, -1, -2), ttnn.TILE_LAYOUT, "tile_transpose"),
        ((2, 1, 100, 47), (0, 1, -1, -2), ttnn.ROW_MAJOR_LAYOUT, "staged_gather"),
        ((2, 3, 4, 5), (3, 2, 1, 0), ttnn.TILE_LAYOUT, "staged_gather"),  # a and b both leading axes
        ((2, 3, 4, 5), (3, 2, 1, 0), ttnn.ROW_MAJOR_LAYOUT, "staged_gather"),
        ((1, 8, 47, 256), (0, 3, 1, 2), ttnn.TILE_LAYOUT, "staged_gather"),
        ((2, 32, 64, 128), (0, 2, 1, 3), ttnn.TILE_LAYOUT, "staged_gather"),  # whole rows move, TILE
    ],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("mem", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["DRAM", "L1"])
def test_regime_pinned(device, shape, dims, layout, regime, mem):
    s, d = canonical(shape, normalize_dims(dims, len(shape)))
    assert select_regime(layout, d) == regime
    _run_into(device, shape, dims, layout, mem)


def test_output_placed_like_input(device):
    """R7: the output is interleaved in the input's buffer type."""
    for mem, buf in ((ttnn.DRAM_MEMORY_CONFIG, ttnn.BufferType.DRAM), (ttnn.L1_MEMORY_CONFIG, ttnn.BufferType.L1)):
        _, t = _input(device, (1, 400, 144), ttnn.TILE_LAYOUT, mem)
        y = permute(t, (0, 2, 1))
        assert y.memory_config().buffer_type == buf
        assert y.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


# ---------------------------------------------------------------------------
# (4) the block knobs are live
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "knobs",
    [{"page_ring_depth": 2}, {"page_ring_depth": 16}, {"plan": (2, 0, 2)}, {"plan": (3, 0, 3)}],
    ids=str,
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
def test_page_gather_knobs(device, knobs, layout):
    _run_into(device, (2, 4, 64, 96), (1, 0, 2, 3), layout, **knobs)


@pytest.mark.parametrize(
    "knobs",
    [
        {"block_tiles": 1},
        {"block_tiles": 8},
        {"block_tiles": 3, "ring_blocks": 3},
        {"plan": (2, 2, 2)},
        {"plan": (1, 4, 1)},
        {"plan": (4, 4, 2), "block_tiles": 2},
    ],
    ids=str,
)
def test_tile_transpose_knobs(device, knobs):
    _run_into(device, (2, 4, 100, 200), (0, 1, 3, 2), ttnn.TILE_LAYOUT, **knobs)


@pytest.mark.parametrize(
    "shape,dims,layout,knobs",
    [
        ((1, 4, 16, 400), (0, 1, 3, 2), ttnn.ROW_MAJOR_LAYOUT, {"gather_threads": 1}),
        ((1, 4, 16, 400), (0, 1, 3, 2), ttnn.ROW_MAJOR_LAYOUT, {"gather_threads": 2, "out_slots": 1}),
        ((1, 4, 16, 400), (0, 1, 3, 2), ttnn.ROW_MAJOR_LAYOUT, {"extent": (1, 2, 16, 64)}),  # ragged on 3
        ((1, 8, 47, 256), (0, 3, 1, 2), ttnn.ROW_MAJOR_LAYOUT, {"extent": (1, 3, 47, 32)}),  # ragged on 1
        ((1, 8, 47, 256), (0, 3, 1, 2), ttnn.TILE_LAYOUT, {"extent": (1, 8, 64, 32)}),  # 2 tiles on h
        ((1, 8, 47, 256), (0, 3, 1, 2), ttnn.TILE_LAYOUT, {"gather_threads": 3, "out_slots": 2}),
        ((4, 8, 32, 47), (0, 2, 3, 1), ttnn.TILE_LAYOUT, {"extent": (2, 8, 32, 32)}),
    ],
    ids=str,
)
def test_staged_gather_knobs(device, shape, dims, layout, knobs):
    _run_into(device, shape, dims, layout, **knobs)
