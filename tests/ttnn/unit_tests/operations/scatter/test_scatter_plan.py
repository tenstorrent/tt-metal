# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design's Work Distribution asks for (op_design.md, "Tests the plan needs"):

1. every thread of the plan ran (probe_skip_thread: thread t leaves its source rows undone);
2. the op in one device session after another program and before a tt-metal program;
3. forced knobs (several blocks per cluster, several source chunks, ragged blocks splitting tiles, fewer
   threads) — no INPUTS shape reaches the L1 clamps on this target;
4. host-only: choose_blocking clamps, and refuses a line that does not fit by name.

Plus the batch axis (rank 3, dim -2) through the program directly: SUPPORTED does not claim it yet, the kernel
already takes (image, face-column) units.
"""

import pytest
import torch

import ttnn
from ttnn.operations.scatter import scatter
from ttnn.operations.scatter.scatter_program_artifacts import (
    SCATTER_THREADS,
    Geometry,
    choose_blocking,
    create_program_artifacts,
    footprint,
)

TILE = 32


def _distinct_index(shape, places, count, seed):
    """Positions along dim -2, no two source rows of a column sent to one place."""
    g = torch.Generator().manual_seed(seed)
    whole = list(shape)
    whole[-2] = places
    return torch.rand(whole, generator=g).argsort(dim=-2).narrow(-2, 0, count).contiguous()


def _inputs(shape, count, seed=0, index=None):
    g = torch.Generator().manual_seed(seed + 1)
    tin = torch.randn(shape, generator=g).to(torch.bfloat16)
    src_shape = list(shape)
    src_shape[-2] = count
    tsrc = torch.randn(src_shape, generator=g).to(torch.bfloat16)
    tidx = _distinct_index(shape, shape[-2], count, seed) if index is None else index
    return tin, tidx, tsrc


def _device(device, tin, tidx, tsrc):
    a = ttnn.from_torch(tin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    i = ttnn.from_torch(tidx.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    s = ttnn.from_torch(tsrc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return a, i, s


def _run_direct(device, tin, tidx, tsrc, **knobs):
    """The program with knob overrides / probes, bypassing validate() (tests only)."""
    a, i, s = _device(device, tin, tidx, tsrc)
    out = ttnn.allocate_tensor_on_device(a.spec, device)
    spec, run_args, indices = create_program_artifacts(a, i, s, out, **knobs)
    return ttnn.to_torch(ttnn.generic_op([a, i, s, out], spec, run_args, indices))


def _reference(tin, tidx, tsrc, keep_rows=None):
    out = tin.clone()
    idx, src = tidx.long(), tsrc
    if keep_rows is not None:
        idx, src = idx[..., keep_rows, :], src[..., keep_rows, :]
    return out.scatter_(-2, idx, src)


def _assert_bitwise(got, expected):
    assert got.shape == expected.shape
    diff = (got.view(torch.int16) != expected.view(torch.int16)).sum().item()
    assert diff == 0, f"{diff} elements differ"


def _rows_of_thread(t, src_rows, chunk_rows, threads):
    """Mirror of scatter_chunk's dealing: per chunk, contiguous ranges, the first `rem` threads one row more."""
    rows = []
    for row0 in range(0, src_rows, chunk_rows):
        n = min(chunk_rows, src_rows - row0)
        q, rem = divmod(n, threads)
        first = t * q + min(t, rem)
        rows += range(row0 + first, row0 + first + q + (1 if t < rem else 0))
    return rows


# --- 1. every thread ran -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("block_src_tile_rows", [None, 1], ids=["one_chunk", "two_chunks"])
@pytest.mark.parametrize("skip", range(SCATTER_THREADS))
def test_each_thread_moves_its_rows(device, skip, block_src_tile_rows):
    """Thread `skip` leaves its source rows undone: exactly their targets keep input's value. A correct result
    alone would not show that six threads ran; this does."""
    shape, count = (70, 512), 64
    tin, tidx, tsrc = _inputs(shape, count, seed=skip)
    chunk_rows = (block_src_tile_rows or (count + TILE - 1) // TILE) * TILE
    skipped = set(_rows_of_thread(skip, count, chunk_rows, SCATTER_THREADS))
    assert skipped, "every thread must be dealt source rows here"
    keep = [r for r in range(count) if r not in skipped]
    got = _run_direct(device, tin, tidx, tsrc, probe_skip_thread=skip, block_src_tile_rows=block_src_tile_rows)
    _assert_bitwise(got, _reference(tin, tidx, tsrc, keep_rows=keep))


# --- 2. one device session ---------------------------------------------------------------------------------------


def test_after_another_program_and_before_to_layout(device):
    """Another program first (a tt-metal to_layout), then scatter, then a tt-metal program on its output, then
    scatter again with another shape: all in one device session."""
    tin, tidx, tsrc = _inputs((70, 512), 64, seed=3)
    pre = ttnn.to_layout(ttnn.from_torch(tin, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
                         ttnn.TILE_LAYOUT)
    _, i, s = _device(device, tin, tidx, tsrc)
    out = scatter(pre, 0, i, s)
    expected = _reference(tin, tidx, tsrc)
    _assert_bitwise(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))), expected)
    tin2, tidx2, tsrc2 = _inputs((128, 512), 17, seed=4)
    a2, i2, s2 = _device(device, tin2, tidx2, tsrc2)
    out2 = scatter(a2, -2, i2, s2)
    _assert_bitwise(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out2, ttnn.ROW_MAJOR_LAYOUT))),
                    _reference(tin2, tidx2, tsrc2))


# --- 3. forced knobs ---------------------------------------------------------------------------------------------

KNOB_CASES = [
    # shape, source rows, knobs
    ((1024, 64), 512, {"block_src_tile_rows": 1}),                     # 16 chunks
    ((1024, 64), 512, {"block_src_tile_rows": 3}),                     # ragged last chunk
    ((64, 8192), 32, {"block_width_faces": 1}),                        # 16 blocks per cluster
    ((64, 8192), 32, {"block_width_faces": 3}),                        # blocks split tiles; ragged last block
    ((128, 512), 17, {"block_width_faces": 1, "block_src_tile_rows": 1}),
    ((70, 512), 64, {"threads": 1}),                                   # the plan at one thread
    ((70, 512), 64, {"threads": 2}),
    ((128, 4096), 100, {"block_width_faces": 5, "block_src_tile_rows": 2, "threads": 4}),
]


@pytest.mark.parametrize("shape,count,knobs", KNOB_CASES, ids=lambda v: str(v).replace(" ", ""))
def test_forced_knobs(device, shape, count, knobs):
    tin, tidx, tsrc = _inputs(shape, count, seed=7)
    _assert_bitwise(_run_direct(device, tin, tidx, tsrc, **knobs), _reference(tin, tidx, tsrc))


def test_positions_outside_the_line_are_skipped(device):
    """R6: a position outside [0, H) (here H, a large one and negatives) names no element; the rest stand."""
    shape, count = (70, 64), 32
    tin, tidx, tsrc = _inputs(shape, count, seed=9)
    bad = tidx.clone()
    bad[0, :] = 70
    bad[1, :] = -1
    bad[2, :] = 1 << 30
    got = _run_direct(device, tin, bad, tsrc)
    _assert_bitwise(got, _reference(tin, tidx, tsrc, keep_rows=list(range(3, count))))


# --- batch axis (rank 3, dim -2): not in SUPPORTED, but the program takes (image, face-column) units ----------


@pytest.mark.parametrize(
    "shape,count,knobs",
    [((4, 128, 512), 64, {}), ((5, 64, 224), 40, {}), ((5, 64, 224), 40, {"block_width_faces": 4})],
    ids=lambda v: str(v).replace(" ", ""),
)
def test_images_along_rows(device, shape, count, knobs):
    tin, tidx, tsrc = _inputs(shape, count, seed=11)
    _assert_bitwise(_run_direct(device, tin, tidx, tsrc, **knobs), _reference(tin, tidx, tsrc))


# --- 4. host-only ------------------------------------------------------------------------------------------------


def test_choose_blocking_clamps_and_refuses():
    geo = Geometry(images=1, out_rows=2766, src_rows=2752, width=2560, tile_bytes=2048, index_page_bytes=2560 * 4)
    whole = footprint(geo, 5, geo.tensor_src_tile_rows)
    assert choose_blocking(geo, 5, whole) == (5, geo.tensor_src_tile_rows)
    wf, sr = choose_blocking(geo, 5, whole - 1)  # the source chunk shrinks first
    assert wf == 5 and 1 <= sr < geo.tensor_src_tile_rows and footprint(geo, wf, sr) <= whole - 1
    wf, sr = choose_blocking(geo, 5, footprint(geo, 2, 1))  # then the width
    assert wf == 2 and sr >= 1
    with pytest.raises(NotImplementedError, match="banded_output_line"):
        choose_blocking(geo, 5, footprint(geo, 1, 1) - 1)
