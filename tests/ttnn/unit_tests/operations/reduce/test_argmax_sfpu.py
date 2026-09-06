# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for the SFPU argmax path: the Blackhole TILE-layout last-dim argmax
that runs phase 1 lane-parallel on the SFPU (all 32 rows of a tile-row per
pass — the batch-shape path) and finishes each row with a scalar phase 2 /
cross-core merge on the dataflow RISC.

ttnn.argmax picks a path on its own and takes no argument that names one, so
the path under test is pinned through the verification-only entry in the
private module (see ttnn/cpp/ttnn/operations/reduction/argmax/argmax_force.hpp).
The same entries supply the scalar-reader golden, which a plain ttnn.argmax
over an eligible TILE bfloat16 last dim does not run on Blackhole.
Automatic routing is covered separately at the bottom of this file.

Gating: architecture only. The kernels are plain SFPU compute kernels that JIT
with the stock toolchain, so these tests run automatically on any Blackhole
host and skip on every other architecture.

Goldens: the SFPU path normalises NaN, signed zero and denormal inputs before
it compares them, and then compares as IEEE fp32 (silicon-measured; see
argmax_sfpu_tile_compute.cpp). That is not the scalar readers' bfloat16_greater
order, which ranks those same values by their raw bit patterns. On finite,
normal data — including every exact tie — the two orders agree bit-for-bit, so
finite cases score against the scalar-reader golden; NaN- and denormal-bearing
cases score against the model of that normalisation implemented here:
  * NaN acts as same-signed infinity (a winning NaN reports its index but
    max value BF16_POS_INF, not the NaN payload; -NaN never wins),
  * denormals and -0 flush to +0 before the compare (first zero's index wins),
  * ties keep the smallest global index.
The measured +2^-127 pack bias on max values below ~2^-118 is not modeled:
no case in this bank can produce a winner in that magnitude range (denormal
winners flush to exactly +0, which reads back 0x0000 unbiased).
"""

import os

import numpy as np
import pytest
import torch
import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.reduce.argmax_common import (
    BF16_NEG_INF,
    BF16_SIGN_BIT,
    CASES,
    SFPU_MIN_ROWS,
    _assert_empty_program_cache,
    _assert_program_cache_active,
    _bits_of,
    _make_case,
    _monotone,
    _ref_argmax_row,
    _single_core_grid,
)

pytestmark = [
    run_for_blackhole("the SFPU argmax path is currently Blackhole-only"),
    # This suite passes under ttsim, but it is far too slow for the sim_bh_p150 budget in
    # .github/time_budget.yaml -- the cost is simulated device cycles, which scale with the
    # reduction width. Skipped for runtime only, not for any capability gap.
    pytest.mark.skipif(
        bool(os.environ.get("TT_METAL_SIMULATOR")),
        reason="too slow for the ttsim time budget (passes, but costs ~20 worker-minutes)",
    ),
]

_force_sfpu = ttnn._ttnn.operations.reduction.argmax_force_sfpu
_force_scalar_reader = ttnn._ttnn.operations.reduction.argmax_force_scalar_reader


# The only bf16 bit pattern this suite needs on its own; the rest come from argmax_common.
BF16_POS_INF = 0x7F80  # +inf, and the magnitude above which a bit pattern is a NaN


def _sfpu_special_values(bits: np.ndarray) -> np.ndarray:
    """The SFPU pipeline's measured normalisation of bf16 special values, applied
    before the compare: NaN -> same-signed infinity, denormals and +/-0 -> +0."""
    bits = bits.astype(np.uint16)
    mag = (bits & 0x7FFF).astype(np.uint16)
    out = bits.copy()
    is_nan = mag > BF16_POS_INF
    out[is_nan] = np.where((bits[is_nan] & BF16_SIGN_BIT) != 0, BF16_NEG_INF, BF16_POS_INF).astype(np.uint16)
    out[mag < 0x0080] = 0x0000  # denormals and both zeros flush to +0
    return out


def _sfpu_argmax_row(bits_row: np.ndarray):
    """SFPU-path semantics: IEEE compare on the normalised values, smallest
    index on ties; the max-value output is the normalised winner."""
    g = _sfpu_special_values(bits_row)
    m = _monotone(g)  # after normalisation there are no NaNs and only +0, so
    i = int(np.argmax(m))  # the monotone bit order is the IEEE order
    return i, int(g[i])


# Classes where the SFPU path's order provably equals the scalar readers' bit order
# (finite normal data, ties included; all_neginf hits both paths' -inf rule).
SCALAR_READER_EXACT_CASES = [c for c in CASES if c not in ("denormal", "nan_bearing")]


def _golden_row(name: str, bits_row: np.ndarray):
    """Score finite classes against the scalar-reader golden (stronger claim) and
    NaN/denormal classes against the special-value normalisation model above."""
    if name in ("denormal", "nan_bearing"):
        return _sfpu_argmax_row(bits_row)
    return _ref_argmax_row(bits_row)


# v boundaries: 32 = single tile (also the num_cores == 1 degenerate multicore
# case); 2016 = 63 tiles, exercising the chunk-remainder branch and an uneven
# multicore split; 2048/8192 = exact multiples of the 64-tile chunk; 32768 =
# the LLM-vocab-class width the path exists for.
@pytest.mark.parametrize("v", [32, 2016, 2048, 8192, 32768])
@pytest.mark.parametrize("b", [1, 8, 32])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("with_maxval", [True, False])
@pytest.mark.parametrize("grid", ["single", "multi"])
def test_argmax_sfpu_special_values(device, v, b, keepdim, with_maxval, grid):
    """Bit-exact index (and optional max-value bits) against the documented
    semantics across planted-max / tie / special-value cases, single-core and
    multicore."""
    sub_core_grids = _single_core_grid() if grid == "single" else None
    rng = np.random.default_rng(1234 + v + 100 * b)
    for name in CASES:
        bits = _make_case(name, v, b, rng)
        x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)

        t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        mv = None
        if with_maxval:
            out_shape = (1, 1, b, 1) if keepdim else (1, 1, b)
            mv = ttnn.from_torch(
                torch.zeros(out_shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

        idx_t = _force_sfpu(t_tile, dim=3, keepdim=keepdim, maxval_tensor=mv, sub_core_grids=sub_core_grids)
        got_idx = ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)
        got_val = _bits_of(ttnn.to_torch(mv).flatten()) if with_maxval else None

        for r in range(b):
            ref_idx, ref_val = _golden_row(name, bits[r])
            assert int(got_idx[r]) == ref_idx, f"case {name} row {r} ({grid}): idx {int(got_idx[r])} != {ref_idx}"
            if with_maxval:
                assert (
                    int(got_val[r]) == ref_val
                ), f"case {name} row {r} ({grid}): val {int(got_val[r]):#06x} != {ref_val:#06x}"


@pytest.mark.parametrize("name", SCALAR_READER_EXACT_CASES)
@pytest.mark.parametrize("v", [32, 2016, 4096])
@pytest.mark.parametrize("b", [1, 32])
def test_argmax_sfpu_matches_upstream_tile_path(device, name, v, b):
    """Index cross-check against the scalar-reader ttnn.argmax on the same TILE
    tensor for every class where the two orders provably agree (finite data
    plus the all--inf corner). The NaN/denormal classes are excluded by
    construction — that divergence is documented, measured, and asserted
    against the normalisation model in the special-values test above."""
    rng = np.random.default_rng(42 + v + 100 * b)
    bits = _make_case(name, v, b, rng)
    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    idx_sfpu = ttnn.to_torch(_force_sfpu(t_tile, dim=3, keepdim=True))
    idx_ref = ttnn.to_torch(_force_scalar_reader(t_tile, dim=3, keepdim=True))
    assert torch.equal(
        idx_sfpu.to(torch.int64), idx_ref.to(torch.int64)
    ), f"SFPU/scalar-reader index mismatch on case {name!r} (v={v}, b={b})"


@pytest.mark.parametrize("grid", ["single", "multi"])
def test_argmax_sfpu_multi_tile_row_batch(device, grid):
    """Rank-4 batch shape with several tile-row passes: outer dims 2 x 3,
    h = 80 rows (three tile-rows, the last one partial). Exercises the
    multi-pass credit flow-control and the padded-row masking; finite random
    data, so results must match the scalar readers' bit order exactly."""
    sub_core_grids = _single_core_grid() if grid == "single" else None
    rng = np.random.default_rng(7)
    outer0, outer1, h, v = 2, 3, 80, 2048
    bits = _make_case("random", v, outer0 * outer1 * h, rng).reshape(outer0, outer1, h, v)
    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16)

    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mv = ttnn.from_torch(
        torch.zeros((outer0, outer1, h), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    idx_t = _force_sfpu(t_tile, dim=3, keepdim=False, maxval_tensor=mv, sub_core_grids=sub_core_grids)
    got_idx = ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)
    got_val = _bits_of(ttnn.to_torch(mv).flatten())

    flat = bits.reshape(-1, v)
    for r in range(flat.shape[0]):
        ref_idx, ref_val = _ref_argmax_row(flat[r])
        assert int(got_idx[r]) == ref_idx, f"row {r} ({grid}): idx {int(got_idx[r])} != {ref_idx}"
        assert int(got_val[r]) == ref_val, f"row {r} ({grid}): val {int(got_val[r]):#06x} != {ref_val:#06x}"


# The three refusals below are unreachable through ttnn.argmax: automatic
# dispatch checks the same preconditions and demotes to the scalar readers
# instead of routing a case the path cannot serve. They are checked through
# the forced entry, which must refuse rather than fall back — a forced leg that
# quietly served another path would make every comparison against it vacuous.


def test_argmax_sfpu_rejects_row_major_input(device, expect_error):
    """The SFPU path is a TILE-layout path; ROW_MAJOR input must be rejected."""
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "requires TILE layout input"):
        _force_sfpu(t_rm, dim=3, keepdim=True)


@pytest.mark.parametrize("rejected_flag", ["use_rvv", "use_sfpu"])
def test_argmax_rejects_path_selection_flags(device, rejected_flag):
    """The path is not caller-selectable: ttnn.argmax has no use_rvv / use_sfpu
    argument, so passing either is a TypeError rather than a knob that could be
    set to a contradictory pair."""
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(TypeError):  # allow-pytest.raises: host binding TypeError (unknown kwarg), not a device fault
        ttnn.argmax(t_tile, dim=3, keepdim=True, **{rejected_flag: True})


def test_argmax_sfpu_rejects_ragged_last_dim(device, expect_error):
    """The reduction dim must be a multiple of the tile width (no W padding)."""
    x = torch.randn(1, 1, 32, 2000).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "multiple of the tile width"):
        _force_sfpu(t_tile, dim=3, keepdim=True)


def test_argmax_sfpu_rejects_sub_core_grid_outside_device(device, expect_error):
    """SFPU core placement must stay inside the device compute grid. This one is
    reachable through ttnn.argmax too — the grid check is path-independent —
    so it is asserted on the public entry."""
    grid = device.compute_with_storage_grid_size()
    outside = ttnn.CoreCoord(grid.x, 0)
    invalid_grid = ttnn.CoreRangeSet([ttnn.CoreRange(outside, outside)])
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must be contained in device compute grid"):
        ttnn.argmax(t_tile, dim=3, keepdim=True, sub_core_grids=invalid_grid)


def test_argmax_sfpu_runs_on_explicit_nonzero_core(device):
    """A valid explicit grid may place the single-core SFPU path away from (0, 0)."""
    grid = device.compute_with_storage_grid_size()
    core = ttnn.CoreCoord(grid.x - 1, grid.y - 1)
    sub_core_grids = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    x = torch.randn(1, 1, 8, 2048).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(_force_sfpu(t_tile, dim=3, keepdim=True, sub_core_grids=sub_core_grids))
    expected = ttnn.to_torch(_force_scalar_reader(t_tile, dim=3, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))


# ---------------------------------------------------------------------------
# Automatic routing
# ---------------------------------------------------------------------------
# ttnn.argmax exposes no path argument, so "which path ran" is read off the
# program cache: warming the expected path through its forced entry means a
# correctly routed ttnn.argmax hits that cached program and leaves the count
# alone, while a mis-route compiles a second program and grows it.


@pytest.mark.parametrize("h", [SFPU_MIN_ROWS, 64], ids=["at_boundary", "above_boundary"])
def test_argmax_auto_routes_to_sfpu_at_batch(device, h):
    """H >= kSfpuMinRows is the SFPU path's shape: one lane-parallel pass
    covers all 32 rows of a tile-row, so its cost is flat in H where the RVV
    scan pays per row, and 32 is where every one of those 32 lanes is finally
    doing useful work. h = 32 is the just-at-the-boundary case: it must route
    to the SFPU, or kSfpuMinRows is not where argmax.cpp says it is.

    The boundary sits at 32 rather than lower because at H = 8 the multicore
    RVV path is measured faster than the multicore SFPU at every core count up
    to 64: the SFPU takes 3.1x as long as RVV on one core and 1.2x as long on
    64. Those two ratios come from test_argmax_path_crossover_bench.py in this
    directory, which is what regenerates them."""
    _assert_empty_program_cache(device)
    x = torch.randn(1, 1, h, 4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_sfpu(t_tile, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, f"auto did not route h={h} to the SFPU path"


@pytest.mark.parametrize(
    "shape,layout",
    [
        # last dim not a multiple of the tile width
        ((1, 1, 32, 2000), ttnn.TILE_LAYOUT),
        # ROW_MAJOR input: the vector paths read TILE directly
        ((1, 1, 32, 2048), ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["ragged_last_dim", "row_major"],
)
def test_argmax_auto_falls_back_to_scalar_readers(device, shape, layout):
    """A case outside the vector paths' preconditions must demote to the
    scalar readers, not raise: automatic dispatch may never turn a servable
    call into an error."""
    _assert_empty_program_cache(device)
    x = torch.randn(*shape).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)

    expected = ttnn.to_torch(_force_scalar_reader(t, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t, dim=3, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, "auto did not fall back to the scalar readers"


def test_argmax_maxval_tensor_on_scalar_reader_route_raises(device, expect_error):
    """maxval_tensor can only be filled by the vector paths. A call the
    heuristic sends to the scalar readers must say so rather than hand the
    buffer back untouched."""
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    mv = ttnn.from_torch(
        torch.zeros((1, 1, 32, 1), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error(RuntimeError, "only produced by the vector paths"):
        ttnn.argmax(t_rm, dim=3, keepdim=True, maxval_tensor=mv)
