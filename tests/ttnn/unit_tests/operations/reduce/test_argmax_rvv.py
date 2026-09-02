# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for the RVV argmax path: the Blackhole TILE-layout last-dim argmax
that runs the reduction on TRISC2's Zve32f vector unit.

ttnn.argmax picks a path on its own and takes no argument that names one, so
the path under test is pinned through the verification-only entry in the
private module (see ttnn/cpp/ttnn/operations/reduction/argmax/argmax_force.hpp).
The same entries supply the scalar-reader golden, which a plain ttnn.argmax
over an eligible TILE bfloat16 last dim does not run on Blackhole.
Automatic routing is covered separately at the bottom of this file.

Gating: architecture only. The RVV kernels JIT-compile with the in-tree opt-in
(ComputeConfigDescriptor::enable_trisc2_rvv, which adds zve32f to the TRISC2
compile), so no special toolchain or environment setup is needed — these tests
run automatically on any Blackhole host and skip on every other architecture.
Compile-side coverage runs device-free in CI
(tests/tt_metal/tt_metal/jit_build/test_trisc2_rvv.cpp).

Multicore: the RVV path splits the reduction dim's tiles across cores and
merges the per-core (index, value) candidates on a gather core. The whole point
of this path is that it is bit-identical to the scalar readers, so the merge
runs bfloat16_greater's bit-pattern order with a smallest-global-index
tie-break — not an IEEE compare. Every golden below is therefore the same
_ref_argmax_row whatever the core count, and the special-value cases are run
across explicit core counts (including ones that leave a ragged last slice)
precisely so a merge that got that order wrong would fail.
"""

import os

import numpy as np
import pytest
import torch
import ttnn

from models.common.utility_functions import run_for_blackhole

pytestmark = [
    run_for_blackhole("the RVV argmax path is Blackhole-only (TRISC2 Zve32f)"),
    # ttsim does not implement TRISC2's Zve32f unit -- it raises
    #   UnsupportedFunctionality: rv32_v_alu: babyrisc non-compliant V extension
    #   is explicitly out of scope
    # and that error path calls _Exit(1), killing the pytest/xdist worker rather than
    # failing a test. run_for_blackhole() alone does not gate it: is_blackhole() is true
    # under ttsim. Same shape as skip_routed_topk_on_sim in test_reduction.py.
    pytest.mark.skipif(
        bool(os.environ.get("TT_METAL_SIMULATOR")),
        reason=(
            "the RVV argmax path runs on TRISC2's Zve32f unit, which ttsim does not "
            "implement (UnsupportedFunctionality: rv32_v_alu)"
        ),
    ),
]

_force_rvv = ttnn._ttnn.operations.reduction.argmax_force_rvv
_force_scalar_reader = ttnn._ttnn.operations.reduction.argmax_force_scalar_reader


# bf16 bit patterns the cases below plant deliberately.
BF16_MAX_FINITE = 0x7F7F  # largest finite bf16: exponent 0xFE with every mantissa bit set
BF16_NEG_INF = 0xFF80  # -inf, which is also the scalar-reader kernel's accumulator init
BF16_POS_NAN = 0x7FC0  # a quiet +NaN; sorts above +inf in the bfloat16_greater bit order
BF16_NEG_NAN = 0xFFC0  # a quiet -NaN; sorts below -inf in that order, so it can never win
BF16_SIGN_BIT = 0x8000  # the sign bit alone is -0.0; OR-ing it makes any value negative


def _monotone(bits: np.ndarray) -> np.ndarray:
    """Monotone uint image of the bfloat16_greater sign-magnitude total order."""
    bits = bits.astype(np.uint32)
    return np.where(bits >= BF16_SIGN_BIT, (~bits) & 0xFFFF, bits | BF16_SIGN_BIT).astype(np.uint32)


_MONO_NEG_INF = 0x007F  # monotone(BF16_NEG_INF): the scalar readers' -inf init


def _ref_argmax_row(bits_row: np.ndarray):
    """Scalar-reader ttnn.argmax semantics: bfloat16_greater total order, smallest
    index on ties, -inf init (a row that never beats -inf reports (0, BF16_NEG_INF))."""
    m = _monotone(bits_row)
    if int(m.max()) <= _MONO_NEG_INF:
        return 0, BF16_NEG_INF
    i = int(np.argmax(m))  # first occurrence == smallest-index tie-break
    return i, int(bits_row[i])


def _bits_of(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16)


def _core_grid(device, num_cores: int, skip: int = 0):
    """A CoreRangeSet naming `num_cores` cores of the compute grid in row-major
    order, starting `skip` cores in. skip > 0 excludes core (0, 0), which is
    what proves the RVV path honours arbitrary placement rather than assuming
    the origin."""
    g = device.compute_with_storage_grid_size()
    assert skip + num_cores <= g.x * g.y, f"grid {g.x}x{g.y} cannot host {num_cores} cores at offset {skip}"
    ranges = []
    for k in range(skip, skip + num_cores):
        c = ttnn.CoreCoord(k % g.x, k // g.x)
        ranges.append(ttnn.CoreRange(c, c))
    return ttnn.CoreRangeSet(ranges)


def _single_core_grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _make_case(name: str, v: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Build the input for one named case: a row-major [b, v] array of bf16 bit
    patterns (uint16), not floats. The caller reinterprets them as bfloat16, so a
    case can plant the exact NaN, signed-zero and denormal encodings that
    generating floats and converting them could not reach.

    b is the number of rows, each of which is reduced independently; v is the
    number of elements in a row, i.e. the width of the reduction. name picks one
    of the scenarios listed in CASES -- which is defined just below this function
    rather than above it, so that is where to look for the legal values.

    Most cases start from a random bf16 background and then overwrite a few
    positions; denormal and all_neginf replace the row wholesale, and
    all_negative rewrites the sign of every element in it. The overwritten
    positions are fractions of v so that one set of positions works at every width
    the tests sweep, and each fraction is there for what it lets an assertion
    catch rather than for the number itself:

      * 5 * v // 8 puts the maximum in the interior of the row. A scan whose
        bounds are off by one at either end can still find a maximum sitting at
        index 0 or at index v - 1, so those two ends are their own cases
        (max_at_zero, max_at_end) and this one keeps the maximum away from both.
      * v // 3 (unique_max) is a decoy one ULP below the planted maximum, placed
        ahead of it in the row. A reader that stopped at the first large value it
        saw, or that compared in the wrong order, answers v // 3 rather than
        5 * v // 8.
      * 7 * v // 8 (tie_first_wins) repeats the very same maximum behind the
        first one, so the row holds two equal maxima and the smallest-index
        tie-break becomes observable: the answer must stay 5 * v // 8.

    One limit worth stating outright: a tile is 32 wide, so at the narrowest
    width in the sweep (v = 32) the whole row is a single tile and both tie
    positions land inside it (5 * 32 // 8 = 20, 7 * 32 // 8 = 28). tie_first_wins
    is a within-tile tie at that width; the wider widths (2016 and up) are what
    make it a cross-tile tie, and a multicore split is what makes it a cross-core
    one.
    """
    x = rng.standard_normal((b, v), dtype=np.float32) * 4.0
    bits = _bits_of(torch.from_numpy(x).bfloat16()).reshape(b, v)
    kdecoy = 0x7F7E  # one ULP below BF16_MAX_FINITE: a decoy the RNG cannot reach
    if name == "random":
        pass
    elif name == "unique_max":
        bits[:, 5 * v // 8] = BF16_MAX_FINITE
        bits[:, v // 3] = kdecoy
    elif name == "tie_first_wins":
        bits[:, 5 * v // 8] = BF16_MAX_FINITE
        bits[:, 7 * v // 8] = BF16_MAX_FINITE
    elif name == "max_at_end":
        bits[:, v - 1] = BF16_MAX_FINITE
    elif name == "max_at_zero":
        bits[:, 0] = BF16_MAX_FINITE
    elif name == "denormal":
        small = rng.integers(0, 0x0080, size=(b, v), dtype=np.uint16)  # denormals and +/-0
        sign = (rng.integers(0, 2, size=(b, v), dtype=np.uint16) << 15).astype(np.uint16)
        bits = (small | sign).astype(np.uint16)
    elif name == "nan_bearing":
        bits[:, v // 4] = BF16_POS_NAN
        bits[:, v // 2] = BF16_NEG_NAN
    elif name == "all_negative":
        bits = (bits | BF16_SIGN_BIT).astype(np.uint16)
        bits[bits == BF16_NEG_INF] = 0xBF80  # -1.0: avoid accidental -inf
    elif name == "all_neginf":
        bits = np.full((b, v), BF16_NEG_INF, dtype=np.uint16)  # the -inf init corner
    else:
        raise ValueError(name)
    return bits


CASES = [
    "random",
    "unique_max",
    "tie_first_wins",
    "max_at_end",
    "max_at_zero",
    "denormal",
    "nan_bearing",
    "all_negative",
    "all_neginf",
]


# v boundaries: 32 = single tile (chunk_pages = min(64, w_tiles) = 1);
# 2016 = 63 tiles, exercising the w_tiles - tiles_done < chunk_pages remainder
# branch; 2048/8192 = exact multiples of the 64-tile chunk.
@pytest.mark.parametrize("v", [32, 2016, 2048, 8192])
@pytest.mark.parametrize("b", [1, 5, 32])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("with_maxval", [True, False])
@pytest.mark.parametrize("grid", ["single", "auto"])
def test_argmax_rvv_special_values(device, v, b, keepdim, with_maxval, grid):
    """Bit-exact index (and optional max-value bits) against the scalar readers'
    semantics across planted-max / tie / special-value cases, pinned to one
    core and at the path's default (multicore) core count. The goldens are
    identical for both: the core count may not be observable in the result."""
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

        idx_t = _force_rvv(t_tile, dim=3, keepdim=keepdim, maxval_tensor=mv, sub_core_grids=sub_core_grids)
        got_idx = ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)
        got_val = _bits_of(ttnn.to_torch(mv).flatten()) if with_maxval else None

        for r in range(b):
            ref_idx, ref_val = _ref_argmax_row(bits[r])
            assert int(got_idx[r]) == ref_idx, f"case {name} row {r} ({grid}): idx {int(got_idx[r])} != {ref_idx}"
            if with_maxval:
                assert (
                    int(got_val[r]) == ref_val
                ), f"case {name} row {r} ({grid}): val {int(got_val[r]):#06x} != {ref_val:#06x}"


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("v", [32, 2016, 4096])
@pytest.mark.parametrize("b", [1, 32])
def test_argmax_rvv_matches_upstream_tile_path(device, name, v, b):
    """Index cross-check against the scalar-reader ttnn.argmax on the same TILE
    tensor — not just random data: every planted-max / tie / special-value
    case from CASES runs through both."""
    rng = np.random.default_rng(42 + v + 100 * b)
    bits = _make_case(name, v, b, rng)
    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    idx_rvv = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True))
    idx_ref = ttnn.to_torch(_force_scalar_reader(t_tile, dim=3, keepdim=True))
    assert torch.equal(
        idx_rvv.to(torch.int64), idx_ref.to(torch.int64)
    ), f"RVV/scalar-reader index mismatch on case {name!r} (v={v}, b={b})"


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("v", [32, 2016, 4096])
def test_argmax_rank1_with_maxval_through_public_entry(device, name, v):
    """A rank-1 [v] TILE bfloat16 last-dim reduction through the public entry,
    carrying a maxval_tensor.

    Rank 1 has no second-to-last dim, so it is the H == 1 shape by
    construction and must reach the RVV path: the scalar readers cannot fill a
    max-value output, so routing rank 1 to them would make this exact call
    raise instead. Both outputs are checked against the host bit-level
    reference, not just the index, because the max value is the half only the
    vector paths can produce."""
    rng = np.random.default_rng(99 + v)
    bits = _make_case(name, v, 1, rng)[0]  # rank-1: one row, no batch dim
    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(v)

    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mv = ttnn.from_torch(
        torch.zeros((1,), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    idx_t = ttnn.argmax(t_tile, dim=-1, keepdim=True, maxval_tensor=mv)
    got_idx = int(ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)[0])
    got_val = int(_bits_of(ttnn.to_torch(mv).flatten())[0])

    ref_idx, ref_val = _ref_argmax_row(bits)
    assert got_idx == ref_idx, f"case {name} (v={v}): idx {got_idx} != {ref_idx}"
    assert got_val == ref_val, f"case {name} (v={v}): val {got_val:#06x} != {ref_val:#06x}"


# ---------------------------------------------------------------------------
# Multicore
# ---------------------------------------------------------------------------
# The RVV path splits the reduction dim's tiles into contiguous per-core slices
# and merges one (global index, max value) candidate per core per row on a
# gather core. Two things can only break there: an index that stayed local to
# its slice, and a merge comparator that is not bfloat16_greater's bit-pattern
# order. Both are visible as a wrong index, so these tests score indices (and
# max-value bits) against the same single-core golden.


@pytest.mark.parametrize("num_cores", [2, 3, 7, 8, 16])
@pytest.mark.parametrize("v", [32, 2016, 2048])
@pytest.mark.parametrize("b", [1, 5])
def test_argmax_rvv_multicore_core_counts(device, num_cores, v, b):
    """The full special-value case bank at explicit core counts, including splits
    that leave a ragged last slice: v=2016 is 63 tiles, so 63 % 2, 7, 8 and 16
    are all non-zero and the leading cores carry one tile more than the rest.
    v=32 is one tile, i.e. fewer tiles than cores — the count is capped and the
    path falls back to its single-core shape.

    This is the bit-exactness proof for the merge: every row's answer travels
    through the exchange buffer, and NaN payloads / -inf / all-negative rows /
    ties all resolve there rather than inside one core's scan."""
    sub_core_grids = _core_grid(device, num_cores)
    rng = np.random.default_rng(555 + v + 100 * b + num_cores)
    for name in CASES:
        bits = _make_case(name, v, b, rng)
        x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)
        t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        mv = ttnn.from_torch(
            torch.zeros((1, 1, b, 1), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        idx_t = _force_rvv(t_tile, dim=3, keepdim=True, maxval_tensor=mv, sub_core_grids=sub_core_grids)
        got_idx = ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)
        got_val = _bits_of(ttnn.to_torch(mv).flatten())

        for r in range(b):
            ref_idx, ref_val = _ref_argmax_row(bits[r])
            assert (
                int(got_idx[r]) == ref_idx
            ), f"case {name} row {r} ({num_cores} cores): idx {int(got_idx[r])} != {ref_idx}"
            assert (
                int(got_val[r]) == ref_val
            ), f"case {name} row {r} ({num_cores} cores): val {int(got_val[r]):#06x} != {ref_val:#06x}"


@pytest.mark.parametrize("num_cores", [1, 4, 13])
def test_argmax_rvv_runs_on_explicit_grid_without_origin(device, num_cores):
    """Arbitrary placement: a grid that excludes core (0, 0) must run, not
    raise — the RVV path must not assume core (0, 0). This is asserted on the
    forced entry — which never falls back — over the whole special-value case
    bank."""
    sub_core_grids = _core_grid(device, num_cores, skip=1)
    v, b = 2016, 5
    rng = np.random.default_rng(31337 + num_cores)
    for name in CASES:
        bits = _make_case(name, v, b, rng)
        x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)
        t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        got = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True, sub_core_grids=sub_core_grids))
        expected = ttnn.to_torch(_force_scalar_reader(t_tile, dim=3, keepdim=True))
        assert torch.equal(
            got.to(torch.int64), expected.to(torch.int64)
        ), f"case {name}: RVV on {num_cores} cores away from (0, 0) diverged from the scalar readers"


def test_argmax_rvv_multicore_placement_does_not_change_the_answer(device):
    """The same tensor over every core count from 1 to 12 must give bit-identical
    results. A merge that silently favoured, say, the gather core's own slice
    would still match the golden at some core counts and not others."""
    v, b = 2016, 3
    rng = np.random.default_rng(2718)
    per_case_expected = {}
    for name in CASES:
        bits = _make_case(name, v, b, rng)
        x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)
        t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        per_case_expected[name] = (t_tile, ttnn.to_torch(_force_scalar_reader(t_tile, dim=3, keepdim=True)))

    for num_cores in range(1, 13):
        sub_core_grids = _core_grid(device, num_cores)
        for name, (t_tile, expected) in per_case_expected.items():
            got = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True, sub_core_grids=sub_core_grids))
            assert torch.equal(
                got.to(torch.int64), expected.to(torch.int64)
            ), f"case {name}: {num_cores}-core result differs from the scalar readers"


@pytest.mark.parametrize("num_cores", [2, 4, 9])
def test_argmax_rvv_multicore_merge_uses_bit_pattern_order(device, num_cores):
    """The cross-core merge must run bfloat16_greater's bit-pattern order — the
    scalar readers' order — rather than the IEEE compare the SFPU path's merge
    uses. Two rows are built whose answers differ between the two orders, with
    the discriminating element deliberately placed in the last tile (index
    2000 of 2016, tile 62), i.e. never in the gather core's own slice:

      * row 0: the largest finite bf16 early on, a +NaN at 2000. IEEE says a
        NaN never compares greater, so an IEEE merge would keep the finite
        index; the bit order sorts +NaN above +inf, so 2000 is the answer.
      * row 1: all -0 except a +0 at 2000. IEEE says +0 == -0, so an IEEE merge
        would tie-break to the smaller index (0); the bit order says +0 > -0,
        so 2000 is the answer again.
      * row 2: all -0 except a -NaN at 2000, which can never win under either
        order — it sorts below -inf — so the answer is 0.

    Reusing the SFPU path's exchange protocol with its comparator would fail
    the first two rows at every core count above 1."""
    sub_core_grids = _core_grid(device, num_cores)
    v = 2016
    rng = np.random.default_rng(4242)
    bits = _make_case("random", v, 3, rng)
    bits[0, 3] = BF16_MAX_FINITE  # in the gather core's slice
    bits[0, 2000] = BF16_POS_NAN  # in the last slice: wins the bit order
    bits[1, :] = BF16_SIGN_BIT  # all -0 ...
    bits[1, 2000] = 0x0000  # ... except a +0 in the last slice
    bits[2, :] = BF16_SIGN_BIT
    bits[2, 2000] = BF16_NEG_NAN  # in the last slice: never wins

    expected = [2000, 2000, 0]
    for r, want in enumerate(expected):
        assert _ref_argmax_row(bits[r])[0] == want, f"reference disagrees with the constructed row {r}"

    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, 3, v)
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mv = ttnn.from_torch(
        torch.zeros((1, 1, 3, 1), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    idx_t = _force_rvv(t_tile, dim=3, keepdim=True, maxval_tensor=mv, sub_core_grids=sub_core_grids)
    got_idx = ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)
    got_val = _bits_of(ttnn.to_torch(mv).flatten())

    for r, want in enumerate(expected):
        ref_val = _ref_argmax_row(bits[r])[1]
        assert int(got_idx[r]) == want, f"row {r} ({num_cores} cores): idx {int(got_idx[r])} != {want}"
        assert int(got_val[r]) == ref_val, f"row {r} ({num_cores} cores): val {int(got_val[r]):#06x} != {ref_val:#06x}"


@pytest.mark.parametrize("num_cores", [1, 5, 9])
def test_argmax_rvv_multicore_multi_tile_row_batch(device, num_cores):
    """Rank-4 batch shape with several tile-row passes: outer dims 2 x 3, h = 80
    rows (three tile-rows, the last one partial). Exercises the per-pass
    slot-reuse credit flow control and the padded-row masking under a
    multi-core split; finite random data, so the scalar readers' bit order is
    the golden."""
    sub_core_grids = _core_grid(device, num_cores)
    rng = np.random.default_rng(7)
    outer0, outer1, h, v = 2, 3, 80, 2016
    bits = _make_case("random", v, outer0 * outer1 * h, rng).reshape(outer0, outer1, h, v)
    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16)

    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mv = ttnn.from_torch(
        torch.zeros((outer0, outer1, h), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    idx_t = _force_rvv(t_tile, dim=3, keepdim=False, maxval_tensor=mv, sub_core_grids=sub_core_grids)
    got_idx = ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)
    got_val = _bits_of(ttnn.to_torch(mv).flatten())

    flat = bits.reshape(-1, v)
    for r in range(flat.shape[0]):
        ref_idx, ref_val = _ref_argmax_row(flat[r])
        assert int(got_idx[r]) == ref_idx, f"row {r} ({num_cores} cores): idx {int(got_idx[r])} != {ref_idx}"
        assert int(got_val[r]) == ref_val, f"row {r} ({num_cores} cores): val {int(got_val[r]):#06x} != {ref_val:#06x}"


def test_argmax_rvv_rejects_row_major_input(device, expect_error):
    """The RVV path is a TILE-layout path; ROW_MAJOR input must be rejected.
    Automatic dispatch never sends such a call here (it demotes to the scalar
    readers), so the refusal is checked through the forced entry — which must
    refuse rather than fall back, or a forced leg would prove nothing."""
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "requires TILE layout input"):
        _force_rvv(t_rm, dim=3, keepdim=True)


# ---------------------------------------------------------------------------
# Automatic routing
# ---------------------------------------------------------------------------
# ttnn.argmax exposes no path argument, so "which path ran" is read off the
# program cache: warming the expected path through its forced entry means a
# correctly routed ttnn.argmax hits that cached program and leaves the count
# alone, while a mis-route compiles a second program and grows it.


def _assert_empty_program_cache(device):
    """Precondition for the delta-0 proxy, checked before anything is warmed.

    "The auto call added no cache entry" only proves the route when the warmed
    path is the only argmax program cached for this shape. A stale entry for a
    different path -- left by an earlier test sharing the device -- would
    absorb a mis-route as a cache hit and the assertion would pass vacuously.
    The `device` fixture is function-scoped (conftest.py), so each of these tests
    starts from an empty cache; assert that, so that marking this file
    `use_module_device` (a natural CI speed-up) fails loudly instead of silently
    gutting every routing test below."""
    msg = (
        "routing tests need a per-test device: the program cache is not empty at test start, so a "
        "mis-route could hit a stale entry for another path and the delta-0 assertions below would "
        "prove nothing (do not mark this file use_module_device)"
    )
    assert device.num_program_cache_entries() == 0, msg


def _assert_program_cache_active(device):
    """The routing assertions read "which path ran" off program-cache growth,
    so an empty (or disabled) cache would make them pass vacuously."""
    msg = "device program cache is empty after warming a path; the routing assertions below would be vacuous"
    assert device.num_program_cache_entries() > 0, msg


# The routing boundary is kSfpuMinRows in argmax.cpp: H < 32 goes to RVV,
# H >= 32 goes to the SFPU. Both sides are asserted -- H = 31 here (just below)
# and H = 32 in test_argmax_sfpu.py::test_argmax_auto_routes_to_sfpu_at_batch
# (just above) -- so moving the constant in either direction fails a test.
SFPU_MIN_ROWS = 32


@pytest.mark.parametrize("h", [1, 8, SFPU_MIN_ROWS - 1], ids=["h1", "h8", "just_below_boundary"])
def test_argmax_auto_routes_to_rvv_below_the_sfpu_boundary(device, h):
    """Every H below kSfpuMinRows is the RVV path's shape: the SFPU
    alternative pays for all 32 lanes whether or not 32 rows are real, and at
    equal core counts it is measured slower for all of these — on one core the
    SFPU takes 12.4x as long as RVV at H = 1, and 3.1x as long at H = 8. Those
    two ratios come from test_argmax_path_crossover_bench.py in this directory,
    which is what regenerates them. h = 31 is the just-below-the-boundary case:
    it must not route to the SFPU, or kSfpuMinRows is not where argmax.cpp says
    it is."""
    _assert_empty_program_cache(device)
    x = torch.randn(1, 1, h, 4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, f"auto did not route H == {h} to the RVV path"


def test_argmax_auto_routes_rank1_to_rvv(device):
    """A rank-1 input has no H dim at all; it counts as H == 1 and must land on
    the RVV path, never on the SFPU (which is the measured loser at H == 1)
    and never on the scalar readers (which cannot fill a maxval_tensor)."""
    _assert_empty_program_cache(device)
    x = torch.randn(4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_rvv(t_tile, dim=-1, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=-1, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, "auto did not route a rank-1 input to the RVV path"


@pytest.mark.parametrize("h", [1, 5, 32, 64])
def test_argmax_exact_special_values_pins_rvv(device, h):
    """exact_special_values excludes the SFPU path, which normalises NaN, signed
    zero and denormal inputs before it compares them and so answers differently
    from the scalar readers for those values. An eligible call therefore lands on
    RVV at every H — including the H >= kSfpuMinRows shapes the default would
    send to the SFPU."""
    _assert_empty_program_cache(device)
    x = torch.randn(1, 1, h, 4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True, exact_special_values=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    msg = f"auto did not route exact_special_values=True (h={h}) to the RVV path"
    assert device.num_program_cache_entries() == entries_before, msg


def test_argmax_exact_special_values_changes_the_route(device):
    """The flag has to change the decision, not merely be accepted: at H = 32 the
    default routes to the SFPU path, and asking for exact special values
    moves the very same call to RVV."""
    _assert_empty_program_cache(device)
    x = torch.randn(1, 1, 32, 4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()

    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True, exact_special_values=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, "exact_special_values did not pin the RVV path"

    # Same tensor, same dim, flag dropped: this must not reuse the RVV program.
    ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True))
    msg = "the default at H = 32 reused the RVV program; exact_special_values would then be a no-op"
    assert device.num_program_cache_entries() > entries_before, msg
