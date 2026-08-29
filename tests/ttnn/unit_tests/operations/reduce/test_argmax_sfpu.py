# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for the SFPU argmax engine: the Blackhole TILE-layout last-dim argmax
that runs phase 1 lane-parallel on the SFPU (all 32 rows of a tile-row per
pass — the batch-shape engine) and finishes each row with a scalar phase 2 /
cross-core merge on the dataflow RISC.

ttnn.argmax picks an engine on its own and takes no argument that names one, so
the engine under test is pinned through the verification-only entry in the
private module (see ttnn/cpp/ttnn/operations/reduction/argmax/argmax_force.hpp).
The same entries supply the incumbent (scalar reader) golden, which a plain
ttnn.argmax over an eligible TILE bfloat16 last dim no longer runs on Blackhole.
Automatic routing is covered separately at the bottom of this file.

Gating: architecture only. The kernels are plain SFPU compute kernels that JIT
with the stock toolchain, so these tests run automatically on any Blackhole
host and skip on every other architecture.

Goldens: the SFPU engine's compare is IEEE-on-fp32 behind a bf16 special-value
gasket (silicon-measured; see argmax_sfpu_tile_compute.cpp), NOT the scalar
readers' bfloat16_greater bit order. On finite, normal data — including every
exact tie — the two orders agree bit-for-bit, so finite cases score against
the incumbent golden; NaN- and denormal-bearing cases score against the
gasket model implemented here:
  * NaN acts as same-signed infinity (a winning NaN reports its index but
    max value 0x7F80, not the NaN payload; -NaN never wins),
  * denormals and -0 flush to +0 before the compare (first zero's index wins),
  * ties keep the smallest global index.
The measured +2^-127 pack bias on max values below ~2^-118 is not modeled:
no case in this bank can produce a winner in that magnitude range (denormal
winners flush to exactly +0, which reads back 0x0000 unbiased).
"""

import numpy as np
import pytest
import torch
import ttnn

from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole("the SFPU argmax engine is currently Blackhole-only")

_force_sfpu = ttnn._ttnn.operations.reduction.argmax_force_sfpu
_force_incumbent = ttnn._ttnn.operations.reduction.argmax_force_incumbent


SINGLE_CORE_GRID = None  # filled lazily (ttnn import-time objects inside fixtures)


def _single_core_grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _monotone(bits: np.ndarray) -> np.ndarray:
    """Monotone uint image of the bf16 sign-magnitude order (no NaNs input)."""
    bits = bits.astype(np.uint32)
    return np.where(bits >= 0x8000, (~bits) & 0xFFFF, bits | 0x8000).astype(np.uint32)


def _gasket_map(bits: np.ndarray) -> np.ndarray:
    """The SFPU pipeline's measured bf16 special-value gasket:
    NaN -> same-signed infinity, denormals and +/-0 -> +0."""
    bits = bits.astype(np.uint16)
    mag = (bits & 0x7FFF).astype(np.uint16)
    out = bits.copy()
    is_nan = mag > 0x7F80
    out[is_nan] = np.where((bits[is_nan] & 0x8000) != 0, 0xFF80, 0x7F80).astype(np.uint16)
    out[mag < 0x0080] = 0x0000  # denormals and both zeros flush to +0
    return out


def _gasket_argmax_row(bits_row: np.ndarray):
    """SFPU-engine semantics: IEEE compare on gasket-mapped values, smallest
    index on ties; the max-value output is the gasket-mapped winner."""
    g = _gasket_map(bits_row)
    m = _monotone(g)  # post-gasket there are no NaNs and only +0, so the
    i = int(np.argmax(m))  # monotone bit order IS the IEEE order
    return i, int(g[i])


_MONO_NEG_INF = 0x007F  # monotone(0xFF80): the incumbent argmax kernel's -inf init


def _ref_argmax_row(bits_row: np.ndarray):
    """Incumbent ttnn.argmax semantics: bfloat16_greater total order, smallest
    index on ties, -inf init (a row that never beats -inf reports (0, 0xFF80))."""
    m = _monotone(bits_row)
    if int(m.max()) <= _MONO_NEG_INF:
        return 0, 0xFF80
    i = int(np.argmax(m))
    return i, int(bits_row[i])


def _bits_of(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16)


def _make_case(name: str, v: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Row-major [b, v] bf16 bit patterns for one battery case (same bank as
    test_argmax_rvv.py)."""
    x = rng.standard_normal((b, v), dtype=np.float32) * 4.0
    bits = _bits_of(torch.from_numpy(x).bfloat16()).reshape(b, v)
    kmax, kdecoy = 0x7F7F, 0x7F7E  # largest finite bf16 + decoy the RNG cannot reach
    if name == "random":
        pass
    elif name == "unique_max":
        bits[:, 5 * v // 8] = kmax
        bits[:, v // 3] = kdecoy
    elif name == "tie_first_wins":
        bits[:, 5 * v // 8] = kmax
        bits[:, 7 * v // 8] = kmax
    elif name == "max_at_end":
        bits[:, v - 1] = kmax
    elif name == "max_at_zero":
        bits[:, 0] = kmax
    elif name == "denormal":
        small = rng.integers(0, 0x0080, size=(b, v), dtype=np.uint16)  # denormals and +/-0
        sign = (rng.integers(0, 2, size=(b, v), dtype=np.uint16) << 15).astype(np.uint16)
        bits = (small | sign).astype(np.uint16)
    elif name == "nan_bearing":
        bits[:, v // 4] = 0x7FC0  # +NaN: wins as +inf on the SFPU path
        bits[:, v // 2] = 0xFFC0  # -NaN: acts as -inf, never wins
    elif name == "all_negative":
        bits = (bits | 0x8000).astype(np.uint16)
        bits[bits == 0xFF80] = 0xBF80  # avoid accidental -inf
    elif name == "all_neginf":
        bits = np.full((b, v), 0xFF80, dtype=np.uint16)  # the -inf init corner
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

# Classes where the gasket order provably equals the incumbent bit order
# (finite normal data, ties included; all_neginf hits both paths' -inf rule).
INCUMBENT_EXACT_CASES = [c for c in CASES if c not in ("denormal", "nan_bearing")]


def _golden_row(name: str, bits_row: np.ndarray):
    """Score finite classes against the incumbent golden (stronger claim) and
    NaN/denormal classes against the gasket model."""
    if name in ("denormal", "nan_bearing"):
        return _gasket_argmax_row(bits_row)
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
def test_argmax_sfpu_battery(device, v, b, keepdim, with_maxval, grid):
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


@pytest.mark.parametrize("name", INCUMBENT_EXACT_CASES)
@pytest.mark.parametrize("v", [32, 2016, 4096])
@pytest.mark.parametrize("b", [1, 32])
def test_argmax_sfpu_matches_upstream_tile_path(device, name, v, b):
    """Index cross-check against the incumbent ttnn.argmax on the same TILE
    tensor for every class where the two orders provably agree (finite data
    plus the all--inf corner). The NaN/denormal classes are excluded by
    construction — that divergence is documented, measured, and asserted
    against the gasket model in the battery test above."""
    rng = np.random.default_rng(42 + v + 100 * b)
    bits = _make_case(name, v, b, rng)
    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    idx_sfpu = ttnn.to_torch(_force_sfpu(t_tile, dim=3, keepdim=True))
    idx_ref = ttnn.to_torch(_force_incumbent(t_tile, dim=3, keepdim=True))
    assert torch.equal(
        idx_sfpu.to(torch.int64), idx_ref.to(torch.int64)
    ), f"SFPU/incumbent index mismatch on case {name!r} (v={v}, b={b})"


@pytest.mark.parametrize("grid", ["single", "multi"])
def test_argmax_sfpu_multi_tile_row_batch(device, grid):
    """Rank-4 batch shape with several tile-row passes: outer dims 2 x 3,
    h = 80 rows (three tile-rows, the last one partial). Exercises the
    multi-pass credit flow-control and the padded-row masking; finite random
    data, so results must match the incumbent bit order exactly."""
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
# instead of routing a case the engine cannot serve. They are checked through
# the forced entry, which must refuse rather than fall back — a forced leg that
# quietly served another engine would make every comparison against it vacuous.


def test_argmax_sfpu_rejects_row_major_input(device, expect_error):
    """The SFPU engine is a TILE-layout engine; ROW_MAJOR input must be rejected."""
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "requires TILE layout input"):
        _force_sfpu(t_rm, dim=3, keepdim=True)


@pytest.mark.parametrize("removed_flag", ["use_rvv", "use_sfpu"])
def test_argmax_rejects_removed_engine_flags(device, removed_flag):
    """The engine is not caller-selectable: the old opt-in flags are gone from
    the public signature, so passing either is a TypeError rather than a knob
    that could be set to a contradictory pair."""
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(TypeError):  # allow-pytest.raises: host binding TypeError (unknown kwarg), not a device fault
        ttnn.argmax(t_tile, dim=3, keepdim=True, **{removed_flag: True})


def test_argmax_sfpu_rejects_ragged_last_dim(device, expect_error):
    """The reduction dim must be a multiple of the tile width (no W padding)."""
    x = torch.randn(1, 1, 32, 2000).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "multiple of the tile width"):
        _force_sfpu(t_tile, dim=3, keepdim=True)


def test_argmax_sfpu_rejects_sub_core_grid_outside_device(device, expect_error):
    """SFPU core placement must stay inside the device compute grid. This one is
    reachable through ttnn.argmax too — the grid check is engine-independent —
    so it is asserted on the public entry."""
    grid = device.compute_with_storage_grid_size()
    outside = ttnn.CoreCoord(grid.x, 0)
    invalid_grid = ttnn.CoreRangeSet([ttnn.CoreRange(outside, outside)])
    x = torch.randn(1, 1, 32, 2048).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must be contained in device compute grid"):
        ttnn.argmax(t_tile, dim=3, keepdim=True, sub_core_grids=invalid_grid)


def test_argmax_sfpu_runs_on_explicit_nonzero_core(device):
    """A valid explicit grid may place the single-core SFPU engine away from (0, 0)."""
    grid = device.compute_with_storage_grid_size()
    core = ttnn.CoreCoord(grid.x - 1, grid.y - 1)
    sub_core_grids = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    x = torch.randn(1, 1, 8, 2048).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(_force_sfpu(t_tile, dim=3, keepdim=True, sub_core_grids=sub_core_grids))
    expected = ttnn.to_torch(_force_incumbent(t_tile, dim=3, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))


# ---------------------------------------------------------------------------
# Automatic routing
# ---------------------------------------------------------------------------
# ttnn.argmax exposes no engine argument, so "which engine ran" is read off the
# program cache: warming the expected engine through its forced entry means a
# correctly routed ttnn.argmax hits that cached program and leaves the count
# alone, while a mis-route compiles a second program and grows it.


def _assert_empty_program_cache(device):
    """Precondition for the delta-0 proxy, checked BEFORE anything is warmed.

    "The auto call added no cache entry" only proves the route when the warmed
    engine is the ONLY argmax program cached for this shape. A stale entry for a
    different engine -- left by an earlier test sharing the device -- would
    absorb a mis-route as a cache hit and the assertion would pass vacuously.
    The `device` fixture is function-scoped (conftest.py), so each of these tests
    starts from an empty cache; assert that, so that marking this file
    `use_module_device` (a natural CI speed-up) fails loudly instead of silently
    gutting every routing test below."""
    msg = (
        "routing tests need a per-test device: the program cache is not empty at test start, so a "
        "mis-route could hit a stale entry for another engine and the delta-0 assertions below would "
        "prove nothing (do not mark this file use_module_device)"
    )
    assert device.num_program_cache_entries() == 0, msg


def _assert_program_cache_active(device):
    """The routing assertions read "which engine ran" off program-cache growth,
    so an empty (or disabled) cache would make them pass vacuously."""
    msg = "device program cache is empty after warming an engine; the routing assertions below would be vacuous"
    assert device.num_program_cache_entries() > 0, msg


# The routing boundary is kSfpuMinRows in argmax.cpp: H >= 32 goes to the SFPU,
# H < 32 goes to RVV. Both sides are asserted -- H = 32 here (just above) and
# H = 31 in test_argmax_rvv.py::test_argmax_auto_routes_to_rvv_below_the_sfpu_boundary
# (just below) -- so moving the constant in either direction fails a test.
SFPU_MIN_ROWS = 32


@pytest.mark.parametrize("h", [SFPU_MIN_ROWS, 64], ids=["at_boundary", "above_boundary"])
def test_argmax_auto_routes_to_sfpu_at_batch(device, h):
    """H >= kSfpuMinRows is the SFPU engine's shape: one lane-parallel pass
    covers all 32 rows of a tile-row, so its cost is flat in H where the RVV
    scan pays per row, and 32 is where every one of those 32 lanes is finally
    doing useful work. h = 32 is the just-at-the-boundary case: it must route
    to the SFPU, or kSfpuMinRows is not where argmax.cpp says it is.

    32, not 8: at H = 8 the multicore RVV engine is measured FASTER than the
    multicore SFPU at every core count up to 64 (SFPU/RVV 3.1x on one core,
    1.2x on 64). The older boundary of 8 came from a table that compared a
    single-core RVV against a multicore SFPU -- see argmax.cpp."""
    _assert_empty_program_cache(device)
    x = torch.randn(1, 1, h, 4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_sfpu(t_tile, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, f"auto did not route h={h} to the SFPU engine"


@pytest.mark.parametrize(
    "shape,layout",
    [
        # last dim not a multiple of the tile width
        ((1, 1, 32, 2000), ttnn.TILE_LAYOUT),
        # ROW_MAJOR input: the accelerated engines read TILE directly
        ((1, 1, 32, 2048), ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["ragged_last_dim", "row_major"],
)
def test_argmax_auto_falls_back_to_incumbent(device, shape, layout):
    """A case outside the accelerated engines' preconditions must demote to the
    scalar readers, not raise: automatic dispatch may never turn a call that
    used to work into an error."""
    _assert_empty_program_cache(device)
    x = torch.randn(*shape).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)

    expected = ttnn.to_torch(_force_incumbent(t, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t, dim=3, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, "auto did not fall back to the scalar readers"


def test_argmax_maxval_tensor_on_incumbent_route_raises(device, expect_error):
    """maxval_tensor can only be filled by the accelerated engines. A call the
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
    with expect_error(RuntimeError, "only produced by the accelerated engines"):
        ttnn.argmax(t_rm, dim=3, keepdim=True, maxval_tensor=mv)
