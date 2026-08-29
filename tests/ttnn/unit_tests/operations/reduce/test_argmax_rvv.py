# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for the RVV argmax engine: the Blackhole TILE-layout last-dim argmax
that runs the reduction on TRISC2's Zve32f vector unit.

ttnn.argmax picks an engine on its own and takes no argument that names one, so
the engine under test is pinned through the verification-only entry in the
private module (see ttnn/cpp/ttnn/operations/reduction/argmax/argmax_force.hpp).
The same entries supply the incumbent (scalar reader) golden, which a plain
ttnn.argmax over an eligible TILE bfloat16 last dim no longer runs on Blackhole.
Automatic routing is covered separately at the bottom of this file.

Gating: architecture only. The RVV kernels JIT-compile with the in-tree opt-in
(ComputeConfigDescriptor::enable_trisc2_rvv, which adds zve32f to the TRISC2
compile), so no special toolchain or environment setup is needed — these tests
run automatically on any Blackhole host and skip on every other architecture.
Compile-side coverage runs device-free in CI
(tests/tt_metal/tt_metal/jit_build/test_trisc2_rvv.cpp).
"""

import numpy as np
import pytest
import torch
import ttnn

from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole("the RVV argmax engine is Blackhole-only (TRISC2 Zve32f)")

_force_rvv = ttnn._ttnn.operations.reduction.argmax_force_rvv
_force_incumbent = ttnn._ttnn.operations.reduction.argmax_force_incumbent


def _monotone(bits: np.ndarray) -> np.ndarray:
    """Monotone uint image of the bfloat16_greater sign-magnitude total order."""
    bits = bits.astype(np.uint32)
    return np.where(bits >= 0x8000, (~bits) & 0xFFFF, bits | 0x8000).astype(np.uint32)


_MONO_NEG_INF = 0x007F  # monotone(0xFF80): the incumbent argmax kernel's -inf init


def _ref_argmax_row(bits_row: np.ndarray):
    """Incumbent ttnn.argmax semantics: bfloat16_greater total order, smallest
    index on ties, -inf init (a row that never beats -inf reports (0, 0xFF80))."""
    m = _monotone(bits_row)
    if int(m.max()) <= _MONO_NEG_INF:
        return 0, 0xFF80
    i = int(np.argmax(m))  # first occurrence == smallest-index tie-break
    return i, int(bits_row[i])


def _bits_of(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16)


def _make_case(name: str, v: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Row-major [b, v] bf16 bit patterns for one battery case."""
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
        bits[:, v // 4] = 0x7FC0  # +NaN sorts above +inf in the bit order
        bits[:, v // 2] = 0xFFC0  # -NaN payload sorts below -inf
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


# v boundaries: 32 = single tile (chunk_pages = min(64, w_tiles) = 1);
# 2016 = 63 tiles, exercising the w_tiles - tiles_done < chunk_pages remainder
# branch; 2048/8192 = exact multiples of the 64-tile chunk.
@pytest.mark.parametrize("v", [32, 2016, 2048, 8192])
@pytest.mark.parametrize("b", [1, 5, 32])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("with_maxval", [True, False])
def test_argmax_rvv_battery(device, v, b, keepdim, with_maxval):
    """Bit-exact index (and optional max-value bits) against the incumbent
    semantics across planted-max / tie / special-value cases."""
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

        idx_t = _force_rvv(t_tile, dim=3, keepdim=keepdim, maxval_tensor=mv)
        got_idx = ttnn.to_torch(idx_t).flatten().numpy().astype(np.uint32)
        got_val = _bits_of(ttnn.to_torch(mv).flatten()) if with_maxval else None

        for r in range(b):
            ref_idx, ref_val = _ref_argmax_row(bits[r])
            assert int(got_idx[r]) == ref_idx, f"case {name} row {r}: idx {int(got_idx[r])} != {ref_idx}"
            if with_maxval:
                assert int(got_val[r]) == ref_val, f"case {name} row {r}: val {int(got_val[r]):#06x} != {ref_val:#06x}"


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("v", [32, 2016, 4096])
@pytest.mark.parametrize("b", [1, 32])
def test_argmax_rvv_matches_upstream_tile_path(device, name, v, b):
    """Index cross-check against the incumbent ttnn.argmax on the same TILE
    tensor — not just random data: every planted-max / tie / special-value
    case from CASES runs through both paths."""
    rng = np.random.default_rng(42 + v + 100 * b)
    bits = _make_case(name, v, b, rng)
    x = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).reshape(1, 1, b, v)
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    idx_rvv = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True))
    idx_ref = ttnn.to_torch(_force_incumbent(t_tile, dim=3, keepdim=True))
    assert torch.equal(
        idx_rvv.to(torch.int64), idx_ref.to(torch.int64)
    ), f"RVV/incumbent index mismatch on case {name!r} (v={v}, b={b})"


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("v", [32, 2016, 4096])
def test_argmax_rank1_with_maxval_through_public_entry(device, name, v):
    """A rank-1 [v] TILE bfloat16 last-dim reduction through the PUBLIC entry,
    carrying a maxval_tensor.

    Rank 1 has no second-to-last dim, so it is the H == 1 shape by
    construction and must reach the RVV engine: the scalar readers cannot fill
    a max-value output, so routing rank 1 to them would make this exact call --
    which the engine served before selection moved in-tree -- raise instead.
    Both outputs are checked against the host bit-level reference, not just the
    index, because the max value is the half only an accelerated engine can
    produce."""
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


def test_argmax_rvv_rejects_row_major_input(device, expect_error):
    """The RVV engine is a TILE-layout engine; ROW_MAJOR input must be rejected.
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


def test_argmax_auto_routes_to_rvv_at_h1(device):
    """H == 1 is the RVV engine's shape: the SFPU alternative would pay for all
    32 lanes to serve a single valid row per tile-row."""
    _assert_empty_program_cache(device)
    x = torch.randn(1, 1, 1, 4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, "auto did not route H == 1 to the RVV engine"


def test_argmax_auto_routes_rank1_to_rvv(device):
    """A rank-1 input has no H dim at all; it counts as H == 1 and must land on
    the RVV engine, never on the SFPU (which is the measured loser at H == 1)
    and never on the scalar readers (which cannot fill a maxval_tensor)."""
    _assert_empty_program_cache(device)
    x = torch.randn(4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_rvv(t_tile, dim=-1, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=-1, keepdim=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, "auto did not route a rank-1 input to the RVV engine"


@pytest.mark.parametrize("h", [1, 5, 32])
def test_argmax_exact_special_values_pins_rvv(device, h):
    """exact_special_values excludes the SFPU engine (its special-value gasket
    diverges), so an eligible call lands on RVV at every H — including the
    H >= 8 shapes the default would send to the SFPU."""
    _assert_empty_program_cache(device)
    x = torch.randn(1, 1, h, 4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()
    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True, exact_special_values=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    msg = f"auto did not route exact_special_values=True (h={h}) to the RVV engine"
    assert device.num_program_cache_entries() == entries_before, msg


def test_argmax_exact_special_values_changes_the_route(device):
    """The flag has to MOVE the decision, not merely be accepted: at H = 32 the
    default routes to the SFPU engine, and asking for exact special values
    moves the very same call to RVV."""
    _assert_empty_program_cache(device)
    x = torch.randn(1, 1, 32, 4096).bfloat16()
    t_tile = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.to_torch(_force_rvv(t_tile, dim=3, keepdim=True))
    _assert_program_cache_active(device)
    entries_before = device.num_program_cache_entries()

    got = ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True, exact_special_values=True))
    assert torch.equal(got.to(torch.int64), expected.to(torch.int64))
    assert device.num_program_cache_entries() == entries_before, "exact_special_values did not pin the RVV engine"

    # Same tensor, same dim, flag dropped: this must NOT reuse the RVV program.
    ttnn.to_torch(ttnn.argmax(t_tile, dim=3, keepdim=True))
    msg = "the default at H = 32 reused the RVV program; exact_special_values would then be a no-op"
    assert device.num_program_cache_entries() > entries_before, msg
