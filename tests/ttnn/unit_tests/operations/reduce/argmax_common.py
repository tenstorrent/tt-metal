# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the two Blackhole vector argmax suites, test_argmax_rvv.py
and test_argmax_sfpu.py: the bf16 bit-pattern case bank, the scalar-reader host
reference both suites score against, and the program-cache preconditions the
automatic-routing tests read their answers off.

Only what is identical to both suites lives here. Path-specific goldens stay in
their own suite -- the SFPU path's special-value normalisation model is in
test_argmax_sfpu.py, because no other path applies it.

This is a plain helper module, not a conftest.py: it holds no fixtures, and the
neighbouring test directories share plain functions and constants this way
(tests/ttnn/unit_tests/operations/eltwise/eltwise_test_utils.py,
tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py). The name does not
match pytest's test_*.py pattern, so it is never collected as a test module.
"""

import numpy as np
import torch
import ttnn


# bf16 bit patterns the cases below plant deliberately.
BF16_MAX_FINITE = 0x7F7F  # largest finite bf16: exponent 0xFE with every mantissa bit set
BF16_NEG_INF = 0xFF80  # -inf, which is also the scalar-reader kernel's accumulator init
# The two NaN constants are read differently by the two paths, and both readings are
# asserted: the bfloat16_greater bit order sorts +NaN above +inf and -NaN below -inf, while
# the SFPU path normalises each to the same-signed infinity. Under either one a +NaN can win
# a row and a -NaN never can.
BF16_POS_NAN = 0x7FC0  # a quiet +NaN
BF16_NEG_NAN = 0xFFC0  # a quiet -NaN
BF16_SIGN_BIT = 0x8000  # the sign bit alone is -0.0; OR-ing it makes any value negative


def _monotone(bits: np.ndarray) -> np.ndarray:
    """Monotone uint image of the bfloat16_greater sign-magnitude total order.

    Every bf16 bit pattern is ordered, NaNs included. A caller that has already
    normalised the NaNs away (the SFPU path's model) is therefore reading the
    IEEE order out of the same mapping."""
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


# The routing boundary is kSfpuMinRows in argmax.cpp: H < 32 goes to RVV, H >= 32 goes to
# the SFPU. Both sides are asserted -- H = 31 in
# test_argmax_rvv.py::test_argmax_auto_routes_to_rvv_below_the_sfpu_boundary (just below)
# and H = 32 in test_argmax_sfpu.py::test_argmax_auto_routes_to_sfpu_at_batch (just at) --
# so moving the constant in either direction fails a test.
SFPU_MIN_ROWS = 32
