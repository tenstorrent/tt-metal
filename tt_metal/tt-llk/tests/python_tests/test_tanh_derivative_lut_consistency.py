# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The legacy tanh-derivative SFPLUT table is written down in three places.

The kernel evaluates `1 - lut(x)^2`, so the coefficient table in
`tanh_derivative_init` *is* its whole approximation. Which kernel is worth being
exact about: not the `calculate_tanh_derivative` sitting next to that init, which
has no callers, but tt-llk's `_calculate_tanh_derivative_`, which the harness pairs
with this init under `SfpuType::tanh_derivative_lut`. The table crosses between them
in LReg0/1/2, so the init lives in one repository and the kernel in another.

`UnarySFPUGolden._tanh_derivative_lut` then models that table by hand, deliberately,
because validating the kernel against an accurate `sech^2` would fail by design. That
leaves three copies that must agree: the Wormhole header, the Blackhole header,
and the Python golden. Nothing in the build couples them, and a retune that
misses one is silent -- the golden would simply describe a kernel that no longer
exists and the comparison would still "pass".

These tests are the coupling. No hardware needed: they read the two headers and
exercise the golden.

`tanh_init` is deliberately NOT checked against these. Both now use the 6-entry
SFPLUTFP32 table, but with different coefficients and for different reasons: that
one is fitted for `tanh`, this one for `sech^2` through `1 - lut^2`. Substituting
either for the other is a regression, so the two must be free to move apart. See
the comment in `tanh_derivative_init`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator

# tests/python_tests/ -> tests/ -> tt-llk/ -> tt_metal/
_TT_METAL = Path(__file__).resolve().parents[3]
_HEADER = "metal/llk_api/llk_sfpu/ckernel_sfpu_tanh_derivative.h"
ARCH_HEADERS = {
    "wormhole_b0": _TT_METAL / "hw/ckernels/wormhole_b0" / _HEADER,
    "blackhole": _TT_METAL / "hw/ckernels/blackhole" / _HEADER,
}

# sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.93701171875f, 0.5869140625f);
# sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(0.0f, 0.183837890625f);
_NUM = r"(-?[0-9.]+(?:[eE][-+]?\d+)?)f"
_LUT16SS = re.compile(rf"vLut16ss\(\s*{_NUM}\s*,\s*{_NUM}\s*\)")
_LUT16II = re.compile(rf"vLut16ii\(\s*{_NUM}\s*,\s*{_NUM}\s*\)")

# TABLE1 upper bounds; the sixth segment runs to infinity.
BREAKPOINTS = [0.5, 1.0, 1.5, 2.0, 3.0]


def _table(path: Path) -> list[tuple[float, float]]:
    """The six (slope, intercept) pairs `tanh_derivative_init` loads.

    Slopes live in LReg0/1/2 and intercepts in LReg4/5/6, each register packing two
    consecutive segments hi/lo, so the pairs interleave rather than reading straight down.
    """
    assert path.is_file(), f"missing kernel header: {path}"
    body = path.read_text()
    start = body.index("inline void tanh_derivative_init()")
    body = body[start:]
    slopes = [float(v) for pair in _LUT16SS.findall(body) for v in pair]
    intercepts = [float(v) for pair in _LUT16II.findall(body) for v in pair]
    assert len(slopes) == 6 and len(intercepts) == 6, (
        f"expected 3 vLut16ss and 3 vLut16ii pairs in {path.name}, "
        f"found {len(slopes)} slopes and {len(intercepts)} intercepts"
    )
    return list(zip(slopes, intercepts))


def _model(pairs: list[tuple[float, float]], x: float) -> float:
    """`1 - lut(x)^2` with SGN_RETAIN, straight from the parsed coefficients."""
    a = abs(x)
    seg = next((i for i, bp in enumerate(BREAKPOINTS) if a < bp), 5)
    slope, intercept = pairs[seg]
    return 1.0 - (slope * a + intercept) ** 2


def test_arch_tables_match():
    """Wormhole and Blackhole must load the same table."""
    wh = _table(ARCH_HEADERS["wormhole_b0"])
    bh = _table(ARCH_HEADERS["blackhole"])
    assert wh == bh, (
        "tanh_derivative_init tables have diverged between architectures:\n"
        f"  wormhole_b0 {wh}\n  blackhole   {bh}"
    )


def test_golden_matches_kernel_table():
    """The hand-written golden must model exactly the table the kernel loads."""
    pairs = _table(ARCH_HEADERS["wormhole_b0"])
    golden = get_golden_generator(UnarySFPUGolden)

    # Both sides of every breakpoint, plus the saturated tail and the origin.
    probes = [0.0, 0.25, 0.4, 0.49, 0.75, 1.25, 1.75, 2.5, 2.99, 4.0, 8.0]
    for bp in BREAKPOINTS:
        probes += [bp - 0.01, bp, bp + 0.01]
    probes += [-p for p in probes if p != 0.0]

    for x in probes:
        expected = _model(pairs, x)
        actual = golden._tanh_derivative_lut(x)
        assert actual == pytest.approx(expected, abs=1e-12), (
            f"golden disagrees with the kernel table at x={x}: "
            f"golden={actual!r}, table gives {expected!r}. "
            "Update UnarySFPUGolden._tanh_derivative_lut and tanh_derivative_init together."
        )


def test_table_is_odd_and_saturating():
    """Structural properties the golden and both kernels rely on."""
    pairs = _table(ARCH_HEADERS["wormhole_b0"])
    slope0, intercept0 = pairs[0]
    assert intercept0 == 0.0, (
        "segment 0 must have a zero intercept, so tanh'(0) is exactly 1: the kernel "
        "returns 1 - lut(0)^2, and a nonzero B makes that 1 - B^2"
    )
    assert pairs[5] == (0.0, 1.0), (
        "the last segment must be the exact constant 1.0 -- it is what makes 1 - lut^2 "
        "collapse to exactly 0 past |x| = 3, and what bounds the kernel at all: any "
        "nonzero slope there sends 1 - lut^2 to -inf as |x| grows"
    )
    assert 0.0 < slope0 <= 1.0


def test_table_is_monotone_and_in_range():
    """`1 - lut^2` must be monotone decreasing in |x| and never leave [0, 1].

    sech^2 is strictly decreasing and positive, so a table that steps the lut *down* at a
    breakpoint makes the derivative rise with |x|, and one that reaches lut > 1 makes it
    negative. Neither is visible in a max-error figure; both are visible here.
    """
    pairs = _table(ARCH_HEADERS["wormhole_b0"])

    # The lut must not step down where the segments meet.
    for bp, (a_lo, b_lo), (a_hi, b_hi) in zip(BREAKPOINTS, pairs[:-1], pairs[1:]):
        step = (a_hi * bp + b_hi) - (a_lo * bp + b_lo)
        assert step >= 0.0, (
            f"the lut steps down by {-step:.6g} at |x| = {bp}, so 1 - lut^2 rises there "
            "and the derivative stops being monotone in |x|"
        )

    xs = [i / 512.0 for i in range(0, 512 * 5)]
    ys = [_model(pairs, x) for x in xs]
    for x, y in zip(xs, ys):
        assert 0.0 <= y <= 1.0, f"1 - lut^2 = {y!r} at x={x}, outside [0, 1]"
    for (x_a, y_a), (x_b, y_b) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        assert y_b <= y_a + 1e-12, (
            f"1 - lut^2 rises from {y_a!r} at x={x_a} to {y_b!r} at x={x_b}"
        )
