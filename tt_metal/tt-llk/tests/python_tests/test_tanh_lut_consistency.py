# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The two tanh SFPLUT tables, held against their copies and their own contracts.

`tanh_init` and `tanh_derivative_init` each load a 6-entry SFPLUTFP32 table, and each
table *is* the whole of its kernel's approximation -- there is no polynomial behind
it to fall back on. Both are written down more than once with nothing in the build
coupling the copies:

  - `tanh_init` is duplicated verbatim in the Wormhole and Blackhole headers.
  - `tanh_derivative_init` is duplicated the same way, and then modelled a third time
    by hand in `UnarySFPUGolden._tanh_derivative_lut`. That golden is deliberate:
    the kernel evaluates `1 - lut(x)^2`, so validating it against an accurate `sech^2`
    would fail by design. A retune that misses the golden is silent -- the golden
    would simply describe a kernel that no longer exists and the comparison would
    still "pass".

Both tables also cross a repository boundary. The coefficients reach the kernel in
LReg0/1/2 (slopes) and LReg4/5/6 (intercepts), so the init lives in tt-metal and the
kernel in tt-llk, coupled by nothing but that register convention -- which is
therefore as much a part of the contract as the numbers are.

These tests are the coupling. No hardware needed: they read the headers and exercise
the golden.

The two tables are NOT checked against each other. They have the same shape and the
same TABLE1 breakpoints, but different coefficients for different reasons: one is
fitted for `tanh`, the other for `sech^2` through `1 - lut^2`. Substituting either
for the other is a regression, so the two must be free to move apart. See the
comment in `tanh_derivative_init`.

What *is* shared is the list of invariants, and those are checked per table: segment
0's intercept, the saturating tail, the register placement, and the two fp32-window
properties -- no segment above 1.0, no downward step at a join -- that an exhaustive
bfloat16 sweep cannot see, because a step of 1.2e-4 is far below the ~2e-3 ulp of a
bfloat16 out there while being ~2048 fp32 ulp of non-monotonicity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator

# tests/python_tests/ -> tests/ -> tt-llk/ -> tt_metal/
# Only valid when tt-llk is vendored inside a tt-metal tree; a standalone tt-llk
# checkout has no such parent, and `_read_tables` skips rather than fails there.
_TT_METAL = Path(__file__).resolve().parents[3]
ARCHES = ("wormhole_b0", "blackhole")


@dataclass(frozen=True)
class LutTable:
    """One SFPLUT table: the init that loads it, and the header it is written in."""

    init: str
    header: str

    def path(self, arch: str) -> Path:
        return _TT_METAL / "hw/ckernels" / arch / "metal/llk_api/llk_sfpu" / self.header


TANH = LutTable("tanh_init", "ckernel_sfpu_tanh.h")
TANH_DERIVATIVE = LutTable("tanh_derivative_init", "ckernel_sfpu_tanh_derivative.h")
TABLES = [TANH, TANH_DERIVATIVE]

# sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.93701171875f, 0.5869140625f);
# sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(0.0f, 0.183837890625f);
#
# The LReg index is captured, not just the literals. Parsing by textual order alone
# would let a retune relabel which register two adjacent assignments target -- loading
# the right numbers into the wrong segments -- with every test below still passing.
_NUM = r"(-?[0-9.]+(?:[eE][-+]?\d+)?)f"
_ASSIGN = re.compile(
    rf"l_reg\[\s*(?:sfpi::)?LRegs::LReg(\d)\s*\]\s*=\s*sfpi::vLut16(ss|ii)"
    rf"\(\s*{_NUM}\s*,\s*{_NUM}\s*\)"
)

# Each vLut16 register packs two consecutive segments, so the register a segment's
# coefficients must live in is fixed: slopes in LReg0/1/2, intercepts in LReg4/5/6,
# each in segment order.
SLOPE_REGS = [0, 1, 2]
INTERCEPT_REGS = [4, 5, 6]

# TABLE1 upper bounds; the sixth segment runs to infinity.
BREAKPOINTS = [0.5, 1.0, 1.5, 2.0, 3.0]

# Segment i covers [LOWER[i], UPPER[i]).
LOWER = [0.0] + BREAKPOINTS
UPPER = BREAKPOINTS + [float("inf")]


def _fp16(value: float) -> float:
    """The value the hardware actually holds: vLut16 coefficients are IEEE halves.

    `ckernel_sfpu_tanh.h` writes its literals truncated to the digits that identify
    the half (0.96191406f is 0.9619140625), so arithmetic on the parsed decimals
    misses the joins by ~1e-8 and "the segments meet exactly" stops being testable.
    Rounding first restores it. `test_coefficients_are_halves` is what makes this
    safe to do -- it pins that the truncation is all that is being undone.
    """
    return float(np.float16(value))


def _init_body(source: str, init: str) -> str:
    """The brace-delimited body of `init`, by brace matching.

    Slicing to end of file instead would be inert today but would quietly start
    reading whatever LUT init is added below it.
    """
    match = re.search(rf"\binline void {re.escape(init)}\s*\([^)]*\)\s*\{{", source)
    assert match, f"could not find 'inline void {init}(...)' -- renamed?"

    depth = 0
    start = source.index("{", match.start())
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise AssertionError(f"unbalanced braces in {init}")


def _table(path: Path, init: str) -> list[tuple[float, float]]:
    """The six (slope, intercept) pairs `init` loads, as written in the header.

    Slopes live in LReg0/1/2 and intercepts in LReg4/5/6, each register packing two
    consecutive segments hi/lo, so the pairs interleave rather than reading straight
    down. Both the values and the register each pair lands in are checked.
    """
    body = _init_body(path.read_text(), init)
    found = _ASSIGN.findall(body)

    slopes: list[float] = []
    intercepts: list[float] = []
    slope_regs: list[int] = []
    intercept_regs: list[int] = []
    for reg, kind, first, second in found:
        target, regs = (
            (slopes, slope_regs) if kind == "ss" else (intercepts, intercept_regs)
        )
        target += [float(first), float(second)]
        regs.append(int(reg))

    assert len(slopes) == 6 and len(intercepts) == 6, (
        f"expected 3 vLut16ss and 3 vLut16ii pairs in {init} ({path.name}), "
        f"found {len(slopes)} slopes and {len(intercepts)} intercepts"
    )
    assert slope_regs == SLOPE_REGS, (
        f"{init} ({path.name}): slopes must be loaded into LReg{SLOPE_REGS} in segment "
        f"order, found LReg{slope_regs}. The kernel reads these registers by index, so "
        "the right coefficients in the wrong register is the wrong table."
    )
    assert intercept_regs == INTERCEPT_REGS, (
        f"{init} ({path.name}): intercepts must be loaded into LReg{INTERCEPT_REGS} in "
        f"segment order, found LReg{intercept_regs}. The kernel reads these registers by "
        "index, so the right coefficients in the wrong register is the wrong table."
    )
    return list(zip(slopes, intercepts))


def _read_tables(table: LutTable) -> dict[str, list[tuple[float, float]]]:
    """Parse both arch headers, or skip if this is not a tt-metal tree."""
    missing = [arch for arch in ARCHES if not table.path(arch).is_file()]
    if missing:
        pytest.skip(
            f"kernel headers not present ({', '.join(missing)}) -- expected in a "
            "standalone tt-llk checkout, where there is nothing to guard"
        )
    return {arch: _table(table.path(arch), table.init) for arch in ARCHES}


def _hw_table(table: LutTable) -> list[tuple[float, float]]:
    """The Wormhole table as the SFPLUT holds it: every coefficient rounded to fp16."""
    return [(_fp16(a), _fp16(b)) for a, b in _read_tables(table)["wormhole_b0"]]


def _lut(pairs: list[tuple[float, float]], x: float) -> float:
    """`sign(x) * (A*|x| + B)`, the SGN_RETAIN SFPLUT, from the parsed coefficients."""
    a = abs(x)
    seg = next((i for i, bp in enumerate(BREAKPOINTS) if a < bp), 5)
    slope, intercept = pairs[seg]
    value = slope * a + intercept
    return -value if x < 0 else value


def _model(pairs: list[tuple[float, float]], x: float) -> float:
    """`1 - lut(x)^2`, the tanh-derivative kernel, from the parsed coefficients."""
    return 1.0 - _lut(pairs, x) ** 2


parametrize_tables = pytest.mark.parametrize(
    "table", TABLES, ids=[t.init for t in TABLES]
)


@parametrize_tables
def test_arch_tables_match(table: LutTable):
    """Wormhole and Blackhole must load the same table."""
    tables = _read_tables(table)
    wh, bh = tables["wormhole_b0"], tables["blackhole"]
    assert wh == bh, (
        f"{table.init} tables have diverged between architectures:\n"
        f"  wormhole_b0 {wh}\n  blackhole   {bh}"
    )


@parametrize_tables
def test_registers_are_in_segment_order(table: LutTable):
    """Slopes in LReg0/1/2, intercepts in LReg4/5/6, both in segment order.

    The assertions live in `_table`, so every test here depends on them; this one
    exists so the failure has a name of its own. It is the relabel failure mode:
    swap two adjacent vLut16 assignments and the numbers are all still present, in
    the wrong segments, silently.
    """
    _read_tables(table)


@parametrize_tables
def test_coefficients_are_halves(table: LutTable):
    """Every literal must be the IEEE half the SFPLUT will hold.

    vLut16 coefficients are fp16. A constant copied straight out of a fitter at
    double precision is silently rounded by the hardware, so the table the reader
    sees is not the table that runs. Written to the digits that identify the half,
    the two agree to ~1e-9; anything that misses by more is a value the header is
    claiming and the hardware is not using.
    """
    for arch, pairs in _read_tables(table).items():
        for seg, (slope, intercept) in enumerate(pairs):
            for name, value in (("slope", slope), ("intercept", intercept)):
                assert abs(value - _fp16(value)) <= 1e-7, (
                    f"{table.init} ({arch}) segment {seg}: {name} {value!r} is not an "
                    f"fp16 value -- the SFPLUT will hold {_fp16(value)!r} instead"
                )


@parametrize_tables
def test_segment_zero_has_a_zero_intercept(table: LutTable):
    """Segment 0's intercept must be exactly 0."""
    _, intercept0 = _hw_table(table)[0]
    assert intercept0 == 0.0, (
        f"{table.init}: segment 0's intercept is {intercept0!r}, not 0. The LUT runs "
        "under SGN_RETAIN and evaluates sign(x) * (A*|x| + B), so a nonzero B is a "
        "jump across the origin -- directly in tanh, and as 1 - B^2 rather than an "
        "exact 1 in the derivative."
    )


@parametrize_tables
def test_segment_zero_slope_is_in_range(table: LutTable):
    """Segment 0's slope alone sets the shape at the origin."""
    slope0, _ = _hw_table(table)[0]
    assert 0.0 < slope0 <= 1.0, (
        f"{table.init}: segment 0's slope is {slope0!r}, outside (0, 1]. With a zero "
        "intercept it is the only thing setting the near-origin shape: <= 0 flattens "
        "or inverts it, and > 1 drives the lut past 1 immediately -- past tanh's own "
        "bound, and negative once it is squared into 1 - lut^2."
    )


@parametrize_tables
def test_tail_is_the_exact_constant_one(table: LutTable):
    """The sixth segment must be exactly (0.0, 1.0)."""
    tail = _hw_table(table)[5]
    assert tail == (0.0, 1.0), (
        f"{table.init}: the last segment is {tail!r}, not (0.0, 1.0). It is what makes "
        "*finite* |x| past 3 saturate -- to exactly 1.0 for tanh, to exactly 0 for "
        "1 - lut^2 -- and it is what bounds the kernel at all: any nonzero slope there "
        "grows without limit as |x| does, sending 1 - lut^2 to -inf. It does not cover "
        "the infinities, where A*inf + B is 0 * inf + 1 = NaN; see test_tanh_specials."
    )


@parametrize_tables
def test_lut_never_exceeds_one(table: LutTable):
    """No segment may reach lut > 1 anywhere in its own range.

    tanh is bounded by 1 and `1 - lut^2` goes negative above it, and neither kernel
    has a `min(result, 1.0f)` to fall back on -- the polynomial tanh path has one
    precisely because it needs it. With a non-negative slope the supremum over a
    half-open segment is at its upper breakpoint, so that is the only point to check.

    A bfloat16 sweep can miss this entirely: an overshoot that peaks at 1.00018
    inside (2.99532, 3.0) has no bfloat16 input landing in the window, but an fp32
    input or an fp32 DEST value in it returns |tanh| > 1.
    """
    pairs = _hw_table(table)
    for seg, ((slope, intercept), upper) in enumerate(zip(pairs, UPPER)):
        assert slope >= 0.0, (
            f"{table.init}: segment {seg} has slope {slope!r} < 0, so the lut falls "
            "inside the segment and the kernel is not monotone in |x|"
        )
        peak = intercept if upper == float("inf") else slope * upper + intercept
        assert peak <= 1.0, (
            f"{table.init}: segment {seg} reaches {peak!r} at |x| = {upper}, above 1. "
            "tanh is bounded by 1 and 1 - lut^2 goes negative above it, and neither "
            "kernel clamps."
        )


@parametrize_tables
def test_lut_does_not_step_down_at_a_join(table: LutTable):
    """The segments must not step the lut down where they meet.

    tanh and its lut are increasing in |x|, and `1 - lut^2` is decreasing; a downward
    step breaks both. It is invisible in a max-error figure and invisible to a
    bfloat16 sweep -- 1.2e-4 is far below the ~2e-3 bfloat16 ulp out there -- but
    neither kernel has a `convert<vFloat16b>`, and both serve the fp32-dest path,
    where that same step is ~2048 fp32 ulp of non-monotonicity.
    """
    pairs = _hw_table(table)
    for bp, (a_lo, b_lo), (a_hi, b_hi) in zip(BREAKPOINTS, pairs[:-1], pairs[1:]):
        step = (a_hi * bp + b_hi) - (a_lo * bp + b_lo)
        assert step >= 0.0, (
            f"{table.init}: the lut steps down by {-step:.6g} at |x| = {bp}, so tanh "
            "falls there and 1 - lut^2 rises"
        )


def test_golden_matches_kernel_table():
    """The hand-written golden must model exactly the table the kernel loads.

    Literal against literal, not rounded to fp16: the point is that the two sources
    agree as written, so a retune cannot update one and leave the other behind.
    """
    pairs = _read_tables(TANH_DERIVATIVE)["wormhole_b0"]
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


def test_derivative_is_monotone_and_in_range():
    """`1 - lut^2` must be monotone decreasing in |x| and never leave [0, 1].

    The per-segment bounds above are on the lut; this is the same properties carried
    through the square, over a dense sweep, which is what the derivative kernel
    actually returns.
    """
    pairs = _hw_table(TANH_DERIVATIVE)

    xs = [i / 512.0 for i in range(0, 512 * 5)]
    ys = [_model(pairs, x) for x in xs]
    for x, y in zip(xs, ys):
        assert 0.0 <= y <= 1.0, f"1 - lut^2 = {y!r} at x={x}, outside [0, 1]"
    for (x_a, y_a), (x_b, y_b) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        assert (
            y_b <= y_a + 1e-12
        ), f"1 - lut^2 rises from {y_a!r} at x={x_a} to {y_b!r} at x={x_b}"


def test_tanh_is_monotone_and_bounded():
    """The tanh lut must be odd, non-decreasing, and inside [-1, 1].

    The dense counterpart to the per-segment bounds, on the kernel's own output:
    `calculate_tanh<APPROXIMATION_MODE=true>` returns the lut unmodified.
    """
    pairs = _hw_table(TANH)

    xs = [i / 512.0 for i in range(0, 512 * 5)]
    ys = [_lut(pairs, x) for x in xs]
    for x, y in zip(xs, ys):
        assert -1.0 <= y <= 1.0, f"tanh lut = {y!r} at x={x}, outside [-1, 1]"
        assert _lut(pairs, -x) == -y, f"tanh lut is not odd at x={x}"
    for (x_a, y_a), (x_b, y_b) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        assert (
            y_b >= y_a
        ), f"tanh lut falls from {y_a!r} at x={x_a} to {y_b!r} at x={x_b}"
