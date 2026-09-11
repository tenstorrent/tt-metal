# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The legacy tanh-derivative SFPLUT table is written down in three places.

`calculate_tanh_derivative` computes `1 - lut(x)^2`, so the coefficient table in
`tanh_derivative_init` *is* that kernel's approximation -- and
`UnarySFPUGolden._tanh_derivative_lut` models it by hand, deliberately, because
validating the kernel against an accurate `sech^2` would fail by design. That
leaves three copies that must agree: the Wormhole header, the Blackhole header,
and the Python golden. Nothing in the build couples them, and a retune that
misses one is silent -- the golden would simply describe a kernel that no longer
exists and the comparison would still "pass".

These tests are the coupling. No hardware needed: they read the two headers and
exercise the golden.

`tanh_init` is deliberately NOT checked against these: it moved to the 6-entry
SFPLUTFP32 table to cut ULP error while this deprecated kernel kept its 3-entry
one, because the objective here is `sech^2` rather than `tanh`. See the comment
in `tanh_derivative_init`.
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

# sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut8si(0.8125f, 0.0f);
_LUT8SI = re.compile(
    r"vLut8si\(\s*(-?[0-9.]+(?:[eE][-+]?\d+)?)f\s*,\s*"
    r"(-?[0-9.]+(?:[eE][-+]?\d+)?)f\s*\)"
)


def _table(path: Path) -> list[tuple[float, float]]:
    """The three (slope, intercept) pairs `tanh_derivative_init` loads."""
    assert path.is_file(), f"missing kernel header: {path}"
    body = path.read_text()
    start = body.index("inline void tanh_derivative_init()")
    pairs = [(float(a), float(b)) for a, b in _LUT8SI.findall(body[start:])]
    assert (
        len(pairs) == 3
    ), f"expected 3 vLut8si pairs in {path.name}, found {len(pairs)}"
    return pairs


def _model(pairs: list[tuple[float, float]], x: float) -> float:
    """`1 - lut(x)^2` with SGN_RETAIN, straight from the parsed coefficients."""
    a = abs(x)
    seg = 0 if a < 1.0 else (1 if a < 2.0 else 2)
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
    probes = [
        0.0,
        0.25,
        0.5,
        0.75,
        0.99,
        1.0,
        1.01,
        1.5,
        1.99,
        2.0,
        2.01,
        2.5,
        3.0,
        8.0,
    ]
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
        "segment 0 must have a zero intercept: SGN_RETAIN computes "
        "sign(x)*(A*|x| + B), so a nonzero B puts a jump across the origin"
    )
    assert pairs[2] == (0.0, 1.0), (
        "segment 2 must be the exact constant 1.0 -- tanh saturates there, and "
        "this is what makes 1 - lut^2 collapse to 0 past |x| = 2"
    )
    assert 0.0 < slope0 <= 1.0
