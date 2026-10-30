# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The theoretical ceiling is SPEC bandwidth over bytes. Nothing is folded into it.

    ceiling = peak_bw / bytes_per_token
    band    = (0.75 * f) * ceiling  ..  f * ceiling      f = 0.80 dense | 0.50 MoE

The sustained fraction used to be multiplied INTO the ceiling, so the number labelled "theoretical"
was already a sustained figure -- and a run could pass it. gemma3 did: 28.7 tok/s/u reported as 84%
of a 34.1 "ceiling", sitting ABOVE its own achievable band of 20.5-27.3, which reads as "done" when
the wall is 512/12 = 42.7 and 28.7 is 67% of it with real headroom left.

To emit one token the chip must read every weight from DRAM and cannot use a weight it has not
fetched, so bytes/bandwidth is a physical bound no software crosses. A ceiling a measurement can beat
is not a ceiling. The fraction has not been deleted -- it now sets the band's TOP, which is exactly
where "80% of spec is achievable on a dense stream" belongs.

Targets are unchanged: the band's top equals the ceiling the tool used to print.

  c1  the ceiling is spec / bytes, for dense and MoE
  c2  the band top lands on the OLD ceiling -- no target moved
  c3  no measurement can exceed the ceiling (the property that makes it a ceiling)
  c4  TP divides the bytes, not the bandwidth
  c5  degenerate inputs stay degenerate
"""

import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

from agent import perf_target as pt  # noqa: E402

PEAK = 512e9


def _facts(params_b, moe=False, dtype="int8"):
    """A model must DECLARE its width now; the ceiling no longer assumes one byte per parameter.

    A ONE-BYTE width is declared, so every figure in this file stays exactly as written: these tests
    are about the SHAPE of the ceiling -- peak / bytes, nothing folded in -- and a 1 B/param model
    makes that arithmetic checkable by eye (512 / 8 GB = 64.0).

    What changed is that the width is now stated by the model rather than assumed for it. The old
    rule applied 1.0 to everything, which is right only for a 1-byte format: voxtral is served bf16
    and streams 2 B/param, so it was handed a ceiling ABOVE what the hardware permits (141.8 tok/s/u
    against a true ~55). See test_the_ceiling_uses_a_measured_width_not_one_byte.
    """
    f = {"total_params": params_b * 1e9, "dominant_dtype": dtype}
    if moe:
        f.update(is_moe=True, active_params=params_b * 1e9)
    return f


def _ceiling_band(params_b, moe=False, tp=1):
    f = _facts(params_b, moe)
    return pt.rate_and_band(pt.simple_active_bytes(f), PEAK, frac=pt.bw_fraction(f), tp_degree=tp)


# --------------------------------------------------------------------------- c1 SPEC
@pytest.mark.parametrize("params_b,expect", [(8, 64.0), (12, 42.7), (27, 19.0), (70, 7.3)])
def test_c1_dense_ceiling_is_peak_over_bytes(params_b, expect):
    c, _ = _ceiling_band(params_b)
    assert c == pytest.approx(512 / params_b, rel=1e-6)
    assert c == pytest.approx(expect, abs=0.1)


@pytest.mark.parametrize("active_b,expect", [(3, 170.7), (13, 39.4)])
def test_c1_moe_ceiling_is_also_spec(active_b, expect):
    """The 0.50 is about what MoE SUSTAINS, not what the bus permits -- it belongs in the band."""
    c, _ = _ceiling_band(active_b, moe=True)
    assert c == pytest.approx(512 / active_b, rel=1e-6)
    assert c == pytest.approx(expect, abs=0.1)


def test_c1_no_fraction_survives_in_the_ceiling():
    """Source guard: the ceiling line must not multiply by the sustained fraction."""
    src = (_PA / "agent" / "perf_target.py").read_text()
    i = src.index("def rate_and_band")
    body = src[i : src.index("\ndef ", i + 10)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "theo = pk / per_dev" in code, "the ceiling is not spec/bytes"
    assert "(pk * fr)" not in code, "the fraction is folded into the ceiling again"


# --------------------------------------------------------------------------- c2 TARGETS HELD
@pytest.mark.parametrize(
    "params_b,moe,old_ceiling",
    [(8, False, 51.2), (12, False, 34.1), (3, True, 85.3), (13, True, 19.7)],
)
def test_c2_band_top_equals_the_old_ceiling(params_b, moe, old_ceiling):
    """Nothing anyone was aiming at has moved -- it is just labelled honestly now."""
    _c, (_lo, hi) = _ceiling_band(params_b, moe)
    assert hi == pytest.approx(old_ceiling, abs=0.15)


def test_c2_dense_band_is_still_sixty_to_eighty_percent():
    c, (lo, hi) = _ceiling_band(12)
    assert lo / c == pytest.approx(0.60, abs=1e-9)
    assert hi / c == pytest.approx(0.80, abs=1e-9)


def test_c2_moe_band_keeps_the_same_shape():
    """0.50 top, and the bottom the same 0.75-of-top ratio dense uses -- not a fourth constant."""
    c, (lo, hi) = _ceiling_band(3, moe=True)
    assert hi / c == pytest.approx(0.50, abs=1e-9)
    assert lo / hi == pytest.approx(0.75, abs=1e-9)


def test_c2_the_gemma3_report_line():
    """The numbers that will appear in RUN_REPORT.md for the next run."""
    c, (lo, hi) = _ceiling_band(12)
    measured = 1000 / 34.82
    assert (round(c, 1), round(lo, 1), round(hi, 1)) == (42.7, 25.6, 34.1)
    assert lo < measured < hi, "28.7 must sit INSIDE the band, not above it"
    assert round(measured / c * 100) == 67


# --------------------------------------------------------------------------- c3 UNBEATABLE
@pytest.mark.parametrize("params_b", [4, 8, 12, 70])
def test_c3_a_measurement_cannot_beat_the_ceiling(params_b):
    """The defining property. bytes/peak is the fastest a token can physically emerge."""
    c, _ = _ceiling_band(params_b)
    fastest_ms = (params_b * 1e9) / PEAK * 1000.0
    assert 1000.0 / fastest_ms == pytest.approx(c, rel=1e-6)


def test_c3_the_old_ceiling_was_beatable_this_one_is_not():
    """Control: 28.7 exceeded the old 34.1-style ceiling's band; against spec it cannot."""
    c, (_lo, hi) = _ceiling_band(12)
    measured = 28.7
    assert measured > 0.80 * hi, "fixture must reproduce a run near the top of the band"
    assert measured < c, "a measurement exceeded the spec ceiling -- it is not a ceiling"


# --------------------------------------------------------------------------- c4 TP
@pytest.mark.parametrize("tp,expect", [(1, 42.7), (2, 85.3), (4, 170.7)])
def test_c4_tp_divides_the_bytes(tp, expect):
    """Fewer bytes per chip, same per-chip bandwidth -- the ceiling scales with TP."""
    c, _ = _ceiling_band(12, tp=tp)
    assert c == pytest.approx(expect, abs=0.15)


def test_c4_band_scales_with_tp_too():
    c1, (lo1, hi1) = _ceiling_band(12, tp=1)
    c4, (lo4, hi4) = _ceiling_band(12, tp=4)
    assert (c4 / c1, lo4 / lo1, hi4 / hi1) == pytest.approx((4.0, 4.0, 4.0), rel=1e-9)


# --------------------------------------------------------------------------- c5 DEGENERATE
@pytest.mark.parametrize("b,peak", [(0, PEAK), (12e9, 0), (-1, PEAK), (12e9, -5)])
def test_c5_unknown_inputs_give_no_ceiling(b, peak):
    c, band = pt.rate_and_band(b, peak, frac=0.80)
    assert c == 0.0 and band == (0.0, 0.0)


def test_c5_zero_fraction_gives_a_ceiling_but_no_band():
    """A missing fraction must not erase the physical wall -- only the target."""
    c, (lo, hi) = pt.rate_and_band(12e9, PEAK, frac=0.0)
    assert c == pytest.approx(42.67, abs=0.05)
    assert (lo, hi) == (0.0, 0.0)
