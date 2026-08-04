"""The Roofline / Overheads / Utilization blocks, stressed on inputs no device is needed to make.

The old section was five lines that conflated two different kinds of number under one "achievable"
label:

    theoretical ceiling : 45.8 tok/s/u
    achievable (60-80%) : 27.5 - 36.6 tok/s/u
    measured            : 29.4 tok/s/u
    measured mem BW     : 329 GB/s
    utilization         : 64%

A roofline row has a spec ceiling and a sustained band -- 60-80% dense, 37.5-50% MoE -- because that
is what silicon delivers against peak. Dispatch and capacity have neither: dispatch is OVERHEAD whose
target is zero, and capacity is a HARD WALL with a safety margin. Printing "achievable" across all of
them invited reading 26% dispatch as a grade rather than as a quarter of every token being wasted,
and 35% capacity as underperformance rather than spare room.

Hence three blocks, and utilization bars grouped by DIRECTION so a reader is not asked to infer that
two of the five want to be small.

Everything here renders from dicts. No device, no profiler, no ledger writes -- the optimize loop is
usually holding the board when a report is drawn, and a report generator that needs hardware is a
report generator that fails exactly when you want it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cc_optimize"))

import summary as S  # noqa: E402

BH_DRAM = 32 * 1024**3


def _prof(total=381.23, host=62.4, counts=True):
    b = [{"id": "matmul", "device_ms": total - host, "count": 2313 if counts else 0}]
    if host:
        b.append({"id": "host_overhead", "device_ms": host, "count": 0})
    return {"device_ms": total, "buckets": b}


def _render(**kw):
    base = dict(
        unit="tok/s/u",
        theo=42.67,
        band=[25.6, 34.1],
        measured=29.46,
        bw_gbps=354.0,
        peak_bw_gbps=512.0,
        active_bytes=12_000_000_000,
        per_unit_ms=33.9435,
        profile=_prof(),
    )
    base.update(kw)
    return "\n".join(S._roofline_tables(**base))


# ---------------------------------------------------------------- the three blocks exist and differ


def test_the_three_blocks_are_separate():
    t = _render()
    assert "Roofline" in t and "Overheads & limits" in t and "Utilization" in t


def test_only_the_roofline_block_claims_an_achievable_band():
    """The whole point of the split: ACHIEVABLE must not span dispatch or capacity."""
    t = _render()
    roof = t[t.index("Roofline") : t.index("Overheads & limits")]
    over = t[t.index("Overheads & limits") : t.index("Utilization")]
    assert "ACHIEVABLE" in roof
    assert "ACHIEVABLE" not in over and "TARGET" in over


def test_the_band_percentage_is_derived_not_hardcoded():
    """MoE sustains 37.5-50%, dense 60-80%. A fixed string would be wrong for half the models."""
    assert "60-80%" in _render()
    moe = _render(theo=170.7, band=[64.0, 85.3])
    assert "37-50%" in moe, moe[:400]  # 64.0/170.7 = 37.49%, rendered %.0f


def test_utilization_is_grouped_by_direction():
    """26% dispatch is bad and 69% bandwidth is good; ungrouped bars invite the opposite reading."""
    t = _render()
    u = t[t.index("Utilization") :]
    assert "higher is better" in u and "lower is better" in u
    assert u.index("higher is better") < u.index("lower is better")
    assert u.index("DRAM bandwidth") < u.index("lower is better") < u.index("Dispatch overhead")


# ---------------------------------------------------------------- alignment


def test_the_column_dividers_line_up():
    """A table whose separator does not meet its dividers reads as corrupted output."""
    t = _render().splitlines()
    rows = [ln for ln in t if "│" in ln and "DRAM bandwidth" in ln]
    seps = [ln for ln in t if "┼" in ln]
    assert rows and seps
    for r in rows:
        for sp in seps:
            assert r.index("│") == sp.index("┼"), (r, sp)


def test_the_bar_is_proportional():
    for frac, filled in ((0.0, 0), (0.5, 10), (1.0, 20)):
        assert S._bar(frac).count("█") == filled


def test_a_bar_never_overflows_its_width():
    """A measurement above the ceiling must not print a longer bar than the scale."""
    assert len(S._bar(3.0)) == S._BAR_W and S._bar(3.0).count("█") == S._BAR_W
    assert len(S._bar(-1.0)) == S._BAR_W and S._bar(-1.0).count("█") == 0


def test_no_data_renders_an_empty_bar_not_a_zero():
    """Zero and unknown are different claims. TTFT is unmeasured, not 0%."""
    assert S._bar(None).count("█") == 0
    assert "—" in _render()


# ---------------------------------------------------------------- missing inputs degrade, not crash


@pytest.mark.parametrize(
    "kw",
    [
        {"measured": None},
        {"bw_gbps": None},
        {"band": [None, None]},
        {"peak_bw_gbps": None},
        {"active_bytes": 0},
        {"per_unit_ms": None},
        {"profile": None},
        {"profile": {}},
        {"profile": {"device_ms": 0, "buckets": []}},
        {"profile": _prof(host=0)},
        {"theo": 0},
    ],
)
def test_a_missing_input_still_renders(kw):
    """A report is a confirmation document; it must state what it lacks rather than fail to print."""
    out = _render(**kw)
    assert "Roofline" in out and "Utilization" in out


def test_a_zero_total_does_not_divide_by_zero():
    assert _render(profile={"device_ms": 0.0, "buckets": [{"id": "host_overhead", "device_ms": 5.0}]})


def test_dispatch_is_omitted_when_there_is_no_host_bucket():
    """No host_overhead means the profiler recorded no op gaps -- inventing 0% would be a claim."""
    out = _render(profile=_prof(host=0))
    over = out[out.index("Utilization") :]
    assert "Dispatch overhead" not in over


# ---------------------------------------------------------------- the numbers are the right ones


def test_capacity_uses_binary_gigabytes():
    """32 GiB of DRAM is 34.4 decimal GB; printing 34 next to a spec sheet saying 32 reads as a bug."""
    out = _render()
    assert "GiB" in out and "/ 32 GiB" in out


def test_capacity_utilization_is_against_the_real_part():
    out = _render(active_bytes=BH_DRAM // 2)
    assert "50%" in out


def test_dispatch_share_is_of_the_headline_unit():
    """host_overhead is 62.4 of 381.23 device_ms = 16%, scaled onto the per-token number."""
    out = _render()
    assert "16%" in out


def test_the_measured_column_flags_out_of_band():
    below = _render(measured=10.0)
    assert "✗" in below


def test_in_band_is_marked_as_such():
    assert "✔" in _render()


# ---------------------------------------------------------------- determinism


def test_rendering_is_deterministic():
    """The report is regenerated every round; a section that churns makes diffs useless."""
    assert len({_render() for _ in range(25)}) == 1


def test_no_device_is_touched():
    """Guards the premise of this file: rendering must not shell out to tt-smi or open a device."""
    src = Path(S.__file__).read_text()
    i = src.index("def _roofline_tables")
    body = src[i : src.index("\ndef ", i + 1)]
    for forbidden in ("tt-smi", "ttnn", "subprocess", "open_device", "MeshDevice"):
        assert forbidden not in body, forbidden
