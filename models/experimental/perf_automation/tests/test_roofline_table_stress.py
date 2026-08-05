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

Hence three blocks, and a DIRECTION marked on each utilization bar so a reader is not asked to infer
that two of them want to be small.

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


def test_each_row_states_its_own_direction():
    """26% dispatch is bad and 69% bandwidth is good; an unmarked bar invites the opposite reading.

    Marked per ROW rather than under a group heading: the heading sat furthest from the bars it
    described, and grouping forced an ordering on rows that otherwise read roofline-then-overhead."""
    u = _render()[_render().index("Utilization") :]
    assert "↑ better" in u and "↓ better" in u
    for name, arrow in (("DRAM bandwidth", "↑"), ("Dispatch overhead", "↓"), ("DRAM capacity", "↓")):
        row = next(l for l in u.splitlines() if name in l)
        assert row.rstrip().endswith("%s better" % arrow), row


def test_an_unmeasured_row_claims_no_direction():
    """TTFT is never measured, so there is no number for 'higher' or 'lower' to be about."""
    row = next(l for l in _render().splitlines() if "Compute (prefill)" in l)
    assert "better" not in row, row


def test_the_direction_column_lines_up():
    u = _render()[_render().index("Utilization") :]
    rows = [l for l in u.splitlines() if "better" in l and l.startswith("  ")]
    assert len({l.index("better") for l in rows}) == 1, rows


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


def test_no_row_carries_trailing_whitespace():
    """RUN_REPORT.md is committed; the repo's pre-commit hook strips trailing whitespace, so a
    generator that emits it makes the file dirty the moment it is written."""
    for line in _render().splitlines():
        assert line == line.rstrip(), repr(line)


# ---------------------------------------------------------------- the overhead rows judge honestly


def test_a_healthy_row_carries_no_verdict():
    """ "ok" read as a verdict against the TARGET beside it while being judged on a hidden 10%
    tolerance -- so a row printed target ~0 ms, measured 2.46 ms, and called itself ok. On a
    threshold row, passing is the silent case."""
    over = _render()[_render().index("Overheads") : _render().index("Utilization")]
    assert " ok" not in over, over


def test_a_breach_is_marked():
    """The exception is the entire signal, so it must be impossible to miss."""
    out = _render(profile=_prof(host=200.0))
    over = out[out.index("Overheads") : out.index("Utilization")]
    assert "✗ OVER" in over, over


def test_the_flag_threshold_is_printed_not_hidden():
    """A verdict the reader cannot check against a stated number is not a verdict."""
    assert "flag >%d%%" % S._DISPATCH_FLAG_PCT in _render()


def _cap_row(**kw):
    """The capacity row alone -- the default profile's dispatch is 16% and already breaching, so a
    whole-render assertion would pass on the wrong row."""
    return next(l for l in _render(**kw).splitlines() if "DRAM capacity" in l and "│" in l)


def test_capacity_breaches_at_the_safety_margin():
    assert "✗ OVER" in _cap_row(active_bytes=int(BH_DRAM * 0.95))
    assert "✗ OVER" not in _cap_row(active_bytes=int(BH_DRAM * 0.5))


def test_the_share_names_the_declared_unit():
    """The MEASURED column says ms/token; calling the same thing a 'step' one column over is two
    names for one unit, side by side."""
    over = _render()[_render().index("Overheads") : _render().index("Utilization")]
    assert "of token" in over and "of step" not in over, over


# ---------------------------------------------------------------- the block states what it lacks


_FID = (
    [("lofi", 7.34e12, 702.0, 10.45), ("hifi2", 0.0, 351.0, 0.0), ("hifi4", 0.0, 176.0, 0.0)],
    10.45,
)


@pytest.fixture()
def fid(monkeypatch):
    monkeypatch.setattr(S, "_fidelity_breakdown", lambda p: _FID)


def test_the_compute_row_uses_a_measured_prefill_stage(fid):
    """It printed a hardcoded "not measured" while trace_replay's prefill stage sat in the state file
    and the block-timing section below rendered it -- the same report saying, in two places, that one
    phase both was and was not measured."""
    out = _render(stage_ms={"prefill": 35.80, "decode": 138.49})
    row = next(l for l in out.splitlines() if "Compute FLOPs" in l)
    assert "35.80 ms" in row and "trace_replay" in row, row
    assert "not measured" not in row, row


def test_a_prefill_above_its_compute_band_is_flagged(fid):
    """35.80 ms against a 13.1-17.4 ms ceiling says prefill is not compute-bound. The tick must not
    say otherwise."""
    row = next(l for l in _render(stage_ms={"prefill": 35.80}).splitlines() if "Compute FLOPs" in l)
    assert row.rstrip().endswith("✗"), row


def test_a_prefill_inside_its_compute_band_ticks(fid):
    row = next(l for l in _render(stage_ms={"prefill": 12.0}).splitlines() if "Compute FLOPs" in l)
    assert row.rstrip().endswith("✔"), row


def _flat(t):
    """Gap reasons are WRAPPED to the table width, so a phrase can straddle two lines. Match against
    the collapsed text or the assertion tests the wrap point, not the wording."""
    return " ".join(t.split())


def test_no_stage_name_is_guessed(fid):
    """Only the DECLARED prefill counts. Matching 'whatever looks like a prefill' would put a
    decode-path number in a TTFT cell on any pipeline that names its stages differently."""
    out = _render(stage_ms={"encode": 35.80, "generate": 138.49})
    row = next(l for l in out.splitlines() if "Compute FLOPs" in l)
    assert "35.80" not in row and "not measured" in row, row


def test_a_zero_flop_rung_is_marked_measured_not_missing(fid):
    """All four fidelities print so the reader sees the whole ladder; that only works if an empty
    rung is visibly EMPTY rather than visibly unknown."""
    out = _render(stage_ms={"prefill": 12.0})
    row = next(l for l in out.splitlines() if "HiFi4" in l)
    assert "no ops at this fidelity" in row, row
    assert "no ops at this fidelity" not in next(l for l in out.splitlines() if "LoFi" in l)


def test_the_report_offers_no_batch_advice():
    """ "21.6 GiB unused - headroom for a larger batch" was editorial, not measurement: batching is an
    emit-e2e decision, and the capacity row already states used-vs-total."""
    assert "headroom" not in _render() and "larger batch" not in _render()


def test_a_dispatch_share_at_or_above_one_is_refused():
    """host_overhead sums per-op GAPS and op intervals OVERLAP, so on a concurrent profile the sum
    runs past total device_ms (634.55 vs 293.20 on gemma-3-12b-it). Scaling that onto a token would
    claim more launch overhead than the step contains, so the row is withheld -- never rendered as a
    plausible-looking number."""
    out = _render(profile={"device_ms": 293.20, "buckets": [{"id": "host_overhead", "device_ms": 634.55}]})
    over = out[out.index("Utilization") :]
    assert "Dispatch overhead" not in over, over


# ---------------------------------------------------------------- an estimate is never a measurement


def test_an_estimate_is_marked_by_a_tilde(fid):
    row = next(l for l in _render(prefill_est_ms=17.0).splitlines() if "Compute FLOPs" in l)
    assert "~17.0 ms" in row, row


def test_an_estimate_earns_no_verdict(fid):
    """A guess cannot be graded, however close to the band it lands -- a tick would assert the
    model IS in band on the strength of a number nobody measured."""
    for est in (14.0, 40.0):
        row = next(l for l in _render(prefill_est_ms=est).splitlines() if "Compute FLOPs" in l)
        assert not row.rstrip().endswith("✗") and not row.rstrip().endswith("✔"), row


def test_the_prefill_row_matches_the_others(fid):
    """achieved / total, same as bandwidth and capacity, and higher is better."""
    row = next(l for l in _render(prefill_est_ms=13.1).splitlines() if "Compute (prefill)" in l)
    assert "10.4 / 13.1 ms" in row and row.rstrip().endswith("↑ better"), row


def test_no_estimate_changes_nothing(fid):
    out = _render()
    assert "~" not in next(l for l in out.splitlines() if "Compute FLOPs" in l)
    assert "TTFT never measured" in out


def test_the_utilization_row_uses_the_measurement(fid):
    """THE INCONSISTENCY: the roofline cell read the measured stage while this row read only the
    estimate, so one report answered the same question two ways -- "15.90 ms (trace_replay)" above
    and "TTFT never measured" below. A guess lit the bar and a measurement did not."""
    out = _render(stage_ms={"prefill": 15.9})
    row = next(l for l in out.splitlines() if "Compute (prefill)" in l)
    assert "10.4 / 15.9 ms" in row and row.rstrip().endswith("↑ better"), row
    assert "never measured" not in row, row


def test_the_measurement_wins_over_an_estimate(fid):
    """Both present: the measured value is the one that renders, in BOTH cells."""
    out = _render(stage_ms={"prefill": 15.9}, prefill_est_ms=13.1)
    assert "15.90 ms (trace_replay)" in out
    row = next(l for l in out.splitlines() if "Compute (prefill)" in l)
    assert "15.9 ms" in row and "13.1" not in row, row


def test_both_cells_agree_on_what_is_known(fid):
    """Whatever the inputs, the two cells must not disagree about whether prefill is known."""
    for kw in ({}, {"stage_ms": {"prefill": 15.9}}, {"prefill_est_ms": 13.1}):
        out = _render(**kw)
        roof_known = "not measured" not in next(l for l in out.splitlines() if "Compute FLOPs" in l)
        util_known = "never measured" not in next(l for l in out.splitlines() if "Compute (prefill)" in l)
        assert roof_known == util_known, (kw, out)
