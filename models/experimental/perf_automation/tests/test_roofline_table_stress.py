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


def test_each_row_states_its_own_direction(fid, facts):
    """26% dispatch is bad and 69% bandwidth is good; an unmarked bar invites the opposite reading.

    Marked per ROW rather than under a group heading: the heading sat furthest from the bars it
    described, and grouping forced an ordering on rows that otherwise read roofline-then-overhead."""
    u = _render(stage_ms={"prefill": 30.0, "decode": 33.9})
    u = u[u.index("Utilization") :]
    assert "↑ better" in u and "↓ better" in u
    rows = [l for l in u.splitlines() if "better" in l]
    # one bar per stage (the BINDING roof), plus dispatch and capacity
    assert len(rows) == 4, rows
    for row in rows:
        want = "\u2193 better" if ("dispatch" in row or "capacity" in row) else "\u2191 better"
        assert row.rstrip().endswith(want), row


def test_an_unmeasured_stage_draws_no_bar():
    """A stage with no measured wall-clock has nothing to divide, so it contributes no utilisation
    row at all -- rather than a 0% bar, which is a claim about the model instead of about the run."""
    out = _render()  # no stage_ms -> prefill never measured
    u = out[out.index("Utilization") :]
    assert "prefill" not in u, u


def test_the_direction_column_lines_up():
    u = _render()[_render().index("Utilization") :]
    rows = [l for l in u.splitlines() if "better" in l and l.startswith("  ")]
    assert len({l.index("better") for l in rows}) == 1, rows


# ---------------------------------------------------------------- alignment


def test_the_column_dividers_line_up():
    """A table whose separator does not meet its dividers reads as corrupted output.

    Checked on EVERY row, not a sampled one. The failures this catches are cells that OVERFLOW their
    field and shunt the divider right -- "not measured" in a 10-wide number slot, "ms, flag >10%" in a
    12-wide unit slot -- and those show up on one row type at a time, so a spot check passes while the
    table is visibly crooked."""
    t = _render(stage_ms={"prefill": 30.0, "decode": 33.9}).splitlines()
    cols = {tuple(i for i, c in enumerate(ln) if c == "\u2502") for ln in t if "\u2502" in ln}
    assert cols == {(31, 49, 79)}, sorted(cols)
    seps = {tuple(i for i, c in enumerate(ln) if c == "\u253c") for ln in t if "\u253c" in ln}
    assert seps == cols, (seps, cols)


def test_the_fidelity_ladder_is_its_own_section(fid, facts):
    """IT IS A WHAT-IF, NOT A MEASUREMENT, and it gets its own section.

    Rendered inside each stage's compute roof it sat in the three-column grid, where its two values
    -- a peak, and what that stage's FLOPs cost at that peak -- landed under THEORETICAL and
    ACHIEVABLE 60-80%, so the second read as a sustained band it is not. It also duplicated per
    stage; the stages are columns now.

    BEHAVIOUR CHANGE: this asserted a SINGLE "in use" marker for the whole table, on the assumption
    that the peaks are identical for every stage. They need not be -- a fidelity lever converts one
    stack at a time -- and the single marker was chosen by whole-profile FLOP share, so the stack
    with the most FLOPs named the rung for all of them. Each stage now marks its own column.
    """
    out = _render(stage_ms={"prefill": 30.0, "decode": 33.9})
    assert "Fidelity ladder" in out, out
    roof = out[out.index("Roofline") : out.index("Fidelity ladder")]
    assert "LoFi" not in roof and "HiFi4" not in roof, roof
    lad = out[out.index("Fidelity ladder") : out.index("Overheads")]
    for rung in ("LoFi", "HiFi2", "HiFi4"):  # the fixture carries three rungs
        assert rung in lad, (rung, lad)
    # the stages are columns, so each rung states both on one row
    assert "prefill ms" in lad and "decode ms" in lad, lad
    # ONE MARKER PER STAGE COLUMN. Both stages share a rung here, so both land on the same row --
    # what matters is that neither is left unmarked and neither is marked twice.
    assert lad.count("← in use") == 2, lad
    _marked = [ln for ln in lad.splitlines() if "← in use" in ln]
    assert len(_marked) == 1 and _marked[0].count("← in use") == 2, _marked


def test_each_stage_marks_its_own_rung_when_they_differ(fid, facts, monkeypatch):
    """THE CASE THE SINGLE MARKER COULD NOT STATE. A fidelity lever converts one stack at a time, so
    the stacks sit on different rungs while it runs. One arrow, chosen by whole-profile FLOP share,
    named the loudest stack's rung and spoke for the others: the table read as though every stack
    had moved, and the headroom still left on the unconverted ones looked already spent."""
    # The stack that got the lever, and the one that did not. Stated per stage rather than derived
    # here, because the point under test is the RENDERER: given two stages on two rungs, does each
    # column mark its own?
    _rung = {"prefill": (702.0e12, "lofi"), "decode": (176.0e12, "hifi4")}
    monkeypatch.setattr(
        S,
        "_peak_for_stage",
        lambda stage, prof, model="", task="": _rung.get(str(stage), (0.0, "")),
    )
    out = _render(stage_ms={"prefill": 30.0, "decode": 33.9})
    lad = out[out.index("Fidelity ladder") : out.index("Overheads")]
    rows = {ln.split("\u2502")[0].strip(): ln for ln in lad.splitlines() if ln.strip().startswith(("LoFi", "HiFi"))}
    # prefill on LoFi, decode on HiFi4 -- two rungs, each marked once, on DIFFERENT rows. The single
    # arrow could only ever have named one of them.
    assert "← in use" in rows["LoFi"], rows["LoFi"]
    assert "← in use" in rows["HiFi4"], rows["HiFi4"]
    assert "← in use" not in rows.get("HiFi2", ""), rows.get("HiFi2")
    assert rows["LoFi"].count("← in use") == 1, rows["LoFi"]
    assert rows["HiFi4"].count("← in use") == 1, rows["HiFi4"]


def test_the_bar_is_proportional():
    for frac, filled in ((0.0, 0), (0.5, 10), (1.0, 20)):
        assert S._bar(frac).count("█") == filled
    # width is an argument: the utilisation panel needs 30 cells to separate 0.1% from 11%
    assert S._bar(0.5, 30).count("█") == 15 and len(S._bar(None, 30)) == 30


def test_a_bar_never_overflows_its_width():
    """A measurement above the ceiling must not print a longer bar than the scale."""
    assert len(S._bar(3.0)) == S._BAR_W and S._bar(3.0).count("█") == S._BAR_W
    assert len(S._bar(-1.0)) == S._BAR_W and S._bar(-1.0).count("█") == 0


def test_no_data_renders_an_empty_bar_not_a_zero():
    """Zero and unknown are different claims."""
    assert S._bar(None).count("█") == 0


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
    assert "dispatch  overhead" not in over


# ---------------------------------------------------------------- the numbers are the right ones


def test_capacity_uses_binary_gigabytes():
    """32 GiB of DRAM is 34.4 decimal GB; printing 34 next to a spec sheet saying 32 reads as a bug.

    Read off the OVERHEADS row now: the utilisation bar for capacity was removed, because its
    percentage was already stated there and a second copy in different furniture is not a second
    fact."""
    out = _render()
    over = out[out.index("Overheads") : out.index("Utilization")]
    assert "GiB" in over and "28.8" in over, over


def test_capacity_utilization_is_against_the_real_part():
    out = _render(active_bytes=BH_DRAM // 2)
    assert "50% used" in out and "50%" in out[out.index("Utilization") :], out


def test_dispatch_share_is_of_the_headline_unit():
    """host_overhead is 62.4 of 381.23 device_ms = 16%, scaled onto the per-token number."""
    out = _render()
    assert "16%" in out


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


def test_a_breach_states_its_share_and_grades_nothing():
    """The share IS the statement. "OVER" was a verdict against a 10% rule printed one column away,
    and the rule line went with the verdict -- with nothing left to grade against, it graded nothing.
    Same change the Roofline made: report the number, let the reader judge."""
    out = _render(profile=_prof(host=200.0))
    over = out[out.index("Overheads") : out.index("Utilization")]
    assert "% of token" in over, over
    assert "OVER" not in over and "flag >" not in over, over


def test_the_target_column_holds_only_targets():
    """It carried "10784 ops" -- a MEASUREMENT, counted over the whole profiling window (one prefill
    plus six decode steps) and printed on a row that reads per token, so wrong by ~7x as well.
    Nobody targets an op count. Op counts live in the Op breakdown table, per class."""
    out = _render()
    over = out[out.index("Overheads") : out.index("Utilization")]
    assert "ops" not in over, over


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


@pytest.fixture()
def facts(monkeypatch):
    """The model's own params and the declared sequence length -- the two inputs the stage compute
    roof is built from. Stubbed rather than read off disk so this file still needs no device and no
    model directory."""
    monkeypatch.setattr(S, "_model_facts", lambda: {"total_params": 11_180_446_320, "layers": 48})
    monkeypatch.setenv("TT_PERF_ISL_TOKENS", "128")
    monkeypatch.setenv("PERF_MCP_ARCH", "blackhole")


def test_the_binding_roof_carries_the_measurement(fid, facts):
    """It printed a hardcoded "not measured" while trace_replay's prefill stage sat in the state file
    and the block-timing section below rendered it -- the same report saying, in two places, that one
    phase both was and was not measured.

    On EVERY roof row, not hoisted to the stage heading. There is one stopwatch per stage, so hoisting
    it looks like the way to avoid repeating it -- but it leaves each roof row holding a bare tick,
    with MEASURED empty exactly where THEORETICAL and ACHIEVABLE have the numbers it is the verdict
    on. A roofline row is read ACROSS, and a column that empties out on the rows that matter is not a
    column."""
    out = _render(stage_ms={"prefill": 35.80, "decode": 138.49})
    body = out[out.index("PREFILL") : out.index("DECODE")]
    row = next(l for l in body.splitlines() if "← binds" in l)
    assert "35.80" in row and "not measured" not in row, row


def test_a_roof_row_is_complete_across_all_three_columns(fid, facts):
    """The verdict glyph is only meaningful next to the pair it judges."""
    out = _render(stage_ms={"prefill": 35.80, "decode": 138.49})
    for ln in out.splitlines():
        if "✗" not in ln and "✔" not in ln:
            continue
        lbl, theo, band, meas = ln[:31], ln[33:49], ln[51:79], ln[81:]
        assert any(c.isdigit() for c in theo), ln
        assert any(c.isdigit() for c in band), ln
        assert any(c.isdigit() for c in meas), ln


def test_no_printed_row_has_a_hole_in_the_measured_column(fid, facts):
    """THE INVARIANT THAT SETTLES THE LAYOUT. Any row that states a THEORETICAL and an ACHIEVABLE
    value must state a MEASURED one; a row that cannot is DROPPED, not printed with a blank.

    The achieved rate is a stage fact -- one run, one number -- so it duplicated when printed under
    both roofs and left a hole when printed under neither. The row itself is what has to go: the
    non-binding roof's ceiling is already stated in ms and in its own currency, and the slack ratio
    carries the comparison."""
    full = _render(stage_ms={"prefill": 35.80, "decode": 138.49})
    # the Roofline block only: the Fidelity ladder is its own section with its own columns
    t = full[full.index("Roofline") : full.index("Fidelity ladder")].splitlines()
    for ln in t:
        if "│" not in ln or "ACHIEVABLE" in ln:
            continue
        theo, band, meas = ln[33:49], ln[51:79], ln[81:]
        if not (theo.strip() and band.strip()):
            continue
        # the fidelity ladder's own header row labels ITS two columns; it states no measurement
        # because a rung that was not run has none, and the rows below say which
        if "fidelity mix" in ln:
            continue
        assert meas.strip(), ln


def test_the_roofline_carries_no_verdict_glyph(fid, facts):
    """THE THREE COLUMNS ARE THE VERDICT.

    Ceiling, band and measurement sit side by side; a tick or cross states nothing they do not, and
    it kept ASSERTING one where the band does not apply -- on an eager-measured stage, and on the
    roof that is not the stage's limit, where the measurement scores an automatic miss against a
    ceiling physics forbids it reaching. The reader compares the numbers.

    Scoped to the Roofline block: the Overheads block keeps its `✗ OVER`, which marks a breach of a
    STATED threshold rather than a position within a band."""
    full = _render(stage_ms={"prefill": 30.0, "decode": 33.9})
    roof = full[full.index("Roofline") : full.index("Overheads")]
    assert "✔" not in roof and "✗" not in roof, roof
    # and the overheads block dropped its own verdict for the same reason
    assert "✗" not in full[full.index("Overheads") : full.index("Utilization")], full


def test_the_roof_that_is_not_the_limit_reduces_to_its_own_currency(fid, facts):
    """NO MS ROW THERE -- the same rule as the rate row.

    Elapsed time is stage-level: compute and memory run at once, so the stage's 33.94 ms cannot be
    split into "X ms memory, Y ms compute". Printed on this roof it was either the stage wall-clock,
    which says something false (compute did not take 33.94 ms, the STAGE did, while the arithmetic
    idled waiting on memory), or a dash. Neither earns a row.

    What is left is TFLOPS, which differs by stage even though the peak does not."""
    out = _render(stage_ms={"decode": 33.9})
    blk = out[out.index("DECODE") : out.index("Overheads")]
    rows = [l for l in blk.splitlines() if l.strip().startswith("compute")]
    assert len(rows) == 1, rows
    assert "TFLOPS" in rows[0] and " ms" not in rows[0], rows[0]
    assert "33.94" not in rows[0] and "—" not in rows[0], rows[0]


def test_the_two_stages_compute_roofs_differ(fid, facts):
    """The guard on the regression above: peak TFLOPS is a device constant, so if the compute roof
    ever reduces to it again, prefill and decode print the same line and the table stops saying
    anything about either stage."""
    out = _render(stage_ms={"prefill": 30.0, "decode": 33.9})
    pf = out[out.index("PREFILL") : out.index("DECODE")]
    dc = out[out.index("DECODE") : out.index("Overheads")]
    a = [l for l in pf.splitlines() if l.strip().startswith("compute")][0]
    b = [l for l in dc.splitlines() if l.strip().startswith("compute")][0]
    assert a != b, a


def test_the_rate_belongs_to_the_binding_roof(fid, facts):
    """THE RATE IS THAT ROOF'S TO EXPLAIN.

    There is one achieved rate and memory sets it. Printed under compute it was a memory-set number
    held against a compute ceiling of 18850-25088 tok/s/u -- not a comparison but a non-sequitur,
    since nothing about the arithmetic can be read off a number the arithmetic did not determine.
    What can be read off it is already one row up, as 0.69 of 702.0 TFLOPS.

    It follows `binds`, so a compute-bound stage gets the mirror image."""
    out = _render(stage_ms={"prefill": 35.80, "decode": 138.49})
    blk = out[out.index("DECODE") : out.index("Overheads")]
    rows = blk.splitlines()
    rate_rows = [i for i, l in enumerate(rows) if "tok/s/u" in l]
    assert len(rate_rows) == 1, [rows[i] for i in rate_rows]
    # and it sits under the roof marked binding, not merely under the first roof
    binds_at = next(i for i, l in enumerate(rows) if "← binds" in l)
    nxt = next(
        (i for i, l in enumerate(rows) if i > binds_at and l.strip().startswith(("memory", "compute"))), len(rows)
    )
    assert binds_at < rate_rows[0] < nxt, (binds_at, rate_rows, nxt)


def test_both_roofs_are_stated_for_both_stages(fid, facts):
    """THE DEFECT THIS REPLACES. `annotate_op` kept only the WINNING floor, so compute was the only
    term that survived to the report -- and the report duly printed a compute band over a stage the
    profile itself marks memory-bound. A roofline with one roof is a line."""
    out = _render(stage_ms={"prefill": 35.80, "decode": 138.49})
    for stage, nxt in (("PREFILL", "DECODE"), ("DECODE", "Overheads")):
        blk = out[out.index(stage) : out.index(nxt)]
        assert "memory" in blk and "compute" in blk, (stage, blk)
        assert "GB/s" in blk and "TFLOPS" in blk, (stage, blk)


def test_exactly_one_roof_per_stage_is_marked_binding(fid, facts):
    """The stage cannot beat its tightest floor, and which floor that is genuinely differs by stage:
    prefill FLOPs scale with the sequence and decode's do not."""
    out = _render(stage_ms={"prefill": 35.80, "decode": 138.49})
    for stage, nxt in (("PREFILL", "DECODE"), ("DECODE", "Overheads")):
        blk = out[out.index(stage) : out.index(nxt)]
        assert blk.count("← binds") == 1, (stage, blk)


def test_the_binding_roof_is_the_slower_one(fid, facts):
    """Decode reads the whole model per token and does ~2 FLOPs per param, so memory binds by three
    orders of magnitude. If compute were marked, the report would send the agent at the fidelity rung
    for a stage no fidelity change can help."""
    out = _render(stage_ms={"decode": 138.49})
    blk = out[out.index("DECODE") : out.index("Overheads")]
    binding = next(l for l in blk.splitlines() if "← binds" in l)
    assert "memory" in binding, binding


def _flat(t):
    """Gap reasons are WRAPPED to the table width, so a phrase can straddle two lines. Match against
    the collapsed text or the assertion tests the wrap point, not the wording."""
    return " ".join(t.split())


def test_no_stage_name_is_guessed(fid, facts):
    """Only the DECLARED prefill counts. Matching 'whatever looks like a prefill' would put a
    decode-path number in a TTFT cell on any pipeline that names its stages differently."""
    out = _render(stage_ms={"encode": 35.80, "generate": 138.49})
    # This model declares ENCODE and GENERATE. It has no prefill, so it now gets no PREFILL row at
    # all -- stronger than the old assertion, which only checked that encode's number stayed out of
    # a prefill cell that should not have existed in the first place.
    assert "PREFILL" not in out, out
    assert "ENCODE" in out and "GENERATE" in out, out
    assert "35.80" in out, "the declared stage was measured and must be shown"


def test_the_compute_peak_is_the_fidelity_the_model_runs(fid, facts):
    """chip_peak_flops defaults to HiFi4 when handed no fidelity, and HiFi4 is a QUARTER of LoFi on
    Blackhole -- so a LoFi model was priced against 175 TFLOPS instead of 702, making its compute roof
    4x too slow and its utilisation 4x too flattering."""
    out = _render(stage_ms={"prefill": 30.0})
    # Only prefill is declared, so PREFILL is the last section -- there is no DECODE heading to slice
    # against any more.
    blk = out[out.index("PREFILL") :]
    row = next(l for l in blk.splitlines() if "TFLOPS" in l and "LoFi" not in l and "HiFi" not in l)
    assert "702" in row, row


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
    assert "dispatch  overhead" not in over, over


# ---------------------------------------------------------------- no estimate can enter the table


def test_the_operator_estimate_is_gone():
    """PERF_MCP_PREFILL_EST_MS filled the prefill cell with a number nobody measured, marked with a
    tilde and a hatched bar to keep it from being quoted as one. The cell is now filled by a roof
    derived from the model's own params and the DECLARED sequence length, so the guess has nothing
    left to do -- and a report with no route for a guess needs no marks warning about one."""
    src = Path(S.__file__).read_text()
    assert "prefill_est_ms" not in src and "PREFILL_EST_MS" not in src


def test_no_cell_is_a_guess(fid, facts):
    """Scoped to the Roofline block. The dispatch TARGET legitimately reads "~0 ms" -- that tilde is
    the target itself, not a hedge on a measurement."""
    for kw in ({}, {"stage_ms": {"prefill": 15.9}}):
        out = _render(**kw)
        roof = out[out.index("Roofline") : out.index("Overheads & limits")]
        assert "~" not in roof, roof


def test_an_unmeasured_stage_says_so_rather_than_vanishing(fid, facts):
    """A report is a confirmation document: the stage must still state its roofs, and say plainly
    that nothing was clocked against them."""
    out = _render()
    blk = out[out.index("PREFILL") : out.index("DECODE")]
    assert "not measured" in blk and "memory" in blk, blk
    assert "✔" not in blk and "✗" not in blk, blk


# ---------------------------------------------------------------- the stage must not vanish


def test_the_prompt_length_is_stated_once(monkeypatch, fid):
    """ONE LITERAL. The emitted test states the benchmark point (128 in / 128 out) because generated
    code cannot import from its generator; everything else READS that literal rather than keeping a
    copy. Two copies is how the roof came to price a length the run never used.

    Precedence is observed, then declared, then the skeleton's own default -- so an overridden ISL is
    priced correctly, and the untouched case is priced as the number the run is about to use."""
    monkeypatch.delenv("TT_PERF_ISL_TOKENS", raising=False)
    monkeypatch.delenv("TT_PERF_SEQ_LEN", raising=False)
    monkeypatch.setattr(S, "_model_facts", lambda: {"total_params": 11_180_446_320, "layers": 48})
    monkeypatch.setenv("PERF_MCP_ARCH", "blackhole")

    from agent.perf_test_gen import DEFAULT_ISL_TOKENS, _skeleton_default

    # the default is READ from the emitted test's own literal, not written twice
    assert DEFAULT_ISL_TOKENS == _skeleton_default("TT_PERF_ISL_TOKENS") > 0

    monkeypatch.setattr(S, "_perf_mcp", lambda: type("m", (), {"read_stage_isl": staticmethod(lambda *a, **k: 0)}))
    assert S._prompt_tokens() == DEFAULT_ISL_TOKENS
    assert "PREFILL" in _render(stage_ms={"prefill": 35.80, "decode": 138.49})

    # ...and an OBSERVED length wins over it, so an overridden ISL is never priced as the default
    monkeypatch.setattr(S, "_perf_mcp", lambda: type("m", (), {"read_stage_isl": staticmethod(lambda *a, **k: 512)}))
    assert S._prompt_tokens() == 512


def test_the_observed_isl_beats_the_default(monkeypatch, fid):
    """OBSERVED, NOT GUESSED. The generated test prints the prompt length it actually tokenized, so a
    run at any ISL prices prefill at that ISL -- without anyone exporting a variable.

    A hardcoded fallback alone is wrong the moment a run uses a different length, which is the whole
    failure mode: the report would price a 512-token prefill's arithmetic as if it were 128."""
    monkeypatch.delenv("TT_PERF_ISL_TOKENS", raising=False)
    monkeypatch.delenv("TT_PERF_SEQ_LEN", raising=False)
    monkeypatch.setattr(S, "_model_facts", lambda: {"total_params": 11_180_446_320, "layers": 48})
    monkeypatch.setenv("PERF_MCP_ARCH", "blackhole")

    # patch through summary's OWN accessor: it resolves perf_mcp itself and caches the module, so
    # patching a separately-imported copy is not the object it calls
    monkeypatch.setattr(S, "_perf_mcp", lambda: type("m", (), {"read_stage_isl": staticmethod(lambda *a, **k: 512)}))
    assert S._prompt_tokens() == 512

    # and the env still overrides nothing above it: observed wins
    monkeypatch.setenv("TT_PERF_ISL_TOKENS", "64")
    assert S._prompt_tokens() == 512


def test_the_model_root_is_given_not_guessed(monkeypatch, fid, tmp_path):
    """THE CALLER'S MODEL DIRECTORY WINS.

    summary is loaded by file path, so perf_mcp's _MODEL_ROOT falls back to "." and
    perf_target_inputs.json is looked for in a directory that does not have it. params came back 0,
    and with no params there is no `2 x params x ISL`: BOTH compute roofs and the entire fidelity
    ladder rendered "not measured" while the file sat in the model dir run.py already knew.

    Third fetched input to fail this way, after the stage timings and the profile. Anything the
    caller can hand over is handed over."""
    (tmp_path / "perf_target_inputs.json").write_text('{"total_params": 7000000000, "layers": 32}')
    monkeypatch.setattr(S, "_MODEL_ROOT_HINT", tmp_path)
    assert (S._model_facts() or {}).get("total_params") == 7_000_000_000

    # and with no hint it does NOT invent one
    monkeypatch.setattr(S, "_MODEL_ROOT_HINT", tmp_path / "nonexistent")
    monkeypatch.setattr(S, "_perf_mcp", lambda: None)
    assert S._model_facts() is None


def test_the_two_stages_never_disagree_about_the_weights_they_share():
    """ONE MODEL, ONE BYTE COUNT. _bytes_for recomputed decode's read set from weight_bytes while the
    HEADLINE ceiling came from the params rule, so a single report carried both:

        stage roof   24.37 GB -> 47.61 ms      (weight_bytes)
        headline     11.18 GB -> 21.84 ms      (params x 1.0 B/param)

    2.18x apart, with the stop gate judging against one number and the reader looking at the other.
    `active_bytes` (the argument) has already been through the agreed precedence -- pinned anchor,
    then measured per-op bytes, then the params rule -- so whatever it decided is THE answer, and
    prefill is that plus only what prefill alone reads."""
    import summary as S

    mf = {
        "weight_bytes": 24374793024,
        "total_params": 11180446320,
        "dominant_dtype": "bfloat16",
        "layers": 48,
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "kv_heads": 8,
        "head_dim": 256,
    }
    _saved_f, _saved_t = S._model_facts, S._prompt_tokens
    try:
        S._model_facts = lambda: mf
        S._prompt_tokens = lambda: 128
        ab = int(11.18e9)
        r = S._stage_roofs(active_bytes=ab, peak_bw_gbps=512.0, tp_degree=1, unit="tok/s/u", profile=None)
        # THE SHARED WEIGHTS ARE THE ANCHOR, and each stage adds only what IT alone reads on top.
        # This asserted decode == anchor exactly, which held only because decode had no per-user term
        # at all: it was priced weights-only, so an 8-user run re-reading eight KV histories counted
        # none of them and batch had nothing to scale. Decode now adds its KV the same way prefill
        # adds its KV and activations -- as a difference against the anchor, never as a second opinion
        # on the weights, which is what this test exists to protect.
        assert r["decode"]["bytes"] >= ab, "decode dropped below the agreed weights figure"
        _kv_only = S._stage_roofs(
            active_bytes=ab, peak_bw_gbps=512.0, tp_degree=1, unit="tok/s/u", profile=None, stage_ms={"decode": 1.0}
        )
        assert _kv_only["decode"]["bytes"] >= ab
        assert r["prefill"]["bytes"] > r["decode"]["bytes"], "prefill must add its KV + activations"
        # and with no context there is nothing extra to add, so it IS the anchor
        _saved_pt = S._prompt_tokens
        S._prompt_tokens = lambda: 0
        try:
            r0 = S._stage_roofs(active_bytes=ab, peak_bw_gbps=512.0, tp_degree=1, unit="tok/s/u", profile=None)
            assert abs(r0["decode"]["bytes"] - ab) < 1.0, "decode invented a second byte count"
        finally:
            S._prompt_tokens = _saved_pt
    finally:
        S._model_facts, S._prompt_tokens = _saved_f, _saved_t


def test_prefill_crosses_from_memory_bound_to_compute_bound_with_the_prompt():
    """The whole reason the two stages get separate roofs: prefill's FLOPs scale with the sequence
    and decode's do not, so the same model binds differently in each stage -- and differently at
    different prompt lengths. A single reused ceiling could not express that."""
    import summary as S

    mf = {
        "weight_bytes": 24374793024,
        "total_params": 11180446320,
        "dominant_dtype": "bfloat16",
        "layers": 48,
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "kv_heads": 8,
        "head_dim": 256,
    }
    _saved_f, _saved_t = S._model_facts, S._prompt_tokens
    try:
        S._model_facts = lambda: mf
        binds = {}
        for isl in (128, 8192):
            S._prompt_tokens = lambda i=isl: i
            r = S._stage_roofs(active_bytes=int(11.18e9), peak_bw_gbps=512.0, tp_degree=1, unit="tok/s/u", profile=None)
            binds[isl] = r["prefill"]["binds"]
        assert binds[128] == "memory", binds
        assert binds[8192] == "compute", binds
    finally:
        S._model_facts, S._prompt_tokens = _saved_f, _saved_t


def test_the_prefill_roof_prices_the_whole_batch_not_one_sequence():
    """THE UNIT OF WORK IS A BATCH, NOT A SEQUENCE.

    Both the byte and the FLOP term came from seq_len alone, so a run at batch 8 was priced as if it
    prefilled 128 tokens when it prefilled 1024. FLOPs are linear in the token count and the
    activation bytes are too, so the roof came out 8x low -- and the stage read memory-bound when it
    was compute-bound. The binding roof is the one thing this table exists to state, and batch was
    deciding it unseen.

    decode is deliberately NOT multiplied: tok/s/u is per USER, so its unit is one token per user
    however many users are in flight."""
    import os

    import summary as S

    mf = {
        "weight_bytes": 24374793024,
        "total_params": 11180446320,
        "dominant_dtype": "bfloat16",
        "layers": 48,
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "kv_heads": 8,
        "head_dim": 256,
    }
    _f, _t, _b = S._model_facts, S._prompt_tokens, os.environ.get("TT_PERF_BATCH")
    try:
        S._model_facts, S._prompt_tokens = (lambda: mf), (lambda: 128)
        got = {}
        for batch in (1, 8):
            os.environ["TT_PERF_BATCH"] = str(batch)
            r = S._stage_roofs(active_bytes=int(11.18e9), peak_bw_gbps=512.0, tp_degree=1, unit="tok/s/u", profile=None)
            got[batch] = r
            assert r["decode"]["tokens"] == 1, "decode is per user; batch must not multiply it"
        assert got[8]["prefill"]["tokens"] == 8 * got[1]["prefill"]["tokens"]
        assert got[8]["prefill"]["compute_ms"] > got[1]["prefill"]["compute_ms"]
        # the case that motivated it: batch 8 flips which roof binds
        assert got[1]["prefill"]["binds"] == "memory", got[1]["prefill"]["binds"]
        assert got[8]["prefill"]["binds"] == "compute", got[8]["prefill"]["binds"]
    finally:
        S._model_facts, S._prompt_tokens = _f, _t
        os.environ.pop("TT_PERF_BATCH", None)
        if _b is not None:
            os.environ["TT_PERF_BATCH"] = _b
