"""A before->after line the tool cannot stand behind must not appear in RUN_REPORT.md at all.

Two real reports motivated this, both printing a line whose two numbers describe different work:

    eager per-op device time (all layers):  before 547.90 ms -> after 547.80 ms
        — NOT COMPARABLE: depth differs: all vs 96
    eager per-op device time (2 layers):  before 296.70 ms -> after 117.40 ms
        — NOT COMPARABLE: depth differs: 2 vs all

The first is the same unoptimized build measured twice (the 'after' is the discovery profile, taken
before a single win landed) -- 0.02% apart, a delta of nothing dressed as a result. The second is
worse: a 2-layer baseline against an all-layer 'after', which reads as a 60% win and is an artifact
of the depth stamp. Printing the disclaimer was not enough; a reader takes the two numbers and
ignores the tail. So the ARROW goes, not the line: what survives is the single latest reading with
its depth attached ("547.80 ms (96 layers)"), which is a real measurement of a real build and cannot
be silently re-read as a delta.

The risk this file guards in the other direction: over-deleting. A COMPARABLE pair must still render
with its delta, or the fix silently blinds the headline the whole report exists to carry.
"""

import importlib
import json
import os

import pytest


@pytest.fixture()
def led_sm(tmp_path, monkeypatch):
    """A ledger rooted in tmp_path, plus the summary module bound to it."""
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    import models.experimental.perf_automation.cc_optimize.measurements as led
    import models.experimental.perf_automation.cc_optimize.summary as sm

    importlib.reload(led)
    importlib.reload(sm)
    return led, sm


def _write(led, model, task, kind, phase, ms, depth, mode="eager", source="test"):
    p = led.ledger_path(model, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "kind": kind,
        "phase": phase,
        "value_ms": ms,
        "depth": depth,
        "mode": mode,
        "source": source,
        "model": model,
        "task": task,
    }
    with p.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- the two real reports


def test_the_gemma3_pair_degrades_to_the_latest_reading(led_sm):
    """all vs 96: the same build profiled twice, 0.09 ms apart. No delta -- but 547.80 is a real
    number and the report should say so, with its depth, and without an arrow."""
    led, sm = led_sm
    _write(led, "gemma3", "main", led.KIND_EAGER, led.PHASE_BEFORE, 547.8951, "all")
    _write(led, "gemma3", "main", led.KIND_EAGER, led.PHASE_AFTER, 547.8010, "96")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "gemma3", "main")
    assert line and "547.80" in line and "96 layers" in line, line
    assert "->" not in line and "547.89" not in line, line
    assert "%" not in line and "x)" not in line, line


def test_the_two_layer_against_all_layer_pair_shows_no_win(led_sm):
    """2 vs all: reads as a 60% win, is a depth artifact. The dangerous one -- the 296.70 must not
    appear at all, or a reader reconstructs the fake delta themselves."""
    led, sm = led_sm
    _write(led, "m2", "main", led.KIND_EAGER, led.PHASE_BEFORE, 296.70, "2")
    _write(led, "m2", "main", led.KIND_EAGER, led.PHASE_AFTER, 117.40, "all")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "m2", "main")
    assert line and "117.40" in line and "all layers" in line, line
    assert "296.70" not in line and "->" not in line, line


def test_no_rendered_line_anywhere_carries_a_disclaimer_or_a_delta(led_sm):
    """The disclaimer existed to excuse a subtraction. With no subtraction there is nothing to
    excuse, and no kind may reintroduce either."""
    led, sm = led_sm
    for kind in (led.KIND_EAGER, led.KIND_TRACE_PASS, led.KIND_FULLPIPE):
        _write(led, "m3", "main", kind, led.PHASE_BEFORE, 100.0, "2")
        _write(led, "m3", "main", kind, led.PHASE_AFTER, 50.0, "all")
        line = sm._ledger_line(kind, kind, "m3", "main")
        assert line and "50.00" in line, (kind, line)
        assert "NOT COMPARABLE" not in line and "->" not in line, (kind, line)


# ---------------------------------------------------------------- the other direction


def test_a_comparable_pair_still_renders_with_its_delta(led_sm):
    """The whole point of the report. Deleting this line would be the worse bug."""
    led, sm = led_sm
    _write(led, "ok1", "main", led.KIND_EAGER, led.PHASE_BEFORE, 200.0, "16")
    _write(led, "ok1", "main", led.KIND_EAGER, led.PHASE_AFTER, 100.0, "16")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "ok1", "main")
    assert line is not None
    assert "200.00" in line and "100.00" in line, line
    assert "16 layers" in line, line
    assert "NOT COMPARABLE" not in line


def test_a_comparable_pair_that_got_slower_still_renders(led_sm):
    """Omission is for pairs that cannot be subtracted, NOT for pairs whose answer is unwelcome.
    Hiding regressions would be the exact dishonesty this change is meant to remove."""
    led, sm = led_sm
    _write(led, "ok2", "main", led.KIND_EAGER, led.PHASE_BEFORE, 100.0, "16")
    _write(led, "ok2", "main", led.KIND_EAGER, led.PHASE_AFTER, 140.0, "16")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "ok2", "main")
    assert line is not None and "140.00" in line, line


def test_before_with_no_after_prints_the_number(led_sm):
    """The normal state for most of a run: one credible reading, shown as a number rather than as a
    half-drawn arrow into nothing."""
    led, sm = led_sm
    _write(led, "ok3", "main", led.KIND_EAGER, led.PHASE_BEFORE, 300.0, "all")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "ok3", "main")
    assert line and "300.00" in line and "->" not in line, line


def test_after_with_no_before_prints_the_number_too(led_sm):
    """Previously omitted entirely. A reading with no anchor is still a reading."""
    led, sm = led_sm
    _write(led, "ok4", "main", led.KIND_EAGER, led.PHASE_AFTER, 88.0, "16")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "ok4", "main")
    assert line and "88.00" in line and "16 layers" in line, line


def test_a_junk_reading_prints_nothing(led_sm):
    """Degrading to a single number must not degrade to a single ZERO -- a 0.0 ms 'measurement' is a
    failed capture, and printing it as the current state is worse than printing nothing."""
    led, sm = led_sm
    _write(led, "ok5", "main", led.KIND_EAGER, led.PHASE_AFTER, 0.0, "16")
    assert sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "ok5", "main") is None


def test_mode_and_stage_mismatches_lose_the_arrow_too(led_sm):
    """comparable() guards three axes. Depth is the one that bit us; the fix must not be depth-only."""
    led, sm = led_sm
    _write(led, "m4", "main", led.KIND_EAGER, led.PHASE_BEFORE, 100.0, "16", mode="eager")
    _write(led, "m4", "main", led.KIND_EAGER, led.PHASE_AFTER, 50.0, "16", mode="tracy-trace")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "m4", "main")
    assert line and "->" not in line, line


# ---------------------------------------------------------------- the whole report


def _render(sm, model, task, tmp_path, **kw):
    """render_summary over an empty kernel log -- the attempt table is irrelevant here, the ledger
    lines are the subject."""
    klog = tmp_path / "kernels.json"
    klog.write_text("[]")
    kwargs = dict(model=model, task=task, finalized=True, metric="device_ms")
    kwargs.update(kw)
    return sm.render_summary(str(klog), **kwargs)


def test_the_rendered_report_has_no_eager_line_and_no_placeholder(led_sm, tmp_path):
    """'not measured (no ledger reading for this run)' is FALSE when the ledger holds two readings;
    the fallback must not fire in place of the omitted line and assert something untrue."""
    led, sm = led_sm
    _write(led, "gemma3", "main", led.KIND_EAGER, led.PHASE_BEFORE, 547.8951, "all")
    _write(led, "gemma3", "main", led.KIND_EAGER, led.PHASE_AFTER, 547.8010, "96")
    _write(led, "gemma3", "main", led.KIND_FULLPIPE, led.PHASE_BEFORE, 84.05, "all", mode="trace")
    _write(led, "gemma3", "main", led.KIND_FULLPIPE, led.PHASE_AFTER, 34.82, "all", mode="trace")
    text = _render(sm, "gemma3", "main", tmp_path)
    assert "NOT COMPARABLE" not in text
    assert "no ledger reading for this run" not in text, text
    # the eager line survives as ONE number; the fake pairing does not
    assert "547.80" in text and "547.89" not in text, text
    # the real result must survive untouched, arrow and all
    assert "84.05" in text and "34.82" in text, text


def test_an_all_uncomparable_report_still_renders(led_sm, tmp_path):
    """Omitting every line must not leave a crash or an empty file -- the report still has a title,
    the roofline and the attempt table to carry."""
    led, sm = led_sm
    for kind in (led.KIND_EAGER, led.KIND_FULLPIPE):
        _write(led, "m5", "main", kind, led.PHASE_BEFORE, 100.0, "2")
        _write(led, "m5", "main", kind, led.PHASE_AFTER, 50.0, "all")
    text = _render(sm, "m5", "main", tmp_path)
    assert text and "m5" in text
    assert "NOT COMPARABLE" not in text


def test_dropping_the_arrow_is_not_a_silent_data_loss(led_sm):
    """The rows stay in the ledger. The report declines to SUBTRACT them; it does not delete them,
    so the pairing is still diagnosable after the fact."""
    led, sm = led_sm
    _write(led, "m6", "main", led.KIND_EAGER, led.PHASE_BEFORE, 547.8951, "all")
    _write(led, "m6", "main", led.KIND_EAGER, led.PHASE_AFTER, 547.8010, "96")
    assert "->" not in (sm._ledger_line(led.KIND_EAGER, "e", "m6", "main") or "")
    rs = led.rows(led.KIND_EAGER, model="m6", task="main")
    assert len(rs) == 2 and {r["phase"] for r in rs} == {"before", "after"}


def test_environment_leaves_no_ledger_in_the_repo(led_sm):
    """This suite writes ledgers. None may land in the repo tree or in the bare shared tempdir --
    a stray perf_measurements_*.jsonl beside a live run has already been mistaken for real data."""
    led, _sm = led_sm
    p = led.ledger_path("m7", "main")
    assert "/tt-metal" not in str(p) or "pytest" in str(p), p
    assert not os.path.exists("/tmp/perf_measurements_m7_main.jsonl")
