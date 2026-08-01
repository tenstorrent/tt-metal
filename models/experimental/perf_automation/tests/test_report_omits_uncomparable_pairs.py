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
ignores the tail. summary.py already argues this case for a missing anchor -- "the value already
appears where it means something (the roofline 'measured'). Omit the line entirely" -- and an
incomparable pair has strictly less to say than a missing one.

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


def test_the_gemma3_pair_of_identical_baselines_renders_nothing(led_sm):
    """all vs 96: the same build profiled twice, 0.09 ms apart. There is no result here."""
    led, sm = led_sm
    _write(led, "gemma3", "main", led.KIND_EAGER, led.PHASE_BEFORE, 547.8951, "all")
    _write(led, "gemma3", "main", led.KIND_EAGER, led.PHASE_AFTER, 547.8010, "96")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "gemma3", "main")
    assert line is None, line


def test_the_two_layer_against_all_layer_pair_renders_nothing(led_sm):
    """2 vs all: reads as a 60% win, is a depth artifact. The dangerous one."""
    led, sm = led_sm
    _write(led, "m2", "main", led.KIND_EAGER, led.PHASE_BEFORE, 296.70, "2")
    _write(led, "m2", "main", led.KIND_EAGER, led.PHASE_AFTER, 117.40, "all")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "m2", "main")
    assert line is None, line


def test_no_rendered_line_anywhere_still_carries_the_disclaimer(led_sm):
    """The disclaimer existed because the line was printed. No line, no disclaimer -- and no path
    may reintroduce it by rendering the text and hoping the reader reaches the tail."""
    led, sm = led_sm
    for kind in (led.KIND_EAGER, led.KIND_TRACE_PASS, led.KIND_FULLPIPE):
        _write(led, "m3", "main", kind, led.PHASE_BEFORE, 100.0, "2")
        _write(led, "m3", "main", kind, led.PHASE_AFTER, 50.0, "all")
        assert sm._ledger_line(kind, kind, "m3", "main") is None, kind


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


def test_before_with_no_after_still_says_so(led_sm):
    """The normal state for most of a run, and a DIFFERENT case: one credible reading exists and the
    report should show it. Only the uncomparable PAIR disappears."""
    led, sm = led_sm
    _write(led, "ok3", "main", led.KIND_EAGER, led.PHASE_BEFORE, 300.0, "all")
    line = sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "ok3", "main")
    assert line is not None and "after not measured yet" in line, line


def test_mode_and_stage_mismatches_are_omitted_too(led_sm):
    """comparable() guards three axes. Depth is the one that bit us; the fix must not be depth-only."""
    led, sm = led_sm
    _write(led, "m4", "main", led.KIND_EAGER, led.PHASE_BEFORE, 100.0, "16", mode="eager")
    _write(led, "m4", "main", led.KIND_EAGER, led.PHASE_AFTER, 50.0, "16", mode="tracy-trace")
    assert sm._ledger_line(led.KIND_EAGER, "eager per-op device time", "m4", "main") is None


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
    assert "eager per-op device time" not in text, text
    assert "no ledger reading for this run" not in text, text
    # the real result must survive untouched
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


def test_omission_is_not_a_silent_data_loss(led_sm):
    """The rows stay in the ledger. The report declines to SUBTRACT them; it does not delete them,
    so the pairing is still diagnosable after the fact."""
    led, sm = led_sm
    _write(led, "m6", "main", led.KIND_EAGER, led.PHASE_BEFORE, 547.8951, "all")
    _write(led, "m6", "main", led.KIND_EAGER, led.PHASE_AFTER, 547.8010, "96")
    assert sm._ledger_line(led.KIND_EAGER, "e", "m6", "main") is None
    rs = led.rows(led.KIND_EAGER, model="m6", task="main")
    assert len(rs) == 2 and {r["phase"] for r in rs} == {"before", "after"}


def test_environment_leaves_no_ledger_in_the_repo(led_sm):
    """This suite writes ledgers. None may land in the repo tree or in the bare shared tempdir --
    a stray perf_measurements_*.jsonl beside a live run has already been mistaken for real data."""
    led, _sm = led_sm
    p = led.ledger_path("m7", "main")
    assert "/tt-metal" not in str(p) or "pytest" in str(p), p
    assert not os.path.exists("/tmp/perf_measurements_m7_main.jsonl")
