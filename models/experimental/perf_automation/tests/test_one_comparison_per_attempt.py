# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One end-to-end comparison per attempt, made once, and everything reads THAT.

The rule, stated the way it should always have worked:

    delta = this attempt's OWN end-to-end  -  the running best
    delta < 0  -> win, best moves      delta >= 0  -> no gain, best unchanged

    best 88 -> grid 87 (-1, win, best:=87) -> X 87 (0, no gain) -> dtype 86 (-1, win, best:=86)
             -> shard 88 (+2, no gain, best stays 86)

Three implementations used to answer this and they disagreed: gate_set_new_best banked, the report's
Δ column subtracted fullpipe_ms - fullpipe_best_ms, and winning_indices re-derived a staircase. On
gemma-3-12b-it that produced 16 raw beat_baseline flags from 14 measurements, 3 rendered ticks, and 2
improvements that actually happened -- with "-41.77 ms" printed beside a dozen rows marked "no gain".

The repetition had a cause: an end-to-end measurement is EXPENSIVE (a full trace replay), so it does
not run per attempt. Attempts that never triggered one re-read the last verdict and inherited its
numbers as their own. A verdict is now attributed ONCE -- the attempt that caused it owns it; later
attempts report `own=False`, carry no delta and no win, and the report prints "n/m" rather than
borrowing someone else's result.

  a1  the worked example above, end to end through the recorder
  a2  a verdict is attributed exactly once
  a3  the three consumers now agree on every row
  a4  the report renders the stamped delta, and distinguishes "no reference" from "not measured"
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _mods(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "gemma3"))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "gemma3").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    out = []
    for name, path in (("pmcp_one", _CC / "perf_mcp.py"), ("meas_one", _CC / "measurements.py")):
        spec = importlib.util.spec_from_file_location(name, str(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out.append(m)
    return out


def _set_verdict(pm, monkeypatch, ms, best=None, bump=[0]):  # noqa: B006
    """Record a NEW end-to-end verdict, with a distinct identity each time.

    measurement_id is what makes it distinct now. It used to be the verdict file's mtime, which moved
    on every recorded verdict rather than on every MEASUREMENT -- so an attempt that ran no trace
    replay could claim one that had (14 times, on gemma-3-12b-it). The id is minted only by
    check_full_pipeline_latency, so a fresh one here stands for a fresh measurement, exactly as a
    fresh mtime used to be assumed to."""
    fp = {
        "status": "ok",
        "full_pipeline_ms": ms,
        "sha": "sha%d" % bump[0],
        "measurement_id": "meas-%d" % bump[0],
    }
    if best is not None:
        fp["best_ms"] = best
    bump[0] += 1
    monkeypatch.setattr(pm, "gate_verdicts", lambda: {"full_pipeline": fp})
    return fp


# --------------------------------------------------------------------------- a1 THE RULE
def test_a1_the_worked_example(monkeypatch, tmp_path):
    """88 -> 87 win -> 87 no gain -> 86 win -> 88 no gain, best never moves backwards."""
    pm, led = _mods(monkeypatch, tmp_path)
    led.record(led.KIND_FULLPIPE, led.PHASE_BEFORE, 88.0, depth="all", mode="trace+1cq", model="gemma3", task="main")

    seq = [(87.0, None, True, -1.0), (87.0, 87.0, False, 0.0), (86.0, 87.0, True, -1.0), (88.0, 86.0, False, 2.0)]
    for ms, best, want_win, want_delta in seq:
        _set_verdict(pm, monkeypatch, ms, best)
        v = pm._attempt_fullpipe_verdict()
        assert v["own"] is True
        assert v["delta"] == pytest.approx(want_delta), f"{ms} vs best {best}"
        assert v["win"] is want_win, f"{ms} vs best {best}"


def test_a1_first_measurement_at_the_baseline_is_no_gain(monkeypatch, tmp_path):
    """The gemma3 case: 87.9294 measured against a baseline of 87.9294 -> 0.00, not a win."""
    pm, led = _mods(monkeypatch, tmp_path)
    led.record(led.KIND_FULLPIPE, led.PHASE_BEFORE, 87.9294, depth="all", mode="trace+1cq", model="gemma3", task="main")
    _set_verdict(pm, monkeypatch, 87.9294)
    v = pm._attempt_fullpipe_verdict()
    assert v["delta"] == pytest.approx(0.0)
    assert v["win"] is False


# --------------------------------------------------------------------------- a2 ATTRIBUTED ONCE
def test_a2_a_verdict_belongs_to_one_attempt_only(monkeypatch, tmp_path):
    """The cause of the repeated numbers: later attempts must NOT inherit this measurement."""
    pm, led = _mods(monkeypatch, tmp_path)
    led.record(led.KIND_FULLPIPE, led.PHASE_BEFORE, 88.0, depth="all", mode="trace+1cq", model="gemma3", task="main")
    _set_verdict(pm, monkeypatch, 46.0)

    first = pm._attempt_fullpipe_verdict()
    assert first["own"] is True and first["win"] is True and first["delta"] == pytest.approx(-42.0)

    for _ in range(5):
        later = pm._attempt_fullpipe_verdict()
        assert later["own"] is False, "a second attempt claimed the same measurement"
        assert later["delta"] is None and later["ms"] is None
        assert later["win"] is False, "inherited a win it did not measure"


def test_a2_a_genuinely_new_measurement_is_owned_again(monkeypatch, tmp_path):
    pm, led = _mods(monkeypatch, tmp_path)
    led.record(led.KIND_FULLPIPE, led.PHASE_BEFORE, 88.0, depth="all", mode="trace+1cq", model="gemma3", task="main")
    _set_verdict(pm, monkeypatch, 46.0)
    assert pm._attempt_fullpipe_verdict()["own"] is True
    assert pm._attempt_fullpipe_verdict()["own"] is False
    _set_verdict(pm, monkeypatch, 45.0, best=46.0)  # a new run of the gate
    v = pm._attempt_fullpipe_verdict()
    assert v["own"] is True and v["win"] is True and v["delta"] == pytest.approx(-1.0)


def test_a2_the_same_ms_after_a_revert_is_still_a_new_measurement(monkeypatch, tmp_path):
    """A revert legitimately re-measures the same number; identity must not collapse on value."""
    pm, led = _mods(monkeypatch, tmp_path)
    led.record(led.KIND_FULLPIPE, led.PHASE_BEFORE, 88.0, depth="all", mode="trace+1cq", model="gemma3", task="main")
    _set_verdict(pm, monkeypatch, 46.0)
    assert pm._attempt_fullpipe_verdict()["own"] is True
    _set_verdict(pm, monkeypatch, 46.0, best=46.0)  # same ms, different verdict
    assert pm._attempt_fullpipe_verdict()["own"] is True


# --------------------------------------------------------------------------- a3 ONE ANSWER
def test_a3_the_consumers_agree(monkeypatch, tmp_path):
    """gate / stamped flag / winning_indices must give the SAME verdict for the same row."""
    pm, led = _mods(monkeypatch, tmp_path)
    led.record(led.KIND_FULLPIPE, led.PHASE_BEFORE, 88.0, depth="all", mode="trace+1cq", model="gemma3", task="main")

    rows, expect = [], []
    for ms, best, want in [(87.0, None, True), (87.0, 87.0, False), (86.0, 87.0, True), (88.0, 86.0, False)]:
        _set_verdict(pm, monkeypatch, ms, best)
        gate = pm.gate_set_new_best()
        v = pm._attempt_fullpipe_verdict()
        assert gate is want and v["win"] is want, f"gate and stamp disagree at {ms}"
        rows.append({"measured_ms": 1.0, "beat_baseline": v["win"], "fullpipe_delta_ms": v["delta"]})
        expect.append(want)

    wins = led.winning_indices(rows)
    assert wins == {i for i, w in enumerate(expect) if w}, "winning_indices disagrees with the stamps"


def test_a3_unstamped_logs_still_use_the_staircase(monkeypatch, tmp_path):
    """Backward compatibility: logs written before the stamp existed must still render ticks."""
    _pm, led = _mods(monkeypatch, tmp_path)
    old = [
        {"measured_ms": 10.0, "beat_baseline": True, "fullpipe_ms": 80.0, "fullpipe_best_ms": 88.0},
        {"measured_ms": 10.0, "beat_baseline": True, "fullpipe_ms": 90.0, "fullpipe_best_ms": 80.0},
    ]
    assert led.winning_indices(old) == {0}


# --------------------------------------------------------------------------- a4 RENDERING
def _render(summary, rows):
    return summary.render_summary(rows, 100.0, model="gemma3", task="main", finalized=True)


def test_a4_report_prints_the_stamped_delta_and_marks_unmeasured(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("summary_one", str(_CC / "summary.py"))
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    log = tmp_path / "k.json"
    log.write_text(
        json.dumps(
            [
                {
                    "op_signature": "Matmul A",
                    "kernel_kind": "grid",
                    "measured_ms": 17.07,
                    "beat_baseline": True,
                    "fullpipe_delta_ms": -41.77,
                    "fullpipe_measured_here": True,
                    "note": "n",
                },
                {
                    "op_signature": "Matmul A",
                    "kernel_kind": "dtype",
                    "measured_ms": 17.07,
                    "beat_baseline": False,
                    "fullpipe_measured_here": False,
                    "note": "n",
                },
            ]
        )
    )
    text = _render(summary, log)
    assert "-41.77 ms" in text, "the stamped delta is not rendered"
    # An attempt with no end-to-end of its own no longer gets an "n/m" row -- it is dropped from the
    # detail table and counted in the footer instead. What this guards is unchanged: it must not
    # silently vanish, and it must not inherit the other row's delta.
    assert "omitted" in text, "an attempt with no end-to-end of its own is not accounted for"
    # the -41.77 must appear ONCE, not copied onto the second row
    assert text.count("-41.77 ms") == 1, "a delta was reused on a row that measured nothing"
