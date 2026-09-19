# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A win must BEAT something. The first measurement of a run is not automatically one.

gate_set_new_best asked:

    return ms > 0 and (prev is None or ms < float(prev))
                       ^^^^^^^^^^^^

`best_ms` does not exist until something has already won, so the FIRST full-pipeline measurement of
every run was credited unconditionally -- the one case the function's own docstring ("did it
actually RATCHET the end-to-end best down") exists to reject.

Observed on gemma-3-12b-it, 2026-07-31:

    #0  fullpipe 87.9294  best None  -> beat_baseline True   <- the BASELINE, improved nothing
    #7  fullpipe 46.1561  best 87.93 -> beat_baseline False  <- the attempt that actually won

The baseline was never missing. The ledger holds `fullpipe_e2e / before` = 87.9294 from the pre-loop
measurement; the gate simply never read it. Now it does, and with no reference at all it fails
CLOSED -- a fabricated win is the one outcome no later measurement can undo.

  r1  the exact gemma3 numbers
  r2  the reference falls back to the ledger baseline, and banked best still wins over it
  r3  fail closed when there is no reference anywhere
  r4  unchanged behaviour once a best exists
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _pm(monkeypatch, tmp_path, model="gemma3"):
    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / model))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    (tmp_path / model).mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("pmcp_win_ref", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _verdict(m, monkeypatch, ms, best):
    fp = {"status": "ok", "full_pipeline_ms": ms}
    if best is not None:
        fp["best_ms"] = best
    monkeypatch.setattr(m, "gate_verdicts", lambda: {"full_pipeline": fp})


def _baseline(m, value, model="gemma3"):
    led = m._ledger()
    led.record(led.KIND_FULLPIPE, led.PHASE_BEFORE, value, depth="all", mode="trace+1cq", model=model, task="main")


# --------------------------------------------------------------------------- r1 THE CASE
def test_r1_the_first_measurement_at_the_baseline_is_not_a_win(monkeypatch, tmp_path):
    """THE defect, with the run's real numbers."""
    m = _pm(monkeypatch, tmp_path)
    _baseline(m, 87.9294)
    _verdict(m, monkeypatch, 87.9294, None)
    assert m.gate_set_new_best() is False, "credited a win for measuring the baseline back"


def test_r1_the_attempt_that_actually_moved_it_is_a_win(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    _baseline(m, 87.9294)
    _verdict(m, monkeypatch, 46.1561, None)
    assert m.gate_set_new_best() is True


def test_r1_slower_than_the_baseline_is_not_a_win(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    _baseline(m, 87.9294)
    _verdict(m, monkeypatch, 92.0, None)
    assert m.gate_set_new_best() is False


# --------------------------------------------------------------------------- r2 REFERENCE
def test_r2_reference_is_the_ledger_baseline_when_nothing_is_banked(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    _baseline(m, 87.9294)
    assert m._fullpipe_reference_ms({}) == pytest.approx(87.9294)


def test_r2_a_banked_best_outranks_the_baseline(monkeypatch, tmp_path):
    """Once something has won, THAT is the bar -- not the original baseline."""
    m = _pm(monkeypatch, tmp_path)
    _baseline(m, 87.9294)
    assert m._fullpipe_reference_ms({"best_ms": 46.1561}) == pytest.approx(46.1561)
    _verdict(m, monkeypatch, 47.0, 46.1561)
    assert m.gate_set_new_best() is False, "47.0 beats the baseline but not the banked best"


def test_r2_the_baseline_is_read_per_model(monkeypatch, tmp_path):
    """A foreign model's baseline must not become this one's bar.

    PERF_MCP_LEDGER (the suite's autouse isolation) pins ONE file and so bypasses (model, task)
    keying entirely; drop it here and keep only the DIRECTORY redirect, so the keyed filenames are
    what separates the two models -- which is the thing under test.
    """
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    m = _pm(monkeypatch, tmp_path, model="gemma3")
    _baseline(m, 87.9294, model="other_model")
    assert m._fullpipe_reference_ms({}) is None, "read another model's baseline as this run's bar"
    _baseline(m, 87.9294, model="gemma3")
    assert m._fullpipe_reference_ms({}) == pytest.approx(87.9294), "own baseline not found once written"


# --------------------------------------------------------------------------- r3 FAIL CLOSED
def test_r3_no_reference_anywhere_is_not_a_win(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    _verdict(m, monkeypatch, 46.1561, None)
    assert m.gate_set_new_best() is False, "a win was invented with nothing to compare against"


@pytest.mark.parametrize("bad", [{"best_ms": "x"}, {"best_ms": None}])
def test_r3_unparseable_reference_is_not_a_win(monkeypatch, tmp_path, bad):
    m = _pm(monkeypatch, tmp_path)
    fp = dict(bad, status="ok", full_pipeline_ms=46.1561)
    monkeypatch.setattr(m, "gate_verdicts", lambda: {"full_pipeline": fp})
    assert m.gate_set_new_best() is False


# --------------------------------------------------------------------------- r4 UNCHANGED
@pytest.mark.parametrize(
    "ms,best,want",
    [(46.0, 87.93, True), (87.93, 87.93, False), (88.0, 87.93, False), (45.0, 46.0, True)],
)
def test_r4_with_a_banked_best_behaviour_is_unchanged(monkeypatch, tmp_path, ms, best, want):
    m = _pm(monkeypatch, tmp_path)
    _verdict(m, monkeypatch, ms, best)
    assert m.gate_set_new_best() is want


def test_r4_a_failed_gate_is_never_a_win(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    _baseline(m, 87.9294)
    monkeypatch.setattr(m, "gate_verdicts", lambda: {"full_pipeline": {"status": "crash", "full_pipeline_ms": 1.0}})
    assert m.gate_set_new_best() is False


def test_r4_zero_or_negative_is_never_a_win(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path)
    _baseline(m, 87.9294)
    for bad in (0.0, -1.0):
        _verdict(m, monkeypatch, bad, None)
        assert m.gate_set_new_best() is False


def test_r4_the_auto_pass_clause_is_gone():
    """Source guard: the shape of the bug, not one spelling of it."""
    src = (_CC / "perf_mcp.py").read_text()
    i = src.index("def gate_set_new_best")
    body = src[i : src.index("\ndef ", i + 10)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "prev is None or" not in code, "the unconditional first-measurement win is back"
    assert "_fullpipe_reference_ms" in code, "the gate no longer resolves a reference to beat"
