"""Each stage keeps the time it started from, so its own first win stays visible.

Qwen-Image-Edit (2026-09-30): the whole-model BEFORE (101949 ms) was pinned, but its per-stage split
lived only in the gate's best-so-far file, which every win overwrites. vision_encode's big win came
first (~63.8 s -> 22.8 s), so no record of its start survived: its history line was flat and every
stage card read baseline == current. The BEFORE reading now pins each stage's share, write-once, and
only when that reading is the run's BEFORE -- a resumed run never files a mid-run split as its start.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

_PA = Path(__file__).resolve().parents[1]


@pytest.fixture()
def run_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_MODEL_NAME", "some_model")
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    import models.experimental.perf_automation.cc_optimize.run as R

    importlib.reload(R)
    return R


def test_the_gate_split_is_read_in_both_shapes(run_mod):
    assert run_mod._stage_ms_of('{"a": 12.5, "b": {"ms": 3.0}, "c": {"best": 1}, "d": 0}') == {"a": 12.5, "b": 3.0}
    assert run_mod._stage_ms_of("not json") == {}
    assert run_mod._stage_ms_of("[]") == {}


def test_the_split_is_pinned_once_with_the_before_reading(run_mod):
    led = run_mod._ledger()
    assert run_mod._ledger_fullpipe(100.0, "trace", "BEFORE") == led.PHASE_BEFORE
    run_mod._ledger_stage_starts({"Alpha": 60.0, "beta": 40.0}, "trace")
    run_mod._ledger_stage_starts({"alpha": 20.0}, "trace")  # a later split never overwrites the start
    got = {r["depth"]: r["value_ms"] for r in led.rows(led.KIND_STAGE_E2E, led.PHASE_BEFORE, "some_model", "main")}
    assert got == {"alpha": 60.0, "beta": 40.0}
    assert run_mod._ledger_fullpipe(80.0, "trace", "committed-best") == led.PHASE_AFTER
    assert run_mod._ledger_fullpipe(90.0, "trace", "BEFORE") == "", "a rerun's BEFORE is not filed"


def test_only_the_reading_that_became_the_before_pins_a_split():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert "if _ledger_fullpipe(ms, mode, label) == _ledger().PHASE_BEFORE:" in src
    assert "_ledger_stage_starts(stages, mode)" in src
    assert "print('FULLPIPE_STAGES=' + json.dumps(r.get('stages') or {}))" in src


def test_the_dashboard_starts_each_stage_from_its_pin():
    from scripts.tt_hw_planner._optimize_dashboard_page import PAGE_HTML
    from scripts.tt_hw_planner.optimize_dashboard import _stage_starts

    stages = [{"name": "alpha", "ms": 22.0, "baseline_ms": 22.0}, {"name": "gamma", "ms": 5.0, "baseline_ms": 5.0}]
    ledger = {"stage_e2e": [{"phase": "before", "depth": "alpha", "value_ms": 64.0}]}
    _stage_starts(stages, ledger)
    assert stages[0] == {"name": "alpha", "ms": 22.0, "baseline_ms": 64.0, "start_ms": 64.0}
    assert stages[1] == {"name": "gamma", "ms": 5.0, "baseline_ms": 5.0}, "no pin: unchanged"
    assert "let runMin = (!ser.best && start != null) ? start : null;" in PAGE_HTML
    assert "line.push([X(0), Y(yOf(start))]);" in PAGE_HTML
