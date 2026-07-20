"""GATE_PCC (PLAN 8.6) — parse_pcc + verdict routing (no hardware)."""

import json

from agent import states
from agent.handlers.gate_pcc import gate_pcc
from agent.loop_context import LoopContext
from agent.pcc_runner import parse_pcc
from agent.run import Run


def test_parse_pcc_extracts_value():
    assert parse_pcc("... PCC: 0.9987 ...") == 0.9987
    assert parse_pcc("assert pcc=0.42 failed") == 0.42
    assert parse_pcc("no number here") is None


# --- run_pcc HONORS the pytest verdict, not just the scraped float (hole ③) ---------------
class _FakeCtx:
    def __init__(self, tmp_path):
        self._root = tmp_path
        self.manifest = {
            "pathmap": {"pcc": {"end_to_end": {"path": "t.py", "threshold": 0.95}}},
            "config": {},
        }

    def model_root(self):
        return self._root


def _patch_run(monkeypatch, tmp_path, stdout, returncode):
    import subprocess

    from agent import gitio, pcc_runner

    monkeypatch.setattr(gitio, "repo_root", lambda p: tmp_path)

    class _R:
        def __init__(self):
            self.stdout, self.stderr, self.returncode = stdout, "", returncode

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    return pcc_runner.run_pcc(_FakeCtx(tmp_path))


def test_run_pcc_ok_when_passed(monkeypatch, tmp_path):
    v = _patch_run(monkeypatch, tmp_path, "e2e PCC=0.999\n1 passed", 0)
    assert v["status"] == "ok" and v["pcc"] == 0.999


def test_run_pcc_high_pcc_nonzero_exit_is_ok(monkeypatch, tmp_path):
    # PCC>=threshold but pytest exited non-zero on a BRING-UP gate (Gate-2 modules-invoked)
    # or nanobind teardown -> NOT an edit-induced regression (fails on the baseline too).
    # The perf loop's correctness signal is PCC, so this must be ok, not crash.
    v = _patch_run(monkeypatch, tmp_path, "e2e PCC=0.999\nGate 2 failed: modules not invoked\n1 failed", 1)
    assert v["status"] == "ok" and v["pcc"] == 0.999


def test_run_pcc_below_threshold_is_pcc_low(monkeypatch, tmp_path):
    v = _patch_run(monkeypatch, tmp_path, "e2e PCC=0.40\n1 failed", 1)
    assert v["status"] == "pcc_low" and v["pcc"] == 0.40


def test_run_pcc_no_pcc_is_crash_with_nanobind_filtered(monkeypatch, tmp_path):
    # Test died before producing PCC -> crash; the nanobind teardown spam must be filtered
    # out of the excerpt so the real error (TT_FATAL) survives for the repair agent.
    noise = "\n".join(["nanobind: leaked 261 functions!"] + ['leaked type "X"'] * 80)
    v = _patch_run(monkeypatch, tmp_path, "TT_FATAL: bad shard spec\n" + noise, 1)
    assert v["status"] == "crash" and "TT_FATAL" in v["error"] and "nanobind" not in v["error"]


def test_run_pcc_skipped_is_crash(monkeypatch, tmp_path):
    # A SKIPPED test verified nothing -- even if a stale "pcc 0.99" string is in the log.
    v = _patch_run(monkeypatch, tmp_path, "reference pcc 0.99 baseline\n1 skipped", 0)
    assert v["status"] == "crash" and "SKIPPED" in v["error"]


def test_get_edit_model_ladder():
    from agent.config import get_edit_model

    assert "haiku" in get_edit_model(0)  # APPLY
    assert "sonnet" in get_edit_model(1)  # repair 1
    assert "opus" in get_edit_model(2)  # repair 2
    assert "opus" in get_edit_model(5)  # capped at top rung


def _ctx(tmp_path, code_fix=0, pcc_fix=0):
    run = Run.create(tmp_path / "runs", config={"config": {}, "pathmap": {}}, run_id="G")
    run.state_path.write_text(
        json.dumps({"state": "GATE_PCC", "code_fix_attempts": code_fix, "pcc_fix_attempts": pcc_fix, "cost_usd": 0.0})
    )
    return LoopContext.from_run(run, index=[])


def test_gate_ok_goes_to_remeasure(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.deps["pcc_runner"] = lambda c: {"status": "ok", "pcc": 0.999}
    assert gate_pcc(ctx) == states.REMEASURE


def test_gate_pcc_low_routes_to_repair_pcc(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.deps["pcc_runner"] = lambda c: {"status": "pcc_low", "pcc": 0.5}
    assert gate_pcc(ctx) == states.REPAIR_PCC


def test_gate_pcc_low_exhausted_discards(tmp_path):
    ctx = _ctx(tmp_path, pcc_fix=2)
    ctx.deps["pcc_runner"] = lambda c: {"status": "pcc_low", "pcc": 0.5}
    assert gate_pcc(ctx) == states.REVERT
    assert ctx.state["last_decision"]["reason"] == "pcc_failed"


def test_gate_crash_routes_to_repair_code(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.deps["pcc_runner"] = lambda c: {"status": "crash", "error": "boom"}
    assert gate_pcc(ctx) == states.REPAIR_CODE


def test_gate_crash_exhausted_discards(tmp_path):
    ctx = _ctx(tmp_path, code_fix=5)
    ctx.deps["pcc_runner"] = lambda c: {"status": "crash"}
    assert gate_pcc(ctx) == states.REVERT
    assert ctx.state["last_decision"]["reason"] == "edit_failed"
