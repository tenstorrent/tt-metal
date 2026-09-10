"""GATE_PCC (PLAN 8.6) — parse_pcc + verdict routing (no hardware)."""

import json

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


# Verbatim pytest -sv output for a device fatal raised INSIDE the compare helper
# (assert_with_pcc calls ttnn.to_torch, which OOMs). No PCC is ever computed, yet the
# traceback echoes the THRESHOLD three ways: the `>` failing line, the def signature, and
# pytest's frame-argument header. Scraping any of them banked pcc=0.99 -> ok on a crash.
_CRASH_ECHOING_THRESHOLD = """t.py::test_e2e_pcc FAILED

=================================== FAILURES ===================================
_________________________________ test_e2e_pcc _________________________________

    def test_e2e_pcc():
>       assert_with_pcc([1.0], [1.0], pcc=0.99)

t.py:5:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

expected = [1.0], actual = [1.0], pcc = 0.99

    def assert_with_pcc(expected, actual, pcc=0.99):
>       raise RuntimeError("TT_FATAL @ allocator.cc:142: Out of Memory")
E       RuntimeError: TT_FATAL @ allocator.cc:142: Out of Memory

t.py:3: RuntimeError
=========================== short test summary info ============================
FAILED t.py::test_e2e_pcc - RuntimeError: TT_FATAL @ allocator.cc:142
1 failed in 0.03s
"""


def test_run_pcc_threshold_echoed_in_traceback_is_crash(monkeypatch, tmp_path):
    # The run computed NOTHING -- banking it as ok (pcc_verified=True) let the optimizer keep
    # an edit on a model that never ran. This is the boundary of the nonzero-exit carve-out
    # above: that carve-out may only fire on a REAL measured value, never on an echoed one.
    v = _patch_run(monkeypatch, tmp_path, _CRASH_ECHOING_THRESHOLD, 1)
    assert v["status"] == "crash", f"crash banked as {v['status']} with pcc={v.get('pcc')}"
    assert "TT_FATAL" in v["error"]


def test_parse_pcc_keeps_value_reported_in_assertion_message():
    # A genuine sub-threshold PCC often survives only in the assertion message under the
    # FAILURES banner. It must stay visible so the run routes to pcc_low (repairable) rather
    # than crash -- the traceback filter drops echoes, not reported values.
    out = (
        "=================================== FAILURES ===================================\n"
        "    def test_e2e_pcc():\n"
        ">       assert_with_pcc(golden, tt_out, pcc=0.99)\n"
        "E       AssertionError: PCC: 0.3120 is below required 0.99\n"
    )
    assert parse_pcc(out) == 0.3120


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
