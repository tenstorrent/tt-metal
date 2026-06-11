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
