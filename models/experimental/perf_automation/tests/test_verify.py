"""VERIFY (PLAN 8.5.1) — parse + import check, no key/hardware."""

import json

from agent import states
from agent.handlers.verify import verify
from agent.loop_context import LoopContext
from agent.run import Run


def _ctx(tmp_path, files, code_fix=0):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    run = Run.create(tmp_path / "runs", config={"config": {"model_root": str(model)}, "pathmap": {}}, run_id="V")
    run.state_path.write_text(
        json.dumps({"state": "VERIFY", "last_edit": {"files": files}, "code_fix_attempts": code_fix, "cost_usd": 0.0})
    )
    return LoopContext.from_run(run, index=[]), model


def test_verify_ok_on_clean_file(tmp_path):
    ctx, model = _ctx(tmp_path, ["m.py"])
    (model / "m.py").write_text("x = 1\n")
    assert verify(ctx) == states.GATE_PCC
    assert ctx.state["last_verdict"]["status"] == "ok"


def test_verify_parse_error_routes_to_repair(tmp_path):
    ctx, model = _ctx(tmp_path, ["m.py"])
    (model / "m.py").write_text("def (:\n")
    assert verify(ctx) == states.REPAIR_CODE
    assert ctx.state["last_verdict"]["status"] == "parse_error"


def test_verify_does_not_run_imports(tmp_path):
    # a bad import is valid SYNTAX -> VERIFY passes; GATE_PCC catches it at runtime.
    ctx, model = _ctx(tmp_path, ["m.py"])
    (model / "m.py").write_text("import a_module_that_does_not_exist_xyz123\n")
    assert verify(ctx) == states.GATE_PCC
    assert ctx.state["last_verdict"]["status"] == "ok"


def test_verify_reverts_when_code_budget_exhausted(tmp_path):
    ctx, model = _ctx(tmp_path, ["m.py"], code_fix=5)
    (model / "m.py").write_text("def (:\n")
    assert verify(ctx) == states.REVERT
    assert ctx.state["last_decision"]["reason"] == "edit_failed"
