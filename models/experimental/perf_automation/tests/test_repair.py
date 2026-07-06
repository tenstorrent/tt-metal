"""REPAIR_CODE / REPAIR_PCC handlers (PLAN 8.5.2) — real self-heal leaves.

The control flow (budget check -> route here, bump counter, -> VERIFY) is the
design; these tests pin the leaf behavior: the editor is re-invoked WITH the
error, the right counter increments, telemetry is recorded, and an editor
exception never crashes the loop.
"""

import json

from agent import states
from agent.handlers.repair_code import repair_code
from agent.handlers.repair_pcc import repair_pcc
from agent.loop_context import LoopContext
from agent.run import Run


def _run(tmp_path, state):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.py").write_text("x = 1\n")
    run = Run.create(
        tmp_path / "runs",
        config={
            "config": {"model_root": str(model)},
            "env": {},
            "pathmap": {
                "model_files": ["model.py"],
                "pcc": {"end_to_end": {"threshold": 0.94}},
            },
        },
        run_id="R",
    )
    base = {"state": "REPAIR_CODE", "selected_lever": "lvr", "iteration": 0, "cost_usd": 0.0}
    base.update(state)
    run.state_path.write_text(json.dumps(base))
    return run


def _ctx(tmp_path, state, editor):
    ctx = LoopContext.from_run(_run(tmp_path, state), index=[])
    ctx.deps["edit_runner"] = editor
    return ctx


# ----------------------------- REPAIR_CODE ---------------------------------
def test_repair_code_passes_error_and_bumps_counter(tmp_path):
    seen = {}

    def editor(*, lever, section, model_files, error=None, spec=None, cwd=None, **kwargs):
        seen["error"], seen["lever"], seen["cwd"] = error, lever, cwd
        return {
            "files": ["model.py"],
            "summary": "fixed",
            "model": "haiku",
            "usage": {"cost_usd": 0.02, "tokens_in": 9, "tokens_out": 3},
        }

    ctx = _ctx(
        tmp_path,
        {"last_verdict": {"status": "parse_error", "error": "bad syntax", "file": "model.py"}, "code_fix_attempts": 0},
        editor,
    )
    assert repair_code(ctx) == states.VERIFY
    assert "bad syntax" in seen["error"] and "model.py" in seen["error"]  # error threaded in
    assert seen["lever"] == "lvr"
    assert seen["cwd"].endswith("/model")  # scoped to model dir
    assert ctx.state["code_fix_attempts"] == 1  # counter bumped
    assert ctx.state.get("pcc_fix_attempts", 0) == 0  # code repair does not touch pcc counter
    assert ctx.state["cost_usd"] == 0.02  # telemetry accumulated


def test_repair_code_survives_editor_exception(tmp_path):
    def editor(**k):
        raise RuntimeError("SDK exploded")

    ctx = _ctx(tmp_path, {"last_verdict": {"status": "edit_failed"}, "code_fix_attempts": 2}, editor)
    assert repair_code(ctx) == states.VERIFY  # loop survives
    assert ctx.state["code_fix_attempts"] == 3  # still counts the attempt


# ----------------------------- REPAIR_PCC ----------------------------------
def test_repair_pcc_passes_pcc_detail_and_bumps_pcc_counter(tmp_path):
    seen = {}

    def editor(*, lever, section, model_files, error=None, spec=None, cwd=None, **kwargs):
        seen["error"] = error
        return {
            "files": ["model.py"],
            "summary": "conservative",
            "model": "haiku",
            "usage": {"cost_usd": 0.01, "tokens_in": 4, "tokens_out": 1},
        }

    ctx = _ctx(tmp_path, {"last_verdict": {"status": "pcc_low", "pcc": 0.81}, "pcc_fix_attempts": 0}, editor)
    assert repair_pcc(ctx) == states.VERIFY
    assert "0.81" in seen["error"] and "0.94" in seen["error"]  # measured + threshold
    assert "conservativ" in seen["error"].lower()  # nudge toward conservative re-apply
    assert ctx.state["pcc_fix_attempts"] == 1  # PCC counter bumped
    assert ctx.state.get("code_fix_attempts", 0) == 0  # code counter untouched


def test_repair_pcc_survives_editor_exception(tmp_path):
    def editor(**k):
        raise RuntimeError("boom")

    ctx = _ctx(tmp_path, {"last_verdict": {"status": "pcc_low", "pcc": 0.5}, "pcc_fix_attempts": 1}, editor)
    assert repair_pcc(ctx) == states.VERIFY
    assert ctx.state["pcc_fix_attempts"] == 2
