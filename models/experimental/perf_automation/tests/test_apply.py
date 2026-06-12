"""APPLY handler (PLAN 8.5) — records clean SHA + calls the injected editor."""

import json
import subprocess

from agent import states
from agent.handlers.apply import apply
from agent.loop_context import LoopContext
from agent.run import Run


def _model_repo(d):
    m = d / "model"
    m.mkdir()
    (m / "model.py").write_text("x = 1\n")
    for cfg in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *cfg], cwd=m, check=True)
    subprocess.run(["git", "add", "."], cwd=m, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=m, check=True)
    return m


def _run(tmp_path):
    model = _model_repo(tmp_path)
    run = Run.create(
        tmp_path / "runs",
        config={"config": {"model_root": str(model)}, "env": {}, "pathmap": {"model_files": ["model.py"]}},
        run_id="R",
    )
    run.state_path.write_text(
        json.dumps({"state": "APPLY", "selected_lever": "some-lever", "iteration": 0, "cost_usd": 0.0})
    )
    return run


def test_apply_records_clean_sha_and_calls_editor(tmp_path):
    run = _run(tmp_path)
    ctx = LoopContext.from_run(run, index=[])
    seen = {}

    def fake_editor(*, lever, section, model_files, spec=None, cwd=None):
        seen["lever"] = lever
        seen["files"] = [str(p) for p in model_files]
        return {
            "files": ["model.py"],
            "summary": "applied",
            "model": "mock",
            "usage": {"cost_usd": 0.01, "tokens_in": 5, "tokens_out": 2},
        }

    ctx.deps["edit_runner"] = fake_editor
    nxt = apply(ctx)

    assert nxt == states.VERIFY
    assert len(ctx.state["git_sha_clean"]) == 40  # clean SHA captured before edit
    assert seen["lever"] == "some-lever"  # lever passed to editor
    assert seen["files"][0].endswith("model/model.py")  # absolute model file path
    assert ctx.state["last_edit"]["files"] == ["model.py"]  # result recorded
    assert ctx.state["cost_usd"] == 0.01  # telemetry accumulated


def test_apply_proceeds_when_editor_errors_but_edits_landed(tmp_path):
    """The tmux failure mode: editor edits on disk then its final JSON is bad.
    git-diff ground truth catches the change, so APPLY proceeds instead of crashing."""
    from pathlib import Path

    run = _run(tmp_path)
    ctx = LoopContext.from_run(run, index=[])

    def editor_edits_then_raises(*, lever, section, model_files, spec=None, cwd=None):
        Path(model_files[0]).write_text("y = 2\n")  # real on-disk edit
        raise RuntimeError("edit result.files must be a non-empty array")

    ctx.deps["edit_runner"] = editor_edits_then_raises
    assert apply(ctx) == states.VERIFY
    assert "model.py" in ctx.state["last_edit"]["files"]


def test_apply_repairs_when_no_edit_lands(tmp_path):
    run = _run(tmp_path)
    ctx = LoopContext.from_run(run, index=[])
    ctx.deps["edit_runner"] = lambda **k: {"files": [], "summary": "", "model": "m", "usage": None}
    assert apply(ctx) == states.REPAIR_CODE
    assert ctx.state["last_verdict"]["status"] == "edit_failed"


def test_apply_reverts_when_no_edit_and_budget_exhausted(tmp_path):
    run = _run(tmp_path)
    s = json.loads(run.state_path.read_text())
    s["code_fix_attempts"] = 5
    run.state_path.write_text(json.dumps(s))
    ctx = LoopContext.from_run(run, index=[])
    ctx.deps["edit_runner"] = lambda **k: {"files": [], "model": "m", "usage": None}
    assert apply(ctx) == states.REVERT
    assert ctx.state["last_decision"]["reason"] == "edit_failed"


# --- deterministic content-anchored patch path (PLAN edits) ----------------
def test_apply_patch_deterministic_no_llm(tmp_path):
    """PLAN edits apply via plain string-replace — the editor LLM is never called."""
    from pathlib import Path

    run = _run(tmp_path)
    s = json.loads(run.state_path.read_text())
    s["edit_spec"] = {"summary": "tweak", "edits": [{"file": "model.py", "old_string": "x = 1", "new_string": "x = 2"}]}
    run.state_path.write_text(json.dumps(s))
    ctx = LoopContext.from_run(run, index=[])
    called = {"n": 0}

    def editor(**k):
        called["n"] += 1
        return {"files": [], "model": "m", "usage": None}

    ctx.deps["edit_runner"] = editor
    assert apply(ctx) == states.VERIFY
    assert called["n"] == 0  # no LLM editor invoked on the fast path
    assert "model.py" in ctx.state["last_edit"]["files"]
    assert (Path(ctx.model_root()) / "model.py").read_text() == "x = 2\n"


def test_apply_patch_miss_routes_to_repair(tmp_path):
    """A stale/missing anchor fails loudly -> REPAIR_CODE (self-heal)."""
    run = _run(tmp_path)
    s = json.loads(run.state_path.read_text())
    s["edit_spec"] = {"summary": "t", "edits": [{"file": "model.py", "old_string": "NOPE", "new_string": "x = 2"}]}
    run.state_path.write_text(json.dumps(s))
    ctx = LoopContext.from_run(run, index=[])
    assert apply(ctx) == states.REPAIR_CODE
    assert ctx.state["last_verdict"]["status"] == "patch_failed"
