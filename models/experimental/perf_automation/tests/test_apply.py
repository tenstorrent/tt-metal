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

    def fake_editor(*, lever, section, model_files):
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
