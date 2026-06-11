"""PLAN agent (PLAN 8.x) — prompt, spec validation, handler (spec / NOOP / fallback)."""

import json

import pytest

from agent import states
from agent.handlers.plan import plan
from agent.loop_context import LoopContext
from agent.plan_agent import PlanError, _validate_spec, build_plan_prompt
from agent.run import Run


def test_plan_prompt_has_lever_section_skeleton():
    p = build_plan_prompt("mlp-fused-activation", "SECTION TEXT", "## mlp.py")
    assert "mlp-fused-activation" in p and "SECTION TEXT" in p and "mlp.py" in p and "JSON" in p


def test_validate_spec_ok():
    s = _validate_spec('{"file": "mlp.py", "location": "M.forward", "change": "fuse gelu"}')
    assert s["file"] == "mlp.py" and s["change"] == "fuse gelu"


def test_validate_spec_rejects_missing_change():
    with pytest.raises(PlanError):
        _validate_spec('{"file": "mlp.py"}')


def test_validate_spec_rejects_non_json():
    with pytest.raises(PlanError):
        _validate_spec("nope")


def _ctx(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "mlp.py").write_text("import ttnn\nclass M:\n    def forward(self, x):\n        return ttnn.linear(x)\n")
    run = Run.create(
        tmp_path / "runs",
        config={"config": {"model_root": str(model)}, "pathmap": {"model_files": ["mlp.py"]}},
        run_id="P",
    )
    run.state_path.write_text(
        json.dumps(
            {"state": "PLAN", "selected_lever": "mlp-fused-activation", "current_bucket": "matmul", "cost_usd": 0.0}
        )
    )
    return LoopContext.from_run(run, index=[])


def test_plan_produces_spec_and_goes_to_apply(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.deps["plan_runner"] = lambda **k: {
        "file": "mlp.py",
        "location": "M.forward",
        "change": "fuse gelu",
        "model": "m",
        "usage": {"cost_usd": 0.03, "tokens_in": 5, "tokens_out": 2},
    }
    assert plan(ctx) == states.APPLY
    assert ctx.state["edit_spec"]["change"] == "fuse gelu"
    assert ctx.state["cost_usd"] == 0.03


def test_plan_noop_routes_to_revert(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.deps["plan_runner"] = lambda **k: {
        "file": "mlp.py",
        "location": "x",
        "change": "NOOP: already applied",
        "model": "m",
        "usage": None,
    }
    assert plan(ctx) == states.REVERT
    assert ctx.state["last_decision"]["reason"] == "already_applied"


def test_plan_failure_falls_back_to_apply(tmp_path):
    ctx = _ctx(tmp_path)

    def boom(**k):
        raise RuntimeError("plan api down")

    ctx.deps["plan_runner"] = boom
    assert plan(ctx) == states.APPLY
    assert ctx.state["edit_spec"] is None
