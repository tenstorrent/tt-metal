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
    s = _validate_spec(
        '{"summary": "fuse gelu", "edits": [{"file": "mlp.py", "old_string": "ttnn.linear(x)", "new_string": "ttnn.linear(x, fused=True)"}]}'
    )
    assert s["summary"] == "fuse gelu"
    assert s["edits"][0]["file"] == "mlp.py" and s["edits"][0]["old_string"] == "ttnn.linear(x)"


def test_validate_spec_noop_empty_edits_ok():
    s = _validate_spec('{"summary": "NOOP: already applied", "edits": []}')
    assert s["edits"] == []


def test_validate_spec_rejects_missing_edits():
    with pytest.raises(PlanError):
        _validate_spec('{"summary": "x"}')


def test_validate_spec_rejects_edit_without_anchor():
    with pytest.raises(PlanError):
        _validate_spec('{"summary": "x", "edits": [{"file": "mlp.py", "new_string": "y"}]}')


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
        "summary": "fuse gelu",
        "edits": [{"file": "mlp.py", "old_string": "ttnn.linear(x)", "new_string": "ttnn.linear(x, fused=True)"}],
        "model": "m",
        "usage": {"cost_usd": 0.03, "tokens_in": 5, "tokens_out": 2},
    }
    assert plan(ctx) == states.APPLY
    assert ctx.state["edit_spec"]["edits"][0]["old_string"] == "ttnn.linear(x)"
    assert ctx.state["cost_usd"] == 0.03


def test_plan_noop_routes_to_revert(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.deps["plan_runner"] = lambda **k: {
        "summary": "NOOP: already applied",
        "edits": [],
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
