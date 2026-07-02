"""Walking-skeleton tests (PLAN 8.1, 8.11).

Proves the engine drives ROUTE -> ... -> terminal end-to-end with mock leaf
handlers — no API key, no hardware. As members replace mocks with real
modules in agent/handlers/__init__.py, these stay green.
"""

import json
from pathlib import Path

import pytest

from agent import engine, states
from agent.handlers import build_handlers
from agent.handlers.route import route as route_handler
from agent.loop_context import LoopContext
from agent.run import Run

FIXTURE = Path(__file__).parent / "fixtures" / "loop" / "after_before_loop"


# A tiny, drift-proof playbook index so ROUTE returns deterministic candidates.
def _entry(anchor):
    e = {"id": anchor, "title": anchor, "file": "f.md", "lever_type": "single-shot"}
    for dim in ("bound", "rank", "fidelity", "grid", "dispatch", "memory", "regime"):
        e[dim] = ["*"]
    e["op_class"] = ["matmul"]
    return e


MOCK_INDEX = [_entry("mlp-fidelity-walk"), _entry("subblock-unlock"), _entry("fuse-activation-matmul")]


def _mk_run(tmp_path, state_overrides=None):
    run = Run.create(tmp_path, config={"config": {}, "env": {}, "pathmap": {}}, run_id="FIXTURE")
    state = json.loads((FIXTURE / "state.json").read_text())
    state.update(state_overrides or {})
    run.state_path.write_text(json.dumps(state))
    (run.profiles_dir / "baseline_profile.json").write_text(
        (FIXTURE / "profiles" / "baseline_profile.json").read_text()
    )
    return run


def test_engine_walks_to_done_with_mocks(tmp_path):
    ctx = LoopContext.from_run(_mk_run(tmp_path), index=MOCK_INDEX)
    final = engine.run(ctx, build_handlers())
    assert final == states.DONE
    assert ctx.state["iteration"] >= 1
    assert ctx.state["metric"]["current"] <= ctx.state["metric"]["target"]


def test_engine_appends_ledger_rows_with_experiment_id(tmp_path):
    run = _mk_run(tmp_path)
    ctx = LoopContext.from_run(run, index=MOCK_INDEX)
    engine.run(ctx, build_handlers())
    rows = ctx.ledger.rows()
    assert rows and rows[0]["experiment_id"].startswith("FIXTURE#")
    assert rows[0]["result"] == "keep"


def test_engine_checkpoints_terminal_to_disk(tmp_path):
    run = _mk_run(tmp_path)
    ctx = LoopContext.from_run(run, index=MOCK_INDEX)
    engine.run(ctx, build_handlers())
    assert json.loads(run.state_path.read_text())["state"] in states.TERMINAL


def test_engine_resumes_from_midstate(tmp_path):
    """Resume = read state from disk and continue; no re-running prior stages."""
    run = _mk_run(
        tmp_path,
        state_overrides={
            "state": "GATE_PCC",
            "selected_lever": "mlp-fidelity-walk",
            "candidates": ["mlp-fidelity-walk"],
        },
    )
    ctx = LoopContext.from_run(run, index=MOCK_INDEX)
    final = engine.run(ctx, build_handlers())
    assert final in (states.DONE, states.STOPPED)


def test_engine_raises_on_unregistered_state(tmp_path):
    ctx = LoopContext.from_run(_mk_run(tmp_path), index=MOCK_INDEX)
    with pytest.raises(engine.EngineError):
        engine.run(ctx, {})  # empty registry


def test_engine_step_cap_guards_runaway(tmp_path):
    ctx = LoopContext.from_run(_mk_run(tmp_path), index=MOCK_INDEX)
    # a registry that never terminates
    loop_handlers = {states.BEFORE_LOOP_DONE: lambda c: states.ROUTE, states.ROUTE: lambda c: states.BEFORE_LOOP_DONE}
    with pytest.raises(engine.EngineError):
        engine.run(ctx, loop_handlers, max_steps=50)


def test_route_handler_is_real_and_picks_top_bucket(tmp_path):
    ctx = LoopContext.from_run(_mk_run(tmp_path), index=MOCK_INDEX)
    assert route_handler(ctx) == states.SELECT
    assert ctx.state["current_bucket"] == "matmul"  # top by device_ms
    assert "mlp-fidelity-walk" in ctx.state["candidates"]


def test_handlers_only_return_declared_transitions(tmp_path):
    """Every registered state is in the TRANSITIONS contract (no rogue edges)."""
    handlers = build_handlers()
    for state in handlers:
        assert state in states.TRANSITIONS, f"{state} missing from states.TRANSITIONS"


def test_current_profile_promoted_on_keep(tmp_path):
    """ROUTE must route on the LATEST committed profile, not the frozen baseline.
    After a kept change, state['current_profile'] points at an iter profile whose
    attacked bucket shrank."""
    run = _mk_run(tmp_path)
    ctx = LoopContext.from_run(run, index=MOCK_INDEX)
    engine.run(ctx, build_handlers())
    assert ctx.state.get("current_profile", "").startswith("profiles/iter_")
    import json as _j

    cur = _j.loads((run.dir / ctx.state["current_profile"]).read_text())
    base = _j.loads((run.profiles_dir / "baseline_profile.json").read_text())
    matmul_now = next(b["device_ms"] for b in cur["buckets"] if b["id"] == "matmul")
    matmul_base = next(b["device_ms"] for b in base["buckets"] if b["id"] == "matmul")
    assert matmul_now < matmul_base  # the committed change moved the profile
