"""Walking-skeleton tests (PLAN 8.1, 8.11).

The engine drives ROUTE -> ... -> terminal end to end. ROUTE, LOG/CHECK_EXIT,
and now APPLY are real; the rest are mocks. APPLY's editor is injected
(ctx.deps["edit_runner"]) so no key/hardware is needed.
"""

import json
import subprocess
from pathlib import Path

import pytest

from agent import engine, states
from agent.handlers import build_handlers
from agent.handlers.route import route as route_handler
from agent.loop_context import LoopContext
from agent.run import Run

FIXTURE = Path(__file__).parent / "fixtures" / "loop" / "after_before_loop"


def _entry(anchor):
    e = {"id": anchor, "title": anchor, "file": "f.md", "lever_type": "single-shot"}
    for dim in ("bound", "rank", "fidelity", "grid", "dispatch", "memory", "regime"):
        e[dim] = ["*"]
    e["op_class"] = ["matmul"]
    return e


MOCK_INDEX = [_entry("mlp-fidelity-walk"), _entry("subblock-unlock"), _entry("fuse-activation-matmul")]


_U = {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_s": 0.0}


def _fake_plan(*, lever, section, skeleton, cwd=None):
    # anchor "x = 1" survives re-application across laps (new_string keeps it)
    return {
        "summary": "tweak",
        "edits": [{"file": "model.py", "old_string": "x = 1", "new_string": "x = 1  # tuned"}],
        "model": "mock",
        "usage": _U,
    }


def _metric(target):
    return {
        "name": "device_ms",
        "unit": "ms",
        "direction": "min",
        "baseline": 12.091,
        "current": 12.091,
        "target": target,
    }


def _fake_editor(*, lever, section, model_files, error=None, spec=None, cwd=None):
    return {
        "files": ["model.py"],
        "summary": "mock edit",
        "model": "mock",
        "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_s": 0.0},
    }


def _fake_select(*, brief, candidates, tried):
    return {
        "lever": candidates[0],
        "reasoning": "mock pick",
        "model": "mock",
        "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_s": 0.0},
    }


def _mk_run(tmp_path, state_overrides=None):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.py").write_text("x = 1\n")
    for cfg in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *cfg], cwd=model, check=True)
    subprocess.run(["git", "add", "."], cwd=model, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=model, check=True)

    run = Run.create(
        tmp_path / "runs",
        config={"config": {"model_root": str(model)}, "env": {}, "pathmap": {"model_files": ["model.py"]}},
        run_id="FIXTURE",
    )
    state = json.loads((FIXTURE / "state.json").read_text())
    state.update(state_overrides or {})
    run.state_path.write_text(json.dumps(state))
    (run.profiles_dir / "baseline_profile.json").write_text(
        (FIXTURE / "profiles" / "baseline_profile.json").read_text()
    )
    return run


def _fake_measure(ctx):
    # re-bucketed profile of the "edited" model: shrink the attacked bucket, drive toward target
    prof = json.loads(json.dumps(ctx.current_profile()))
    before = ctx.state["metric"]["current"]
    target = ctx.state["metric"].get("target") or (before - 1.0)
    newdev = round(max(target, before - 1.0), 4)
    for b in prof["buckets"]:
        if b["id"] == ctx.state.get("current_bucket"):
            b["device_ms"] = round(b["device_ms"] * 0.5, 4)
    prof["device_ms"] = newdev
    return [prof]


def _ctx(run):
    ctx = LoopContext.from_run(run, index=MOCK_INDEX)
    ctx.deps["edit_runner"] = _fake_editor  # APPLY is real; editor injected
    ctx.deps["select_runner"] = _fake_select  # SELECT is real; picker injected
    ctx.deps["pcc_runner"] = lambda c: {"status": "ok", "pcc": 0.999}  # GATE_PCC real; measure injected
    ctx.deps["measure_runner"] = _fake_measure  # REMEASURE real; tracy injected
    ctx.deps["plan_runner"] = _fake_plan  # PLAN real; planner injected
    return ctx


def test_engine_walks_to_done_with_mocks(tmp_path):
    ctx = _ctx(_mk_run(tmp_path))
    final = engine.run(ctx, build_handlers())
    assert final == states.DONE
    assert ctx.state["iteration"] >= 1
    assert ctx.state["metric"]["current"] <= ctx.state["metric"]["target"]


def test_engine_appends_ledger_rows_with_experiment_id(tmp_path):
    run = _mk_run(tmp_path)
    ctx = _ctx(run)
    engine.run(ctx, build_handlers())
    rows = ctx.ledger.rows()
    assert rows and rows[0]["experiment_id"].startswith("FIXTURE#")
    assert rows[0]["result"] == "keep"


def test_engine_checkpoints_terminal_to_disk(tmp_path):
    run = _mk_run(tmp_path)
    engine.run(_ctx(run), build_handlers())
    assert json.loads(run.state_path.read_text())["state"] in states.TERMINAL


def test_apply_records_clean_sha_in_loop(tmp_path):
    run = _mk_run(tmp_path)
    engine.run(_ctx(run), build_handlers())
    assert len(json.loads(run.state_path.read_text())["git_sha_clean"]) == 40


def test_engine_resumes_from_midstate(tmp_path):
    run = _mk_run(
        tmp_path,
        state_overrides={
            "state": "GATE_PCC",
            "selected_lever": "mlp-fidelity-walk",
            "candidates": ["mlp-fidelity-walk"],
        },
    )
    final = engine.run(_ctx(run), build_handlers())
    assert final in (states.DONE, states.STOPPED)


def test_engine_raises_on_unregistered_state(tmp_path):
    with pytest.raises(engine.EngineError):
        engine.run(_ctx(_mk_run(tmp_path)), {})


def test_engine_step_cap_guards_runaway(tmp_path):
    ctx = _ctx(_mk_run(tmp_path))
    loop_handlers = {states.BEFORE_LOOP_DONE: lambda c: states.ROUTE, states.ROUTE: lambda c: states.BEFORE_LOOP_DONE}
    with pytest.raises(engine.EngineError):
        engine.run(ctx, loop_handlers, max_steps=50)


def test_route_handler_is_real_and_picks_top_bucket(tmp_path):
    ctx = _ctx(_mk_run(tmp_path))
    assert route_handler(ctx) == states.SELECT
    assert ctx.state["current_bucket"] == "matmul"
    assert "mlp-fidelity-walk" in ctx.state["candidates"]


def test_current_profile_promoted_on_keep(tmp_path):
    run = _mk_run(tmp_path)
    ctx = _ctx(run)
    engine.run(ctx, build_handlers())
    assert ctx.state.get("current_profile", "").startswith("profiles/iter_")
    cur = json.loads((run.dir / ctx.state["current_profile"]).read_text())
    base = json.loads((run.profiles_dir / "baseline_profile.json").read_text())
    matmul_now = next(b["device_ms"] for b in cur["buckets"] if b["id"] == "matmul")
    matmul_base = next(b["device_ms"] for b in base["buckets"] if b["id"] == "matmul")
    assert matmul_now < matmul_base


def test_handlers_only_return_declared_transitions(tmp_path):
    for state in build_handlers():
        assert state in states.TRANSITIONS, f"{state} missing from states.TRANSITIONS"


def test_route_writes_decision_brief(tmp_path):
    """ROUTE appends a route_briefs.jsonl row with candidates + section texts for SELECT."""
    from agent.events import read_jsonl_last

    ctx = _ctx(_mk_run(tmp_path))
    route_handler(ctx)
    rid = ctx.state["route_brief_id"]
    brief = read_jsonl_last(ctx.run.dir / "route_briefs.jsonl", route_brief_id=rid)
    assert brief is not None
    assert brief["route_brief_id"] == rid
    assert brief["row_type"] == "route_brief"
    assert brief["bucket"]["id"] == "matmul"
    assert "mlp-fidelity-walk" in [c["id"] for c in brief["candidates"]]
    assert brief["model_map"]  # the filtered ast skeleton is included
    assert any(s["id"] == "mlp-fidelity-walk" and s["text"] for s in brief["sections"])


def test_engine_stop_after_route_parks_before_select(tmp_path):
    """--until ROUTE: ROUTE runs (brief written), SELECT/APPLY never do (no key path)."""
    run = _mk_run(tmp_path)
    ctx = _ctx(run)
    parked = engine.run(ctx, build_handlers(), stop_after={states.ROUTE})
    assert parked == states.SELECT
    assert ctx.state.get("route_brief_id")  # ROUTE produced the brief
    assert ctx.state.get("git_sha_clean") is None  # APPLY never ran


# --------------------------- repair-path integration -----------------------
def _scripted_editor(*steps):
    """Editor whose Nth call runs steps[N](model_py_path). Each step writes the
    model file and returns a runner-shaped result. The last step repeats."""
    calls = {"n": 0}

    def editor(*, lever, section, model_files, error=None, spec=None, cwd=None):
        i = min(calls["n"], len(steps) - 1)
        calls["n"] += 1
        return steps[i](Path(model_files[0]))

    editor.calls = calls
    return editor


def _writes(text):
    def step(model_py):
        model_py.write_text(text)
        return {
            "files": ["model.py"],
            "summary": "edit",
            "model": "mock",
            "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_s": 0.0},
        }

    return step


def _role_count(run, role):
    f = run.dir / "agent_calls.jsonl"
    if not f.exists():
        return 0
    return sum(1 for ln in f.read_text().splitlines() if ln.strip() and json.loads(ln).get("role") == role)


def test_engine_repair_code_self_heals_syntax_error(tmp_path):
    """PLAN lands a syntactically broken edit -> VERIFY parse_error -> REPAIR_CODE
    editor fixes it -> VERIFY ok -> ... -> DONE. (single lap via target=11.5)"""
    run = _mk_run(tmp_path, state_overrides={"metric": _metric(11.5)})
    ctx = _ctx(run)
    ctx.deps["plan_runner"] = lambda **k: {
        "summary": "break",
        "edits": [{"file": "model.py", "old_string": "x = 1", "new_string": "def (:"}],
        "model": "mock",
        "usage": _U,
    }
    ctx.deps["edit_runner"] = _scripted_editor(_writes("x = 2\n"))  # repair editor fixes it
    final = engine.run(ctx, build_handlers())
    assert final == states.DONE
    assert _role_count(run, "repair_code") == 1  # exactly one code self-heal ran
    assert _role_count(run, "repair_pcc") == 0


def test_engine_repair_pcc_recovers_low_pcc(tmp_path):
    """Valid edit -> GATE_PCC pcc_low -> REPAIR_PCC -> GATE_PCC ok -> ... -> DONE."""
    run = _mk_run(tmp_path, state_overrides={"metric": _metric(11.5)})
    ctx = _ctx(run)  # _fake_plan emits a valid edit
    ctx.deps["edit_runner"] = _scripted_editor(_writes("x = 1  # tuned\n"))  # repair editor (valid)
    pcc_calls = {"n": 0}

    def pcc_runner(c):
        pcc_calls["n"] += 1
        return {"status": "pcc_low", "pcc": 0.80} if pcc_calls["n"] == 1 else {"status": "ok", "pcc": 0.999}

    ctx.deps["pcc_runner"] = pcc_runner
    final = engine.run(ctx, build_handlers())
    assert final == states.DONE
    assert _role_count(run, "repair_pcc") == 1  # exactly one PCC recovery ran
    assert _role_count(run, "repair_code") == 0
    assert pcc_calls["n"] >= 2  # gate failed then passed


def test_engine_repair_code_exhausts_budget_then_reverts(tmp_path):
    """Edit can never be fixed: REPAIR_CODE runs up to MAX_CODE_FIX, then discards
    (edit_failed) -> REVERT."""
    run = _mk_run(tmp_path)
    ctx = _ctx(run)
    ctx.deps["plan_runner"] = lambda **k: {
        "summary": "break",
        "edits": [{"file": "model.py", "old_string": "x = 1", "new_string": "def (:"}],
        "model": "mock",
        "usage": _U,
    }
    ctx.deps["edit_runner"] = _scripted_editor(_writes("def (:\n"))  # always broken
    final = engine.run(ctx, build_handlers())
    assert final in states.TERMINAL
    assert _role_count(run, "repair_code") >= states.MAX_CODE_FIX
    assert ctx.state["last_decision"]["reason"] == "edit_failed"
