"""REMEASURE (PLAN 8.7) — median (never mean), variance, iter profile, measure_failed."""

import json

from agent import states
from agent.handlers.remeasure import remeasure
from agent.loop_context import LoopContext
from agent.run import Run


def _ctx(tmp_path, current=12.0):
    run = Run.create(tmp_path / "runs", config={"config": {}, "pathmap": {}}, run_id="RM")
    run.state_path.write_text(
        json.dumps(
            {
                "state": "REMEASURE",
                "iteration": 2,
                "cost_usd": 0.0,
                "metric": {"name": "device_ms", "direction": "min", "current": current},
                "last_verdict": {"status": "ok", "pcc": 0.99},
            }
        )
    )
    return LoopContext.from_run(run, index=[]), run


def _prof(dev):
    return {
        "device_ms": dev,
        "wall_ms": 0,
        "buckets": [{"id": "matmul", "device_ms": dev, "pct": 100, "count": 1, "tags": {}}],
    }


def test_remeasure_median_spread_and_iter_profile(tmp_path):
    ctx, run = _ctx(tmp_path, current=12.0)
    ctx.deps["measure_runner"] = lambda c: [_prof(11.0), _prof(11.4), _prof(11.2)]
    assert remeasure(ctx) == states.DECIDE
    d = ctx.state["last_decision"]
    assert d["after"] == 11.2  # median of {11.0, 11.2, 11.4}
    assert d["spread"] == 0.4
    assert d["runs"] == 3
    assert d["pcc"] == 0.99  # carried from the gate verdict
    assert (run.dir / d["profile"]).exists()


def test_remeasure_uses_median_not_mean(tmp_path):
    ctx, _ = _ctx(tmp_path)
    ctx.deps["measure_runner"] = lambda c: [_prof(10.0), _prof(10.0), _prof(40.0)]  # mean 20, median 10
    remeasure(ctx)
    assert ctx.state["last_decision"]["after"] == 10.0


def test_remeasure_crash_is_measure_failed(tmp_path):
    ctx, _ = _ctx(tmp_path)

    def boom(c):
        raise RuntimeError("tracy crashed")

    ctx.deps["measure_runner"] = boom
    assert remeasure(ctx) == states.REVERT
    assert ctx.state["last_decision"]["reason"] == "measure_failed"


def test_remeasure_empty_is_measure_failed(tmp_path):
    ctx, _ = _ctx(tmp_path)
    ctx.deps["measure_runner"] = lambda c: []
    assert remeasure(ctx) == states.REVERT
    assert ctx.state["last_decision"]["reason"] == "measure_failed"
