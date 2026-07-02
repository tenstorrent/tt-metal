"""DECIDE (PLAN 8.8) — pure keep/discard per direction + noise floor."""

import json

from agent import states
from agent.handlers.decide import decide
from agent.loop_context import LoopContext
from agent.run import Run


def _ctx(tmp_path, before, after, direction="min"):
    run = Run.create(tmp_path / "runs", config={"config": {}, "pathmap": {}}, run_id="D")
    run.state_path.write_text(
        json.dumps(
            {
                "state": "DECIDE",
                "metric": {"name": "device_ms", "direction": direction},
                "last_decision": {"before": before, "after": after},
            }
        )
    )
    return LoopContext.from_run(run, index=[])


def test_decide_keep_when_faster(tmp_path):
    ctx = _ctx(tmp_path, 12.0, 11.0)
    assert decide(ctx) == states.COMMIT
    assert ctx.state["last_decision"]["result"] == "keep"


def test_decide_discard_within_noise_floor(tmp_path):
    ctx = _ctx(tmp_path, 12.0, 11.99)  # 0.01 < 0.05 floor
    assert decide(ctx) == states.REVERT
    assert ctx.state["last_decision"]["result"] == "discard"
    assert ctx.state["last_decision"]["reason"] == "no_gain"


def test_decide_discard_when_slower(tmp_path):
    ctx = _ctx(tmp_path, 12.0, 12.5)
    assert decide(ctx) == states.REVERT


def test_decide_max_direction_keeps_higher(tmp_path):
    ctx = _ctx(tmp_path, 100.0, 150.0, direction="max")  # fps: higher is better
    assert decide(ctx) == states.COMMIT
