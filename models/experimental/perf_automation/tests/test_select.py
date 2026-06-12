"""SELECT agent (PLAN 8.3) — prompt, choice validation, handler + fallbacks."""

import json

import pytest

from agent import states
from agent.handlers.select import select
from agent.loop_context import LoopContext
from agent.run import Run
from agent.select_agent import SelectError, _validate_choice, build_select_prompt


def _ctx(tmp_path, candidates, tried=()):
    run = Run.create(tmp_path / "runs", config={"config": {}, "pathmap": {}}, run_id="S")
    run.state_path.write_text(
        json.dumps(
            {
                "state": "SELECT",
                "candidates": candidates,
                "tried": list(tried),
                "iteration": 0,
                "cost_usd": 0.0,
                "code_fix_attempts": 3,
                "pcc_fix_attempts": 1,
            }
        )
    )
    return LoopContext.from_run(run, index=[])


def test_prompt_has_brief_candidates_and_tried():
    p = build_select_prompt("BRIEF TEXT", ["a", "b"], ["c"])
    assert "BRIEF TEXT" in p and "a" in p and "b" in p and "c" in p and "JSON" in p


def test_validate_accepts_valid_untried():
    assert _validate_choice('{"lever": "a", "reasoning": "x"}', ["a", "b"], [])["lever"] == "a"


def test_validate_rejects_not_a_candidate():
    with pytest.raises(SelectError):
        _validate_choice('{"lever": "z"}', ["a"], [])


def test_validate_rejects_already_tried():
    with pytest.raises(SelectError):
        _validate_choice('{"lever": "a"}', ["a"], ["a"])


def test_validate_rejects_non_json():
    with pytest.raises(SelectError):
        _validate_choice("not json", ["a"], [])


def test_select_uses_agent_choice_and_resets_counters(tmp_path):
    ctx = _ctx(tmp_path, ["a", "b", "c"], tried=["a"])
    ctx.deps["select_runner"] = lambda **k: {
        "lever": "c",
        "reasoning": "best for matmul",
        "model": "m",
        "usage": {"cost_usd": 0.02, "tokens_in": 3, "tokens_out": 1},
    }
    assert select(ctx) == states.PLAN
    assert ctx.state["selected_lever"] == "c"
    assert ctx.state["select_reasoning"] == "best for matmul"
    assert ctx.state["code_fix_attempts"] == 0 and ctx.state["pcc_fix_attempts"] == 0
    assert ctx.state["cost_usd"] == 0.02


def test_select_falls_back_on_invalid_pick(tmp_path):
    ctx = _ctx(tmp_path, ["a", "b"], tried=[])
    ctx.deps["select_runner"] = lambda **k: {"lever": "NOPE", "model": "m", "usage": None}
    select(ctx)
    assert ctx.state["selected_lever"] == "a"  # untried[0]


def test_select_falls_back_on_runner_error(tmp_path):
    ctx = _ctx(tmp_path, ["a", "b"], tried=[])

    def boom(**k):
        raise RuntimeError("api down")

    ctx.deps["select_runner"] = boom
    select(ctx)
    assert ctx.state["selected_lever"] == "a"


def test_select_persists_prompt_and_response(tmp_path):
    ctx = _ctx(tmp_path, ["a", "b"], tried=[])
    ctx.deps["select_runner"] = lambda **k: {
        "lever": "a",
        "reasoning": "r",
        "model": "m",
        "usage": {"cost_usd": 0.0},
        "prompt": "PROMPT TEXT",
        "response": "RESPONSE TEXT",
    }
    select(ctx)
    import json as _j

    rows = [_j.loads(l) for l in (ctx.run.dir / "agent_calls.jsonl").read_text().splitlines()]
    call = rows[-1]
    assert "agent_call_id" in call
    assert call["prompt_sha"] and call["response_sha"]
    assert "prompt_file" not in call  # no per-call files; join by agent_call_id
    # full payload lives in the SINGLE prompts.jsonl, linked by agent_call_id
    payloads = [_j.loads(l) for l in (ctx.run.dir / "prompts.jsonl").read_text().splitlines()]
    body = next(b for b in payloads if b["agent_call_id"] == call["agent_call_id"])
    assert body["prompt"] == "PROMPT TEXT"
    assert body["response"] == "RESPONSE TEXT"
