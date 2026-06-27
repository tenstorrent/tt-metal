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


def test_code_fix_budget_offmenu_is_larger():
    # Off-menu invents WHAT + places WHERE in one budget -> larger than a known lever's.
    assert states.code_fix_budget("shard-activation-to-l1") == states.MAX_CODE_FIX
    assert states.code_fix_budget(states.FROM_PRINCIPLES) == states.MAX_CODE_FIX_PRINCIPLES
    assert states.code_fix_budget(states.FROM_PRINCIPLES) > states.MAX_CODE_FIX
    assert states.code_fix_budget(None) == states.MAX_CODE_FIX


# --- OFF-MENU: the brain can choose from-principles even when real levers exist ----------
def test_prompt_offers_off_menu_from_principles():
    # The select prompt must present FROM_PRINCIPLES as a valid choice + tell the brain WHEN
    # to use it (no listed lever targets the dominant op).
    p = build_select_prompt("BRIEF", ["shard-activation-to-l1", states.FROM_PRINCIPLES], [])
    assert states.FROM_PRINCIPLES in p and "first principles" in p.lower()


def test_validate_accepts_off_menu_pick():
    # Off-menu IS selectable: when FROM_PRINCIPLES is a candidate, picking it is valid (not
    # rejected like an invalid lever). This is what unlocks "levers exist but none fits".
    out = _validate_choice(
        json.dumps({"lever": states.FROM_PRINCIPLES, "reasoning": "no lever targets the tilize cost"}),
        ["shard-activation-to-l1", states.FROM_PRINCIPLES],
        [],
    )
    assert out["lever"] == states.FROM_PRINCIPLES


def test_validate_parses_skip_list():
    # skip = irrelevant levers to prune; keep only valid candidates, drop the chosen one + junk.
    out = _validate_choice(
        json.dumps({"lever": "a", "skip": ["b", "a", "zzz"], "reasoning": "a fits, b/c don't"}),
        ["a", "b", "c"],
        [],
    )
    assert out["lever"] == "a" and out["skip"] == ["b"]  # 'a'=chosen dropped, 'zzz'=not a candidate dropped


def test_validate_skip_optional_defaults_empty():
    out = _validate_choice('{"lever": "a", "reasoning": "x"}', ["a", "b"], [])
    assert out["skip"] == []


def test_select_handler_prunes_skipped_levers(tmp_path):
    # The brain picks 'a' and prunes 'b' as irrelevant -> 'b' is marked tried (won't be re-offered).
    ctx = _ctx(tmp_path, ["a", "b", "c"], tried=[])
    ctx.deps["select_runner"] = lambda **k: {
        "lever": "a",
        "skip": ["b"],
        "reasoning": "a fits; b irrelevant",
        "model": "m",
        "usage": None,
    }
    select(ctx)
    assert ctx.state["selected_lever"] == "a"
    assert "b" in ctx.state["tried"]  # pruned
    assert "c" not in ctx.state["tried"]  # not pruned, still available


def test_select_handler_honors_off_menu_over_real_levers(tmp_path):
    # A real lever exists AND from-principles is on the menu; the brain chooses off-menu.
    # The handler must HONOR that choice, not override it with a lever.
    ctx = _ctx(tmp_path, ["shard-activation-to-l1", states.FROM_PRINCIPLES], tried=[])
    ctx.deps["select_runner"] = lambda **k: {
        "lever": states.FROM_PRINCIPLES,
        "reasoning": "none of the levers address the dominant layout-conversion cost",
        "model": "m",
        "usage": None,
    }
    assert select(ctx) == states.PLAN
    assert ctx.state["selected_lever"] == states.FROM_PRINCIPLES


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
