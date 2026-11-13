import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from agentic_collect_swe import collect, load_attempts


def attempt(task="a", start=10, end=30, ids=(), solved=False):
    return {
        "task": task,
        "trial_name": f"{task}-{start}",
        "start": start,
        "end": end,
        "response_ids": set(ids),
        "solved": solved,
    }


def native(rid="r", statement="unique statement a", start=12, end=15, cached=2):
    boundary = {
        "id": rid,
        "start": start,
        "end": end,
        "request": json.dumps({"messages": [{"role": "user", "content": statement}], "max_tokens": 32768}),
    }
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": cached}}
    body = {"id": rid, "usage": usage, "choices": [{"message": {"content": "reply"}, "finish_reason": "length"}]}
    return ["eval_call=" + json.dumps(boundary), "response_body={" + json.dumps(body) + "}"]


def test_discarded_reply_recovered_by_statement():
    result, calls, mapping = collect(native(), [attempt()], {"a": "unique statement a"}, 1, 8, 0, 40)
    call = calls["tasks"][0]["calls"][0]
    assert call["absent_from_native_trajectory"] is True
    assert call["uncached_input"] == 8 and call["finish_reason"] == "length"
    assert calls["tasks"][0]["outside_s"] == 17
    assert mapping["tasks"][0]["calls"] == [{"id": "r"}]
    assert result["solution_fetch_audit"].startswith("pending")


def test_first_attempt_only_and_all_attempt_digest():
    attempts = [attempt(start=10, end=20, ids=["r"]), attempt(start=21, end=30, ids=["s"], solved=True)]
    result, calls, mapping = collect(
        native() + native("s", start=22, end=25), attempts, {"a": "unique statement a"}, 1, 8, 0, 40
    )
    assert result["score"]["solved"] == 0
    assert len(calls["tasks"][0]["calls"]) == 1 and len(mapping["tasks"]) == 2
    assert [r["counted"] for r in result["attempts"]] == [True, False]


def test_ambiguous_statement_never_guessed():
    result, calls, _ = collect(
        native(statement="unique statement a unique statement b"),
        [attempt(), attempt("b")],
        {"a": "unique statement a", "b": "unique statement b"},
        2,
        8,
        0,
        40,
    )
    assert result["accounting_complete"] is False
    assert not any(t["calls"] for t in calls["tasks"])


def test_missing_cache_is_unknown_and_keeps_digest_id():
    result, calls, mapping = collect(
        native(cached=None), [attempt(ids=["r"])], {"a": "unique statement a"}, 1, 8, 0, 40
    )
    assert not result["accounting_complete"]
    assert "uncached_input" not in calls["tasks"][0]["calls"][0]
    assert mapping["tasks"][0]["calls"][0]["id"] == "r"


def test_id_statement_contradiction():
    result, _, _ = collect(
        native(),
        [attempt(ids=[]), attempt("b", ids=["r"])],
        {"a": "unique statement a", "b": "unique statement b"},
        2,
        8,
        0,
        40,
    )
    assert any("contradicts" in i["reason"] for i in result["issues"])


def test_load_real_schema_shape_preserves_limits_and_exception(tmp_path):
    path = tmp_path / "trial"
    (path / "agent").mkdir(parents=True)
    raw = {
        "trial_name": "trial",
        "task_name": "swe-bench/swebench-verified__a",
        "config": {"task": {"path": "/tasks/a"}, "agent": {"override_timeout_sec": 21600}},
        "started_at": "2026-11-13T00:00:00+00:00",
        "finished_at": "2026-11-13T00:01:00+00:00",
        "verifier_result": {"rewards": {"reward": 0}},
        "exception_info": {"exception_type": "AgentTimeoutError"},
    }
    (path / "result.json").write_text(json.dumps(raw))
    (path / "agent/mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "messages": [{"role": "assistant", "extra": {"response": {"id": "r"}}}],
                "info": {"exit_status": "FormatError"},
            }
        )
    )
    rows = load_attempts(tmp_path, ["a"])
    assert rows[0]["response_ids"] == {"r"} and rows[0]["solved"] is False
    assert rows[0]["limits_config"]["override_timeout_sec"] == 21600
    assert rows[0]["trajectory_info"]["exit_status"] == "FormatError"


def test_global_overhead_not_assigned_to_short_concurrent_task():
    from agentic_collect_swe import explain_with_global_overhead

    rows = [attempt("a", 10, 20), attempt("b", 10, 30)]
    result, calls, _ = collect([], rows, {"a": "statement a", "b": "statement b"}, 2, 8, 0, 40)
    assert result["global_outside_trials_s"] == 20
    assert [t["outside_s"] for t in calls["tasks"]] == [10, 20]
    rates = {"rows": [{"batch": 8, "input_tokens": 1024, "prefill_tok_s": 100, "decode_tok_s_user": 10}]}
    replay = explain_with_global_overhead(rates, calls)
    # explain_hours rounds task replay to four decimal hours (<=0.18s error).
    assert abs(replay["explained_hours"] * 3600 - 40) < 0.2


def test_trial_outside_run_rejected(expect_error):
    with expect_error(ValueError, "outside run"):
        collect([], [attempt(start=0, end=50)], {"a": "statement a"}, 1, 8, 10, 40)
