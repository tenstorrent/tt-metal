# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from models.autoports.google_gemma_4_31b_it.tools.agentic_collect_gpqa import collect, explain_with_global_overhead, sha


def fixture(cached=3):
    question = "Which particle has charge zero?"
    tasks = {"tasks": [{"doc_id": 7, "record_id": "rec7", "question_sha256": sha(question)}]}
    samples = [{"doc_id": 7, "doc": {"Question": question}, "exact_match": 1, "resps": [["PRIVATE TEXT"]]}]
    boundary = {
        "id": "r1",
        "start": 100,
        "end": 120,
        "request": json.dumps({"messages": [{"role": "user", "content": "Question: " + question + "\nA neutron"}]}),
    }
    body = {
        "id": "r1",
        "choices": [{"message": {"content": "PRIVATE TEXT"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": cached}},
    }
    lines = ["eval_call=" + json.dumps(boundary), "response_body={" + json.dumps(body) + "}"]
    return lines, samples, tasks


def test_exact_join_usage_score_and_private_output():
    result, calls = collect(*fixture(), 10, 32)
    assert result["accounting_complete"]
    assert result["score"]["solved"] == 1
    call = calls["tasks"][0]["calls"][0]
    assert (call["id"], call["uncached_input"], call["output"], call["context"]) == ("r1", 7, 5, 15)
    assert "PRIVATE TEXT" not in json.dumps(result)
    assert calls["tasks"][0]["outside_s"] == 0


def test_wrong_document_id_is_rejected(expect_error):
    lines, samples, tasks = fixture()
    samples[0]["doc"]["Question"] += " WRONG"
    with expect_error(ValueError, "hash"):
        collect(lines, samples, tasks, 10, 32)


def test_missing_cache_count_is_unknown_not_zero():
    result, calls = collect(*fixture(cached=None), 10, 32)
    assert not result["accounting_complete"]
    assert not calls["tasks"][0]["calls"]


def test_duplicate_log_lines_do_not_duplicate_calls():
    lines, samples, tasks = fixture()
    result, calls = collect(lines + lines, samples, tasks, 10, 32)
    assert len(calls["tasks"][0]["calls"]) == 1
    assert result["score"]["n"] == 1


def test_missing_task_keeps_denominator():
    lines, samples, tasks = fixture()
    tasks["tasks"].append({"doc_id": 9, "question_sha256": sha("another question")})
    result, calls = collect(lines, samples, tasks, 10, 32)
    assert result["score"]["n"] == 2
    assert not result["accounting_complete"]


def test_shared_run_overhead_counted_once_and_old_calls_filtered():
    lines, samples, tasks = fixture()
    old_boundary = {
        "id": "old",
        "start": 50,
        "end": 70,
        "request": json.dumps({"prompt": samples[0]["doc"]["Question"]}),
    }
    result, calls = collect(
        lines + ["eval_call=" + json.dumps(old_boundary)], samples, tasks, 10, 32, run_start=90, run_end=130
    )
    assert result["accounting_complete"]
    assert result["measured_hours"] == 40 / 3600
    assert calls["global_overhead_s"] == result["global_overhead_s"] == 20
    assert calls["tasks"][0]["outside_s"] == 0
    assert [c["id"] for c in calls["tasks"][0]["calls"]] == ["r1"]


def test_boundary_that_cuts_a_call_is_rejected(expect_error):
    with expect_error(ValueError, "beyond run end"):
        collect(*fixture(), 10, 32, run_start=90, run_end=110)


def test_concurrent_long_other_task_keeps_global_overhead_after_replay():
    lines, samples, tasks = fixture()
    question = "Which particle is charged?"
    tasks["tasks"].append({"doc_id": 8, "question_sha256": sha(question)})
    samples.append({"doc_id": 8, "doc": {"Question": question}, "exact_match": 0})
    boundary = {"id": "r2", "start": 101, "end": 200, "request": json.dumps({"prompt": question})}
    body = {
        "id": "r2",
        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 900, "prompt_tokens_details": {"cached_tokens": 3}},
    }
    lines += ["eval_call=" + json.dumps(boundary), "response_body={" + json.dumps(body) + "}"]
    result, calls = collect(lines, samples, tasks, 2, 8, run_start=90, run_end=230)
    assert calls["global_overhead_s"] == 40
    assert all(task["outside_s"] == 0 for task in calls["tasks"])
    rates = {"rows": [{"batch": 8, "input_tokens": 1024, "prefill_tok_s": 100, "decode_tok_s_user": 10}]}
    replay = explain_with_global_overhead(rates, calls)
    assert replay["global_overhead_s"] == 40
    assert replay["explained_hours"] == pytest.approx(replay["task_replay_hours"] + 40 / 3600)
    assert replay["task_replay_hours"] == round(max(t["server_s"] for t in replay["tasks"]) / 3600, 4)
    assert replay["tasks"][1]["server_s"] > replay["tasks"][0]["server_s"] + 40
    assert result["measured_hours"] == 140 / 3600


def test_replay_requires_complete_accounting_and_run_boundaries(expect_error):
    rates = {"rows": []}
    _, calls = collect(*fixture(cached=None), 2, 8)
    with expect_error(ValueError, "incomplete"):
        explain_with_global_overhead(rates, calls)
    _, calls = collect(*fixture(), 2, 8)
    with expect_error(ValueError, "boundaries"):
        explain_with_global_overhead(rates, calls)
