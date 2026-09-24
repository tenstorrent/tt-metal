# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Client timing includes reasoning; scoring receives only final-answer text."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx

from models.demos.gemma4_31b_qb2.tests import benchmark


def stream_response(events):
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return httpx.Response(200, text=body + "data: [DONE]\n\n")


def run_stream(events):
    async def run():
        transport = httpx.MockTransport(lambda request: stream_response(events))
        async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
            return await benchmark.complete(client, {}, chat=True)

    return asyncio.run(run())


def test_reasoning_and_final_answer_share_timing_but_keep_distinct_text(monkeypatch):
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    row = run_stream(
        [
            {"choices": [{"delta": {"reasoning": "Work through the choices."}}]},
            {"choices": [{"delta": {"content": "Answer: (B)"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 4}},
        ]
    )
    assert row["text"] == "Answer: (B)"
    assert row["reasoning"] == "Work through the choices."
    assert row["ttft_ms"] == 1000
    assert row["decode_tokens_per_s"] == 3
    assert row["elapsed_s"] == 3


def test_special_token_only_response_preserves_usage_and_has_no_text_timing():
    row = run_stream(
        [
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 1}},
        ]
    )
    summary = benchmark.summarize([row], "start", "end", 2)
    assert row["text"] == ""
    assert row["usage"]["completion_tokens"] == 1
    assert summary["mean_ttft_ms"] is None
    assert summary["mean_decode_tokens_per_s"] is None
    assert summary["aggregate_output_tokens_per_s"] == 0.5


def test_truncated_stream_without_usage_fails(expect_error):
    with expect_error(RuntimeError, "Incomplete stream"):
        run_stream([{"choices": [{"delta": {"content": "Answer: (B)"}, "finish_reason": "stop"}]}])


def test_stream_error_is_not_scored_as_a_model_answer(expect_error):
    with expect_error(RuntimeError, "response details withheld"):
        run_stream([{"error": "engine stopped"}])


def test_single_user_performance_never_submits_a_batch(monkeypatch, tmp_path):
    """Capacity-one results contain four measured requests and two separate warmups."""
    from transformers import AutoTokenizer

    tokenizer = SimpleNamespace(bos_token_id=2, encode=lambda *args, **kwargs: [3, 4, 5])
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    active = calls = 0

    async def complete(client, payload, *, chat):
        nonlocal active, calls
        active += 1
        calls += 1
        assert active == 1
        assert not chat and payload["max_tokens"] == 128 and payload["ignore_eos"]
        await asyncio.sleep(0)
        active -= 1
        return {
            "usage": {"prompt_tokens": len(payload["prompt"]), "completion_tokens": 128},
            "finish_reason": "length",
            "elapsed_s": 4,
            "ttft_ms": 50,
            "decode_tokens_per_s": 40,
        }

    monkeypatch.setattr(benchmark, "complete", complete)
    rows = asyncio.run(benchmark.run_performance(None, tmp_path, 1, (128, 1024)))
    assert calls == 6
    assert [(r["input_tokens"], r["concurrency"], r["requests"]) for r in rows] == [(128, 1, 2), (1024, 1, 2)]
    assert all(r["server_capacity"] == 1 for r in rows)
    raw = [json.loads(line) for line in (tmp_path / "performance-responses.jsonl").read_text().splitlines()]
    assert len(raw) == 4 and all(r["server_capacity"] == r["batch"] == 1 for r in raw)
    inputs = json.loads((tmp_path / "performance-inputs.json").read_text())
    assert [len(row["prompt"]) for row in inputs] == [128, 1024]
    assert [r["prompt_sha256"] for r in rows] == [row["sha256"] for row in inputs]


def gpqa_args(output_dir, *, prepare_only=False):
    return SimpleNamespace(
        output_dir=output_dir,
        mode="gpqa",
        server_capacity=32,
        performance_input_lengths=(128,),
        prepare_only=prepare_only,
        base_url="http://test",
    )


def test_gpqa_artifacts_and_logs_contain_only_metadata(monkeypatch, tmp_path, capsys):
    # Invented sentinels, not dataset examples. Scoring must still receive text.
    prompt, answer, reasoning = "PRIVATE_PROMPT_SENTINEL", "PRIVATE_ANSWER_SENTINEL", "PRIVATE_REASONING_SENTINEL"
    cases = [{"id": i, "doc": {"question": prompt, "answer": answer}, "prompt": prompt} for i in range(10)]
    scored = []

    def score(doc, responses):
        assert doc == {"question": prompt, "answer": answer}
        assert responses == [answer]
        scored.append(doc)
        return {"exact_match": 1}

    async def complete(client, payload, *, chat):
        assert chat and payload["messages"][0]["content"] == prompt
        return {
            "text": answer,
            "reasoning": reasoning,
            "unexpected_response_field": prompt,
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "unexpected_usage_field": answer},
            "finish_reason": "stop",
            "elapsed_s": 1,
            "ttft_ms": 10,
            "decode_tokens_per_s": 40,
        }

    monkeypatch.setattr(benchmark, "load_gpqa", lambda: (SimpleNamespace(process_results=score), cases))
    monkeypatch.setattr(benchmark, "complete", complete)
    assert benchmark.run(gpqa_args(tmp_path, prepare_only=True)) == 0
    manifest = [json.loads(line) for line in (tmp_path / "inputs.jsonl").read_text().splitlines()]
    assert manifest == [
        {"id": case["id"], "sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()}
        for case in cases
    ]
    assert benchmark.run(gpqa_args(tmp_path)) == 0
    assert len(scored) == 10
    rows = [json.loads(line) for line in (tmp_path / "gpqa-responses.jsonl").read_text().splitlines()]
    assert len(rows) == 10 and all(row["correct"] == 1 for row in rows)
    assert all(row["usage"] == {"prompt_tokens": 20, "completion_tokens": 4} for row in rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["gpqa_result"]["accuracy"] == 1
    assert summary["gpqa_result"]["mean_decode_tokens_per_s"] == 40
    capture = capsys.readouterr()
    public_output = capture.out + capture.err + "".join(p.read_text() for p in tmp_path.iterdir())
    assert all(secret not in public_output for secret in (prompt, answer, reasoning))


def test_benchmark_failure_withholds_exception_content(monkeypatch, tmp_path, capsys):
    async def fail(args):
        raise ValueError("PRIVATE_PROMPT_OR_RESPONSE_SENTINEL")

    monkeypatch.setattr(benchmark, "main", fail)
    assert benchmark.run(gpqa_args(tmp_path)) == 1
    error = json.loads((tmp_path / "error.json").read_text())
    assert error["error_type"] == "ValueError"
    capture = capsys.readouterr()
    assert "PRIVATE_PROMPT_OR_RESPONSE_SENTINEL" not in capture.out + capture.err + json.dumps(error)


def test_stream_error_withholds_server_response(expect_error, capsys):
    with expect_error(RuntimeError, "response details withheld"):
        run_stream([{"error": {"message": "PRIVATE_SERVER_RESPONSE_SENTINEL"}}])
    capture = capsys.readouterr()
    assert "PRIVATE_SERVER_RESPONSE_SENTINEL" not in capture.out + capture.err


def test_startup_status_never_copies_log_content():
    from models.demos.gemma4_31b_qb2.tests.startup_status import startup_status

    text = "\n".join(
        (
            "PRIVATE_PROMPT_SENTINEL",
            "Loading Gemma4 layer 59",
            "Gemma4 model loaded",
            "Warming model trace",
            "Traceback (most recent call last): PRIVATE_EXCEPTION_SENTINEL",
            "Loading Gemma4 layer 999999 PRIVATE_ANSWER_SENTINEL",
        )
    )
    assert startup_status(text) == {
        "last_stage": "weight_loading",
        "last_layer_started": 59,
        "exception_logged": True,
    }
    assert startup_status("PRIVATE_REASONING_SENTINEL") == {
        "last_stage": "unknown",
        "last_layer_started": None,
        "exception_logged": False,
    }
