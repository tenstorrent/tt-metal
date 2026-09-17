# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Client timing includes reasoning; scoring receives only final-answer text."""

import asyncio
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
    with expect_error(RuntimeError, "engine stopped"):
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
