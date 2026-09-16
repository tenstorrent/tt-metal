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
