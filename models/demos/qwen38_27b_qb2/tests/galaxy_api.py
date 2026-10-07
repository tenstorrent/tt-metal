# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small live serving checks before spending time on full reference evaluation."""

import asyncio
import json
from pathlib import Path

import httpx

from models.demos.qwen38_27b_qb2.tests.benchmark import MODEL, complete, response_metrics


async def validate_api(base_url, output, *, concurrency=128):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve the previous API validation receipt")
    report = dict(state="running", passed=False, checks=[], concurrency=concurrency)

    def passed(name, **details):
        report["checks"].append(dict(name=name, passed=True, **details))
        output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        async with httpx.AsyncClient(base_url=base_url, timeout=900, limits=limits) as client:
            response = await client.get("/health")
            response.raise_for_status()
            passed("health")
            response = await client.get("/v1/models")
            response.raise_for_status()
            assert MODEL in {row["id"] for row in response.json()["data"]}, "Expected served model is absent"
            passed("model_listing")
            payload = dict(
                model=MODEL,
                messages=[dict(role="user", content="Reply with the single word READY.")],
                temperature=0,
                top_k=1,
                seed=42,
                max_tokens=32,
                chat_template_kwargs={"enable_thinking": False},
            )
            response = await client.post("/v1/chat/completions", json={**payload, "stream": False})
            response.raise_for_status()
            body = response.json()
            answer = body["choices"][0]["message"]["content"]
            assert answer and "ready" in answer.lower(), "Nonstreaming chat did not follow the simple instruction"
            assert body["choices"][0]["finish_reason"] == "stop", "Short chat unexpectedly exhausted its budget"
            assert body["usage"]["completion_tokens"] > 0
            passed("nonstreaming_chat", usage=body["usage"])
            streamed = await complete(client, payload, chat=True)
            assert streamed["text"] == answer, "Streaming and nonstreaming greedy answers differ"
            assert streamed["finish_reason"] == "stop"
            passed("streaming_greedy_equivalence", **response_metrics(streamed))
            multiturn = {
                **payload,
                "messages": [
                    dict(role="user", content="Remember the word cedar. Reply only OK."),
                    dict(role="assistant", content="OK"),
                    dict(role="user", content="What word did I ask you to remember? Reply with that word only."),
                ],
            }
            remembered = await complete(client, multiturn, chat=True)
            assert "cedar" in remembered["text"].lower(), "Multiturn chat did not retain the supplied history"
            assert remembered["finish_reason"] == "stop"
            passed("multiturn_chat", **response_metrics(remembered))
            replies = await asyncio.gather(*(complete(client, payload, chat=True) for _ in range(concurrency)))
            assert all(row["text"] == answer and row["finish_reason"] == "stop" for row in replies)
            passed("concurrent_greedy_repeatability", responses=[response_metrics(row) for row in replies])
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error_type=type(error).__name__)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report
