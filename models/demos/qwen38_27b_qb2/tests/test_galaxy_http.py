# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HTTP checks must inspect actual replies and preserve partial failure receipts."""

import asyncio
import json

import httpx

from models.demos.qwen38_27b_qb2.tests import galaxy_api
from models.demos.qwen38_27b_qb2.tests.galaxy_http_sweep import summarize_http
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, render


def test_http_metrics_do_not_multiply_per_user_speed_into_aggregate():
    sample = dict(
        elapsed_s=20,
        responses=[
            dict(decode_tokens_per_s=40, ttft_ms=1000, usage={"completion_tokens": 128}),
            dict(decode_tokens_per_s=20, ttft_ms=5000, usage={"completion_tokens": 128}),
        ],
    )
    summary = summarize_http([sample] * 3)
    assert summary["tokens_per_second_per_user"] == 30
    assert summary["aggregate_e2e_tokens_per_second"] == 12.8
    assert "aggregate_decode_tokens_per_second" not in summary
    assert summary["ttft_p50_s"] == 3 and summary["ttft_p90_s"] == 5


def test_http_missing_token_timing_is_not_invented(expect_error):
    with expect_error(ValueError, "Cannot measure"):
        summarize_http([dict(elapsed_s=2, responses=[dict(decode_tokens_per_s=None, ttft_ms=None)])])


def test_http_plot_labels_the_actual_end_to_end_rate(tmp_path):
    report = make_plan(8)
    report.update(measurement_mode="http", state="running")
    report["cells"][0].update(
        status="completed",
        summary=dict(
            tokens_per_second_per_user=30,
            aggregate_e2e_tokens_per_second=12.8,
            ttft_p50_s=3,
            ttft_p90_s=5,
            tpot_ms=1000 / 30,
        ),
    )
    render(report, tmp_path)
    document = (tmp_path / "index.html").read_text()
    assert "Aggregate end-to-end throughput" in document and "End-to-end tokens/s" in document
    headers = (tmp_path / "sweep.csv").read_text().splitlines()[0].split(",")
    assert headers.count("aggregate_e2e_tokens_per_second") == 1
    assert "aggregate_decode_tokens_per_second" not in headers


def install_api_transport(monkeypatch, *, wrong_memory=False):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": galaxy_api.MODEL}]})
        body = json.loads(request.content)
        text = "cedar" if len(body["messages"]) > 1 else "READY"
        if wrong_memory and text == "cedar":
            text = "wrong"
        usage = dict(prompt_tokens=40, completion_tokens=2)
        if not body.get("stream"):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": usage}
            )
        events = [
            {"choices": [{"delta": {"content": text}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]
        return httpx.Response(
            200, text="".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        galaxy_api.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs)
    )
    return requests


def test_live_api_workflow_checks_streaming_and_multiturn(monkeypatch, tmp_path):
    requests = install_api_transport(monkeypatch)
    report = asyncio.run(galaxy_api.validate_api("http://test", tmp_path / "api.json", concurrency=8))
    assert report["passed"] and len(report["checks"]) == 6
    assert len(requests) == 13
    assert len(report["checks"][-1]["responses"]) == 8


def test_live_api_failure_keeps_completed_checks(monkeypatch, tmp_path, expect_error):
    install_api_transport(monkeypatch, wrong_memory=True)
    path = tmp_path / "api.json"
    with expect_error(AssertionError, "Multiturn"):
        asyncio.run(galaxy_api.validate_api("http://test", path, concurrency=8))
    report = json.loads(path.read_text())
    assert report["state"] == "failed" and report["passed"] is False
    assert len(report["checks"]) == 4
    with expect_error(FileExistsError, "Preserve"):
        asyncio.run(galaxy_api.validate_api("http://test", path, concurrency=8))
