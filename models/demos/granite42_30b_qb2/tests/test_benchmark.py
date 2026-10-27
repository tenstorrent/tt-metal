# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Client timing includes reasoning; scoring receives only final-answer text."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from models.demos.granite42_30b_qb2.tests import benchmark


def stream_response(events):
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return httpx.Response(200, text=body + "data: [DONE]\n\n")


def run_stream(events):
    async def run():
        transport = httpx.MockTransport(lambda request: stream_response(events))
        async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
            return await benchmark.complete(client, {}, chat=True)

    return asyncio.run(run())


def run_nonstream(payload):
    async def run():
        response = httpx.Response(200, json=payload)
        transport = httpx.MockTransport(lambda request: response)
        async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
            return await benchmark.complete_nonstreaming(client, {}, chat=True)

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


def test_nonstreaming_keeps_reasoning_separate_from_scored_text():
    row = run_nonstream(
        {
            "choices": [
                {
                    "message": {"content": "Answer: (B)", "reasoning_content": "Private reasoning"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 4},
        }
    )
    assert row["text"] == "Answer: (B)"
    assert row["reasoning"] == "Private reasoning"
    assert row["ttft_ms"] is row["decode_tokens_per_s"] is None


@pytest.mark.parametrize("checkpoint", [None, "/local/granite-checkpoint"])
def test_single_user_performance_never_submits_a_batch(monkeypatch, tmp_path, checkpoint):
    """Capacity-one results contain four measured requests and two separate warmups."""
    from transformers import AutoTokenizer

    tokenizer = SimpleNamespace(bos_token_id=2, encode=lambda *args, **kwargs: [3, 4, 5])
    if checkpoint is None:
        monkeypatch.delenv("HF_MODEL", raising=False)
    else:
        monkeypatch.setenv("HF_MODEL", checkpoint)

    def load_tokenizer(source, **kwargs):
        assert source == (checkpoint or benchmark.MODEL)
        assert kwargs == {"revision": benchmark.MODEL_REVISION, "local_files_only": True}
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
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
    rows = asyncio.run(benchmark.run_performance(None, tmp_path, 16, 1, (128, 1024)))
    assert calls == 6
    assert [(r["input_tokens"], r["concurrency"], r["requests"]) for r in rows] == [(128, 1, 2), (1024, 1, 2)]
    assert all(r["server_capacity"] == 16 for r in rows)
    raw = [json.loads(line) for line in (tmp_path / "performance-responses.jsonl").read_text().splitlines()]
    assert len(raw) == 4 and all(r["server_capacity"] == 16 and r["batch"] == 1 for r in raw)
    inputs = json.loads((tmp_path / "performance-inputs.json").read_text())
    assert [len(row["prompt"]) for row in inputs] == [128, 1024]
    assert [r["prompt_sha256"] for r in rows] == [row["sha256"] for row in inputs]


def gpqa_args(output_dir, *, prepare_only=False):
    return SimpleNamespace(
        output_dir=output_dir,
        mode="gpqa",
        server_capacity=16,
        performance_concurrency=None,
        performance_input_lengths=(128,),
        prepare_only=prepare_only,
        base_url="http://test",
    )


def test_gpqa_artifacts_and_logs_contain_only_metadata(monkeypatch, tmp_path, capsys):
    # Invented sentinels, not dataset examples. Scoring must still receive text.
    prompt, answer, reasoning = "PRIVATE_PROMPT_SENTINEL", "PRIVATE_ANSWER_SENTINEL", "PRIVATE_REASONING_SENTINEL"
    cases = [{"id": i, "doc": {"question": prompt, "answer": answer}, "prompt": prompt} for i in range(10)]
    scored = []
    active = max_active = 0

    def score(doc, responses):
        assert doc == {"question": prompt, "answer": answer}
        assert responses == [answer]
        scored.append(doc)
        return {"exact_match": 1}

    async def complete_nonstreaming(client, payload, *, chat):
        nonlocal active, max_active
        assert chat and payload["messages"][0]["content"] == prompt
        assert payload["seed"] == benchmark.GPQA_SEED
        assert payload["chat_template_kwargs"] == {"enable_thinking": True, "low_effort": False}
        assert "repetition_detection" not in payload
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
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

    task = SimpleNamespace(process_results=score)
    monkeypatch.setattr(benchmark, "load_gpqa", lambda: (task, cases))
    monkeypatch.setattr(benchmark, "complete_nonstreaming", complete_nonstreaming)
    assert benchmark.run(gpqa_args(tmp_path, prepare_only=True)) == 0
    manifest = [json.loads(line) for line in (tmp_path / "inputs.jsonl").read_text().splitlines()]
    assert manifest == [
        {"id": case["id"], "sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()}
        for case in cases
    ]
    assert benchmark.run(gpqa_args(tmp_path)) == 0
    assert len(scored) == 10
    assert max_active == 5
    rows = [json.loads(line) for line in (tmp_path / "gpqa-responses.jsonl").read_text().splitlines()]
    assert len(rows) == 10 and all(row["correct"] == 1 for row in rows)
    assert all(row["usage"] == {"prompt_tokens": 20, "completion_tokens": 4} for row in rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["task"] == "r1_gpqa_diamond"
    assert summary["gpqa_result"]["accuracy"] == 1
    assert summary["gpqa_result"]["concurrency"] == 5
    assert summary["gpqa_result"]["mean_decode_tokens_per_s"] == 40
    assert summary["gpqa"]["seed"] == benchmark.GPQA_SEED
    assert summary["gpqa"]["thinking"] is True
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


def test_report_preserves_run_start_and_every_shape(monkeypatch, tmp_path):
    import importlib

    from infra.data_collection.pydantic_models import BenchmarkMeasurement, PartialBenchmarkRun
    from models.perf import benchmarking_utils

    report = importlib.import_module("models.demos.granite42_30b_qb2.tests.report")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmarking_utils, "IS_CI_ENV", True)
    monkeypatch.setattr(benchmarking_utils, "BenchmarkMeasurement", BenchmarkMeasurement, raising=False)
    monkeypatch.setattr(benchmarking_utils, "PartialBenchmarkRun", PartialBenchmarkRun, raising=False)
    monkeypatch.setattr(report, "IS_CI_ENV", True)
    # Exercise the real writer and filename handling without deserializing pickle.
    monkeypatch.setattr(
        benchmarking_utils, "pickle", SimpleNamespace(dumps=lambda record: record.model_dump_json().encode())
    )
    row = dict(
        measurement_start="2026-09-25T10:01:00+00:00",
        measurement_end="2026-09-25T10:02:00+00:00",
        mean_ttft_ms=70,
        mean_decode_tokens_per_s=40,
        aggregate_output_tokens_per_s=39,
        concurrency=1,
        requests=2,
        input_tokens=128,
        output_tokens=128,
    )
    summary = dict(
        run_start="2026-09-25T10:00:00+00:00",
        server_capacity=1,
        checkpoint_revision="checkpoint",
        input_sha256="input",
        scope="synthetic regression",
        performance_results=[row],
    )
    report.report(summary)
    summary.update(
        server_capacity=16,
        gpqa_result={**row, "concurrency": 1, "requests": 10, "accuracy": 0.9},
        performance_results=[row, {**row, "concurrency": 16, "requests": 32}],
    )
    report.report(summary)
    files = list((tmp_path / "generated/benchmark_data").glob("partial_run_*.pkl"))
    assert len(files) == 4, "Shapes sharing a run start must not overwrite each other"
    records = [PartialBenchmarkRun.model_validate_json(path.read_text()) for path in files]
    assert {r.batch_size for r in records} == {1, 16}
    for record in records:
        assert record.run_start_ts.isoformat() == summary["run_start"]
        assert record.run_end_ts.isoformat() == row["measurement_end"]
        assert all(m.step_start_ts.isoformat() == row["measurement_start"] for m in record.measurements)
        assert record.config_params["server_capacity"] in (1, 16)


def test_gpqa_delegates_final_answer_extraction_to_the_r1_task():
    doc = {"answer": "(B)"}
    final_answer = r"The final answer is \boxed{B}."

    def score(actual_doc, responses):
        assert actual_doc is doc
        assert responses == [final_answer]
        return {"exact_match": 1}

    # r1_gpqa_diamond owns extraction; it needs no separate named filter.
    task = SimpleNamespace(process_results=score)
    assert benchmark.score_answer(task, doc, final_answer) == 1
