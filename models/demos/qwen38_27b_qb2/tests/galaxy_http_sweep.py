# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-Galaxy HTTP measurements with independent serving engines resident."""

import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import httpx

from models.demos.qwen38_27b_qb2.demo.galaxy_serving import OPTIMIZATION_ENV
from models.demos.qwen38_27b_qb2.tests.benchmark import complete, response_metrics
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, render, save_report


def summarize_http(samples):
    speeds = [row["decode_tokens_per_s"] for sample in samples for row in sample["responses"]]
    if not speeds or any(speed is None or speed <= 0 for speed in speeds):
        raise ValueError("Cannot measure client decode speed without first-token and completion timestamps")
    if any(row["ttft_ms"] is None for sample in samples for row in sample["responses"]):
        raise ValueError("Cannot measure TTFT without first-token timestamps")
    ttfts = [row["ttft_ms"] / 1000 for sample in samples for row in sample["responses"]]
    tsu = statistics.median(speeds)
    return dict(
        tokens_per_second_per_user=tsu,
        aggregate_e2e_tokens_per_second=statistics.median(
            sum(row["usage"]["completion_tokens"] for row in sample["responses"]) / sample["elapsed_s"]
            for sample in samples
        ),
        ttft_p50_s=statistics.median(ttfts),
        ttft_p90_s=statistics.quantiles(ttfts, n=10, method="inclusive")[8],
        tpot_ms=1000 / tsu,
        elapsed_s=statistics.median(sample["elapsed_s"] for sample in samples),
        samples=len(samples),
    )


async def run_http_sweep(base_url, directory, *, deployment, priority_only=False):
    from transformers import AutoTokenizer

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Use a fresh directory for the Galaxy HTTP sweep")
    directory.mkdir(parents=True)
    report = (
        make_plan(8, batches=(4, 8, 16), input_lengths=(32768, 16384, 131072, 262016))
        if priority_only
        else make_plan(8)
    )
    pool_tokens = int(OPTIMIZATION_ENV["QWEN_VLLM_KV_POOL_TOKENS"])
    for cell in report["cells"]:
        if cell["pool_tokens_per_replica"] > pool_tokens:
            cell.update(
                status="capacity_guard", reason=f"Exceeds the configured {pool_tokens:,}-token per-replica KV pool"
            )
    report.update(
        measurement_mode="http",
        state="running",
        deployment=deployment,
        kv_pool_tokens_per_replica=pool_tokens,
        capacity_note=(
            f"Capacity-guard cells exceed the configured {pool_tokens:,}-token per-replica KV pool and are untested. "
            "Per-replica batch is nominal; the framework balances the actual global request burst."
        ),
        http_connection_policy="Fresh connections per request; no retries or connection-pool reuse",
        methodology=(
            f"Eight independent TP4 vLLM engines; actual HTTP bursts at total concurrency {[8*b for b in report['batches']]}. "
            "Fresh full prefills, no prefix reuse; fixed 128-token outputs through EOS; one warmup burst "
            "then three measured bursts per shape. Per-user speed is the median client rate from first "
            "visible token until stream finish. Aggregate throughput includes queueing, prefill, decode "
            "and transport. Prefill and decode can overlap across engines, so no artificial global decode "
            "phase is inferred. The serving optimization knobs differ from the initial native TP4 baseline. "
            "Three bursts are a small sample, not a tail-latency load study."
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_WEIGHTS_DIR"], local_files_only=True)
    passage = (
        "The scientific method tests explanations against observations. "
        "Describe an experiment, its controls, and the evidence needed to evaluate the result. "
    )
    base = tokenizer.encode(passage, add_special_tokens=False)
    active_cell = None
    try:
        # Avoid stale pooled-connection races between long, synchronized bursts.
        # This changes client transport only; failed requests are never retried.
        limits = httpx.Limits(max_connections=128, max_keepalive_connections=0)
        async with httpx.AsyncClient(base_url=base_url, timeout=3600, limits=limits) as client:
            for cell in report["cells"]:
                if cell["status"] == "capacity_guard":
                    continue
                active_cell = cell
                cell["status"] = "running"
                save_report(report, directory)
                render(report, directory)
                length, concurrency = cell["input_tokens"], cell["concurrency"]
                prompt = (base * (length // len(base) + 1))[:length]
                cell["prompt_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
                payload = dict(
                    prompt=prompt, max_tokens=128, ignore_eos=True, temperature=0, seed=42, add_special_tokens=False
                )

                async def burst():
                    started = time.perf_counter()
                    tasks = [asyncio.create_task(complete(client, payload, chat=False)) for _ in range(concurrency)]
                    try:
                        responses = await asyncio.gather(*tasks)
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                    elapsed = time.perf_counter() - started
                    for row in responses:
                        assert row["usage"]["prompt_tokens"] == length
                        assert row["usage"]["completion_tokens"] == 128 and row["finish_reason"] == "length"
                    return dict(
                        elapsed_s=elapsed,
                        responses=[response_metrics(row) for row in responses],
                        output_sha256=[hashlib.sha256(row["text"].encode()).hexdigest() for row in responses],
                    )

                print(f"HTTP_SWEEP_BEGIN isl={length} concurrency={concurrency}", flush=True)
                cell["warmup"] = await burst()
                cell["samples"] = []
                save_report(report, directory)
                print(
                    f"HTTP_SWEEP_WARMUP_COMPLETE isl={length} concurrency={concurrency} elapsed_s={cell['warmup']['elapsed_s']:.6f}",
                    flush=True,
                )
                for repeat in range(report["measured_runs"]):
                    cell["samples"].append(await burst())
                    save_report(report, directory)
                    print(
                        f"HTTP_SWEEP_SAMPLE_COMPLETE isl={length} concurrency={concurrency} repeat={repeat + 1} "
                        f"elapsed_s={cell['samples'][-1]['elapsed_s']:.6f}",
                        flush=True,
                    )
                cell["summary"] = summarize_http(cell["samples"])
                cell["status"] = "completed"
                save_report(report, directory)
                render(report, directory)
                print(f"HTTP_SWEEP_COMPLETE {json.dumps(cell['summary'])}", flush=True)
        report["state"] = "completed"
    except BaseException as error:
        report.update(state="failed", error_type=type(error).__name__)
        if active_cell is not None:
            active_cell["status"] = "failed"
        for cell in report["cells"]:
            if cell["status"] == "queued":
                cell["status"] = "not_run"
        raise
    finally:
        save_report(report, directory)
        render(report, directory)
    return report
