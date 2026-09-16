# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Score a fixed IFEval subset and time those same serial streaming requests."""

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from huggingface_hub import hf_hub_download
from lm_eval.tasks.ifeval.utils import process_results

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
META_REVISION = "d768f59e351416d894a9277a47195a0609ea58b7"
INSTRUCTIONS_REVISION = "5a5661c2a35488308556cf4453dc074d1eba91a0"
META_FILE = "Llama-3.1-8B-Instruct-evals/Details_ifeval__strict_2024-07-22T22-35-31.576573.parquet.gzip"


def load_cases():
    """Preserve Meta's row order and rendered chat prompts; never select by score."""
    meta = hf_hub_download(f"{MODEL}-evals", META_FILE, repo_type="dataset", revision=META_REVISION)
    original = hf_hub_download(
        "wis-k/instruction-following-eval", "input_data.jsonl", repo_type="dataset", revision=INSTRUCTIONS_REVISION
    )
    lookup = {row["prompt"]: row for row in map(json.loads, Path(original).read_text().splitlines())}
    cases = []
    for row in pd.read_parquet(meta).iloc[:28].to_dict("records"):
        question = json.loads(row["input_question"])["dialog"][0]["body"]
        # These two spelling corrections are part of Meta's published join.
        question = question.replace("Is it True that the first song", "Is it true that the first song")
        question = question.replace("Is the following True", "Is the following true")
        cases.append({**lookup[question], "prompt": row["input_final_prompts"][0]})
    return cases


def complete(client, prompt, max_tokens):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stop": [],
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": True,
    }
    first = last = usage = finish_reason = None
    pieces = []
    start = time.perf_counter()
    with client.stream("POST", "/v1/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice["text"]:
                    last = time.perf_counter()
                    first = last if first is None else first
                    pieces.append(choice["text"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    elapsed = time.perf_counter() - start
    if first is None or usage is None or finish_reason is None:
        raise RuntimeError("Incomplete streaming response: expected text, usage and finish reason")
    tokens = usage["completion_tokens"]
    return {
        "text": "".join(pieces),
        "usage": usage,
        "finish_reason": finish_reason,
        "elapsed_s": elapsed,
        "ttft_ms": 1000 * (first - start),
        "decode_tokens_per_s": (tokens - 1) / (last - first) if tokens > 1 and last > first else None,
    }


def run(base_url, output_dir, cases):
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases)
    (output_dir / "inputs.jsonl").write_text(inputs)
    results = []
    workload = cases * 2
    run_start = datetime.now(timezone.utc).isoformat()
    with httpx.Client(base_url=base_url, timeout=900) as client:
        setup_start = time.perf_counter()
        for case in cases:
            complete(client, case["prompt"], 1)
        setup_s = time.perf_counter() - setup_start
        print(f"Prompt warmup: {setup_s:.2f}s", flush=True)
        with (output_dir / "responses.jsonl").open("w") as responses:
            measurement_start = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            for case in workload:
                result = {"key": case["key"], **complete(client, case["prompt"], 1280)}
                results.append(result)
                responses.write(json.dumps(result) + "\n")
                responses.flush()
                print(
                    f"{result['key']}: {result['usage']['completion_tokens']} tokens in {result['elapsed_s']:.2f}s",
                    flush=True,
                )
            measured_s = time.perf_counter() - start
            measurement_end = datetime.now(timezone.utc).isoformat()
    scores = [process_results(case, [result["text"]]) for case, result in zip(workload, results)]
    replay_matches = all(a["text"] == b["text"] for a, b in zip(results[:28], results[28:]))
    accuracy = {}
    for name in scores[0]:
        values = [score[name] for score in scores]
        if isinstance(values[0], list):
            values = [value for row in values for value in row]
        accuracy[name] = statistics.mean(values)
    mean_accuracy = statistics.mean(accuracy.values())
    decode_tps = statistics.mean(result["decode_tokens_per_s"] for result in results if result["decode_tokens_per_s"])
    summary = {
        "run_start": run_start,
        "measurement_start": measurement_start,
        "measurement_end": measurement_end,
        "requests": len(results),
        "unique_prompts": len(cases),
        "passes": 2,
        "input_sha256": hashlib.sha256(inputs.encode()).hexdigest(),
        "meta_revision": META_REVISION,
        "instructions_revision": INSTRUCTIONS_REVISION,
        "warmup_s": setup_s,
        "measured_s": measured_s,
        "accuracy": accuracy,
        "mean_accuracy": mean_accuracy,
        "mean_decode_tokens_per_s": decode_tps,
        "mean_ttft_ms": statistics.mean(result["ttft_ms"] for result in results),
        "aggregate_output_tokens_per_s": sum(result["usage"]["completion_tokens"] for result in results) / measured_s,
        "gates": {
            "mean_accuracy": mean_accuracy >= 0.75,
            "mean_decode_tokens_per_s": decode_tps >= 110,
            "identical_replay": replay_matches,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not all(summary["gates"].values()):
        raise AssertionError(f"Benchmark gates failed: {summary['gates']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Use a previously prepared copy of the pinned 28 requests")
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.inputs.read_text().splitlines()] if args.inputs else load_cases()
    if len(cases) != 28:
        raise ValueError("The benchmark requires the fixed 28-request subset")
    run(args.base_url, args.output_dir, cases)
