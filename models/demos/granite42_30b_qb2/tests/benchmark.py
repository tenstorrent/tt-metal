# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Time and score the fixed GPQA CI subset; optionally measure fixed-length serving."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

MODEL = "ibm-granite/granite-4.2-30b"
MODEL_REVISION = "9e668ce1c538387ef24d3644e9b0606647762636"
DATASET_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
HARNESS_REVISION = "321e3bb68cb750a58c76606ab57832533302be73"
GPQA_TOKENS = 32768
GPQA_COUNT = 10
GPQA_CONCURRENCY = 5
GPQA_SEED = 42
GPQA_THRESHOLD = 0.6


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_gpqa():
    import datasets
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config

    # Choice permutations must come from this seed, not a previous map cache.
    datasets.disable_caching()
    random.seed(42)
    path = TaskManager().task_index["r1_gpqa_diamond"]["yaml_path"]
    config = load_yaml_config(path)
    config["dataset_kwargs"] = {"revision": DATASET_REVISION}
    task = ConfigurableTask(config=config)
    docs = task.validation_docs()
    if len(docs) != 198:
        raise ValueError(f"Expected 198 GPQA Diamond questions, got {len(docs)}")
    cases = [{"id": i, "doc": docs[i], "prompt": task.doc_to_text(docs[i])} for i in range(GPQA_COUNT)]
    return task, cases


async def complete(client, payload, *, chat):
    """TTFT starts at submission and includes queueing; decode includes reasoning tokens."""
    first = last = usage = finish = None
    text, reasoning = [], []
    start = time.perf_counter()
    endpoint = "/v1/chat/completions" if chat else "/v1/completions"
    payload = {"model": MODEL, **payload, "stream": True, "stream_options": {"include_usage": True}}
    async with client.stream("POST", endpoint, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError("Model server returned a streaming error; response details withheld")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") if chat else choice.get("text")
                thought = delta.get("reasoning") or delta.get("reasoning_content")
                if content or thought:
                    last = time.perf_counter()
                    first = last if first is None else first
                    if content:
                        text.append(content)
                    if thought:
                        reasoning.append(thought)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if usage is None or finish is None:
        raise RuntimeError("Incomplete stream: expected usage and a finish reason")
    tokens = usage["completion_tokens"]
    return {
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish,
        "elapsed_s": time.perf_counter() - start,
        "ttft_ms": 1000 * (first - start) if first is not None else None,
        "decode_tokens_per_s": (
            (tokens - 1) / (last - first) if tokens > 1 and first is not None and last > first else None
        ),
    }


async def complete_nonstreaming(client, payload, *, chat):
    """Match the validated release eval request path; response text stays in memory."""
    start = time.perf_counter()
    endpoint = "/v1/chat/completions" if chat else "/v1/completions"
    response = await client.post(endpoint, json={"model": MODEL, **payload, "stream": False})
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError("Model server returned an error; response details withheld")
    choices = data.get("choices", [])
    usage = data.get("usage")
    if len(choices) != 1 or usage is None or choices[0].get("finish_reason") is None:
        raise RuntimeError("Incomplete response: expected usage and one finished choice")
    choice = choices[0]
    message = choice.get("message", {}) if chat else {}
    return {
        "text": (message.get("content") if chat else choice.get("text")) or "",
        "reasoning": (message.get("reasoning") or message.get("reasoning_content")) if chat else "",
        "usage": usage,
        "finish_reason": choice["finish_reason"],
        "elapsed_s": time.perf_counter() - start,
        "ttft_ms": None,
        "decode_tokens_per_s": None,
    }


def summarize(rows, start, end, elapsed):
    ttfts = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    speeds = [r["decode_tokens_per_s"] for r in rows if r["decode_tokens_per_s"] is not None]
    return {
        "measurement_start": start,
        "measurement_end": end,
        "measured_s": elapsed,
        "requests": len(rows),
        "mean_ttft_ms": statistics.mean(ttfts) if ttfts else None,
        "mean_decode_tokens_per_s": statistics.mean(speeds) if speeds else None,
        "aggregate_output_tokens_per_s": sum(r["usage"]["completion_tokens"] for r in rows) / elapsed,
    }


def response_metrics(response):
    """Only numeric metrics and a fixed finish category may leave the evaluator."""
    return {
        "usage": {key: int(response["usage"][key]) for key in ("prompt_tokens", "completion_tokens")},
        "finish_reason": response["finish_reason"] if response["finish_reason"] in ("stop", "length") else "other",
        **{
            key: float(response[key]) if response[key] is not None else None
            for key in ("elapsed_s", "ttft_ms", "decode_tokens_per_s")
        },
    }


def score_answer(task, doc, text):
    """Use r1_gpqa_diamond's own final-answer extraction and exact-match scorer."""
    return task.process_results(doc, [text])["exact_match"]


async def run_gpqa(client, output_dir, task, cases):
    payloads = [
        {
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": GPQA_TOKENS,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": GPQA_SEED,
            "chat_template_kwargs": {"enable_thinking": True, "low_effort": False},
            "stop": [],
        }
        for case in cases
    ]
    rows = []
    semaphore = asyncio.Semaphore(GPQA_CONCURRENCY)
    with (output_dir / "gpqa-responses.jsonl").open("w") as log:

        async def generate(case, payload):
            async with semaphore:
                response = await complete_nonstreaming(client, payload, chat=True)
            score = score_answer(task, case["doc"], response["text"])
            row = {
                "id": case["id"],
                **response_metrics(response),
                "correct": int(score),
                "reasoning_present": bool(response["reasoning"]),
            }
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"GPQA {case['id']}: score={row['correct']}, tokens={row['usage']['completion_tokens']}", flush=True)

        start, tick = utc_now(), time.perf_counter()
        await asyncio.gather(*(generate(case, payload) for case, payload in zip(cases, payloads)))
        elapsed, end = time.perf_counter() - tick, utc_now()
    if len(rows) != GPQA_COUNT or len({r["id"] for r in rows}) != GPQA_COUNT:
        raise AssertionError("The CI result must include every selected question exactly once")
    accuracy = sum(r["correct"] for r in rows) / GPQA_COUNT
    return {
        **summarize(rows, start, end, elapsed),
        "accuracy": accuracy,
        "correct": sum(r["correct"] for r in rows),
        "completed_samples": GPQA_COUNT,
        "dataset_samples": 198,
        "concurrency": GPQA_CONCURRENCY,
        "max_output_tokens": GPQA_TOKENS,
        "passed": accuracy >= GPQA_THRESHOLD,
    }


def performance_shapes(request_concurrency, input_lengths):
    return [(length, request_concurrency) for length in input_lengths]


async def run_performance(client, output_dir, server_capacity, request_concurrency, input_lengths):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("HF_MODEL", MODEL), revision=MODEL_REVISION, local_files_only=True
    )
    passage = "The scientific method tests explanations against observations. Describe an experiment and its controls. "
    source = tokenizer.encode(passage, add_special_tokens=False)
    rows, summaries, inputs = [], [], []
    for length, batch in performance_shapes(request_concurrency, input_lengths):
        prompt = (source * (length // len(source) + 1))[:length]
        prompt_sha256 = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        inputs.append({"input_tokens": length, "concurrency": batch, "prompt": prompt, "sha256": prompt_sha256})
        (output_dir / "performance-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        payload = {
            "prompt": prompt,
            "max_tokens": 128,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 42,
            "stop": [],
            "add_special_tokens": False,
        }
        await asyncio.gather(*(complete(client, payload, chat=False) for _ in range(batch)))
        case_rows = []
        start, tick = utc_now(), time.perf_counter()
        for repeat in range(2):
            results = await asyncio.gather(*(complete(client, payload, chat=False) for _ in range(batch)))
            for result in results:
                if result["usage"]["prompt_tokens"] != length or result["usage"]["completion_tokens"] != 128:
                    raise AssertionError(f"Fixed-length benchmark returned unexpected token counts: {result['usage']}")
                case_rows.append(
                    {
                        "server_capacity": server_capacity,
                        "input_tokens": length,
                        "batch": batch,
                        "prompt_sha256": prompt_sha256,
                        "repeat": repeat,
                        **response_metrics(result),
                    }
                )
        elapsed, end = time.perf_counter() - tick, utc_now()
        rows.extend(case_rows)
        summaries.append(
            {
                "server_capacity": server_capacity,
                "prompt_sha256": prompt_sha256,
                "input_tokens": length,
                "output_tokens": 128,
                "concurrency": batch,
                "warmup_requests": batch,
                "repeats": 2,
                "ignore_eos": True,
                **summarize(case_rows, start, end, elapsed),
            }
        )
        (output_dir / "performance-responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps(summaries[-1]), flush=True)
    return summaries


async def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_concurrency = args.performance_concurrency or args.server_capacity
    task, cases = load_gpqa() if args.mode != "performance" else (None, [])
    inputs = "".join(json.dumps(c, sort_keys=True) + "\n" for c in cases)
    # GPQA terms prohibit publishing examples. Keep reproducibility hashes,
    # including the shuffled choices, without saving documents or prompts.
    manifest = [
        {"id": case["id"], "sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()}
        for case in cases
    ]
    (args.output_dir / "inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
    protocol = {
        "model": MODEL,
        "task": "r1_gpqa_diamond",
        "prompt_mode": "chat",
        "performance_prompt_mode": "raw token continuation stress",
        "server_capacity": args.server_capacity,
        "checkpoint_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "harness_revision": HARNESS_REVISION,
        "selection": "first 10 Diamond rows, choice shuffle seed 42",
        "scope": (
            "fixed-length performance only"
            if args.mode == "performance"
            else "10/198 CI subset; 32768-token output budget bounds weekly runtime"
        ),
        "gpqa": {
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": GPQA_SEED,
            "thinking": True,
            "max_output_tokens": GPQA_TOKENS,
            "concurrency": GPQA_CONCURRENCY,
            "accuracy_threshold": GPQA_THRESHOLD,
        },
        "performance": {
            "shapes": [
                [length, 128, batch]
                for length, batch in performance_shapes(request_concurrency, args.performance_input_lengths)
            ],
            "warmup": "one burst per shape",
            "repeats": 2,
            "temperature": 0,
            "ignore_eos": True,
        },
        "input_sha256": hashlib.sha256(inputs.encode()).hexdigest(),
        "run_start": utc_now(),
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if args.prepare_only:
        return
    summary = dict(protocol)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=7200) as client:
        if args.mode != "performance":
            summary["gpqa_result"] = await run_gpqa(client, args.output_dir, task, cases)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.mode != "gpqa":
            summary["performance_results"] = await run_performance(
                client,
                args.output_dir,
                args.server_capacity,
                request_concurrency,
                args.performance_input_lengths,
            )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if "gpqa_result" in summary and not summary["gpqa_result"]["passed"]:
        raise AssertionError(f"GPQA CI accuracy below {GPQA_THRESHOLD}: {summary['gpqa_result']['accuracy']}")


def run(args):
    try:
        asyncio.run(main(args))
    except Exception as error:
        # Library/server exceptions can include prompts or generated text.
        # Preserve the failure category, never its message or chained traceback.
        failure = {"error_type": type(error).__name__, "details": "Withheld to protect evaluation content"}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "error.json").write_text(json.dumps(failure) + "\n")
        print(json.dumps(failure), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("gpqa", "performance", "all"), default="all")
    parser.add_argument(
        "--server-capacity",
        type=int,
        choices=(1, 8, 16),
        default=16,
        help="The server max-num-seqs setting.",
    )
    parser.add_argument(
        "--performance-concurrency",
        type=int,
        choices=(1, 8, 16),
        help="Measured request concurrency; defaults to --server-capacity.",
    )
    parser.add_argument("--performance-input-lengths", type=int, nargs="+", choices=(128, 1024), default=(128, 1024))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args))
