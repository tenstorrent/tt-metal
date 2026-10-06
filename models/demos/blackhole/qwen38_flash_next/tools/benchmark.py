# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Score and time the same natural-EOS GSM8K responses through the live server."""

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx

MODEL = "Qwen/Qwen3.8-Flash-Next"
CHECKPOINT_REVISION = "2741eec155d03a8ce151b993ccce1a7b1e398d6b"
DATASET_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATASET_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
DATASET_SAMPLES = 1319
ACCURACY_FLOOR = 0.80
INSTRUCTION = "\nSolve the problem step by step. End your answer with #### followed by the final number."
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def answer(text):
    """Only the final explicit answer marker is scored; missing/malformed answers fail."""
    if "####" not in text:
        return None
    value = text.rsplit("####", 1)[1].strip()
    if value.endswith("."):
        value = value[:-1]
    if not NUMBER.fullmatch(value):
        return None
    return Decimal(value.replace(",", ""))


def load_cases(path, limit):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA256:
        raise ValueError("GSM8K file does not match the pinned dataset")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != DATASET_SAMPLES or limit not in (64, DATASET_SAMPLES):
        raise ValueError("Use the complete 1319-item test set or the predefined first-64 smoke subset")
    return [dict(id=index, **row) for index, row in enumerate(rows[:limit])]


def request_payload(question, max_tokens=512):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "seed": 42,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "repetition_penalty": 1,
        "max_tokens": max_tokens,
        "ignore_eos": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def complete(client, payload):
    first = last = usage = finish = None
    content, reasoning = [], []
    done = False
    start = time.perf_counter()
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"Server stream failed: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content")
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if text:
                    last = time.perf_counter()
                    first = last if first is None else first
                    content.append(text)
                if thought:
                    reasoning.append(thought)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if not done or usage is None or finish is None:
        raise RuntimeError("Incomplete response: require [DONE], usage, and a finish reason")
    tokens = int(usage["completion_tokens"])
    return {
        "text": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish,
        "elapsed_s": time.perf_counter() - start,
        "ttft_s": first - start if first is not None else None,
        "content_span_s": last - first if first is not None else None,
        "stream_decode_tokens_per_s": (
            (tokens - 1) / (last - first) if tokens > 1 and first is not None and last > first else None
        ),
    }


def wilson_interval(correct, count):
    z = 1.959963984540054
    p = correct / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return [center - radius, center + radius]


def summarize(rows, requested, started, elapsed):
    correct = sum(row["correct"] for row in rows)
    ttft = [row["ttft_s"] for row in rows if row["ttft_s"] is not None]
    speeds = [row["stream_decode_tokens_per_s"] for row in rows if row["stream_decode_tokens_per_s"] is not None]
    return {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "dataset_samples": DATASET_SAMPLES,
        "requested_samples": requested,
        "completed_samples": len(rows),
        "scope": "full test set" if requested == DATASET_SAMPLES else "first 64/1319 smoke subset",
        "measurement_start": started,
        "measurement_end": utc_now(),
        "measured_s": elapsed,
        "concurrency": 1,
        "max_output_tokens": 512,
        "ignore_eos": False,
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "accuracy_95pct_wilson_interval": wilson_interval(correct, len(rows)) if rows else None,
        "accuracy_floor": ACCURACY_FLOOR,
        "passed": len(rows) == requested and correct / requested >= ACCURACY_FLOOR,
        "truncated_responses": sum(row["finish_reason"] == "length" for row in rows),
        "mean_ttft_s": statistics.mean(ttft) if ttft else None,
        "mean_stream_decode_tokens_per_s": statistics.mean(speeds) if speeds else None,
        "aggregate_output_tokens_per_s": sum(row["usage"]["completion_tokens"] for row in rows) / elapsed,
        "timing_note": "Streaming content timing; decode rate is a client estimate, not a device measurement.",
    }


async def run(args):
    from transformers import AutoTokenizer

    cases = load_cases(args.dataset, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payloads = []
    for case in cases:
        payload = request_payload(case["question"] + INSTRUCTION)
        ids = tokenizer.apply_chat_template(
            payload["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        payloads.append(
            {"id": case["id"], "payload": payload, "prompt_token_ids": ids, "expected": str(answer(case["answer"]))}
        )
    (args.output_dir / "inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in payloads))
    rows = []
    async with httpx.AsyncClient(base_url=args.url, timeout=1800) as client:
        warmup = await complete(client, request_payload("What is 2 + 2?", max_tokens=16))
        (args.output_dir / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
        started, tick = utc_now(), time.perf_counter()
        try:
            with (args.output_dir / "responses.jsonl").open("w") as log:
                for item in payloads:
                    response = await complete(client, item["payload"])
                    actual = answer(response["text"])
                    correct = actual is not None and actual == Decimal(item["expected"])
                    row = {
                        "id": item["id"],
                        **response,
                        "answer": None if actual is None else str(actual),
                        "correct": correct,
                    }
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(
                        f"GSM8K {item['id']}: correct={correct}, tokens={response['usage']['completion_tokens']}",
                        flush=True,
                    )
        finally:
            summary = summarize(rows, args.limit, started, time.perf_counter() - tick)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(64, DATASET_SAMPLES), default=DATASET_SAMPLES)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
