# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Latency sweep against a live OpenAI API where every request calls a tool.

This is intentionally separate from the generator-level fixed-OSL sweep. A
plain token loop cannot prove that vLLM loaded the model-owned parser or that a
coding client receives a structured call. This runner measures the API path,
validates every response, and records the actual prompt/completion token counts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

HF_MODEL_ID = "meta-models/Muse-Glimmer-30B"
WEIGHT_REVISION = "f84ecc3a0ea984a4c04542a84269e3d065350a6e"
HF_ADVERTISED_CONTEXT = 131072
DEFAULT_MAX_TOKENS = 512
DEFAULT_REPEATS = 3
DEFAULT_ISLS = (512, 1024, 4096, 8192, 16384, 32768, 65536, 130560)
FILLER = " context"
INSTRUCTION = "\nCall record_latency_probe exactly once with payload ready."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_latency_probe",
            "description": "Record one benchmark probe after reading the supplied context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "string",
                        "description": "Must be exactly ready",
                    }
                },
                "required": ["payload"],
            },
        },
    }
]


def prompt_tokens(tokenizer, content: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return len(input_ids)


def exact_prompt(tokenizer, target_isl: int) -> str:
    """Build a tool-enabled chat prompt with exactly ``target_isl`` tokens."""
    base = prompt_tokens(tokenizer, INSTRUCTION)
    if target_isl < base:
        raise ValueError(f"target ISL {target_isl} is below the tool-template floor of {base}")
    # For this pinned tokenizer, each leading ``' context'`` contributes exactly
    # one token. Assert that contract instead of silently reporting the wrong ISL
    # if a future tokenizer revision changes its segmentation.
    content = FILLER * (target_isl - base) + INSTRUCTION
    actual = prompt_tokens(tokenizer, content)
    if actual != target_isl:
        raise RuntimeError(
            f"could not construct exact ISL {target_isl}: tokenizer produced {actual}; "
            "update FILLER for the pinned tokenizer revision"
        )
    return content


def _append_tool_delta(calls: dict[int, dict[str, str]], delta: dict[str, Any]) -> None:
    for item in delta.get("tool_calls") or []:
        index = int(item.get("index", 0))
        call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        call["id"] += item.get("id") or ""
        function = item.get("function") or {}
        call["name"] += function.get("name") or ""
        call["arguments"] += function.get("arguments") or ""


def measure_request(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    content: str,
    expected_isl: int,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "tools": TOOLS,
        "tool_choice": "required",
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_semantic: float | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    calls: dict[int, dict[str, str]] = {}
    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    with client.stream(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(f"server returned an SSE error: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                semantic = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                )
                if semantic and first_semantic is None:
                    first_semantic = time.perf_counter()
                if delta.get("content"):
                    content_parts.append(delta["content"])
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    reasoning_parts.append(reasoning)
                _append_tool_delta(calls, delta)

    finished = time.perf_counter()
    if first_semantic is None:
        raise RuntimeError("stream completed without content, reasoning, or a tool-call delta")
    ordered_calls = [calls[index] for index in sorted(calls)]
    if finish_reason != "tool_calls":
        raise RuntimeError(f"expected finish_reason='tool_calls', got {finish_reason!r}")
    if len(ordered_calls) != 1 or ordered_calls[0]["name"] != "record_latency_probe":
        raise RuntimeError(f"expected one record_latency_probe call, got {ordered_calls!r}")
    try:
        arguments = json.loads(ordered_calls[0]["arguments"])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"tool arguments are not JSON: {ordered_calls[0]['arguments']!r}") from exc
    if arguments != {"payload": "ready"}:
        raise RuntimeError(f"unexpected tool arguments: {arguments!r}")

    actual_isl = int(usage.get("prompt_tokens", -1))
    completion_tokens = int(usage.get("completion_tokens", -1))
    if actual_isl != expected_isl:
        raise RuntimeError(f"server counted {actual_isl} prompt tokens; expected exact ISL {expected_isl}")
    if completion_tokens < 1:
        raise RuntimeError(f"missing/invalid completion token usage: {usage!r}")

    ttft_ms = (first_semantic - started) * 1000.0
    e2el_ms = (finished - started) * 1000.0
    tpot_ms = (e2el_ms - ttft_ms) / max(completion_tokens - 1, 1)
    return {
        "isl": actual_isl,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms_derived": tpot_ms,
        "e2el_ms": e2el_ms,
        "tokens_per_second_per_user_derived": 1000.0 / tpot_ms if tpot_ms > 0 else 0.0,
        "finish_reason": finish_reason,
        "tool_name": ordered_calls[0]["name"],
        "tool_arguments": arguments,
        "content": "".join(content_parts),
        "reasoning_chars": len("".join(reasoning_parts)),
        "tool_call_pass": True,
    }


def median_row(target_isl: int, samples: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = ("completion_tokens", "ttft_ms", "tpot_ms_derived", "e2el_ms", "tokens_per_second_per_user_derived")
    row = {"isl": target_isl, "repeats": len(samples), "tool_call_pass": all(s["tool_call_pass"] for s in samples)}
    for key in numeric:
        row[key] = statistics.median(float(sample[key]) for sample in samples)
    row["samples"] = samples
    return row


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Checkpoint a sweep so completed rows survive a later failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:20000")
    parser.add_argument("--model", default=HF_MODEL_ID)
    parser.add_argument("--revision", default=WEIGHT_REVISION)
    parser.add_argument("--profile", required=True, choices=("p150", "p150x2", "p150x4"))
    parser.add_argument("--isl", type=int, nargs="*", default=list(DEFAULT_ISLS))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--source-revision", help="Exact source commit serving this sweep")
    parser.add_argument("--image-digest", help="Exact packaged image digest, when measuring a package")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    for isl in args.isl:
        if isl + args.max_tokens > HF_ADVERTISED_CONTEXT:
            raise SystemExit(f"ISL {isl} + max_tokens {args.max_tokens} exceeds context {HF_ADVERTISED_CONTEXT}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
    )
    prompts = {isl: exact_prompt(tokenizer, isl) for isl in args.isl}
    timeout = httpx.Timeout(args.timeout, connect=15.0)
    rows: list[dict[str, Any]] = []
    artifact: dict[str, Any] | None = None

    with httpx.Client(timeout=timeout) as client:
        health = client.get(f"{args.base_url.rstrip('/')}/health")
        health.raise_for_status()
        models_response = client.get(f"{args.base_url.rstrip('/')}/v1/models")
        models_response.raise_for_status()
        server_models = [str(item.get("id")) for item in models_response.json().get("data", [])]
        if args.model not in server_models:
            raise RuntimeError(f"server exposes models {server_models!r}; expected {args.model!r}")

        artifact = {
            "schema": 2,
            "status": "in_progress",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": args.profile,
            "base_url": args.base_url,
            "model": args.model,
            "server_models": server_models,
            "weight_revision": args.revision,
            "source_revision": args.source_revision,
            "image_digest": args.image_digest,
            "context_limit": HF_ADVERTISED_CONTEXT,
            "planned_isls": list(args.isl),
            "completed_isls": [],
            "max_tokens": args.max_tokens,
            "batch_size": 1,
            "repeats": args.repeats,
            "warmup_per_shape": not args.no_warmup,
            "measurement": {
                "ttft": "client start to first semantic SSE delta (reasoning, content, or tool call)",
                "e2el": "client start through SSE completion",
                "tpot": "derived as (E2EL - TTFT) / (completion_tokens - 1)",
            },
            "tool_contract": {"name": "record_latency_probe", "arguments": {"payload": "ready"}},
            "rows": rows,
        }
        write_artifact(args.out, artifact)
        try:
            for isl in args.isl:
                if not args.no_warmup:
                    measure_request(
                        client,
                        base_url=args.base_url,
                        model=args.model,
                        content=prompts[isl],
                        expected_isl=isl,
                        max_tokens=args.max_tokens,
                    )
                samples = [
                    measure_request(
                        client,
                        base_url=args.base_url,
                        model=args.model,
                        content=prompts[isl],
                        expected_isl=isl,
                        max_tokens=args.max_tokens,
                    )
                    for _ in range(args.repeats)
                ]
                row = median_row(isl, samples)
                rows.append(row)
                artifact["completed_isls"] = [item["isl"] for item in rows]
                write_artifact(args.out, artifact)
                print(
                    f"{isl:>7,} in | {row['completion_tokens']:>5.0f} out | "
                    f"TTFT {row['ttft_ms']:>9.1f} ms | TPOT {row['tpot_ms_derived']:>7.2f} ms | "
                    f"E2E {row['e2el_ms']:>10.1f} ms | tool PASS",
                    flush=True,
                )
        except Exception as exc:
            artifact["status"] = "failed"
            artifact["error"] = f"{type(exc).__name__}: {exc}"
            write_artifact(args.out, artifact)
            raise

    assert artifact is not None
    artifact["status"] = "complete"
    artifact.pop("error", None)
    write_artifact(args.out, artifact)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
