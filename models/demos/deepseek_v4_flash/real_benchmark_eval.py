# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Small benchmark/eval harness over the real DSV4 Flash server adapter.

This intentionally remains a two-layer stepping-stone artifact. It batches
JSON-shaped adapter requests, reports compact per-request summaries, and emits
aggregate correctness and latency metrics for future eval/CI glue.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from models.demos.deepseek_v4_flash.real_decode_logits_smoke import DEFAULT_DECODE_LOGITS_TOP_K
from models.demos.deepseek_v4_flash.real_decode_stack_logits_smoke import DEFAULT_DECODE_STACK_LAYERS
from models.demos.deepseek_v4_flash.real_generation_eval_runner import DEFAULT_REAL_SNAPSHOT_DIR
from models.demos.deepseek_v4_flash.real_input_stack_logits_smoke import DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS
from models.demos.deepseek_v4_flash.real_multi_token_decode_smoke import (
    DEFAULT_MULTI_TOKEN_MAX_BYTES,
    DEFAULT_MULTI_TOKEN_MAX_TENSORS,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import DEFAULT_SEQUENCE_LENGTH
from models.demos.deepseek_v4_flash.real_server_adapter import (
    REAL_SERVER_ADAPTER_NAME,
    REAL_SERVER_LIMITATION_FLAGS,
    RealServerRequest,
    ensure_real_server_request,
    run_real_server_request,
)

REAL_BENCHMARK_EVAL_SCHEMA_VERSION = 1
REAL_BENCHMARK_EVAL_NAME = "deepseek_v4_flash_real_benchmark_eval"


def load_request_json(path: str | Path) -> list[RealServerRequest]:
    """Load a single ``RealServerRequest`` JSON object."""

    request_path = Path(path).expanduser()
    with request_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{request_path} must contain a JSON object request")
    return [RealServerRequest.from_mapping(payload)]


def load_requests_jsonl(path: str | Path) -> list[RealServerRequest]:
    """Load one ``RealServerRequest`` JSON object per non-empty JSONL line."""

    request_path = Path(path).expanduser()
    requests: list[RealServerRequest] = []
    with request_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{request_path}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{request_path}:{line_number} must contain a JSON object request")
            requests.append(RealServerRequest.from_mapping(payload))
    if not requests:
        raise ValueError(f"{request_path} did not contain any requests")
    return requests


def run_real_benchmark_eval(
    requests: Sequence[RealServerRequest | Mapping[str, Any]],
    *,
    request_source: str = "direct",
    request_runner: Callable[[RealServerRequest], Mapping[str, Any]] = run_real_server_request,
) -> dict[str, Any]:
    normalized_requests = [ensure_real_server_request(request) for request in requests]
    if not normalized_requests:
        raise ValueError("At least one request is required")

    summaries: list[dict[str, Any]] = []
    for request in normalized_requests:
        request_start = time.perf_counter()
        response = request_runner(request)
        request_latency_seconds = _seconds_since(request_start)
        summaries.append(
            compact_real_benchmark_request_summary(
                response,
                request_latency_seconds=request_latency_seconds,
            )
        )

    aggregate = aggregate_real_benchmark_summaries(summaries)
    return {
        "schema_version": REAL_BENCHMARK_EVAL_SCHEMA_VERSION,
        "harness": {
            "name": REAL_BENCHMARK_EVAL_NAME,
            "adapter": REAL_SERVER_ADAPTER_NAME,
            "request_source": request_source,
            "scope": "two_layer_stepping_stone",
            "limitations": dict(REAL_SERVER_LIMITATION_FLAGS),
        },
        "aggregate": aggregate,
        "metrics": aggregate["metrics"],
        "per_request": summaries,
    }


def compact_real_benchmark_request_summary(
    response: Mapping[str, Any],
    *,
    request_latency_seconds: float,
) -> dict[str, Any]:
    timing = response["timing"]
    tokens = response["tokens"]
    vocab = response["vocab"]
    prompt_tokens = len(tokens["prefill_token_ids"])
    generated_ids = [int(value) for value in tokens["generated_token_ids"]]
    generated_tokens = len(generated_ids)
    decode_latency_seconds = _decode_latency_seconds(timing)
    mean_decode_latency_per_token = (
        _round_float(decode_latency_seconds / generated_tokens)
        if decode_latency_seconds is not None and generated_tokens > 0
        else None
    )

    return {
        "request_id": response["request_id"],
        "passed": bool(response["passed"]),
        "mode": response["mode"],
        "tokens": {
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated_tokens),
        },
        "generated_ids": generated_ids,
        "reference_generated_ids": [int(value) for value in tokens["reference_generated_token_ids"]],
        "ttnn_generated_ids": [int(value) for value in tokens["ttnn_generated_token_ids"]],
        "top1_ids_match": response["correctness"]["top1_ids_match"],
        "top_k": response["top_k"],
        "timing": {
            "end_to_end_latency_seconds": _round_float(request_latency_seconds),
            "adapter_end_to_end_wall_seconds": _optional_round_float(timing.get("end_to_end_wall_seconds")),
            "runner_wall_seconds": _optional_round_float(timing.get("runner_wall_seconds")),
            "decode_latency_seconds": decode_latency_seconds,
            "mean_decode_latency_per_token_seconds": mean_decode_latency_per_token,
            "tokens_per_sec_per_user": _optional_round_float(timing.get("tokens_per_sec_per_user")),
            "decode_tokens_per_sec_denominator": timing.get("decode_tokens_per_sec_denominator"),
        },
        "tokens_per_sec_per_user": _optional_round_float(timing.get("tokens_per_sec_per_user")),
        "vocab": {
            "mode": vocab["mode"],
            "vocab_start": int(vocab["vocab_start"]),
            "vocab_size": int(vocab["vocab_size"]),
            "full_vocab_size": int(vocab["full_vocab_size"]),
            "deterministic_slice": vocab["deterministic_slice"],
        },
        "payload_bytes": _numeric_mapping(response["payload_bytes"]),
        "limitation_flags": dict(response["limitation_flags"]),
    }


def aggregate_real_benchmark_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("At least one request summary is required")

    request_count = len(summaries)
    passed_count = sum(1 for summary in summaries if summary["passed"])
    top1_values = [
        bool(summary["top1_ids_match"]) for summary in summaries if summary.get("top1_ids_match") is not None
    ]
    top1_match_count = sum(1 for value in top1_values if value)
    total_prompt_tokens = sum(int(summary["tokens"]["prompt_tokens"]) for summary in summaries)
    total_generated_tokens = sum(int(summary["tokens"]["generated_tokens"]) for summary in summaries)

    end_to_end_latencies = [float(summary["timing"]["end_to_end_latency_seconds"]) for summary in summaries]
    decode_latency_seconds_total = 0.0
    decode_latency_token_count = 0
    for summary in summaries:
        decode_latency_seconds = summary["timing"]["decode_latency_seconds"]
        if decode_latency_seconds is None:
            continue
        generated_tokens = int(summary["tokens"]["generated_tokens"])
        decode_latency_seconds_total += float(decode_latency_seconds)
        decode_latency_token_count += generated_tokens

    mean_decode_latency_per_token = (
        _round_float(decode_latency_seconds_total / decode_latency_token_count)
        if decode_latency_token_count > 0
        else None
    )
    aggregate_decode_tokens_per_sec_per_user = (
        _round_float(decode_latency_token_count / decode_latency_seconds_total)
        if decode_latency_seconds_total > 0.0
        else None
    )
    payload_byte_totals = _sum_payload_bytes(summaries)

    aggregate = {
        "request_count": int(request_count),
        "passed_count": int(passed_count),
        "pass_rate": _rate(passed_count, request_count),
        "top1_match_available_count": len(top1_values),
        "top1_match_count": int(top1_match_count),
        "top1_match_rate": _rate(top1_match_count, len(top1_values)) if top1_values else None,
        "total_prompt_tokens": int(total_prompt_tokens),
        "total_generated_tokens": int(total_generated_tokens),
        "mean_end_to_end_latency_seconds": _round_float(statistics.fmean(end_to_end_latencies)),
        "p50_end_to_end_latency_seconds": _round_float(statistics.median(end_to_end_latencies)),
        "max_end_to_end_latency_seconds": _round_float(max(end_to_end_latencies)),
        "decode_latency_seconds_total": _round_float(decode_latency_seconds_total),
        "decode_latency_token_count": int(decode_latency_token_count),
        "mean_decode_latency_per_token_seconds": mean_decode_latency_per_token,
        "aggregate_decode_tokens_per_sec_per_user": aggregate_decode_tokens_per_sec_per_user,
        "vocab_modes_used": sorted({str(summary["vocab"]["mode"]) for summary in summaries}),
        "payload_byte_totals": payload_byte_totals,
    }
    aggregate["metrics"] = {
        "request_count": aggregate["request_count"],
        "passed_count": aggregate["passed_count"],
        "pass_rate": aggregate["pass_rate"],
        "top1_match_available_count": aggregate["top1_match_available_count"],
        "top1_match_count": aggregate["top1_match_count"],
        "top1_match_rate": aggregate["top1_match_rate"],
        "prompt_tokens": aggregate["total_prompt_tokens"],
        "generated_tokens": aggregate["total_generated_tokens"],
        "mean_end_to_end_latency_s": aggregate["mean_end_to_end_latency_seconds"],
        "p50_end_to_end_latency_s": aggregate["p50_end_to_end_latency_seconds"],
        "max_end_to_end_latency_s": aggregate["max_end_to_end_latency_seconds"],
        "mean_decode_latency_per_token_s": aggregate["mean_decode_latency_per_token_seconds"],
        "aggregate_decode_tokens_s_per_user": aggregate["aggregate_decode_tokens_per_sec_per_user"],
        "payload_bytes_total": payload_byte_totals.get("total", 0),
    }
    return aggregate


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a compact DeepSeek V4 Flash real benchmark/eval harness over real_server_adapter."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--request-json", type=Path, help="Path to one RealServerRequest JSON object.")
    input_group.add_argument("--requests-jsonl", type=Path, help="Path to JSONL RealServerRequest objects.")
    parser.add_argument("--request-id", default="benchmark-request-0")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(DEFAULT_REAL_SNAPSHOT_DIR))),
    )
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--input-ids-path", type=Path)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_DECODE_STACK_LAYERS))
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODE_LOGITS_TOP_K)
    parser.add_argument("--embedding-mode", choices=("slice", "full"), default="slice")
    parser.add_argument("--max-embedding-rows", type=int, default=DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS)
    parser.add_argument(
        "--vocab-mode",
        choices=("full", "slice"),
        default=os.environ.get("DSV4_FLASH_REAL_BENCHMARK_EVAL_VOCAB_MODE", "slice"),
    )
    parser.add_argument("--full-vocab-smoke", action="store_true")
    parser.add_argument(
        "--vocab-start",
        type=int,
        default=_optional_int_env("DSV4_FLASH_REAL_BENCHMARK_EVAL_VOCAB_START", 0),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=_optional_int_env("DSV4_FLASH_REAL_BENCHMARK_EVAL_VOCAB_SIZE"),
    )
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_MULTI_TOKEN_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MULTI_TOKEN_MAX_BYTES)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    requests, request_source = _requests_from_args(args)
    result = run_real_benchmark_eval(requests, request_source=request_source)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    if result["aggregate"]["passed_count"] != result["aggregate"]["request_count"]:
        raise SystemExit(1)


def _requests_from_args(args: argparse.Namespace) -> tuple[list[RealServerRequest], str]:
    if args.request_json is not None:
        return load_request_json(args.request_json), "request_json"
    if args.requests_jsonl is not None:
        return load_requests_jsonl(args.requests_jsonl), "requests_jsonl"
    return [_direct_request_from_args(args)], "direct_flags"


def _direct_request_from_args(args: argparse.Namespace) -> RealServerRequest:
    return RealServerRequest(
        request_id=args.request_id,
        snapshot_dir=args.snapshot_dir,
        input_ids=args.input_ids,
        input_ids_path=args.input_ids_path,
        prompt_path=args.prompt_path,
        max_tokens=args.max_tokens,
        decode_steps=args.decode_steps,
        prefill_seq_len=args.prefill_seq_len,
        layers=args.layers,
        top_k=args.top_k,
        embedding_mode=args.embedding_mode,
        max_embedding_rows=args.max_embedding_rows,
        vocab_mode=args.vocab_mode,
        full_vocab=args.full_vocab_smoke,
        vocab_start=args.vocab_start,
        vocab_size=args.vocab_size,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
    )


def _decode_latency_seconds(timing: Mapping[str, Any]) -> float | None:
    denominator = timing.get("decode_tokens_per_sec_denominator")
    if isinstance(denominator, str) and timing.get(denominator) is not None:
        return _round_float(float(timing[denominator]))
    for key in ("ttnn_decode_total_seconds", "decode_build_total_seconds"):
        if timing.get(key) is not None:
            return _round_float(float(timing[key]))
    return None


def _sum_payload_bytes(summaries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for summary in summaries:
        for key, value in summary["payload_bytes"].items():
            totals[key] = totals.get(key, 0) + int(value)
    return dict(sorted(totals.items()))


def _numeric_mapping(data: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value) for key, value in data.items() if isinstance(value, int) and not isinstance(value, bool)
    }


def _rate(count: int, total: int) -> float:
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    return _round_float(float(count) / float(total))


def _seconds_since(start: float) -> float:
    return max(0.0, time.perf_counter() - start)


def _optional_round_float(value: Any) -> float | None:
    if value is None:
        return None
    return _round_float(float(value))


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _optional_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    main()
