# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Laguna prefix-cache qualification against a running vLLM server.

This is intentionally a host-only client.  It does not import TTNN, start a server,
or reset devices.  Run the two phases against fresh, otherwise-identical p150x2
servers and preserve both JSON artifacts::

    python -m models.autoports.poolside_laguna_xs_2_1.tests.prefix_cache_qualification \
        off --output /tmp/laguna-prefix/off.json

    python -m models.autoports.poolside_laguna_xs_2_1.tests.prefix_cache_qualification \
        on --oracle /tmp/laguna-prefix/off.json \
        --output /tmp/laguna-prefix/on.json

The cache-on server must expose prompt-token details (vLLM
``--enable-prompt-tokens-details``); the cache-off oracle may omit them. Requests
use raw token ids, cache salts, streaming TTFT/TPOT measurement, and returned
prompt/output token ids. The ``off`` phase records exact-token oracles and cold
performance. The ``on`` phase proves cold, full-hit, and partial-hit correctness
and applies the accuracy-first production gates:

* cached-token counts exactly match canonical 8192-token admission;
* 32K full-hit TTFT speedup >= 3x;
* 65K full-hit TTFT speedup >= 2x;
* full-hit TTFT is lower than cache-on cold TTFT;
* cache-on cold TTFT regression <= 5%; and
* cache-on cold and hit TPOT regression <= 2%.

The default suite is deliberately long-running: it includes a near-context-cap
partial hit in addition to the 32K/65K performance shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import secrets
import statistics
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCHEMA_VERSION = 1
DEFAULT_MODEL = "poolside/Laguna-XS-2.1"
DEFAULT_VOCAB_SIZE = 100_352
DEFAULT_BLOCK_SIZE = 64
DEFAULT_CACHE_ADMISSION_GRANULARITY = 8_192
DEFAULT_MAX_CONTEXT = 131_072
DEFAULT_SEED = 1234
DEFAULT_EXCLUDED_TOKEN_IDS = (2, 9, 24)
PREFIX_CACHE_QUERIES_METRIC = "vllm:prefix_cache_queries_total"
PREFIX_CACHE_HITS_METRIC = "vllm:prefix_cache_hits_total"
POISON_SEED_LEN = 2_048
POISON_TARGET_LEN = 32_768
DECODE_BOUNDARY_PROMPT_HEADROOM = 12
DECODE_BOUNDARY_SEED_OUTPUT_LEN = 16


@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    prefix_len: int
    suffix_len: int
    suffix_seed: int

    @property
    def prompt_len(self) -> int:
        return self.prefix_len + self.suffix_len


@dataclass(frozen=True)
class PerformanceCase:
    name: str
    prompt_len: int
    minimum_speedup: float


@dataclass(frozen=True)
class PoisonOrderStep:
    name: str
    prompt_len: int
    output_len: int
    raw_candidate_cached_tokens: int
    expected_cached_tokens: int
    compare_with_full_32k_oracle: bool


@dataclass(frozen=True)
class DecodeBoundarySpec:
    admission_boundary: int
    seed_prompt_len: int
    seed_output_len: int
    target_appended_decode_tokens: int
    target_prompt_len: int
    potential_poisoned_hit_tokens: int
    expected_cached_tokens: int


CORRECTNESS_CASES = (
    CorrectnessCase("partial_2k", 2_048, 65, 20_481),
    CorrectnessCase("partial_32k", 32_768, 257, 32_769),
    CorrectnessCase("partial_65k", 65_536, 257, 65_537),
    CorrectnessCase("partial_near_cap", 129_984, 64, 130_049),
)

PERFORMANCE_CASES = (
    PerformanceCase("full_32k", 32_768, 3.0),
    PerformanceCase("full_65k", 65_536, 2.0),
)


@dataclass
class CompletionResult:
    request_id: str | None
    prompt_tokens: int
    output_tokens: int
    cached_tokens: int | None
    prompt_token_ids_sha256: str
    token_ids: list[int]
    token_ids_sha256: str
    finish_reason: str | None
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float


class QualificationError(RuntimeError):
    """A server response violated the qualification interface or contract."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def token_ids_sha256(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<I", int(token)))
    return digest.hexdigest()


def deterministic_token_ids(
    length: int,
    *,
    seed: int,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    excluded: Iterable[int] = DEFAULT_EXCLUDED_TOKEN_IDS,
) -> list[int]:
    """Generate exact, reproducible in-vocabulary ids without tokenizer round trips."""
    if length <= 0:
        raise ValueError(f"token length must be positive, got {length}")
    excluded_set = {int(token) for token in excluded}
    candidates = tuple(token for token in range(vocab_size) if token not in excluded_set)
    if not candidates:
        raise ValueError("vocabulary contains no usable token ids")
    rng = random.Random(seed)
    return [candidates[rng.randrange(len(candidates))] for _ in range(length)]


def build_base_tokens(max_length: int, *, vocab_size: int, seed: int) -> list[int]:
    return deterministic_token_ids(max_length, seed=seed, vocab_size=vocab_size)


def correctness_target(
    case: CorrectnessCase,
    base_tokens: Sequence[int],
    *,
    vocab_size: int,
) -> list[int]:
    if case.prefix_len > len(base_tokens):
        raise ValueError(f"{case.name} prefix exceeds generated base")
    suffix = deterministic_token_ids(
        case.suffix_len,
        seed=case.suffix_seed,
        vocab_size=vocab_size,
    )
    target = [*base_tokens[: case.prefix_len], *suffix]
    if target[case.prefix_len :] == list(base_tokens[case.prefix_len : case.prompt_len]):
        raise AssertionError(f"{case.name} suffix unexpectedly equals the base continuation")
    return target


def expected_full_hit_tokens(prompt_len: int, block_size: int) -> int:
    """vLLM recomputes the final block of an otherwise complete prompt hit."""
    if prompt_len <= 0 or block_size <= 0:
        raise ValueError("prompt_len and block_size must be positive")
    return ((prompt_len - 1) // block_size) * block_size


def expected_admitted_hit_tokens(
    raw_candidate_cached_tokens: int,
    admission_granularity: int = DEFAULT_CACHE_ADMISSION_GRANULARITY,
) -> int:
    """Floor a raw prefix match to the canonical safe-admission boundary."""
    if raw_candidate_cached_tokens < 0:
        raise ValueError("raw_candidate_cached_tokens must be non-negative")
    if admission_granularity <= 0:
        raise ValueError("admission_granularity must be positive")
    return (raw_candidate_cached_tokens // admission_granularity) * admission_granularity


def poison_order_plan(
    *,
    output_len: int,
    block_size: int,
    admission_granularity: int = DEFAULT_CACHE_ADMISSION_GRANULARITY,
) -> tuple[PoisonOrderStep, ...]:
    """Exercise the 2K-to-32K oldest-hash poisoning order on one cache salt."""
    if output_len <= 0:
        raise ValueError("output_len must be positive")
    repeat_raw = expected_full_hit_tokens(POISON_TARGET_LEN, block_size)
    return (
        PoisonOrderStep(
            "seed_2k",
            POISON_SEED_LEN,
            1,
            0,
            0,
            False,
        ),
        PoisonOrderStep(
            "target_32k_after_2k",
            POISON_TARGET_LEN,
            output_len,
            POISON_SEED_LEN,
            expected_admitted_hit_tokens(POISON_SEED_LEN, admission_granularity),
            True,
        ),
        PoisonOrderStep(
            "repeat_32k",
            POISON_TARGET_LEN,
            output_len,
            repeat_raw,
            expected_admitted_hit_tokens(repeat_raw, admission_granularity),
            True,
        ),
    )


def decode_boundary_spec(
    admission_granularity: int = DEFAULT_CACHE_ADMISSION_GRANULARITY,
) -> DecodeBoundarySpec:
    """Describe a prompt+decode crossing whose generated block must not be reusable."""
    if admission_granularity <= DECODE_BOUNDARY_PROMPT_HEADROOM:
        raise ValueError("admission_granularity is too small for the decode-boundary regression")
    seed_prompt_len = admission_granularity - DECODE_BOUNDARY_PROMPT_HEADROOM
    target_appended_decode_tokens = DECODE_BOUNDARY_PROMPT_HEADROOM + 1
    if DECODE_BOUNDARY_SEED_OUTPUT_LEN < target_appended_decode_tokens:
        raise AssertionError("decode-boundary seed does not generate enough target prompt tokens")
    return DecodeBoundarySpec(
        admission_boundary=admission_granularity,
        seed_prompt_len=seed_prompt_len,
        seed_output_len=DECODE_BOUNDARY_SEED_OUTPUT_LEN,
        target_appended_decode_tokens=target_appended_decode_tokens,
        target_prompt_len=admission_granularity + 1,
        potential_poisoned_hit_tokens=admission_granularity,
        expected_cached_tokens=0,
    )


def build_decode_boundary_target(
    seed_prompt: Sequence[int],
    seed_output_token_ids: Sequence[int],
    spec: DecodeBoundarySpec,
) -> list[int]:
    if len(seed_prompt) != spec.seed_prompt_len:
        raise ValueError(f"decode-boundary seed prompt has {len(seed_prompt)} tokens, expected {spec.seed_prompt_len}")
    if len(seed_output_token_ids) < spec.target_appended_decode_tokens:
        raise ValueError(
            "decode-boundary seed returned too few tokens: "
            f"got {len(seed_output_token_ids)}, need {spec.target_appended_decode_tokens}"
        )
    target = [
        *seed_prompt,
        *seed_output_token_ids[: spec.target_appended_decode_tokens],
    ]
    if len(target) != spec.target_prompt_len:
        raise AssertionError("decode-boundary target length does not match its specification")
    return target


def build_completion_payload(
    *,
    model: str,
    prompt: Sequence[int],
    output_len: int,
    cache_salt: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": [int(token) for token in prompt],
        "add_special_tokens": False,
        "max_tokens": int(output_len),
        "temperature": 0,
        "top_p": 1,
        "seed": int(seed),
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }


def parse_prometheus_counter(text: str, metric: str) -> float:
    """Sum a Prometheus counter over label sets."""
    total = 0.0
    found = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name_and_labels, raw_value = fields[:2]
        name = name_and_labels.split("{", 1)[0]
        if name != metric:
            continue
        try:
            value = float(raw_value)
        except (IndexError, ValueError) as exc:
            raise QualificationError(f"invalid Prometheus sample for {metric}: {line}") from exc
        if not math.isfinite(value):
            raise QualificationError(f"non-finite Prometheus sample for {metric}: {line}")
        total += value
        found = True
    if not found:
        raise QualificationError(f"/metrics did not expose {metric}")
    return total


def _median(values: Sequence[float]) -> float:
    if not values:
        raise QualificationError("cannot summarize an empty result set")
    return float(statistics.median(values))


def summarize_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "count": len(runs),
        "median_ttft_ms": _median([float(run["ttft_ms"]) for run in runs]),
        "median_tpot_ms": _median([float(run["tpot_ms"]) for run in runs]),
        "median_e2e_ms": _median([float(run["e2e_ms"]) for run in runs]),
    }


def performance_gate(
    *,
    oracle: Mapping[str, Any],
    cold: Mapping[str, Any],
    hit: Mapping[str, Any],
    minimum_speedup: float,
) -> dict[str, Any]:
    off_ttft = float(oracle["median_ttft_ms"])
    off_tpot = float(oracle["median_tpot_ms"])
    cold_ttft = float(cold["median_ttft_ms"])
    cold_tpot = float(cold["median_tpot_ms"])
    hit_ttft = float(hit["median_ttft_ms"])
    hit_tpot = float(hit["median_tpot_ms"])
    samples = (off_ttft, off_tpot, cold_ttft, cold_tpot, hit_ttft, hit_tpot)
    if not all(math.isfinite(value) and value > 0 for value in samples):
        raise QualificationError("performance metrics must all be positive")
    speedup_vs_oracle = off_ttft / hit_ttft
    speedup_vs_cold = cold_ttft / hit_ttft
    ratios = {
        "hit_ttft_speedup": min(speedup_vs_oracle, speedup_vs_cold),
        "hit_ttft_speedup_vs_oracle": speedup_vs_oracle,
        "hit_ttft_speedup_vs_candidate_cold": speedup_vs_cold,
        "cold_ttft_ratio": cold_ttft / off_ttft,
        "cold_tpot_ratio": cold_tpot / off_tpot,
        "hit_tpot_ratio": hit_tpot / off_tpot,
    }
    checks = {
        "hit_ttft_speedup": ratios["hit_ttft_speedup"] >= minimum_speedup,
        "hit_ttft_improves_candidate_cold": hit_ttft < cold_ttft,
        "cold_ttft_regression": ratios["cold_ttft_ratio"] <= 1.05,
        "cold_tpot_regression": ratios["cold_tpot_ratio"] <= 1.02,
        "hit_tpot_regression": ratios["hit_tpot_ratio"] <= 1.02,
    }
    return {
        "minimum_speedup": minimum_speedup,
        **ratios,
        "checks": checks,
        "passed": all(checks.values()),
    }


class OpenAIClient:
    def __init__(self, base_url: str, *, api_key: str | None, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _open(self, request: Request):
        try:
            return urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise QualificationError(f"HTTP {exc.code} for {request.full_url}: {body}") from exc
        except URLError as exc:
            raise QualificationError(f"request failed for {request.full_url}: {exc}") from exc

    def get_json(self, path: str) -> Any:
        request = Request(self.base_url + path, headers=self.headers, method="GET")
        with self._open(request) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_text(self, path: str) -> str:
        request = Request(self.base_url + path, headers=self.headers, method="GET")
        with self._open(request) as response:
            return response.read().decode("utf-8")

    def completion(self, payload: Mapping[str, Any]) -> CompletionResult:
        request = Request(
            self.base_url + "/v1/completions",
            data=_json_bytes(payload),
            headers=self.headers,
            method="POST",
        )
        started = time.perf_counter()
        token_ids: list[int] = []
        token_times: list[float] = []
        returned_prompt_token_ids: list[int] | None = None
        usage: Mapping[str, Any] | None = None
        request_id = None
        finish_reason = None
        with self._open(request) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise QualificationError(f"invalid SSE JSON: {data[:200]}") from exc
                if "error" in chunk:
                    raise QualificationError(f"completion stream error: {chunk['error']}")
                request_id = chunk.get("id", request_id)
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if int(choice.get("index", 0)) != 0:
                        raise QualificationError(f"completion returned unexpected choice index {choice.get('index')!r}")
                    prompt_ids = choice.get("prompt_token_ids")
                    if prompt_ids is not None:
                        candidate_prompt_ids = [int(token) for token in prompt_ids]
                        if returned_prompt_token_ids is not None and candidate_prompt_ids != returned_prompt_token_ids:
                            raise QualificationError("stream returned inconsistent prompt_token_ids")
                        returned_prompt_token_ids = candidate_prompt_ids
                    delta_ids = choice.get("token_ids")
                    if delta_ids:
                        arrived = time.perf_counter()
                        token_ids.extend(int(token) for token in delta_ids)
                        token_times.extend([arrived] * len(delta_ids))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        finished = time.perf_counter()
        if not token_ids:
            raise QualificationError("completion returned no token_ids; server must honor return_token_ids")
        if returned_prompt_token_ids is None:
            raise QualificationError("completion returned no prompt_token_ids; server must honor return_token_ids")
        exact_prompt = [int(token) for token in payload["prompt"]]
        if returned_prompt_token_ids != exact_prompt:
            raise QualificationError(
                "server-returned prompt token ids differ from the exact raw prompt: "
                f"got={token_ids_sha256(returned_prompt_token_ids)}, "
                f"expected={token_ids_sha256(exact_prompt)}"
            )
        if usage is None:
            raise QualificationError("stream returned no final usage chunk")
        details = usage.get("prompt_tokens_details")
        cached_tokens = (
            int(details["cached_tokens"])
            if isinstance(details, Mapping) and details.get("cached_tokens") is not None
            else None
        )
        prompt_tokens = int(usage.get("prompt_tokens", -1))
        output_tokens = int(usage.get("completion_tokens", -1))
        if prompt_tokens != len(payload["prompt"]):
            raise QualificationError(
                f"server counted {prompt_tokens} prompt tokens for exact raw prompt of {len(payload['prompt'])}"
            )
        if output_tokens != len(token_ids):
            raise QualificationError(
                f"server usage reports {output_tokens} output tokens but returned {len(token_ids)} token ids"
            )
        ttft_ms = (token_times[0] - started) * 1000.0
        tpot_ms = (token_times[-1] - token_times[0]) * 1000.0 / (len(token_times) - 1) if len(token_times) > 1 else 0.0
        return CompletionResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            prompt_token_ids_sha256=token_ids_sha256(returned_prompt_token_ids),
            token_ids=token_ids,
            token_ids_sha256=token_ids_sha256(token_ids),
            finish_reason=finish_reason,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_ms=(finished - started) * 1000.0,
        )


def _result_dict(result: CompletionResult) -> dict[str, Any]:
    data = asdict(result)
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        data[key] = round(float(data[key]), 6)
    return data


def _assert_result(
    result: CompletionResult,
    *,
    expected_output_len: int,
    expected_cached_tokens: int,
    allow_missing_cached_tokens: bool = False,
    oracle_token_ids: Sequence[int] | None = None,
) -> None:
    if result.output_tokens != expected_output_len:
        raise QualificationError(f"expected {expected_output_len} output tokens, got {result.output_tokens}")
    if result.finish_reason != "length":
        raise QualificationError(f"expected finish_reason=length, got {result.finish_reason!r}")
    if result.cached_tokens is None and not allow_missing_cached_tokens:
        raise QualificationError(
            "usage.prompt_tokens_details.cached_tokens is absent; launch cache-on vLLM with "
            "--enable-prompt-tokens-details"
        )
    if result.cached_tokens is not None and result.cached_tokens != expected_cached_tokens:
        raise QualificationError(f"expected {expected_cached_tokens} cached tokens, got {result.cached_tokens}")
    if oracle_token_ids is not None and list(result.token_ids) != list(oracle_token_ids):
        raise QualificationError(
            "generated token ids differ from cache-off oracle: "
            f"got={result.token_ids_sha256}, expected={token_ids_sha256(oracle_token_ids)}"
        )


def _server_metadata(client: OpenAIClient, model: str, required_context: int) -> dict[str, Any]:
    health = client.get_text("/health")
    models = client.get_json("/v1/models")
    entries = [entry for entry in models.get("data", []) if entry.get("id") == model]
    if len(entries) != 1:
        raise QualificationError(f"/v1/models did not advertise exactly one {model!r} entry")
    max_model_len = int(entries[0].get("max_model_len", 0))
    if max_model_len < required_context:
        raise QualificationError(f"server max_model_len={max_model_len} is below required {required_context}")
    return {"health_body": health, "model": entries[0]}


def _metric_snapshot(client: OpenAIClient) -> dict[str, float]:
    text = client.get_text("/metrics")
    return {
        "prefix_cache_queries": parse_prometheus_counter(text, PREFIX_CACHE_QUERIES_METRIC),
        "prefix_cache_hits": parse_prometheus_counter(text, PREFIX_CACHE_HITS_METRIC),
    }


def _metric_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {key: float(after[key]) - float(before[key]) for key in before}


def _request(
    client: OpenAIClient,
    *,
    model: str,
    prompt: Sequence[int],
    output_len: int,
    salt: str,
    seed: int,
) -> CompletionResult:
    return client.completion(
        build_completion_payload(
            model=model,
            prompt=prompt,
            output_len=output_len,
            cache_salt=salt,
            seed=seed,
        )
    )


def _oracle_output(oracle: Mapping[str, Any], category: str, name: str) -> list[int]:
    try:
        if category == "correctness":
            entry = oracle[category][name]["oracle"]
            tokens = [int(token) for token in entry["token_ids"]]
            expected_hash = entry["token_ids_sha256"]
        else:
            entry = oracle[category][name]
            tokens = [int(token) for token in entry["reference_token_ids"]]
            expected_hash = entry["reference_token_ids_sha256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(f"oracle artifact is missing {category}.{name} output ids") from exc
    actual_hash = token_ids_sha256(tokens)
    if expected_hash != actual_hash:
        raise QualificationError(
            f"oracle artifact {category}.{name} token hash is invalid: " f"stored={expected_hash}, actual={actual_hash}"
        )
    return tokens


def _salt(args: argparse.Namespace, *parts: str) -> str:
    return "-".join(("laguna-prefix", args.run_id, *parts))


def _base_artifact(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "run_id": args.run_id,
        "created_unix": int(time.time()),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "block_size": args.block_size,
            "cache_admission_granularity": args.cache_admission_granularity,
            "cache_admission_policy": "canonical_floor_v1",
            "vocab_size": args.vocab_size,
            "seed": args.seed,
            "performance_output_len": args.performance_output_len,
            "correctness_output_len": args.correctness_output_len,
            "repetitions": args.repetitions,
            "cold_ttft_max_ratio": 1.05,
            "tpot_max_ratio": 1.02,
            "expected_prefix_cache_enabled": mode == "on",
        },
        "suite": {
            "excluded_token_ids": list(DEFAULT_EXCLUDED_TOKEN_IDS),
            "correctness_cases": [asdict(case) for case in CORRECTNESS_CASES],
            "performance_cases": [asdict(case) for case in PERFORMANCE_CASES],
            "poison_order": [
                asdict(step)
                for step in poison_order_plan(
                    output_len=args.performance_output_len,
                    block_size=args.block_size,
                    admission_granularity=args.cache_admission_granularity,
                )
            ],
            "decode_boundary": asdict(decode_boundary_spec(args.cache_admission_granularity)),
        },
        "correctness": {},
        "poison_order": {},
        "decode_boundary": {},
        "performance": {},
        "failures": [],
        "verdicts": {
            "oracle": {"passed": False},
            "correctness": {"passed": False},
            "performance": {"passed": False},
            "metrics": {"passed": False},
            "health": {"passed": False},
            "overall": {"passed": False},
        },
    }


def _maximum_prompt_length(args: argparse.Namespace) -> int:
    prompt_lengths = [
        POISON_TARGET_LEN,
        decode_boundary_spec(args.cache_admission_granularity).target_prompt_len,
    ]
    prompt_lengths.extend(case.prompt_len for case in CORRECTNESS_CASES)
    prompt_lengths.extend(case.prompt_len for case in PERFORMANCE_CASES)
    return max(prompt_lengths)


def _required_context(args: argparse.Namespace) -> int:
    boundary = decode_boundary_spec(args.cache_admission_granularity)
    request_lengths = [
        POISON_TARGET_LEN + args.performance_output_len,
        boundary.seed_prompt_len + boundary.seed_output_len,
        boundary.target_prompt_len + args.correctness_output_len,
    ]
    request_lengths.extend(case.prompt_len + args.correctness_output_len for case in CORRECTNESS_CASES)
    request_lengths.extend(case.prompt_len + args.performance_output_len for case in PERFORMANCE_CASES)
    return max(request_lengths)


def run_off(args: argparse.Namespace, client: OpenAIClient) -> dict[str, Any]:
    artifact = _base_artifact(args, "off")
    max_prompt = _maximum_prompt_length(args)
    artifact["server"] = _server_metadata(client, args.model, _required_context(args))
    metrics_before = _metric_snapshot(client)
    base = build_base_tokens(max_prompt, vocab_size=args.vocab_size, seed=args.seed)

    for case in CORRECTNESS_CASES:
        target = correctness_target(case, base, vocab_size=args.vocab_size)
        result = _request(
            client,
            model=args.model,
            prompt=target,
            output_len=args.correctness_output_len,
            salt=_salt(args, "off", case.name),
            seed=args.seed,
        )
        _assert_result(
            result,
            expected_output_len=args.correctness_output_len,
            expected_cached_tokens=0,
            allow_missing_cached_tokens=True,
        )
        artifact["correctness"][case.name] = {
            "prefix_len": case.prefix_len,
            "suffix_len": case.suffix_len,
            "prompt_len": case.prompt_len,
            "prompt_sha256": token_ids_sha256(target),
            "oracle": _result_dict(result),
            "passed": True,
        }

    for case in PERFORMANCE_CASES:
        prompt = base[: case.prompt_len]
        runs = []
        reference_ids = None
        for index in range(args.repetitions):
            result = _request(
                client,
                model=args.model,
                prompt=prompt,
                output_len=args.performance_output_len,
                salt=_salt(args, "off", case.name, str(index)),
                seed=args.seed,
            )
            _assert_result(
                result,
                expected_output_len=args.performance_output_len,
                expected_cached_tokens=0,
                allow_missing_cached_tokens=True,
                oracle_token_ids=reference_ids,
            )
            if reference_ids is None:
                reference_ids = list(result.token_ids)
            runs.append(_result_dict(result))
        assert reference_ids is not None
        artifact["performance"][case.name] = {
            "prompt_len": case.prompt_len,
            "prompt_sha256": token_ids_sha256(prompt),
            "minimum_speedup": case.minimum_speedup,
            "reference_token_ids": reference_ids,
            "reference_token_ids_sha256": token_ids_sha256(reference_ids),
            "cold_runs": runs,
            "cold_summary": summarize_runs(runs),
            "passed": True,
        }

    metrics_after = _metric_snapshot(client)
    artifact["metrics"] = {
        "before": metrics_before,
        "after": metrics_after,
        "delta": _metric_delta(metrics_before, metrics_after),
    }
    if artifact["metrics"]["delta"]["prefix_cache_hits"] != 0:
        raise QualificationError("cache-off phase recorded prefix-cache hits")
    artifact["server"]["final_health_body"] = client.get_text("/health")
    artifact["verdicts"] = {
        "oracle": {"passed": True},
        "correctness": {"passed": True},
        "performance": {"passed": True},
        "metrics": {"passed": True},
        "health": {"passed": True},
        "overall": {"passed": True},
    }
    artifact["passed"] = True
    return artifact


def _check_oracle_compatibility(args: argparse.Namespace, oracle: Mapping[str, Any]) -> None:
    if oracle.get("schema_version") != SCHEMA_VERSION or oracle.get("mode") != "off":
        raise QualificationError("--oracle is not a compatible cache-off artifact")
    if oracle.get("passed") is not True:
        raise QualificationError("--oracle cache-off artifact did not pass")
    config = oracle.get("config", {})
    compared = (
        "model",
        "block_size",
        "vocab_size",
        "seed",
        "performance_output_len",
        "correctness_output_len",
        "repetitions",
    )
    for key in compared:
        if config.get(key) != getattr(args, key):
            raise QualificationError(f"candidate {key}={getattr(args, key)!r} differs from oracle {config.get(key)!r}")


def run_on(
    args: argparse.Namespace,
    client: OpenAIClient,
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    _check_oracle_compatibility(args, oracle)
    artifact = _base_artifact(args, "on")
    artifact["verdicts"]["oracle"]["passed"] = True
    artifact["oracle_path"] = str(Path(args.oracle).resolve())
    max_prompt = _maximum_prompt_length(args)
    artifact["server"] = _server_metadata(client, args.model, _required_context(args))
    metrics_before = _metric_snapshot(client)
    base = build_base_tokens(max_prompt, vocab_size=args.vocab_size, seed=args.seed)
    minimum_metric_hits = 0

    for case in CORRECTNESS_CASES:
        target = correctness_target(case, base, vocab_size=args.vocab_size)
        try:
            oracle_prompt_hash = oracle["correctness"][case.name]["prompt_sha256"]
        except (KeyError, TypeError) as exc:
            raise QualificationError(f"oracle artifact is missing correctness.{case.name} prompt hash") from exc
        if oracle_prompt_hash != token_ids_sha256(target):
            raise QualificationError(f"{case.name} prompt does not match oracle artifact")
        oracle_ids = _oracle_output(oracle, "correctness", case.name)
        cold = _request(
            client,
            model=args.model,
            prompt=target,
            output_len=args.correctness_output_len,
            salt=_salt(args, "on", "cold", case.name),
            seed=args.seed,
        )
        _assert_result(
            cold,
            expected_output_len=args.correctness_output_len,
            expected_cached_tokens=0,
            oracle_token_ids=oracle_ids,
        )
        hit_salt = _salt(args, "on", "hit", case.name)
        seed_result = _request(
            client,
            model=args.model,
            prompt=base[: case.prefix_len],
            output_len=1,
            salt=hit_salt,
            seed=args.seed,
        )
        _assert_result(seed_result, expected_output_len=1, expected_cached_tokens=0)
        raw_candidate_cached_tokens = case.prefix_len
        expected_cached_tokens = expected_admitted_hit_tokens(
            raw_candidate_cached_tokens,
            args.cache_admission_granularity,
        )
        hit = _request(
            client,
            model=args.model,
            prompt=target,
            output_len=args.correctness_output_len,
            salt=hit_salt,
            seed=args.seed,
        )
        _assert_result(
            hit,
            expected_output_len=args.correctness_output_len,
            expected_cached_tokens=expected_cached_tokens,
            oracle_token_ids=oracle_ids,
        )
        minimum_metric_hits += expected_cached_tokens
        artifact["correctness"][case.name] = {
            "prefix_len": case.prefix_len,
            "suffix_len": case.suffix_len,
            "prompt_len": case.prompt_len,
            "prompt_sha256": token_ids_sha256(target),
            "raw_candidate_cached_tokens": raw_candidate_cached_tokens,
            "expected_cached_tokens": expected_cached_tokens,
            "cold": _result_dict(cold),
            "seed": _result_dict(seed_result),
            "hit": _result_dict(hit),
            "passed": True,
        }

    boundary = decode_boundary_spec(args.cache_admission_granularity)
    boundary_seed_prompt = base[: boundary.seed_prompt_len]
    boundary_poison_salt = _salt(args, "on", "decode-boundary", "poison")
    boundary_seed = _request(
        client,
        model=args.model,
        prompt=boundary_seed_prompt,
        output_len=boundary.seed_output_len,
        salt=boundary_poison_salt,
        seed=args.seed,
    )
    _assert_result(
        boundary_seed,
        expected_output_len=boundary.seed_output_len,
        expected_cached_tokens=0,
    )
    boundary_target = build_decode_boundary_target(
        boundary_seed_prompt,
        boundary_seed.token_ids,
        boundary,
    )
    boundary_cold_salt = _salt(args, "on", "decode-boundary", "cold")
    boundary_cold = _request(
        client,
        model=args.model,
        prompt=boundary_target,
        output_len=args.correctness_output_len,
        salt=boundary_cold_salt,
        seed=args.seed,
    )
    _assert_result(
        boundary_cold,
        expected_output_len=args.correctness_output_len,
        expected_cached_tokens=0,
    )
    boundary_lookup = _request(
        client,
        model=args.model,
        prompt=boundary_target,
        output_len=args.correctness_output_len,
        salt=boundary_poison_salt,
        seed=args.seed,
    )
    _assert_result(
        boundary_lookup,
        expected_output_len=args.correctness_output_len,
        expected_cached_tokens=boundary.expected_cached_tokens,
        oracle_token_ids=boundary_cold.token_ids,
    )
    artifact["decode_boundary"] = {
        **asdict(boundary),
        "seed_prompt_sha256": token_ids_sha256(boundary_seed_prompt),
        "target_prompt_sha256": token_ids_sha256(boundary_target),
        "poison_cache_salt": boundary_poison_salt,
        "cold_cache_salt": boundary_cold_salt,
        "oracle_kind": "isolated_cache_on_cold_exact_token_ids",
        "seed": _result_dict(boundary_seed),
        "cold_oracle": _result_dict(boundary_cold),
        "poisoned_lookup": _result_dict(boundary_lookup),
        "passed": True,
    }

    try:
        poison_oracle_case = oracle["performance"]["full_32k"]
    except (KeyError, TypeError) as exc:
        raise QualificationError("oracle artifact is missing performance.full_32k") from exc
    poison_oracle_ids = _oracle_output(oracle, "performance", "full_32k")
    poison_prompt = base[:POISON_TARGET_LEN]
    poison_prompt_hash = token_ids_sha256(poison_prompt)
    if poison_oracle_case.get("prompt_sha256") != poison_prompt_hash:
        raise QualificationError("poison-order 32K prompt does not match oracle artifact")
    poison_salt = _salt(args, "on", "poison", "2k-to-full32")
    poison_steps = []
    for step in poison_order_plan(
        output_len=args.performance_output_len,
        block_size=args.block_size,
        admission_granularity=args.cache_admission_granularity,
    ):
        prompt = base[: step.prompt_len]
        result = _request(
            client,
            model=args.model,
            prompt=prompt,
            output_len=step.output_len,
            salt=poison_salt,
            seed=args.seed,
        )
        _assert_result(
            result,
            expected_output_len=step.output_len,
            expected_cached_tokens=step.expected_cached_tokens,
            oracle_token_ids=poison_oracle_ids if step.compare_with_full_32k_oracle else None,
        )
        minimum_metric_hits += step.expected_cached_tokens
        poison_steps.append(
            {
                **asdict(step),
                "prompt_sha256": token_ids_sha256(prompt),
                "result": _result_dict(result),
                "passed": True,
            }
        )
    artifact["poison_order"] = {
        "name": "2k_seed_then_32k_target_then_repeat",
        "cache_salt": poison_salt,
        "oldest_hash_regression_covered": True,
        "steps": poison_steps,
        "passed": True,
    }

    for case in PERFORMANCE_CASES:
        prompt = base[: case.prompt_len]
        oracle_case = oracle["performance"][case.name]
        oracle_ids = _oracle_output(oracle, "performance", case.name)
        if oracle_case.get("prompt_sha256") != token_ids_sha256(prompt):
            raise QualificationError(f"{case.name} prompt does not match oracle artifact")
        cold_runs = []
        for index in range(args.repetitions):
            result = _request(
                client,
                model=args.model,
                prompt=prompt,
                output_len=args.performance_output_len,
                salt=_salt(args, "on", "cold", case.name, str(index)),
                seed=args.seed,
            )
            _assert_result(
                result,
                expected_output_len=args.performance_output_len,
                expected_cached_tokens=0,
                oracle_token_ids=oracle_ids,
            )
            cold_runs.append(_result_dict(result))

        hit_salt = _salt(args, "on", "perf", case.name)
        seed_result = _request(
            client,
            model=args.model,
            prompt=prompt,
            output_len=1,
            salt=hit_salt,
            seed=args.seed,
        )
        _assert_result(seed_result, expected_output_len=1, expected_cached_tokens=0)
        raw_candidate_cached_tokens = expected_full_hit_tokens(case.prompt_len, args.block_size)
        expected_cached = expected_admitted_hit_tokens(
            raw_candidate_cached_tokens,
            args.cache_admission_granularity,
        )
        hit_runs = []
        for _ in range(args.repetitions):
            result = _request(
                client,
                model=args.model,
                prompt=prompt,
                output_len=args.performance_output_len,
                salt=hit_salt,
                seed=args.seed,
            )
            _assert_result(
                result,
                expected_output_len=args.performance_output_len,
                expected_cached_tokens=expected_cached,
                oracle_token_ids=oracle_ids,
            )
            hit_runs.append(_result_dict(result))
        minimum_metric_hits += expected_cached * args.repetitions

        cold_summary = summarize_runs(cold_runs)
        hit_summary = summarize_runs(hit_runs)
        gate = performance_gate(
            oracle=oracle_case["cold_summary"],
            cold=cold_summary,
            hit=hit_summary,
            minimum_speedup=case.minimum_speedup,
        )
        artifact["performance"][case.name] = {
            "prompt_len": case.prompt_len,
            "prompt_sha256": token_ids_sha256(prompt),
            "raw_candidate_cached_tokens": raw_candidate_cached_tokens,
            "expected_cached_tokens": expected_cached,
            "seed": _result_dict(seed_result),
            "cold_runs": cold_runs,
            "cold_summary": cold_summary,
            "hit_runs": hit_runs,
            "hit_summary": hit_summary,
            "gate": gate,
            "passed": gate["passed"],
        }
        if not gate["passed"]:
            artifact["failures"].append(f"performance.{case.name}")

    metrics_after = _metric_snapshot(client)
    delta = _metric_delta(metrics_before, metrics_after)
    metrics_passed = delta["prefix_cache_hits"] >= minimum_metric_hits
    artifact["metrics"] = {
        "before": metrics_before,
        "after": metrics_after,
        "delta": delta,
        "minimum_expected_prefix_cache_hits": minimum_metric_hits,
        "passed": metrics_passed,
    }
    if not metrics_passed:
        artifact["failures"].append("metrics.insufficient_prefix_cache_hits")
    artifact["server"]["final_health_body"] = client.get_text("/health")
    performance_passed = all(case["passed"] for case in artifact["performance"].values())
    artifact["verdicts"] = {
        "oracle": {"passed": True},
        "correctness": {"passed": True},
        "performance": {"passed": performance_passed},
        "metrics": {"passed": metrics_passed},
        "health": {"passed": True},
        "overall": {"passed": not artifact["failures"]},
    }
    artifact["passed"] = not artifact["failures"]
    return artifact


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("off", "on"), help="cache-disabled oracle or cache-enabled candidate")
    parser.add_argument("--output", type=Path, required=True, help="JSON artifact to create")
    parser.add_argument("--oracle", type=Path, help="cache-off JSON artifact (required for mode=on)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=1_800.0)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--cache-admission-granularity",
        type=int,
        default=DEFAULT_CACHE_ADMISSION_GRANULARITY,
        help="canonical safe-admission boundary used for exact cached-token assertions",
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--run-id",
        default=secrets.token_hex(32),
        help="unique cache-salt namespace (random by default; record it for repeatability)",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--performance-output-len", type=int, default=128)
    parser.add_argument("--correctness-output-len", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode == "on" and args.oracle is None:
        parser.error("mode=on requires --oracle")
    if args.mode == "off" and args.oracle is not None:
        parser.error("mode=off does not accept --oracle")
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if args.performance_output_len < 2:
        parser.error("--performance-output-len must be at least 2 so TPOT is measurable")
    if args.correctness_output_len <= 0:
        parser.error("--correctness-output-len must be positive")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.cache_admission_granularity <= 0:
        parser.error("--cache-admission-granularity must be positive")
    if args.cache_admission_granularity <= DECODE_BOUNDARY_PROMPT_HEADROOM:
        parser.error("--cache-admission-granularity is too small for decode-boundary qualification")
    if args.cache_admission_granularity % args.block_size:
        parser.error("--cache-admission-granularity must be a multiple of --block-size")
    if not args.run_id.strip():
        parser.error("--run-id must be non-empty")

    client = OpenAIClient(args.base_url, api_key=args.api_key, timeout=args.timeout)
    try:
        if args.mode == "off":
            artifact = run_off(args, client)
        else:
            oracle = json.loads(args.oracle.read_text())
            artifact = run_on(args, client, oracle)
        _write_artifact(args.output, artifact)
    except (QualificationError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        failure = _base_artifact(args, args.mode)
        failure["failures"].append(str(exc))
        failure["passed"] = False
        _write_artifact(args.output, failure)
        print(f"PREFIX_CACHE_QUALIFICATION FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"PREFIX_CACHE_QUALIFICATION {'PASS' if artifact['passed'] else 'FAIL'} "
        f"mode={args.mode} artifact={args.output}"
    )
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
