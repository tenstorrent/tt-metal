# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only qualification for Laguna's opt-in serving-envelope probes.

This client never opens a device or starts/stops a server.  It drives exact raw
token IDs through a separately launched p150x2 vLLM server and writes one JSON
artifact.  Two modes correspond to the launcher's fail-closed probes:

* ``multi-seq`` compares two sequential B=1 oracles with the same two requests
  issued concurrently against the 65K/two-sequence uniform-KV server.
* ``context-262k`` repeats a request just beyond the qualified 131K boundary,
  then fills the advertised 262,144-token request contract exactly.  Its TTFT
  gate rejects work beyond the model's causal-attention scaling bound, rather
  than incorrectly assuming that full-attention prefill is linear in tokens.

Pass ``--server-log`` for a qualification run.  The harness validates the
launch header before requests and scans only newly appended log bytes for hard
faults afterwards.  The known once-only active-trace allocator advisory is not
a fault marker.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Barrier
from typing import Any, Sequence

from . import prefix_cache_qualification as prefix_q

SCHEMA_VERSION = 1
DEFAULT_MODEL = prefix_q.DEFAULT_MODEL
DEFAULT_VOCAB_SIZE = prefix_q.DEFAULT_VOCAB_SIZE
DEFAULT_SEED = prefix_q.DEFAULT_SEED
FULL_ATTENTION_LAYERS = 10
SLIDING_ATTENTION_LAYERS = 30
SLIDING_WINDOW = 512
HARD_FAULT_MARKERS = (
    "traceback (most recent call last)",
    "fatal error",
    "device hang",
    "watcher exception",
    "out of memory",
    "program cache miss forbidden",
    "compile under the resident trace",
)


class EnvelopeQualificationError(RuntimeError):
    """A serving-envelope gate failed."""


def _finite_positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise EnvelopeQualificationError(f"{label} must be finite and positive, got {value}")
    return value


def _causal_attention_work_upper_bound(prompt_len: int) -> float:
    """Return a conservative architecture-aware prefill work estimate.

    Laguna has ten full-attention and thirty 512-token sliding-attention
    layers.  Full causal attention grows quadratically with prompt length;
    projections, MoE, norms, and communication are linear and therefore can
    only reduce the ratio between a longer prompt and a shorter prompt.  The
    estimate is consequently an upper bound for expected length scaling, which
    is the appropriate denominator for detecting an additional padding or
    specialization cliff.
    """

    tokens = int(prompt_len)
    if tokens <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    causal_pairs = tokens * (tokens + 1) / 2
    sliding_pairs = tokens * min(tokens, SLIDING_WINDOW)
    return FULL_ATTENTION_LAYERS * causal_pairs + SLIDING_ATTENTION_LAYERS * sliding_pairs


def _result_dict(result: prefix_q.CompletionResult) -> dict[str, Any]:
    data = asdict(result)
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        data[key] = round(float(data[key]), 6)
    return data


def _assert_completion(
    result: prefix_q.CompletionResult,
    *,
    prompt_len: int,
    output_len: int,
    oracle_ids: Sequence[int] | None = None,
) -> None:
    if result.prompt_tokens != int(prompt_len):
        raise EnvelopeQualificationError(f"server counted {result.prompt_tokens} prompt tokens, expected {prompt_len}")
    if result.output_tokens != int(output_len):
        raise EnvelopeQualificationError(f"server returned {result.output_tokens} output tokens, expected {output_len}")
    if result.finish_reason != "length":
        raise EnvelopeQualificationError(f"expected finish_reason='length', got {result.finish_reason!r}")
    if result.cached_tokens not in (None, 0):
        raise EnvelopeQualificationError(
            f"cache-off envelope unexpectedly reported {result.cached_tokens} cached tokens"
        )
    _finite_positive(result.ttft_ms, "TTFT")
    _finite_positive(result.e2e_ms, "E2E")
    if output_len > 1:
        _finite_positive(result.tpot_ms, "TPOT")
    if oracle_ids is not None and list(result.token_ids) != [int(token) for token in oracle_ids]:
        raise EnvelopeQualificationError(
            "generated token IDs differ from the sequential oracle: "
            f"got={result.token_ids_sha256}, "
            f"expected={prefix_q.token_ids_sha256(oracle_ids)}"
        )


def _payload(
    *,
    model: str,
    prompt: Sequence[int],
    output_len: int,
    run_id: str,
    label: str,
    seed: int,
) -> dict[str, Any]:
    return prefix_q.build_completion_payload(
        model=model,
        prompt=prompt,
        output_len=output_len,
        cache_salt=f"laguna-envelope-{run_id}-{label}",
        seed=seed,
    )


def _server_metadata(
    client: prefix_q.OpenAIClient,
    *,
    model: str,
    expected_context: int,
) -> dict[str, Any]:
    health = client.get_text("/health")
    models = client.get_json("/v1/models")
    entries = [entry for entry in models.get("data", []) if entry.get("id") == model]
    if len(entries) != 1:
        raise EnvelopeQualificationError(f"/v1/models did not advertise exactly one {model!r} entry")
    advertised = int(entries[0].get("max_model_len", 0))
    if advertised != int(expected_context):
        raise EnvelopeQualificationError(
            f"server advertised max_model_len={advertised}, expected exactly {expected_context}"
        )
    return {"initial_health_body": health, "model": entries[0]}


def _log_contract(mode: str) -> tuple[str, ...]:
    if mode == "multi-seq":
        return (
            "profile: p150x2",
            "context: 65536 | seqs: 2",
            "hybrid KV: 0",
            "prefix cache: 0",
            "TT_LAGUNA_MULTI_SEQ_POOL=1 (qualified=0)",
        )
    if mode == "context-262k":
        return (
            "profile: p150x2",
            "context: 262144 | seqs: 1",
            "hybrid KV: 1",
            "prefix cache: 0",
            "TT_LAGUNA_CONTEXT_PROBE=1 (qualified=0)",
        )
    raise ValueError(f"unknown qualification mode {mode!r}")


def _prepare_log(path: str | None, mode: str) -> tuple[Path | None, int, dict[str, Any]]:
    if not path:
        return None, 0, {"validated": False, "reason": "--server-log omitted"}
    log_path = Path(path).expanduser().resolve()
    if not log_path.is_file():
        raise EnvelopeQualificationError(f"server log does not exist: {log_path}")
    initial = log_path.read_text(encoding="utf-8", errors="replace")
    missing = [fragment for fragment in _log_contract(mode) if fragment not in initial]
    if missing:
        raise EnvelopeQualificationError(f"server log does not match {mode} launch contract; missing {missing}")
    return (
        log_path,
        log_path.stat().st_size,
        {
            "validated": True,
            "path": str(log_path),
            "initial_size": log_path.stat().st_size,
            "required_fragments": list(_log_contract(mode)),
        },
    )


def _scan_log_tail(log_path: Path | None, offset: int) -> dict[str, Any]:
    if log_path is None:
        return {"scanned": False, "reason": "--server-log omitted"}
    with log_path.open("rb") as stream:
        stream.seek(offset)
        tail = stream.read().decode("utf-8", errors="replace")
    lowered = tail.lower()
    faults = [marker for marker in HARD_FAULT_MARKERS if marker in lowered]
    if faults:
        raise EnvelopeQualificationError(f"post-start server log contains hard fault markers: {faults}")
    return {
        "scanned": True,
        "new_bytes": len(tail.encode("utf-8")),
        "hard_fault_markers": faults,
    }


def run_multi_seq(args: argparse.Namespace, client: prefix_q.OpenAIClient) -> dict[str, Any]:
    prompts = [
        prefix_q.deterministic_token_ids(
            args.prompt_len,
            seed=args.seed + index * 997,
            vocab_size=args.vocab_size,
        )
        for index in range(2)
    ]
    oracles = []
    for index, prompt in enumerate(prompts):
        result = client.completion(
            _payload(
                model=args.model,
                prompt=prompt,
                output_len=args.output_len,
                run_id=args.run_id,
                label=f"oracle-{index}",
                seed=args.seed,
            )
        )
        _assert_completion(
            result,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
        )
        oracles.append(result)

    barrier = Barrier(2)

    def concurrent_request(index: int) -> tuple[float, float, prefix_q.CompletionResult]:
        barrier.wait(timeout=30)
        started = time.perf_counter()
        result = client.completion(
            _payload(
                model=args.model,
                prompt=prompts[index],
                output_len=args.output_len,
                run_id=args.run_id,
                label=f"concurrent-{index}",
                seed=args.seed,
            )
        )
        return started, time.perf_counter(), result

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(concurrent_request, index) for index in range(2)]
        concurrent = [future.result(timeout=args.timeout + 60) for future in futures]
    concurrent_wall_ms = (time.perf_counter() - wall_start) * 1000.0

    intervals = []
    concurrent_results = []
    for index, (started, finished, result) in enumerate(concurrent):
        _assert_completion(
            result,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            oracle_ids=oracles[index].token_ids,
        )
        intervals.append((started, finished))
        concurrent_results.append(result)
    overlap_ms = max(
        0.0,
        (min(finished for _, finished in intervals) - max(started for started, _ in intervals)) * 1000.0,
    )
    if overlap_ms <= 0:
        raise EnvelopeQualificationError("the two qualification requests did not overlap")

    sequential_wall_ms = sum(float(result.e2e_ms) for result in oracles)
    wall_speedup = sequential_wall_ms / concurrent_wall_ms
    worst_tpot_ratio = max(concurrent_results[index].tpot_ms / oracles[index].tpot_ms for index in range(2))
    checks = {
        "exact_tokens": all(concurrent_results[index].token_ids == oracles[index].token_ids for index in range(2)),
        "requests_overlap": overlap_ms > 0,
        "aggregate_wall_speedup": wall_speedup >= args.minimum_wall_speedup,
        "per_request_tpot": worst_tpot_ratio <= args.maximum_tpot_ratio,
    }
    if not all(checks.values()):
        raise EnvelopeQualificationError(
            "two-sequence performance gate failed: "
            f"checks={checks}, wall_speedup={wall_speedup:.4f}, "
            f"worst_tpot_ratio={worst_tpot_ratio:.4f}"
        )
    return {
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "prompt_hashes": [prefix_q.token_ids_sha256(prompt) for prompt in prompts],
        "sequential_oracles": [_result_dict(result) for result in oracles],
        "concurrent_results": [_result_dict(result) for result in concurrent_results],
        "sequential_wall_ms": round(sequential_wall_ms, 6),
        "concurrent_wall_ms": round(concurrent_wall_ms, 6),
        "overlap_ms": round(overlap_ms, 6),
        "wall_speedup": wall_speedup,
        "worst_tpot_ratio": worst_tpot_ratio,
        "thresholds": {
            "minimum_wall_speedup": args.minimum_wall_speedup,
            "maximum_tpot_ratio": args.maximum_tpot_ratio,
        },
        "checks": checks,
    }


def run_context_262k(args: argparse.Namespace, client: prefix_q.OpenAIClient) -> dict[str, Any]:
    maximum_prompt = max(args.boundary_prompt_len, args.cap_prompt_len)
    base = prefix_q.deterministic_token_ids(
        maximum_prompt,
        seed=args.seed,
        vocab_size=args.vocab_size,
    )
    boundary_prompt = base[: args.boundary_prompt_len]
    boundary_runs = []
    reference_ids = None
    for index in range(args.boundary_repetitions):
        result = client.completion(
            _payload(
                model=args.model,
                prompt=boundary_prompt,
                output_len=args.boundary_output_len,
                run_id=args.run_id,
                label=f"boundary-{index}",
                seed=args.seed,
            )
        )
        _assert_completion(
            result,
            prompt_len=args.boundary_prompt_len,
            output_len=args.boundary_output_len,
            oracle_ids=reference_ids,
        )
        if reference_ids is None:
            reference_ids = list(result.token_ids)
        boundary_runs.append(result)

    if args.cap_prompt_len + args.cap_output_len != 262144:
        raise EnvelopeQualificationError("the exact-cap request must satisfy cap_prompt_len + cap_output_len == 262144")
    cap_prompt = base[: args.cap_prompt_len]
    cap_result = client.completion(
        _payload(
            model=args.model,
            prompt=cap_prompt,
            output_len=args.cap_output_len,
            run_id=args.run_id,
            label="exact-cap",
            seed=args.seed,
        )
    )
    _assert_completion(
        cap_result,
        prompt_len=args.cap_prompt_len,
        output_len=args.cap_output_len,
    )

    boundary_ttft_ms = statistics.median(result.ttft_ms for result in boundary_runs)
    ttft_ratio = cap_result.ttft_ms / boundary_ttft_ms
    boundary_attention_work = _causal_attention_work_upper_bound(args.boundary_prompt_len)
    cap_attention_work = _causal_attention_work_upper_bound(args.cap_prompt_len)
    attention_work_ratio = cap_attention_work / boundary_attention_work
    normalized_ratio = ttft_ratio / attention_work_ratio
    checks = {
        "boundary_repeat_exact": all(result.token_ids == boundary_runs[0].token_ids for result in boundary_runs),
        "crosses_qualified_boundary": args.boundary_prompt_len > 131072,
        "fills_exact_request_cap": args.cap_prompt_len + args.cap_output_len == 262144,
        "no_power_of_two_ttft_cliff": normalized_ratio <= args.maximum_normalized_ttft_ratio,
    }
    if not all(checks.values()):
        raise EnvelopeQualificationError(
            "262K context performance gate failed: "
            f"checks={checks}, ttft_ratio={ttft_ratio:.4f}, "
            f"attention_work_ratio={attention_work_ratio:.4f}, normalized={normalized_ratio:.4f}"
        )
    return {
        "boundary_prompt_len": args.boundary_prompt_len,
        "boundary_output_len": args.boundary_output_len,
        "boundary_prompt_hash": prefix_q.token_ids_sha256(boundary_prompt),
        "boundary_runs": [_result_dict(result) for result in boundary_runs],
        "cap_prompt_len": args.cap_prompt_len,
        "cap_output_len": args.cap_output_len,
        "cap_prompt_hash": prefix_q.token_ids_sha256(cap_prompt),
        "cap_result": _result_dict(cap_result),
        "boundary_median_ttft_ms": boundary_ttft_ms,
        "ttft_ratio": ttft_ratio,
        "boundary_attention_work": boundary_attention_work,
        "cap_attention_work": cap_attention_work,
        "attention_work_ratio": attention_work_ratio,
        "normalized_ttft_ratio": normalized_ratio,
        "thresholds": {
            "maximum_normalized_ttft_ratio": args.maximum_normalized_ttft_ratio,
        },
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    multi = subparsers.add_parser("multi-seq")
    multi.add_argument("--prompt-len", type=int, default=8192)
    multi.add_argument("--output-len", type=int, default=64)
    multi.add_argument("--minimum-wall-speedup", type=float, default=1.05)
    multi.add_argument("--maximum-tpot-ratio", type=float, default=1.10)

    context = subparsers.add_parser("context-262k")
    context.add_argument("--boundary-prompt-len", type=int, default=131136)
    context.add_argument("--boundary-output-len", type=int, default=16)
    context.add_argument("--boundary-repetitions", type=int, default=2)
    context.add_argument("--cap-prompt-len", type=int, default=262112)
    context.add_argument("--cap-output-len", type=int, default=32)
    context.add_argument("--maximum-normalized-ttft-ratio", type=float, default=1.15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "run_id": args.run_id,
        "created_unix": int(time.time()),
        "passed": False,
    }
    try:
        log_path, log_offset, log_contract = _prepare_log(args.server_log, args.mode)
        artifact["log_contract"] = log_contract
        client = prefix_q.OpenAIClient(
            args.base_url,
            api_key=args.api_key or None,
            timeout=args.timeout,
        )
        expected_context = 65536 if args.mode == "multi-seq" else 262144
        artifact["server"] = _server_metadata(
            client,
            model=args.model,
            expected_context=expected_context,
        )
        if args.mode == "multi-seq":
            artifact["results"] = run_multi_seq(args, client)
        else:
            artifact["results"] = run_context_262k(args, client)
        artifact["server"]["final_health_body"] = client.get_text("/health")
        artifact["post_start_log"] = _scan_log_tail(log_path, log_offset)
        artifact["passed"] = True
    except Exception as exc:  # noqa: BLE001 - persist a useful failed qualification artifact
        artifact["failure"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
