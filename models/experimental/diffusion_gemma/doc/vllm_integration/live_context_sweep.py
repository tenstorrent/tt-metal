# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run one real tenstorrent/vllm DiffusionGemma context sweep on QB2.

The harness launches the patched OpenAI server itself, reserves the Metal trace
region before mesh open, creates exact tokenizer-derived prompt-token prefixes,
and issues serial block-diffusion completion requests. It consumes the
``DG_VLLM_METRIC`` and ``DG_TRACE_METRIC`` server markers to report prefill,
capture-inclusive block-0 TTFT, warmed block latency, position progression,
trace capture/replay/release counts, and DRAM.

One invocation owns one ``--max-model-len`` server. Run separate invocations to
distinguish actual prompt-length scaling from allocation scaling, and one
isolated server lifecycle per ``--max-denoise-steps`` budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CANVAS_LENGTH = 256
MODEL_NAME = "diffusiongemma-26B-A4B-it"
DEFAULT_TRACE_REGION_SIZE = 10 * 2**30
MIN_TRACE_REGION_SIZE = 2 * 2**30
MAX_DENOISE_STEPS = 48
FILLER_UNIT = (
    "Diffusion language models refine a fixed token canvas while preserving a "
    "deterministic frozen prompt context for this live serving benchmark. "
)


def _http_json(base_url: str, path: str, payload: dict | None = None, *, timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {body}") from exc


def _marker_records(log_path: Path, marker: str) -> list[dict]:
    if not log_path.exists():
        return []
    decoder = json.JSONDecoder()
    records = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].lstrip()
        try:
            record, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            continue
        records.append(record)
    return records


def _wait_for_marker_count(
    log_path: Path,
    marker: str,
    event: str,
    previous_count: int,
    *,
    timeout: float,
) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        records = _marker_records(log_path, marker)
        matching = [record for record in records if record.get("event") == event]
        if len(matching) > previous_count:
            return records
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {marker}{event!r} in {log_path}")


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _server_env(args) -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    tt_pythonpath = f"{args.tt_metal}:{args.tt_metal / 'ttnn'}"
    selected = {
        "PYTHONPATH": tt_pythonpath,
        "TT_METAL_HOME": str(args.tt_metal),
        "TT_METAL_RUNTIME_ROOT": str(args.tt_metal),
        "MESH_DEVICE": args.mesh,
        "ARCH_NAME": "blackhole",
        "DG_CKPT": str(args.checkpoint),
        "DG_VLLM_TRACE": "1",
        "DG_VLLM_GUMBEL_MODE": "argmax",
        "DG_SPARSE_MOE": "1",
        "DG_DEDUP_ARGMAX": "1",
        "DG_SPARSE_MOE_TUNED": "1",
        "DG_VLLM_MAX_DENOISE_STEPS": str(args.max_denoise_steps),
        "DG_TRACE_REGION_SIZE": str(args.trace_region_size),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_RPC_TIMEOUT": "1800000",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TT_LOGGER_LEVEL": "ERROR",
        "LOGURU_LEVEL": "INFO",
        "PYTHONUNBUFFERED": "1",
    }
    env.update(selected)
    # Force the requested single-step scheme: one Metal trace per denoise step.
    for key in (
        "DG_DENOISE_TRACED",
        "DG_DENOISE_TRACED_MULTISTEP",
        "DG_DENOISE_MULTISTEP_GROUP",
        "DG_DENOISE_EARLY_HALT",
        "DG_DENOISE_EARLY_HALT_WINDOW",
        "DG_DENOISE_DEVICE_LOOP",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_WATCHER",
        "TRACY_NO_INVARIANT_CHECK",
        # Leave the newly selected self-conditioning defaults selector-free.
        "DG_SELFCOND_PRECHUNK_EMBED",
        "DG_SELFCOND_LOGITS_L1",
        "DG_DENOISE_FROZEN_PREFIX",
    ):
        env.pop(key, None)
    # New default serving perf config (set AFTER the pop so the sweep's choice is authoritative):
    # early-halt (bit-exact; converges post tanh-fix so it fires) + frozen-prefix capture-once
    # (restores steady speed by reusing the block-0 trace instead of per-block recapture).
    env["DG_DENOISE_EARLY_HALT"] = "0" if args.fixed_budget else "1"
    if not args.growing_prefix:
        env["DG_DENOISE_FROZEN_PREFIX"] = "1"
    return env, selected


def _server_command(args) -> list[str]:
    tt_config = {
        "tt": {
            "sample_on_device_mode": "all",
            "enable_model_warmup": False,
            "trace_mode": "all",
            "trace_region_size": args.trace_region_size,
        }
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.checkpoint),
        "--served-model-name",
        MODEL_NAME,
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_model_len),
        "--max-num-seqs",
        "1",
        "--block-size",
        "64",
        "--additional-config",
        json.dumps(tt_config, separators=(",", ":")),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    # Right-size the KV block pool. Without this vLLM's gpu_memory_utilization grabs ~all free DRAM
    # for KV blocks, which OOMs the build at large max_model_len (256K KV + bf16 weights + trace
    # region overflow 32 GB/chip). ceil(max_model_len/block_size) blocks hold the full context for
    # batch 1 (like the demo's page_max_num_blocks), freeing DRAM for the prefill activation.
    if args.num_gpu_blocks_override is not None:
        command += ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
    return command


def _wait_for_server(proc: subprocess.Popen, base_url: str, *, timeout: float) -> float:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = proc.poll()
        if returncode is not None:
            raise RuntimeError(f"OpenAI server exited during startup with code {returncode}")
        try:
            _http_json(base_url, "/health", timeout=2)
            return time.perf_counter() - started
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"OpenAI server did not become healthy within {timeout}s")


def _stop_server(proc: subprocess.Popen) -> dict:
    if proc.poll() is not None:
        return {"method": "already_exited", "returncode": proc.returncode}
    os.killpg(proc.pid, signal.SIGINT)
    try:
        return {"method": "SIGINT", "returncode": proc.wait(timeout=180)}
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            return {"method": "SIGINT_then_SIGTERM", "returncode": proc.wait(timeout=60)}
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"server pid {proc.pid} did not stop after SIGINT/SIGTERM") from exc


def _token_prefix(base_url: str, target: int) -> tuple[list[int], dict]:
    repeats = max(1, target // 12)
    while True:
        source = FILLER_UNIT * repeats
        tokenized = _http_json(
            base_url,
            "/tokenize",
            {"model": MODEL_NAME, "prompt": source, "add_special_tokens": True},
            timeout=120,
        )
        ids = [int(token) for token in tokenized["tokens"]]
        if len(ids) >= target:
            prompt_ids = ids[:target]
            return prompt_ids, {
                "recipe": "prefix of tokenizer(FILLER_UNIT * repeats, add_special_tokens=True)",
                "repeats": repeats,
                "source_token_count": len(ids),
                "source_char_count": len(source),
                "prompt_token_sha256": hashlib.sha256(
                    json.dumps(prompt_ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        repeats *= 2


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _run_completion(base_url: str, args, payload: dict) -> tuple[dict, float, list[dict], list[dict], str]:
    before_model = _marker_records(args.server_log, "DG_VLLM_METRIC ")
    before_trace = _marker_records(args.server_log, "DG_TRACE_METRIC ")
    before_release_count = sum(event.get("event") == "request_release" for event in before_model)
    log_start = args.server_log.stat().st_size
    request_t0 = time.perf_counter()
    response = _http_json(base_url, "/v1/completions", payload, timeout=args.request_timeout)
    request_wall_s = time.perf_counter() - request_t0
    after_model = _wait_for_marker_count(
        args.server_log,
        "DG_VLLM_METRIC ",
        "request_release",
        before_release_count,
        timeout=args.release_timeout,
    )
    after_trace = _marker_records(args.server_log, "DG_TRACE_METRIC ")
    with args.server_log.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(log_start)
        log_segment = handle.read()
    return (
        response,
        request_wall_s,
        after_model[len(before_model) :],
        after_trace[len(before_trace) :],
        log_segment,
    )


def _request_summary(
    *,
    target: int,
    prompt_meta: dict,
    response: dict,
    request_wall_s: float,
    model_events: list[dict],
    trace_events: list[dict],
    blocks_requested: int,
    max_denoise_steps: int,
    log_segment: str,
    repeat_index: int,
    require_no_compile_markers: bool = False,
) -> dict:
    session_events = [event for event in model_events if event.get("event") == "session_create"]
    block0_events = [event for event in model_events if event.get("event") == "prefill_block0"]
    decode_events = [event for event in model_events if event.get("event") == "decode_block"]
    release_events = [event for event in model_events if event.get("event") == "request_release"]
    captures = [event for event in trace_events if event.get("event") == "capture"]
    replays = [event for event in trace_events if event.get("event") == "replay"]
    trace_releases = [event for event in trace_events if event.get("event") == "release"]

    # Lenient trace-count validation: the traced-denoise recapture pattern changed with growing-prefix
    # correctness (commit ec5b64b4891, after the 2026-07-10 sweep), so a request may now emit more than
    # one capture / trace-release. The timing metrics below come from the DG_VLLM_METRIC block markers
    # and are independent of the trace-event counts; the actual counts are recorded in the "trace"
    # section for provenance instead of hard-asserted to 1.
    if not (len(session_events) == len(block0_events) == len(release_events) == 1):
        raise AssertionError(
            "expected one session/block0/release event, got "
            f"{len(session_events)}/{len(block0_events)}/{len(release_events)}"
        )
    # Accept any traced denoise variant. Early-halt is now the default traced/serving variant
    # (traced_early_halt_block); multistep and single-step are the other traced loops. Only an
    # eager fallback (or a non-traced path) is unexpected here.
    valid_denoise_paths = {"traced_denoise_block", "traced_denoise_multistep_block", "traced_early_halt_block"}
    if session_events[0]["denoise_path"] not in valid_denoise_paths:
        raise AssertionError(f"unexpected denoise path: {session_events[0]}")
    if int(session_events[0]["max_denoise_steps"]) != max_denoise_steps:
        raise AssertionError(
            f"server session used {session_events[0]['max_denoise_steps']} denoise steps, "
            f"expected {max_denoise_steps}"
        )
    if "falling back to eager" in log_segment.lower():
        raise AssertionError("server log contains eager-fallback marker")
    compile_marker_count = log_segment.count("Building trisc")
    if require_no_compile_markers and compile_marker_count:
        raise AssertionError(f"timed request contains {compile_marker_count} first-use kernel compile markers")

    block0 = block0_events[0]
    blocks = [
        {
            "block_idx": 0,
            "latency_s": block0["block_latency_s"],
            "denoise_latency_s": block0["denoise_latency_s"],
            "commit_latency_s": block0["commit_latency_s"],
            "denoise_steps": block0["denoise_steps"],
            "committed_tokens": block0["committed_tokens"],
            "start_pos": block0["start_pos"],
            "next_pos": block0["next_pos"],
            "capture_inclusive": True,
        }
    ]
    blocks.extend(
        {
            "block_idx": event["block_idx"],
            "latency_s": event["block_latency_s"],
            "denoise_latency_s": event["denoise_latency_s"],
            "commit_latency_s": event["commit_latency_s"],
            "denoise_steps": event["denoise_steps"],
            "committed_tokens": event["committed_tokens"],
            "start_pos": event["start_pos"],
            "next_pos": event["next_pos"],
            "capture_inclusive": False,
        }
        for event in decode_events
    )
    blocks.sort(key=lambda block: block["block_idx"])
    if len(blocks) != blocks_requested:
        raise AssertionError(f"expected {blocks_requested} real blocks, got {len(blocks)}")

    # Trace-event provenance (recorded, not strictly asserted — see the lenient note above).
    trace_ids = captures[0]["trace_ids"] if captures else []
    expected_execute_calls = max_denoise_steps * blocks_requested
    denoise_steps = [int(block["denoise_steps"]) for block in blocks]
    # Early-halt (the default traced/serving variant) commits fewer than the full budget once the
    # trajectory converges; the realized per-block step counts are recorded below. Only a count
    # ABOVE the budget is unexpected.
    if any(steps > max_denoise_steps for steps in denoise_steps):
        raise AssertionError(f"denoise steps exceed budget {max_denoise_steps}: {denoise_steps}")

    usage = response["usage"]
    actual_prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    if actual_prompt_tokens != target:
        raise AssertionError(f"tokenizer prefix target {target}, server usage {actual_prompt_tokens}")
    expected_completion = blocks_requested * CANVAS_LENGTH
    if completion_tokens != expected_completion:
        raise AssertionError(f"expected {expected_completion} committed output tokens, got {completion_tokens}")

    steady_latencies = [float(block["latency_s"]) for block in blocks[1:]]
    steady_denoise_latencies = [float(block["denoise_latency_s"]) for block in blocks[1:]]
    steady_commit_latencies = [float(block["commit_latency_s"]) for block in blocks[1:]]
    steady_s = _mean(steady_latencies)
    choice = response["choices"][0]
    output_ids = choice.get("token_ids") or []
    return {
        "target_logical_prompt_tokens": target,
        "actual_logical_prompt_tokens": actual_prompt_tokens,
        "aligned_cache_tokens": int(block0["cache_len"]),
        "repeat_index": repeat_index,
        "max_denoise_steps": max_denoise_steps,
        "prompt": prompt_meta,
        "request_wall_s": round(request_wall_s, 6),
        "prefill_s": float(block0["prefill_s"]),
        "block0_ttft_s": float(block0["ttft_s"]),
        "blocks": blocks,
        "steady": {
            "latencies_s": steady_latencies,
            "mean_s": round(steady_s, 6),
            "median_s": round(statistics.median(steady_latencies), 6),
            "p99_s": round(_nearest_rank_percentile(steady_latencies, 99), 6),
            "pstdev_s": round(statistics.pstdev(steady_latencies), 6),
            "blocks_per_s": round(1 / steady_s, 6),
            "output_tokens_per_s": round(CANVAS_LENGTH / steady_s, 6),
            "denoise_mean_s": round(_mean(steady_denoise_latencies), 6),
            "denoise_ms_per_step": round(1000 * _mean(steady_denoise_latencies) / max_denoise_steps, 6),
            "commit_mean_s": round(_mean(steady_commit_latencies), 6),
            "commit_p99_s": round(_nearest_rank_percentile(steady_commit_latencies, 99), 6),
            "p99_method": "nearest-rank",
            "samples": len(steady_latencies),
        },
        "compile_markers_in_request": compile_marker_count,
        "denoise_steps_per_block": denoise_steps,
        "committed_tokens": completion_tokens,
        "position_progression": [blocks[0]["start_pos"], *[block["next_pos"] for block in blocks]],
        "trace": {
            "capture_events": len(captures),
            "replay_events": len(replays),
            "release_events": len(trace_releases),
            "metal_traces_captured": max_denoise_steps,
            "block_replays_total": blocks_requested,
            "steady_block_replays": blocks_requested - 1,
            "execute_trace_calls_total": expected_execute_calls,
            "steady_execute_trace_calls": max_denoise_steps * (blocks_requested - 1),
            "trace_ids": trace_ids,
            "released": len(trace_releases) > 0,
            "eager_fallback": "falling back to eager" in log_segment.lower(),
            "recapture_after_block0": len(captures) > 1,
        },
        "dram": {
            "trace_resident_after_block0": block0["dram"],
            "after_request_release": release_events[0]["dram"],
        },
        "response": {
            "id": response.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "output_token_count_returned": len(output_ids),
            "output_token_sha256": (
                hashlib.sha256(json.dumps(output_ids, separators=(",", ":")).encode()).hexdigest()
                if output_ids
                else None
            ),
            "text_chars": len(choice.get("text", "")),
            "text_sha256": hashlib.sha256(choice.get("text", "").encode()).hexdigest(),
        },
    }


def _aggregate_target_requests(target: int, requests: list[dict]) -> dict:
    steady_latencies = [latency for request in requests for latency in request["steady"]["latencies_s"]]
    steady_denoise_latencies = [
        float(block["denoise_latency_s"]) for request in requests for block in request["blocks"][1:]
    ]
    steady_commit_latencies = [
        float(block["commit_latency_s"]) for request in requests for block in request["blocks"][1:]
    ]
    mean_s = _mean(steady_latencies)
    max_denoise_steps = int(requests[0]["max_denoise_steps"])
    return {
        "logical_prompt_tokens": target,
        "aligned_cache_tokens": requests[0]["aligned_cache_tokens"],
        "max_denoise_steps": max_denoise_steps,
        "measured_requests": len(requests),
        "steady_block_samples": len(steady_latencies),
        "mean_s": round(mean_s, 6),
        "median_s": round(statistics.median(steady_latencies), 6),
        "p99_s": round(_nearest_rank_percentile(steady_latencies, 99), 6),
        "p99_method": "nearest-rank",
        "pstdev_s": round(statistics.pstdev(steady_latencies), 6),
        "blocks_per_s": round(1 / mean_s, 6),
        "output_tokens_per_s": round(CANVAS_LENGTH / mean_s, 6),
        "denoise_mean_s": round(_mean(steady_denoise_latencies), 6),
        "denoise_ms_per_step": round(1000 * _mean(steady_denoise_latencies) / max_denoise_steps, 6),
        "commit_mean_s": round(_mean(steady_commit_latencies), 6),
        "commit_p99_s": round(_nearest_rank_percentile(steady_commit_latencies, 99), 6),
        "compile_markers_in_timed_requests": sum(request["compile_markers_in_request"] for request in requests),
    }


def _mark_interrupted(result: dict, exc: BaseException) -> None:
    result["status"] = "interrupted"
    result["interruption"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "completed_requests": len(result["requests"]),
        "completed_targets": sorted({request["actual_logical_prompt_tokens"] for request in result["requests"]}),
    }


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(item) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("prompt lengths must be positive comma-separated integers")
    return lengths


def _parse_max_denoise_steps(value: str) -> int:
    try:
        steps = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max denoise steps must be an integer in [1, 48]") from exc
    if not 1 <= steps <= MAX_DENOISE_STEPS:
        raise argparse.ArgumentTypeError("max denoise steps must be in [1, 48]")
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--prompt-lengths", type=_parse_lengths, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--max-denoise-steps", type=_parse_max_denoise_steps, default=MAX_DENOISE_STEPS)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--fixed-budget",
        action="store_true",
        help="Disable early-halt: run the full --max-denoise-steps budget every block (baseline).",
    )
    parser.add_argument(
        "--growing-prefix",
        action="store_true",
        help="Disable DG_DENOISE_FROZEN_PREFIX: per-block trace recapture (multi-block-correct, ~4x slower).",
    )
    parser.add_argument("--require-no-compile-markers", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    parser.add_argument("--tt-metal", type=Path, default=Path("/home/zni/tt-metal"))
    parser.add_argument("--vllm-root", type=Path, default=Path("/home/zni/tt-vllm"))
    parser.add_argument("--mesh", default="P150x4")
    parser.add_argument("--trace-region-size", type=int, default=DEFAULT_TRACE_REGION_SIZE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--release-timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--label", default=None)
    return parser


def run(args) -> dict:
    if args.warmup_requests < 0 or args.repetitions <= 0:
        raise ValueError("warmup_requests must be >= 0 and repetitions must be > 0")
    args.max_denoise_steps = _parse_max_denoise_steps(str(args.max_denoise_steps))
    generated_tokens = args.blocks * CANVAS_LENGTH
    for prompt_len in args.prompt_lengths:
        aligned = ((prompt_len + 31) // 32) * 32
        if aligned + generated_tokens > args.max_model_len:
            raise ValueError(
                f"aligned prompt {aligned} + {generated_tokens} generated exceeds "
                f"max_model_len={args.max_model_len}"
            )
    # The verified 48-step denoise trace is ~1.44 GiB resident; the 10 GiB default is the proven
    # short/mid-context headroom. Long-context (>=128K) prefill needs that DRAM back for the
    # single-chunk activation (the trace region is reserved from DRAM, so a smaller region directly
    # widens the prefill headroom), so allow any region from a safe 2 GiB floor up to the 10 GiB
    # verified default. The trace footprint is context-independent (~1.44 GiB), so a smaller-but-
    # sufficient region does not change the denoise timing for contexts that already fit.
    if not (MIN_TRACE_REGION_SIZE <= args.trace_region_size <= DEFAULT_TRACE_REGION_SIZE):
        raise ValueError(
            f"trace region must be in [{MIN_TRACE_REGION_SIZE}, {DEFAULT_TRACE_REGION_SIZE}] bytes; "
            f"got {args.trace_region_size}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    command = _server_command(args)
    env, recorded_env = _server_env(args)
    result = {
        "schema_version": 1,
        "status": "running",
        "label": args.label,
        "max_model_len": args.max_model_len,
        "prompt_length_targets": args.prompt_lengths,
        "blocks_requested": args.blocks,
        "warmup_requests_per_target": args.warmup_requests,
        "measured_repetitions_per_target": args.repetitions,
        "require_no_compile_markers": args.require_no_compile_markers,
        "canvas_length": CANVAS_LENGTH,
        "max_denoise_steps": args.max_denoise_steps,
        "tt_metal_head": _git_head(args.tt_metal),
        "vllm_head": _git_head(args.vllm_root),
        "server": {
            "cwd": str(args.vllm_root),
            "command": command,
            "env": recorded_env,
            "additional_config": json.loads(command[command.index("--additional-config") + 1]),
            "log_path": str(args.server_log),
        },
        "warmups": [],
        "requests": [],
        "aggregates": [],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    base_url = f"http://{args.host}:{args.port}"
    with args.server_log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=args.vllm_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
    try:
        result["server"]["pid"] = proc.pid
        result["server"]["startup_s"] = round(_wait_for_server(proc, base_url, timeout=args.startup_timeout), 6)
        result["server"]["health"] = "ok"
        result["server"]["models"] = _http_json(base_url, "/v1/models", timeout=30)
        model_markers = _marker_records(args.server_log, "DG_VLLM_METRIC ")
        builds = [event for event in model_markers if event.get("event") == "model_build"]
        if len(builds) != 1:
            raise AssertionError(f"expected one model_build marker, got {len(builds)}")
        result["server"]["model_build"] = builds[0]

        for target in args.prompt_lengths:
            prompt_ids, prompt_meta = _token_prefix(base_url, target)
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt_ids,
                "max_tokens": generated_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "seed": 0,
                "return_token_ids": True,
            }
            for warmup_index in range(args.warmup_requests):
                response, request_wall_s, model_events, trace_events, log_segment = _run_completion(
                    base_url, args, payload
                )
                warmup = _request_summary(
                    target=target,
                    prompt_meta=prompt_meta,
                    response=response,
                    request_wall_s=request_wall_s,
                    model_events=model_events,
                    trace_events=trace_events,
                    blocks_requested=args.blocks,
                    max_denoise_steps=args.max_denoise_steps,
                    log_segment=log_segment,
                    repeat_index=warmup_index,
                )
                result["warmups"].append(
                    {
                        "logical_prompt_tokens": target,
                        "warmup_index": warmup_index,
                        "block_latencies_s": [block["latency_s"] for block in warmup["blocks"]],
                        "compile_markers": warmup["compile_markers_in_request"],
                        "trace_released": warmup["trace"]["released"],
                    }
                )

            target_requests = []
            for repeat_index in range(args.repetitions):
                response, request_wall_s, model_events, trace_events, log_segment = _run_completion(
                    base_url, args, payload
                )
                summary = _request_summary(
                    target=target,
                    prompt_meta=prompt_meta,
                    response=response,
                    request_wall_s=request_wall_s,
                    model_events=model_events,
                    trace_events=trace_events,
                    blocks_requested=args.blocks,
                    max_denoise_steps=args.max_denoise_steps,
                    log_segment=log_segment,
                    repeat_index=repeat_index,
                    require_no_compile_markers=args.require_no_compile_markers,
                )
                result["requests"].append(summary)
                target_requests.append(summary)
                args.output.write_text(json.dumps(result, indent=2) + "\n")

            aggregate = _aggregate_target_requests(target, target_requests)
            result["aggregates"].append(aggregate)
            print(
                "DG_LIVE_CONTEXT_RESULT "
                + json.dumps(
                    {
                        "max_model_len": args.max_model_len,
                        "prompt_len": target,
                        "cache_len": aggregate["aligned_cache_tokens"],
                        "max_denoise_steps": args.max_denoise_steps,
                        "steady_samples": aggregate["steady_block_samples"],
                        "steady_s": aggregate["mean_s"],
                        "p99_s": aggregate["p99_s"],
                        "output_tokens_per_s": aggregate["output_tokens_per_s"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        result["status"] = "passed"
    except KeyboardInterrupt as exc:
        _mark_interrupted(result, exc)
        raise
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            result["server"]["shutdown"] = _stop_server(proc)
        except Exception as shutdown_exc:
            result["server"]["shutdown_error"] = {
                "type": type(shutdown_exc).__name__,
                "message": str(shutdown_exc),
            }
            if result["status"] == "passed":
                result["status"] = "failed"
        if args.server_log.exists():
            result["server"]["log_bytes"] = args.server_log.stat().st_size
            result["server"]["log_sha256"] = hashlib.sha256(args.server_log.read_bytes()).hexdigest()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        "DG_LIVE_CONTEXT_SWEEP "
        + json.dumps(
            {
                "status": result["status"],
                "max_model_len": result["max_model_len"],
                "max_denoise_steps": result["max_denoise_steps"],
                "requests": len(result["requests"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
