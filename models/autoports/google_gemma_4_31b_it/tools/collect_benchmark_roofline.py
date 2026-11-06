#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect modeled work over model-owned, consecutive host completion boundaries."""
import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from benchmark_client import exchange
from benchmark_work import ASSUMPTIONS, TP, decode_work, prefill_work

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / "tt"))
from benchmark_phases import socket_path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def healthy(value):
    if value.get("error") or value.get("errors") or value.get("pending_steps"):
        raise ValueError(f"Phase observer is not quiescent and healthy: {value}")
    if value.get("clock") != "perf_counter_ns":
        raise ValueError("Unknown observer clock")


def check(args, server):
    endpoint = socket_path(args.run_dir / f"phase-b{args.concurrency}.jsonl")
    before = exchange(endpoint, "check")
    healthy(before)
    identity = before["identity"]
    if (
        identity["layers"] != list(range(60))
        or identity["precision"]["config_id"] != "head_bfp4"
        or len(identity["device_ids"]) != TP
    ):
        raise ValueError("Observer model identity does not match full selected model")
    # A real, bounded token-out request proves both async completion paths before
    # accuracy. It is outside all measured windows and retained as setup evidence.
    exchange(endpoint, "reset")
    payload = {
        "model": server["model"],
        "prompt": [2] + [100] * 31,
        "max_tokens": 2,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        server["base_url"].rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    observed = exchange(endpoint, "drain")
    healthy(observed)
    phases = {e["phase"] for e in observed["events"] if e["event"] == "complete"}
    if phases != {"prefill", "decode"}:
        raise ValueError(f"Probe did not observe both completed phases: {phases}")
    write(
        args.run_dir / f"roofline-b{args.concurrency}-readiness.json",
        {
            "status": "ready",
            "request": payload,
            "response": result,
            "observer": observed,
            "device_profiler_enabled": False,
            "added_device_synchronization": False,
        },
    )


def account(raw, start, end):
    healthy(start)
    healthy(end)
    events = end["events"]
    submitted = {e["step_id"]: e for e in events if e["event"] == "submit"}
    completed = sorted((e for e in events if e["event"] == "complete"), key=lambda e: e["monotonic_ns"])
    if len(completed) != len(submitted) or {e["step_id"] for e in completed} != set(submitted):
        raise ValueError("Missing or duplicate async completion evidence")
    # Client monotonic clocks share the host perf_counter basis. Use first actual
    # request start through last observed client token, excluding process import.
    first = min(raw["start_times"])
    last = max(s + t + sum(itls) for s, t, itls in zip(raw["start_times"], raw["ttfts"], raw["itls"]))
    if not 0 < last - first <= raw["duration"] + 0.001:
        raise ValueError("Invalid client observation interval")
    lower = int(first * 1e9)
    upper = max(int(last * 1e9), max((e["monotonic_ns"] for e in completed), default=0))
    if (upper - lower) / 1e9 > raw["duration"]:
        raise ValueError("Executed async work extends beyond benchmark duration; cannot omit its time")
    if lower < start["boundary_ns"] or upper > end["boundary_ns"]:
        raise ValueError("Client interval is outside observer capture")
    if any(e["monotonic_ns"] < lower for e in completed):
        raise ValueError("Non-measured request was captured (warmup/ready check)")
    seconds = {"prefill": 0.0, "decode": 0.0}
    work = {"prefill": 0, "decode": 0}
    rows, prefill_tokens, decode_tokens = [], 0, 0
    previous = lower
    for event in completed:
        submit = submitted[event["step_id"]]
        if submit["monotonic_ns"] > event["monotonic_ns"]:
            raise ValueError("Completion precedes submission")
        phase = event["phase"]
        boundary = event["monotonic_ns"]
        elapsed = max(0, boundary - previous) / 1e9
        previous = max(previous, boundary)
        if phase == "prefill":
            detail = prefill_work(event["starts"], event["ends"])
            prefill_tokens += detail["tokens"]
            amount = detail["flops"]
        else:
            detail = decode_work(event["effective_positions"], event["batch_size"])
            decode_tokens += detail["active_tokens"]
            amount = detail["dram_bytes"]
        seconds[phase] += elapsed
        work[phase] += amount
        rows.append(
            {
                "step_id": event["step_id"],
                "phase": phase,
                "submit_ns": submit["monotonic_ns"],
                "completion_ns": event["monotonic_ns"],
                "interval_seconds": elapsed,
                "sampling_mode": event["sampling_mode"],
                "work": detail,
            }
        )
    if not completed or previous > upper:
        raise ValueError("Invalid phase window")
    tail = (upper - previous) / 1e9
    seconds[completed[-1]["phase"]] += tail
    # Cached prefixes reduce actual prefill work and remain explicit. A random
    # workload should have no substantial reuse; do not invent missing work.
    if not 0 < prefill_tokens <= raw["total_input_tokens"]:
        raise ValueError("Invalid measured prefill token coverage")
    minimum_decode = raw["total_output_tokens"] - raw["completed"]
    if decode_tokens < minimum_decode:
        raise ValueError(f"Decode work missing: {decode_tokens} < {minimum_decode}")
    return {
        "window_start_ns": lower,
        "window_end_ns": upper,
        "window_seconds": (upper - lower) / 1e9,
        "window_method": "First upstream request start through later of last received token (start+TTFT+sum ITLs) and last async completion, same-host perf_counter. Client process import/reporting excluded.",
        "async_completion_tail_after_client_seconds": max(0, upper / 1e9 - last),
        "prefill_tokens_executed": prefill_tokens,
        "input_tokens_cached_or_not_executed": raw["total_input_tokens"] - prefill_tokens,
        "decode_active_tokens_executed": decode_tokens,
        "delivered_decode_tokens": minimum_decode,
        "extra_decode_tokens_executed": decode_tokens - minimum_decode,
        "final_client_tail_seconds": tail,
        "seconds": seconds,
        "work": work,
        "steps": rows,
    }


def collect(args, server):
    directory, batch = args.run_dir, args.concurrency
    raw_path = directory / f"perf-b{batch}.json"
    raw = json.loads(raw_path.read_text())
    start_path, end_path = [directory / f"phase-b{batch}-window-{part}.json" for part in ("start", "end")]
    start, end = [json.loads(p.read_text()) for p in (start_path, end_path)]
    # Collector runs before server stop: verify the same exporter is still alive.
    live = exchange(socket_path(directory / f"phase-b{batch}.jsonl"), "check")
    healthy(live)
    if live["pid"] != end["pid"] or live["identity"] != end["identity"]:
        raise ValueError("Running observer changed since measured profile")
    if raw["failed"] or raw["completed"] != raw["num_prompts"]:
        raise ValueError("Incomplete performance request population")
    detail = account(raw, start, end)
    peak_path = MODEL / "doc/optimized_decoder/peak_basis.json"
    peak = json.loads(peak_path.read_text())
    detail.update(
        assumptions=ASSUMPTIONS,
        observer_identity=end["identity"],
        observer_pid=end["pid"],
        raw_events_file=end_path.name,
        raw_events_sha256=sha(end_path),
        start_file=start_path.name,
        start_sha256=sha(start_path),
        peak_basis=peak,
        device_time_available=False,
        participating_devices=TP,
    )
    evidence_path = directory / f"roofline-b{batch}-evidence.json"
    write(evidence_path, detail)
    if evidence_path.stat().st_size > 400000:
        compressed = evidence_path.with_suffix(".json.gz")
        compressed.write_bytes(gzip.compress(evidence_path.read_bytes(), mtime=0))
        evidence_path.unlink()
        evidence_path = compressed
    timing = (
        "Disjoint intervals between consecutive adapter-observed async completions, attributed to the ending completion phase; first interval starts at first client request. "
        "Plugin host sampling/output/scheduling, communication and idle gaps after adapter return fall inside following intervals; final client tail assigned to last phase. "
        "Operator decision4 approved attribution. No extra synchronization or live device profiler. Host wall time, not device time."
    )
    common = {
        "timing_scope": "full_phase_wall_time",
        "timing_method": timing,
        "work_method": "tools/benchmark_work.py; per-completed-step executed shape accounting, assumptions and component totals in evidence",
        "peak_source": "doc/optimized_decoder/peak_basis.json: 4 whole P300c Blackhole ASICs, 120 cores/ASIC,1.35GHz,4096 LoFi FLOPs/core/cycle;512GB/s DRAM/ASIC",
        "evidence": evidence_path.name,
        "evidence_sha256": sha(evidence_path),
    }
    row = {
        "performance_sha256": sha(raw_path),
        "server_identity_sha256": digest(server),
        "requests": raw["completed"],
        "input_tokens": raw["total_input_tokens"],
        "output_tokens": raw["total_output_tokens"],
        "prefill": dict(
            common,
            flops=detail["work"]["prefill"],
            seconds=detail["seconds"]["prefill"],
            peak_flops_per_second=TP * peak["peak_flops_per_s"],
        ),
        "decode": dict(
            common,
            dram_bytes=detail["work"]["decode"],
            seconds=detail["seconds"]["decode"],
            peak_dram_bytes_per_second=TP * peak["peak_dram_bytes_per_s"],
        ),
    }
    path = directory / "roofline.json"
    profiles = json.loads(path.read_text()) if path.exists() else {}
    profiles[str(batch)] = row
    write(path, profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--action", choices=("check", "collect"), required=True)
    args = parser.parse_args()
    server = json.loads((args.run_dir / f"perf-b{args.concurrency}-server.json").read_text())
    (check if args.action == "check" else collect)(args, server)


if __name__ == "__main__":
    main()
