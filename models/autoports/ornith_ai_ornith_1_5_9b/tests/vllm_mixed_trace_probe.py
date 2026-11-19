# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stream mixed requests and join their timing with generator trace evidence."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import requests


def stream_lines(response):
    """Consume HTTP chunks promptly without bytewise SSE-line reconstruction."""
    # urllib3 amt=None yields each HTTP chunk, but can wait for EOF on a
    # non-chunked response. Refuse that transport instead of reporting false ITL.
    encodings = response.headers.get("Transfer-Encoding", "").lower().split(",")
    if "chunked" not in [encoding.strip() for encoding in encodings]:
        raise ValueError("Mixed latency measurement requires chunked HTTP streaming")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("Mixed latency measurement requires uncompressed streaming")
    return response.iter_lines(chunk_size=None)


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * fraction
    low = int(at)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (at - low)


def events(path):
    result = []
    for line in path.read_text(errors="replace").splitlines():
        if "ORNITH_SERVING " in line:
            payload = line.split("ORNITH_SERVING ", 1)[1]
            try:
                result.append(json.JSONDecoder().raw_decode(payload)[0])
            except ValueError:
                pass
    return result


def run_phase(request, phase, lengths, host, concurrency, admission_timeout=300):
    """Admit long prompts only after a short request starts streaming."""
    items = [(phase, length, host) for length in lengths]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        if host:
            return list(pool.map(request, items)), None
        assert concurrency >= 3, "Mixed admission requires one decoder plus two prefill workers"
        seed = next(i for i, length in enumerate(lengths) if length < 1024)
        long_indices = [i for i, length in enumerate(lengths) if length >= 1024]
        assert len(long_indices) == 2
        ready = Event()
        ordering = {"phase": phase, "seed_index": seed, "seed_length": lengths[seed], "admissions": []}

        def first_token(timestamp, received_tokens):
            ordering["seed_first_token_monotonic"] = timestamp
            ordering["seed_received_tokens_at_admission_signal"] = received_tokens
            ready.set()

        seed_future = pool.submit(request, items[seed], first_token)
        # Wake on a failed/empty seed too, then propagate the original failure.
        seed_future.add_done_callback(lambda _: ready.set())
        if not ready.wait(admission_timeout):
            raise TimeoutError(f"{phase}: no short-request first token within {admission_timeout}s")
        if seed_future.done():
            seed_future.result()
            raise RuntimeError(f"{phase}: short request completed before long-prompt admission")
        assert ordering["seed_received_tokens_at_admission_signal"] < 70
        futures = {seed: seed_future}
        remaining = long_indices + [i for i in range(len(items)) if i != seed and i not in long_indices]
        for index in remaining:
            ordering["admissions"].append(
                {"index": index, "length": lengths[index], "submitted_monotonic": time.monotonic()}
            )
            futures[index] = pool.submit(request, items[index])
        rows = [futures[i].result() for i in range(len(items))]
        ordering["seed_completed_monotonic"] = rows[seed]["completed_monotonic"]
        ordering["long_prompts_started_during_short_decode"] = all(
            ordering["seed_first_token_monotonic"] < rows[i]["started_monotonic"] < rows[seed]["completed_monotonic"]
            for i in long_indices
        )
        return rows, ordering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--length-offset", type=int, default=0)
    parser.add_argument("--long-prompts", type=int, nargs=2, default=(1033, 2051))
    args = parser.parse_args()
    if min(args.long_prompts) < 1024:
        parser.error("Both long prompts must contain at least 1024 tokens")

    def metrics():
        response = requests.get(args.server_url + "/metrics", timeout=30)
        response.raise_for_status()
        return [
            line
            for line in response.text.splitlines()
            if line.startswith(
                ("vllm:num_requests_running{", "vllm:num_requests_waiting{", "vllm:kv_cache_usage_perc{")
            )
        ]

    metrics_before = metrics()
    baseline = events(args.server_log)
    started = time.time()
    results = []

    def request(item, first_token=None):
        phase, length, host = item
        payload = {
            "model": "ornith-ai/Ornith-1.5-9B",
            "prompt": [9707 + i % 13 for i in range(length)],
            "max_tokens": 70,
            "ignore_eos": True,
            "temperature": 0,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if host:
            payload["logprobs"] = 1
        start = time.perf_counter()
        started_monotonic = time.monotonic()
        chunks, ticks, ids, text = [], [], [], ""
        with requests.post(
            args.server_url + "/v1/completions",
            json=payload,
            headers={"Accept-Encoding": "identity"},
            stream=True,
            timeout=1800,
        ) as response:
            response.raise_for_status()
            for line in stream_lines(response):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                elapsed = time.perf_counter() - start
                data = json.loads(line[6:])
                chunks.append({"seconds": elapsed, "response": data})
                for choice in data.get("choices", []):
                    token_ids = choice.get("token_ids") or []
                    if token_ids:
                        ticks.append(elapsed)
                        ids.extend(token_ids)
                        if first_token is not None:
                            first_token(time.monotonic(), len(ids))
                            first_token = None
                    text += choice.get("text", "")
        intervals = [(b - a) * 1000 for a, b in zip(ticks, ticks[1:])]
        return {
            "phase": phase,
            "length": length,
            "host": host,
            "request": payload,
            "chunks": chunks,
            "tokens": ids,
            "text": text,
            "ttft_ms": ticks[0] * 1000 if ticks else None,
            "tpot_ms": (ticks[-1] - ticks[0]) * 1000 / (len(ids) - 1) if len(ids) > 1 else None,
            "itl_ms": intervals,
            "max_decode_pause_ms": max(intervals, default=None),
            "seconds": time.perf_counter() - start,
            "started_monotonic": started_monotonic,
            "completed_monotonic": time.monotonic(),
        }

    phases = [
        ("first", [129, args.long_prompts[0], 131, 130, 129, args.long_prompts[1]], False),
        ("host", [131, 129, 130], True),
        ("return", [args.long_prompts[1], 130, 129, 131, args.long_prompts[0], 129], False),
    ]
    phases = [(phase, [n + args.length_offset for n in lengths], host) for phase, lengths, host in phases]
    admission_ordering = []
    for phase, lengths, host in phases:
        rows, ordering = run_phase(request, phase, lengths, host, args.concurrency)
        results.extend(rows)
        if ordering is not None:
            admission_ordering.append(ordering)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"status": "running", "requests": results, "admission_ordering": admission_ordering}, indent=2)
            + "\n"
        )
        print(phase, [(r["length"], len(r["tokens"])) for r in results if r["phase"] == phase], flush=True)
    duration = time.time() - started
    after = events(args.server_log)
    workload = after[len(baseline) :]
    captures = [e for e in after if e["event"] == "trace_capture"]
    # The exact generator event spelling and signature are retained verbatim.
    baseline_traces = [e for e in baseline if "signature" in e and "capture_ms" in e]
    new_traces = [e for e in workload if "signature" in e and "capture_ms" in e]
    signatures = Counter(json.dumps(e["signature"], sort_keys=True) for e in baseline_traces + new_traces)
    references = {r["length"]: r["tokens"] for r in results if r["phase"] == "first"}
    prefill = [e for e in workload if e["event"] == "prefill"]
    decode = [e for e in workload if e["event"] == "decode"]
    starts = [start for e in prefill for start in e["starts"]]
    mixed = any(
        prefill[i]["counters"]["model_replays"] > prefill[i - 1]["counters"]["model_replays"]
        and any(prefill[i]["starts"])
        for i in range(1, len(prefill))
    )
    checks = {
        "completed_70_tokens": all(len(r["tokens"]) == 70 for r in results),
        "greedy_exact_across_order_and_modes": all(r["tokens"] == references[r["length"]] for r in results),
        "trace_baseline_present": bool(baseline_traces),
        "one_capture_per_signature": bool(signatures) and max(signatures.values()) == 1,
        "no_new_captures": not new_traces,
        "no_retirement": bool(prefill) and all(e["counters"].get("releases") == 0 for e in prefill),
        "changed_continuation_offsets": len(set(starts) - {0}) >= 2,
        "concurrent_decode_during_chunks": mixed and bool(decode),
        "short_streaming_before_long_admission": len(admission_ordering) == 2
        and all(row["long_prompts_started_during_short_decode"] for row in admission_ordering),
        "same_server": len({e["pid"] for e in baseline + workload}) == 1,
    }
    ttft = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    itl = [value for r in results for value in r["itl_ms"]]
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "workload": {"phases": phases, "concurrency": args.concurrency, "outputs": 70},
        "admission_ordering": admission_ordering,
        "ttft_ms": {f"p{p}": percentile(ttft, p / 100) for p in (50, 95, 99)},
        "itl_ms": {f"p{p}": percentile(itl, p / 100) for p in (50, 95, 99)},
        "max_decode_pause_ms": max(itl, default=None),
        "completed_requests_per_second": len(results) / duration,
        "seconds": duration,
        "metrics_before": metrics_before,
        "metrics_after": metrics(),
        "requests": results,
        "baseline_traces": baseline_traces,
        "workload_events": workload,
        "capture_events": captures,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], checks, flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
