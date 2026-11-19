# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stream mixed request shapes and retain generator trace-lifetime evidence."""

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def events(path):
    result = []
    for line in path.read_text().splitlines():
        if "QWEN_VLLM_EVENT " in line:
            result.append(json.loads(line.split("QWEN_VLLM_EVENT ", 1)[1]))
    return result


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    rank = (len(values) - 1) * p / 100
    lo = int(rank)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (rank - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, help="Prior identical workload with async scheduling disabled")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--include-eval-sampling", action="store_true")
    args = parser.parse_args()
    model = requests.get(args.server_url + "/v1/models", timeout=30).json()["data"][0]["id"]
    baseline = events(args.server_log)
    rows = []
    waves = []

    def request(length, variant, sampling_top_k=0, first=None):
        sampled = sampling_top_k > 0
        pattern = ([785, 13458, 13], [785, 3432, 13], [785, 9403, 13])[variant % 3]
        prompt = (pattern * ((length + 2) // 3))[:length]
        start = time.monotonic()
        row = dict(
            length=length,
            variant=variant,
            sampled=sampled,
            sampling_top_k=sampling_top_k or 1,
            start=start,
            text="",
            token_ids=[],
            arrivals=[],
            usage=None,
        )
        body = dict(
            model=model,
            prompt=prompt,
            temperature=1.0 if sampling_top_k == 40 else 0.8 if sampled else 0,
            top_k=sampling_top_k or 1,
            top_p=0.95 if sampled else 1.0,
            max_tokens=96,
            ignore_eos=True,
            stream=True,
            stream_options={"include_usage": True},
            return_token_ids=True,
        )
        if sampling_top_k == 40:
            body["seed"] = 42
        with requests.post(args.server_url + "/v1/completions", json=body, stream=True, timeout=1200) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if item.get("choices"):
                    row["token_ids"].extend(item["choices"][0].get("token_ids") or [])
                    text = item["choices"][0].get("text", "")
                    if text:
                        row["text"] += text
                        row["arrivals"].append(time.monotonic())
                        if first is not None:
                            first.set()
                if item.get("usage"):
                    row["usage"] = item["usage"]
        row["end"] = time.monotonic()
        row["ttft_ms"] = 1000 * (row["arrivals"][0] - start) if row["arrivals"] else None
        row["stream_gaps_ms"] = [1000 * (b - a) for a, b in zip(row["arrivals"], row["arrivals"][1:])]
        row["passed"] = (
            row["usage"] is not None
            and row["usage"]["prompt_tokens"] == length
            and row["usage"]["completion_tokens"] == 96
        )
        return row

    started = time.monotonic()
    # First encounters have no request-specific warmup. Delay the second
    # arrival until streaming proves the first request is decoding.
    shapes = [(129, 513, 0, 0), (130, 515, 0, 0), (131, 257, 0, 0), (130, 129, 20, 20), (513, 129, 0, 0)]
    if args.include_eval_sampling:
        shapes += [(129, 131, 0, 40), (130, 129, 0, 0), (129, 515, 40, 20), (131, 129, 0, 40)]
    for a, b, left_k, right_k in shapes:
        first = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(request, a, 0, left_k, first)
            if not first.wait(600):
                raise RuntimeError("First request did not reach streaming decode")
            overlap_at_submit = not left.done()
            right = pool.submit(request, b, 1, right_k)
            left_row, right_row = left.result(), right.result()
        waves.append(dict(indices=[len(rows), len(rows) + 1], overlap_at_submit=overlap_at_submit))
        rows.extend((left_row, right_row))
    ended = time.monotonic()
    observed = events(args.server_log)
    workload = observed[len(baseline) :]
    captures = [e for e in observed if e["kind"] == "capture"]
    signatures = Counter(json.dumps(e["signature"]) for e in captures)
    retirements = [e for e in workload if "release" in e["kind"] or "retir" in e["kind"]]
    submissions = [e for e in workload if e["kind"] == "prefill_submission"]
    decode = [e for e in workload if e["kind"] == "decode_submission"]
    intermediate = [e for e in workload if e["kind"] == "prefill_result" and not any(e["sample_mask"])]
    starts = sorted({s for e in submissions for s in e["starts"]})
    checks = dict(
        requests=all(r["passed"] for r in rows),
        client_overlap=all(w["overlap_at_submit"] for w in waves),
        continuation_offsets=len([s for s in starts if s > 0]) >= 2,
        concurrent_decode=any(e["active_decode_slots"] for e in submissions),
        captures_recorded=bool(captures),
        at_most_one_capture_per_signature=all(count == 1 for count in signatures.values()),
        zero_retirements=not retirements,
        device_sampling_only=all(
            e["sampling_mode"] == "device" or (e["sampling_mode"] == "none" and not any(e.get("sample_mask", [True])))
            for e in submissions + decode
        ),
        intermediate_no_output=bool(intermediate)
        and all(
            not e["output_present"] and e["counters"]["readbacks"] == e["counters"]["sampling_replays"] == 0
            for e in intermediate
        ),
        no_oov=all(e["out_of_vocabulary"] == 0 for e in workload if e["kind"] == "sampled_tokens"),
    )
    comparisons = []
    if args.control:
        controls = json.loads(args.control.read_text())["requests"]
        for index, (row, control) in enumerate(zip(rows, controls, strict=True)):
            if not row["sampled"]:
                assert len(row["token_ids"]) == len(control["token_ids"]) == 96
                comparisons.append(dict(index=index, identical=row["token_ids"] == control["token_ids"]))
        checks["nonoverlapped_greedy_control"] = all(c["identical"] for c in comparisons)
    else:
        checks["nonoverlapped_greedy_control"] = None
    ttfts = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
    gaps = [gap for r in rows for gap in r["stream_gaps_ms"]]
    result = dict(
        passed=all(v for v in checks.values() if v is not None),
        checks=checks,
        reduced=args.reduced,
        control=str(args.control) if args.control else None,
        correctness_scope="request lengths, trace lifetime and greedy control; raw token stress is not qualitative evidence",
        workload=(
            f"{len(shapes)} staggered pairs; 129/130/131/257/513/515 input tokens; 96 output tokens; concurrency2"
        ),
        include_eval_sampling=args.include_eval_sampling,
        requests=rows,
        waves=waves,
        comparisons=comparisons,
        ttft_ms={f"p{p}": percentile(ttfts, p) for p in (50, 95, 99)},
        stream_gap_ms={f"p{p}": percentile(gaps, p) for p in (50, 95, 99, 100)},
        completed_requests_per_second=len(rows) / (ended - started),
        baseline_events=baseline,
        workload_events=workload,
        capture_signatures=dict(signatures),
        scheduler_start_positions=starts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], checks=checks, output=str(args.output))))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
