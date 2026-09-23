# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure warmed streaming 128/128/c1 and check request turnover in one server."""

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turnover", action="store_true")
    args = parser.parse_args()
    model = requests.get(args.url + "/v1/models", timeout=10).json()["data"][0]["id"]

    def request(index=0):
        payload = dict(
            model=model,
            prompt=[100 + ((i + index * 13) % 256) for i in range(128)],
            max_tokens=128,
            temperature=0,
            seed=17,
            ignore_eos=True,
            stream=True,
            stream_options={"include_usage": True},
        )
        begin = time.perf_counter()
        first = last = None
        fragments = []
        usage = None
        with requests.post(args.url + "/v1/completions", json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    if choice.get("text"):
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        fragments.append(choice["text"])
        assert usage and usage["prompt_tokens"] == 128 and usage["completion_tokens"] == 128, usage
        assert first is not None and last > first
        return dict(
            index=index,
            ttft_ms=(first - begin) * 1000,
            tpot_ms=(last - first) / 127 * 1000,
            tokens_per_second=127 / (last - first),
            text="".join(fragments),
            usage=usage,
        )

    warmup = request()
    measured = [request() for _ in range(8)]
    assert all(row["text"] == warmup["text"] for row in measured), "Repeated B1 requests changed output"
    turnover = []
    if args.turnover:
        controls = [request(i) for i in range(5)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            concurrent_rows = list(pool.map(request, range(5)))
        # Batch shapes may change floating-point rounding. Cross-shape text
        # equality is diagnostic; returning to the same B1 shape must match.
        for row, control in zip(concurrent_rows, controls):
            row["matches_sequential_text"] = row["text"] == control["text"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            full_rows = list(pool.map(request, range(16)))
        turnover = concurrent_rows + full_rows + [request()]
        assert turnover[-1]["text"] == controls[0]["text"]
    result = dict(warmup=warmup, measured=measured, turnover=turnover)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            dict(
                median_tokens_per_second=statistics.median(r["tokens_per_second"] for r in measured),
                min_tokens_per_second=min(r["tokens_per_second"] for r in measured),
                median_ttft_ms=statistics.median(r["ttft_ms"] for r in measured),
                requests=len(measured),
                turnover_checked=bool(turnover),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
