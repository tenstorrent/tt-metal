# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched native vLLM short-prompt TTFT and occupancy measurements."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 33, 64, 127, 128, 129, 256])
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--occupancy", action="store_true")
    parser.add_argument(
        "--warm-occupancy", action="store_true", help="Warm one full C8/C32 burst before each occupancy case"
    )
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 900
    while True:
        try:
            with urlopen("http://127.0.0.1:8000/health", timeout=2):
                break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("Server not ready")
            time.sleep(2)
    shapes = [(length, args.output_len, 1, args.requests) for length in args.lengths]
    shapes += [(128, 128, 1, 3)]
    if args.occupancy:
        shapes += [(128, 32, 8, 8), (100, 100, 32, 32)]
    for length, output, concurrency, count in shapes:
        name = f"s{length}-o{output}-c{concurrency}.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_performance.py")),
            "--backend",
            "vllm",
            "--model",
            "google/gemma-4-26B-A4B-it",
            "--base-url",
            "http://127.0.0.1:8000",
            "--endpoint",
            "/v1/completions",
            "--dataset-name",
            "random",
            "--random-input-len",
            str(length),
            "--random-output-len",
            str(output),
            "--random-range-ratio",
            "0.0",
            "--num-prompts",
            str(count),
            "--max-concurrency",
            str(concurrency),
            "--request-rate",
            "inf",
            "--ignore-eos",
            "--temperature",
            "0",
            "--seed",
            "4101",
            "--num-warmups",
            str(concurrency if args.warm_occupancy and concurrency > 1 else args.warmups),
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95,99",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(args.output),
            "--result-filename",
            name,
        ]
        print("RUN", name, flush=True)
        with (args.output / name.replace(".json", ".log")).open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads((args.output / name).read_text())
        if result.get("completed") != count:
            raise RuntimeError(f"{name}: only {result.get('completed')} of {count} requests completed")
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in ("completed", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_itl_ms")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
