#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the adopted release row on the last, already warm owned B1 server."""

import json
import os
import subprocess
import time
from pathlib import Path

from benchmark_client import CLIENT, TOKENIZER

MODEL = Path(__file__).resolve().parents[1]
BASE = MODEL / "doc/benchmark"
RUN = BASE / "run"


def main():
    timing = json.loads((BASE / "stage_timing.json").read_text())
    deadline = timing["started_monotonic"] + timing["budget_seconds"]
    dest = RUN / "release-matrix"
    dest.mkdir(exist_ok=False)
    for name, count in [("warmup", 1), ("128-128-c1", 8)]:
        command = [
            str(CLIENT),
            "bench",
            "serve",
            "--tokenizer",
            TOKENIZER,
            "--model",
            "google/gemma-4-31B-it",
            "--base-url",
            "http://127.0.0.1:8000",
            "--endpoint",
            "/v1/completions",
            "--backend",
            "vllm",
            "--dataset-name",
            "random",
            "--random-input-len",
            "128",
            "--random-output-len",
            "128",
            "--random-range-ratio",
            "0.0",
            "--num-prompts",
            str(count),
            "--max-concurrency",
            "1",
            "--request-rate",
            "inf",
            "--ignore-eos",
            "--temperature",
            "0",
            "--seed",
            str(7300 + count),
            "--ready-check-timeout-sec",
            "0",
            "--num-warmups",
            "0",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95,99",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(dest),
            "--result-filename",
            name + ".json",
        ]
        (dest / (name + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
        with (dest / (name + ".log")).open("w") as log:
            subprocess.run(
                command,
                env=dict(os.environ, VLLM_PLUGINS=""),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(1, deadline - time.monotonic() - 120),
                check=True,
            )
        path = dest / (name + ".json")
        raw = json.loads(path.read_text())
        assert raw["completed"] == count and raw["failed"] == 0
        assert raw["input_lens"] == [128] * count and raw["output_lens"] == [128] * count
        path.write_text(json.dumps(raw) + "\n")
    matrix_path = BASE / "release_benchmark_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    row = matrix["rows"][0]
    row.update(
        actual_seconds=raw["duration"],
        measurement="run/release-matrix/128-128-c1.json",
        completed_requests=raw["completed"],
    )
    row.pop("missing_reason", None)
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
