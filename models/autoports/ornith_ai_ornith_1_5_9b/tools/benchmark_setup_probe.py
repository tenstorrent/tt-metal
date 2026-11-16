# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded pre-invocation collector and benchmark protocol pilot."""

import json
import subprocess
import time
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL / "doc/benchmark"
EVAL = "/home/ttuser/ornith-9b/benchmark-clients/eval/bin/python"


def run(command, log, timeout):
    started = time.monotonic()
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    record = {"command": command, "wall_seconds": time.monotonic() - started, "exit": result.returncode}
    log.with_suffix(".command.json").write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"Setup command failed: {log}")
    return record


def main():
    config = json.loads((ROOT / "run_config.draft.json").read_text())
    output = ROOT / "pilot_performance"
    output.mkdir(exist_ok=False)
    records = []
    records.append(
        run(
            [
                *config["performance_server_command"],
                "--max-num-seqs",
                "32",
                "--base-url",
                config["base_url"],
                "--output",
                str(output / "perf-b32-server.json"),
            ],
            output / "server-selection.log",
            1900,
        )
    )
    collector = [*config["roofline_command"], "--run-dir", str(output), "--concurrency", "32"]
    records.append(run([*collector, "--action", "check"], output / "collector-check.log", 60))
    for name, requests, seed in (("perf-b32-warmup", 32, 9001), ("perf-b32", 96, 9002)):
        command = [
            *config["benchmark_command"],
            "--backend",
            "vllm",
            "--model",
            config["model"],
            "--base-url",
            config["base_url"],
            "--endpoint",
            "/v1/completions",
            "--dataset-name",
            "random",
            "--random-input-len",
            "4096",
            "--random-output-len",
            "128",
            "--random-range-ratio",
            "0.0",
            "--num-prompts",
            str(requests),
            "--max-concurrency",
            "32",
            "--request-rate",
            "inf",
            "--ignore-eos",
            "--temperature",
            "0",
            "--seed",
            str(seed),
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95,99",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            str(output),
            "--result-filename",
            name + ".json",
        ]
        records.append(run(command, output / (name + ".log"), 600))
    records.append(run([*collector, "--action", "collect"], output / "collector-collect.log", 60))
    pilot = ROOT / "pilot_accuracy"
    command = [
        EVAL,
        "-m",
        "benchmark_stage",
        "evaluate",
        "--model",
        config["model"],
        "--base-url",
        config["base_url"],
        "--manifest",
        str(ROOT / "pilot_subset/manifest.json"),
        "--task",
        ",".join(config["tasks"]),
        "--output",
        str(pilot),
        "--generation",
        json.dumps(config["generation"]["mmlu_pro"]),
        "--concurrency",
        "32",
        "--shared",
    ]
    records.append(run(command, ROOT / "pilot_accuracy.log", 1800))
    (ROOT / "setup_probe.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
