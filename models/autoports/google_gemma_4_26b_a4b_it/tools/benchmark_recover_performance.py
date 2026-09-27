# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recover serving measurements after failed accuracy without resetting its deadline.

This deliberately cannot complete the stage. It preserves the failed accuracy
attempt and runs only the original two performance profiles within the original
absolute monotonic budget. The caller must stop its accuracy client first.
"""

import argparse
import json
import math
import re
import signal
import time
from pathlib import Path
from urllib.request import urlopen

from benchmark_control import query
from benchmark_server import STATE, process_alive, process_start
from benchmark_stage.evidence import validate_performance, validate_server
from benchmark_stage.report import write_report
from benchmark_stage.roofline import load_roofline
from benchmark_stage.run import command
from benchmark_stage.subsets import digest


def confirm_idle(server, deadline):
    """Read actual scheduler gauges and owned process identity, never infer idle."""
    if time.monotonic() >= deadline:
        raise TimeoutError("Original benchmark deadline expired before recovery")
    owner = json.loads(STATE.read_text())
    if (
        owner["slots"] != 32
        or not process_alive(owner["pid"])
        or process_start(owner["pid"]) != owner["process_start"]
        or owner["control"] != server["phase_control"]
    ):
        raise ValueError("Recovery requires the existing owned 32-slot server")
    with urlopen(server["base_url"] + "/metrics", timeout=min(10, deadline - time.monotonic())) as response:
        metrics = response.read().decode()
    observed = {}
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        values = [
            float(match.group(1))
            for match in re.finditer(r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+([^\s]+)", metrics, re.MULTILINE)
        ]
        if not values or any(not math.isfinite(v) or v != 0 for v in values):
            raise ValueError(f"Existing server not confirmed idle: {name}={values}")
        observed[name] = values
    phase = query(server["phase_control"])
    if phase["pending"] or phase["errors"]:
        raise ValueError("Observer has pending work or errors; recovery cannot begin")
    return {
        "pid": owner["pid"],
        "process_start": owner["process_start"],
        "scheduler_gauges": observed,
        "observer_pending": [],
        "monotonic": time.monotonic(),
    }


def performance_argv(config, output, capacity, warmup):
    name = f"perf-b{capacity}" + ("-warmup" if warmup else "")
    requests = capacity if warmup else max(8, capacity * 3)
    args = [
        *config.get("benchmark_command", [config.get("vllm_cli", "vllm"), "bench", "serve"]),
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
        str(capacity),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--temperature",
        "0",
        "--seed",
        str(4100 + capacity + int(warmup)),
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
    return name, requests, args


def recover(output, started):
    output = Path(output).resolve()
    config = json.loads((output / "run_config.json").read_text())
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "failed":
        raise ValueError("Recovery requires the original runner to have terminated and written failed summary")
    if not math.isfinite(started) or started <= 0 or started > time.monotonic():
        raise ValueError("Invalid original monotonic start")
    if config.get("output_tokens", 128) != 128:
        raise ValueError("Recovery requires unchanged 4096/128 workload")
    budget = min(float(config.get("budget_seconds", 3600)), 3600)
    if not 0 < budget <= 3600:
        raise ValueError("Invalid original stage budget")
    deadline = started + budget
    if time.monotonic() >= deadline:
        raise TimeoutError("Original benchmark deadline has expired; cannot restart clock")
    original = output / "summary-before-performance-recovery.json"
    if original.exists():
        raise ValueError("Performance recovery was already attempted; preserve its evidence")
    for n in (32, 1):
        for warm in (True, False):
            name, _, _ = performance_argv(config, output, n, warm)
            if (output / (name + ".json")).exists():
                raise ValueError("Refusing to overwrite prior performance measurement: " + name)
    baseline = json.loads((output / "perf-b32-server.json").read_text())
    validate_server(baseline, 32, config["model"], config["base_url"], output=output)
    idle = confirm_idle(baseline, deadline)
    original.write_bytes(summary_path.read_bytes())
    report_path = output / "REPORT.md"
    if report_path.exists():
        (output / "REPORT-before-performance-recovery.md").write_bytes(report_path.read_bytes())
    original_error = summary.get("error", "Accuracy incomplete")
    summary["status"] = "failed"
    summary["error"] = original_error
    summary["performance_recovery"] = {
        "status": "running",
        "original_started_monotonic": started,
        "absolute_deadline_monotonic": deadline,
        "idle_confirmation": idle,
        "accuracy_retried": False,
        "original_error": original_error,
    }

    def interrupted(signum, frame):
        raise InterruptedError(f"Performance recovery interrupted by signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for n in (32, 1):
            server = baseline
            if n == 1:
                server_path = output / "perf-b1-server.json"
                command(
                    [
                        *config["performance_server_command"],
                        "--max-num-seqs",
                        "1",
                        "--base-url",
                        config["base_url"],
                        "--output",
                        str(server_path),
                    ],
                    output / "perf-b1-server.log",
                    deadline,
                )
                server = json.loads(server_path.read_text())
                validate_server(server, 1, config["model"], config["base_url"], baseline=baseline, output=output)
                command(
                    [*config["roofline_command"], "--run-dir", str(output), "--concurrency", "1", "--action", "check"],
                    output / "roofline-b1-check.log",
                    deadline,
                )
            for warmup in (True, False):
                name, requests, argv = performance_argv(config, output, n, warmup)
                command(argv, output / (name + ".log"), deadline)
                raw = json.loads((output / (name + ".json")).read_text())
                validate_performance(raw, requests, 128)
                if raw.get("model_id") != config["model"] or raw.get("max_concurrency") != n:
                    raise ValueError("Performance workload model/concurrency mismatch")
                if not warmup:
                    summary.setdefault("performance", {})[str(n)] = {
                        "concurrency": n,
                        "requested_input_tokens": 4096,
                        "requested_output_tokens": 128,
                        "server_max_num_seqs": server["max_num_seqs"],
                        "server_identity_sha256": digest(server),
                        "requests": requests,
                        "completed": raw["completed"],
                        **{
                            k: v
                            for k, v in raw.items()
                            if k.startswith(("mean_", "median_", "p95_", "p99_"))
                            or k
                            in (
                                "duration",
                                "request_throughput",
                                "output_throughput",
                                "total_token_throughput",
                                "total_input_tokens",
                                "total_output_tokens",
                            )
                        },
                    }
            command(
                [*config["roofline_command"], "--run-dir", str(output), "--concurrency", str(n), "--action", "collect"],
                output / f"roofline-b{n}-collect.log",
                deadline,
            )
            load_roofline(output, required=(str(n),))
        load_roofline(output, required=("1", "32"))
        summary["performance_recovery"]["status"] = "measurements_collected_stage_still_failed"
    except BaseException as exc:
        summary["performance_recovery"]["status"] = "failed"
        summary["performance_recovery"]["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            # The report always sees failed status and the original accuracy error.
            write_report(output, config, summary)
        except Exception as exc:
            summary["report_error"] = f"{type(exc).__name__}: {exc}"
        summary["status"] = "failed"
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["performance_recovery"]["within_original_budget"] = summary["elapsed_seconds"] < budget
        with (output / "REPORT.md").open("a") as report:
            report.write(f"\nFinal status: failed. {original_error}\n")
            report.write(
                "\nPerformance-only recovery retained the original monotonic deadline; accuracy was not retried.\n"
            )
            report.write(
                f"\nTotal original client-stage wall time through reporting: {summary['elapsed_seconds']:.1f} seconds.\n"
            )
        # Include report append time in the authoritative total as well.
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["performance_recovery"]["within_original_budget"] = summary["elapsed_seconds"] < budget
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        signal.signal(signal.SIGTERM, previous)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--started-monotonic", type=float, required=True)
    args = parser.parse_args()
    recover(args.run_dir, args.started_monotonic)
    raise SystemExit(1)  # Incomplete accuracy never becomes a successful stage.
