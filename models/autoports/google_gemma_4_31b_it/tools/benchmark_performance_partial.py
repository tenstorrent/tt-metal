#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Retain latency-only diagnostics within the original incomplete stage budget.

This never marks the benchmark stage complete and does not run accuracy. The
caller must supply the original benchmark_stage invocation's monotonic timing.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from benchmark_stage.evidence import validate_performance, validate_server
from benchmark_stage.subsets import digest

SNAPSHOT = "/home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def execute(argv, log, deadline, *, client=False):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Original benchmark stage deadline exhausted")
    environment = dict(os.environ)
    if client:
        environment["VLLM_PLUGINS"] = ""
    start = time.monotonic()
    record = {
        "argv": argv,
        "started_monotonic": start,
        "remaining_seconds_at_launch": remaining,
        "client_plugins_disabled": client,
    }
    save(log.with_suffix(".command.json"), record)
    with log.open("w") as stream:
        stream.write(json.dumps(argv) + "\n")
        stream.flush()
        proc = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        try:
            code = proc.wait(timeout=max(0.001, deadline - time.monotonic()))
            record["exit_code"] = code
            if code:
                raise RuntimeError(f"Command exited {code}; see {log}")
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            # Only the host client/control hook is killed. The server hook
            # starts its bounded device launcher in a separate session.
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
            raise
        finally:
            record["finished_monotonic"] = time.monotonic()
            record["elapsed_seconds"] = record["finished_monotonic"] - start
            save(log.with_suffix(".command.json"), record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    timing = json.loads(args.timing.read_text())
    started = float(timing["started_monotonic"])
    deadline = min(float(timing["deadline_monotonic"]), started + 7200)
    if not 0 < started <= time.monotonic() or deadline <= started:
        raise ValueError("Invalid original monotonic stage timestamps")
    if Path("/proc/sys/kernel/random/boot_id").exists() and timing.get("boot_id"):
        assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == timing["boot_id"]
    directory = args.run_dir.resolve() / "partial-performance"
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "run_config.json", config)
    save(directory / "original_stage_timing.json", timing)
    summary = {
        "status": "incomplete_missing_roofline",
        "accuracy_run": False,
        "missing_reasons": [
            "Full-phase accounting cannot yet include the stock asynchronous post-return completion boundary; latency measurements alone do not complete the stage."
        ],
        "performance": {},
        "failed_attempts": [],
        "started_monotonic": started,
        "deadline_monotonic": deadline,
        "diagnostic_started_monotonic": time.monotonic(),
        "budget_seconds": deadline - started,
        "tokenizer": SNAPSHOT,
    }
    baseline = None
    batches = sorted(config.get("concurrencies", [1, 8, 16, 32]), reverse=True)
    if set(batches) - {1, 8, 16, 32}:
        raise ValueError("Only Stage 10 validated batches are supported")
    try:
        for batch in batches:
            identity_path = directory / f"perf-b{batch}-server.json"
            execute(
                [
                    *config["performance_server_command"],
                    "--max-num-seqs",
                    str(batch),
                    "--base-url",
                    config["base_url"],
                    "--output",
                    str(identity_path),
                ],
                directory / f"perf-b{batch}-server.log",
                deadline,
            )
            identity = json.loads(identity_path.read_text())
            validate_server(identity, batch, config["model"], config["base_url"], baseline=baseline, output=directory)
            if baseline is None:
                baseline = identity
            for warmup in (True, False):
                name = f"perf-b{batch}" + ("-warmup" if warmup else "")
                requests = batch if warmup else max(8, 3 * batch)
                argv = [
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
                    str(batch),
                    "--request-rate",
                    "inf",
                    "--ignore-eos",
                    "--temperature",
                    "0",
                    "--seed",
                    str(4100 + batch + int(warmup)),
                    "--percentile-metrics",
                    "ttft,tpot,itl,e2el",
                    "--metric-percentiles",
                    "50,95,99",
                    "--save-result",
                    "--save-detailed",
                    "--result-dir",
                    str(directory),
                    "--result-filename",
                    f"{name}.json",
                    "--tokenizer",
                    SNAPSHOT,
                ]
                execute(argv, directory / f"{name}.log", deadline, client=True)
                raw_path = directory / f"{name}.json"
                raw = json.loads(raw_path.read_text())
                validate_performance(raw, requests, 128)
                if raw.get("model_id") != config["model"] or raw.get("max_concurrency") != batch:
                    raise ValueError(f"{name}: raw model/concurrency mismatch")
                if not warmup:
                    summary["performance"][str(batch)] = {
                        "concurrency": batch,
                        "server_max_num_seqs": identity["max_num_seqs"],
                        "prefix_caching": identity["prefix_caching"],
                        "requests": requests,
                        "requested_input_tokens": 4096,
                        "requested_output_tokens": 128,
                        "server_identity_sha256": digest(identity),
                        "performance_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
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
                    save(directory / "summary.json", summary)
    except BaseException as exc:
        summary["failed_attempts"].append(
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        )
        raise
    finally:
        now = time.monotonic()
        summary["elapsed_seconds_from_original_invocation"] = now - started
        summary["remaining_seconds_for_reporting"] = max(0, deadline - now)
        summary["within_original_budget"] = now <= deadline
        summary["unmeasured_batches"] = [b for b in batches if str(b) not in summary["performance"]]
        save(directory / "summary.json", summary)
        print(
            json.dumps(
                {
                    "summary": str(directory / "summary.json"),
                    "status": summary["status"],
                    "remaining_seconds": max(0, deadline - now),
                }
            )
        )


if __name__ == "__main__":
    main()
