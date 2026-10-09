# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Supervise eight TP4 engines, live API checks, full GPQA and the HTTP sweep.

Run only through run_galaxy_serving.sh, which owns the device lock. The endpoint
stays resident after successful qualification until the bounded service stops.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

from models.demos.qwen38_27b_qb2.demo.galaxy_serving import (
    MODEL_NAME,
    PLUGIN_REVISION,
    qualified_groups,
    qualified_runtime_environment,
    server_command,
    verify_qualified_source,
    verify_worker_bindings,
    verify_worker_precision,
)
from models.demos.qwen38_27b_qb2.tests.galaxy_api import validate_api
from models.demos.qwen38_27b_qb2.tests.galaxy_http_sweep import run_http_sweep


def write_receipt(path, report):
    report["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def stop_child(process):
    if process is None:
        return
    # Both server and evaluator are launched in dedicated process groups.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def main(args):
    task, output = args.task_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "deployment.json"
    report = dict(state="preflight", passed=False, model=MODEL_NAME, endpoint=f"http://127.0.0.1:{args.port}")
    server = evaluator = None
    server_log = None

    def terminate(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        qualification_bytes = args.qualification.read_bytes()
        qualification = json.loads(qualification_bytes)
        groups = qualified_groups(qualification)
        source = Path(__file__).resolve().parents[1]
        verify_qualified_source(qualification, source)
        if not isinstance(qualification.get("precision"), dict):
            raise ValueError("Qualification must record the effective model precision")
        plugin = subprocess.check_output(
            ["git", "-C", str(task / "vllm-plugin"), "rev-parse", "HEAD"], text=True
        ).strip()
        if plugin != PLUGIN_REVISION:
            raise ValueError("The serving launch contract requires the pinned plugin revision")
        if not importlib.metadata.version("vllm").startswith("0.26.0"):
            raise ValueError("The serving launch contract requires vLLM 0.26.0")
        # A healthy unrelated endpoint must not satisfy this job's readiness.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        checkpoint = Path(os.environ["MODEL_WEIGHTS_DIR"]).resolve()
        command = server_command(task, checkpoint, groups, port=args.port)
        runtime_environment = qualified_runtime_environment(qualification["precision"])
        report.update(
            plugin_revision=plugin,
            versions={name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "numpy")},
            qualified_groups=groups,
            qualification_sha256=hashlib.sha256(qualification_bytes).hexdigest(),
            checkpoint=str(checkpoint),
            checkpoint_config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
            command=command,
            environment=runtime_environment,
            precision=qualification.get("precision"),
            data_parallel_size=8,
            per_replica_capacity=16,
            total_capacity=128,
            model_source_sha256={
                str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(source.rglob("*.py"))
                if "__pycache__" not in path.parts
            },
        )
        environment = {key: value for key, value in os.environ.items() if not key.startswith("QWEN_")}
        environment.update(runtime_environment)
        environment.update(
            EXTRA_MODELS_DIR=str(source.parent),
            MESH_DEVICE="(8, 4)",
            HF_HOME=str(task / "hf-eval"),
            HF_HUB_OFFLINE="1",
            VLLM_DEBUG_LOG_API_SERVER_RESPONSE="false",
            VLLM_NO_USAGE_STATS="1",
            TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES="0",
        )
        for key in (
            "TT_MESH_GRAPH_DESC_PATH",
            "TT_VISIBLE_DEVICES",
            "TT_METAL_SLOW_DISPATCH_MODE",
            "TT_METAL_ALLOCATOR_MODE_HYBRID",
            "TT_METAL_DEVICE_PROFILER",
            "TT_METAL_PROFILER_DIR",
            "TT_METAL_PROFILER_MID_RUN_DUMP",
            "TT_METAL_PROFILER_CPP_POST_PROCESS",
        ):
            environment.pop(key, None)
        report["state"] = "starting"
        write_receipt(receipt, report)
        server_log = (output / "server.log").open("w")
        server = subprocess.Popen(
            command,
            cwd=source.parents[2],
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        report["server_pid"] = server.pid
        write_receipt(receipt, report)
        deadline = time.monotonic() + args.readiness_timeout
        with httpx.Client(timeout=3) as client:
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited before readiness with code {server.returncode}")
                verify_worker_precision(
                    (output / "server.log").read_text(errors="replace"), qualification["precision"], require_all=False
                )
                try:
                    ready = client.get(report["endpoint"] + "/health").is_success
                except httpx.HTTPError:
                    ready = False
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Galaxy serving readiness deadline exceeded")
                time.sleep(5)
        report["worker_bindings"] = verify_worker_bindings((output / "server.log").read_text(errors="replace"), groups)
        report["worker_precision"] = verify_worker_precision(
            (output / "server.log").read_text(errors="replace"), qualification["precision"]
        )
        report["state"] = "api_checks"
        write_receipt(receipt, report)
        asyncio.run(validate_api(report["endpoint"], output / "api.json", concurrency=128))
        report["state"] = "gpqa"
        write_receipt(receipt, report)
        eval_command = [
            str(task / "eval_env/bin/python"),
            str(source / "tests/benchmark.py"),
            "--mode",
            "gpqa",
            "--base-url",
            report["endpoint"],
            "--server-capacity",
            "16",
            "--data-parallel-size",
            "8",
            "--gpqa-count",
            "198",
            "--gpqa-concurrency",
            "128",
            "--gpqa-max-tokens",
            str(args.gpqa_max_tokens),
            "--gpqa-threshold",
            "0.892",
            "--gpqa-csv",
            str(task / "gpqa-data/gpqa_diamond.csv"),
            "--output-dir",
            str(output / "gpqa"),
        ]
        if args.retain_raw_responses:
            eval_command.append("--retain-raw-responses")
        report["evaluation_command"] = eval_command
        write_receipt(receipt, report)
        with (output / "evaluator.log").open("w") as log:
            evaluator = subprocess.Popen(
                eval_command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + args.evaluation_timeout
            while evaluator.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("Serving process exited during GPQA")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Full GPQA deadline exceeded")
                time.sleep(5)
        report["evaluation_exit_code"] = evaluator.returncode
        evaluator = None
        summary_path = output / "gpqa/summary.json"
        if not summary_path.exists():
            raise RuntimeError("GPQA did not produce a complete summary")
        report["gpqa"] = json.loads(summary_path.read_text())["gpqa_result"]
        if report["gpqa"]["completed_samples"] != 198 or not report["gpqa"]["full_dataset"]:
            raise RuntimeError("GPQA result does not cover all 198 questions")
        if args.tau_source is not None:
            report["state"] = "tau_pilot"
            command = [
                str(args.tau_python),
                str(source / "tests/tau_benchmark.py"),
                "--source",
                str(args.tau_source),
                "--output",
                str(output / "tau"),
                "--base-url",
                report["endpoint"],
            ]
            report["tau_command"] = command
            write_receipt(receipt, report)
            tau_environment = dict(environment)
            tau_environment["LD_LIBRARY_PATH"] = (
                str(args.tau_source.parent / "portaudio/root/usr/lib/x86_64-linux-gnu")
                + ":"
                + environment.get("LD_LIBRARY_PATH", "")
            )
            with (output / "tau.log").open("w") as log:
                evaluator = subprocess.Popen(
                    command, env=tau_environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                deadline = time.monotonic() + 2820
                while evaluator.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError("Serving process exited during Tau3")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Bounded Tau3 pilot deadline exceeded")
                    time.sleep(5)
            report["tau_exit_code"] = evaluator.returncode
            evaluator = None
            tau_summary = output / "tau/summary.json"
            if tau_summary.exists():
                report["tau"] = json.loads(tau_summary.read_text())
            else:
                report["tau"] = {"state": "failed", "error": "No terminal summary"}
            # Preserve unsuccessful pilot outcomes separately from the GPQA gate.
            write_receipt(receipt, report)
        if args.sweep_before_exit:
            report["state"] = "http_sweep"
            write_receipt(receipt, report)
            try:
                asyncio.run(
                    asyncio.wait_for(
                        run_http_sweep(
                            report["endpoint"], output / "http-sweep", deployment=str(receipt), priority_only=True
                        ),
                        timeout=5400,
                    )
                )
                report["http_sweep_passed"] = True
            except (httpx.HTTPError, TimeoutError) as error:
                report["http_sweep_passed"] = False
                report["http_sweep_error"] = type(error).__name__
                # Keep the failed measurement; never retry a scored burst. A
                # live health check plus owned-worker shutdown lets the next
                # independent control reset and start from a clean process.
                if server.poll() is not None:
                    raise
                with httpx.Client(timeout=5) as client:
                    client.get(report["endpoint"] + "/health").raise_for_status()
                report["healthy_after_sweep_failure"] = True
        if args.exit_after_eval:
            report.update(
                state="evaluation_completed",
                passed=report["gpqa"]["passed"]
                and report["evaluation_exit_code"] == 0
                and report.get("http_sweep_passed", True),
                resident_endpoint=False,
            )
            print(f"GALAXY_EVALUATION_FINISHED passed={report['passed']}", flush=True)
            return
        # Preserve and measure a healthy deployment even if the accuracy gate
        # misses; never relabel that result as qualified.
        report["state"] = "http_sweep"
        write_receipt(receipt, report)
        asyncio.run(run_http_sweep(report["endpoint"], output / "http-sweep", deployment=str(receipt)))
        report.update(state="serving", passed=report["gpqa"]["passed"] and report["evaluation_exit_code"] == 0)
        write_receipt(receipt, report)
        print(f"GALAXY_QUALIFICATION_FINISHED passed={report['passed']} endpoint={report['endpoint']}", flush=True)
        while server.poll() is None:
            time.sleep(5)
        raise RuntimeError(f"Serving process exited with code {server.returncode}")
    except BaseException as error:
        report.update(
            state="stopped" if isinstance(error, InterruptedError) else "failed", error_type=type(error).__name__
        )
        raise
    finally:
        # Once shutdown begins, allow cleanup to run instead of interrupting it
        # with a second signal from the enclosing service.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_child(evaluator)
        stop_child(server)
        if server_log is not None:
            server_log.close()
        report["owned_processes_stopped"] = True
        write_receipt(receipt, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--readiness-timeout", type=int, default=5400)
    parser.add_argument("--evaluation-timeout", type=int, default=14400)
    parser.add_argument("--gpqa-max-tokens", type=int, default=32768)
    parser.add_argument("--retain-raw-responses", action="store_true")
    parser.add_argument(
        "--sweep-before-exit",
        action="store_true",
        help="Run the priority-context physical Galaxy sweep before bounded shutdown",
    )
    parser.add_argument("--tau-source", type=Path, help="Pinned Tau3 source; run the bounded pilot after full GPQA")
    parser.add_argument("--tau-python", type=Path, help="Isolated interpreter containing the pinned Tau3 package")
    parser.add_argument(
        "--exit-after-eval",
        action="store_true",
        help="Stop owned workers after full GPQA; omit HTTP sweep and resident serving",
    )
    arguments = parser.parse_args()
    if (arguments.tau_source is None) != (arguments.tau_python is None):
        parser.error("--tau-source and --tau-python must be provided together")
    main(arguments)
