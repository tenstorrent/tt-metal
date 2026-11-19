# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run full-model tracked serving history and mixed-shape checks on an owned server."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .run_optimized_vllm_chunk_sweep import devices_free, get, idle_metrics, owned_processes, process_snapshot
from .run_optimized_vllm_measurement import cleanup_owner_decision, runtime_identity, shutdown_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--async-control", type=Path, help="Synchronous history JSON; enables async scheduling")
    args = parser.parse_args()
    if args.budget < 9 or args.output_dir.exists():
        parser.error("Use budget >=9 and a fresh evidence directory")
    if args.async_control and not args.async_control.is_file():
        parser.error("Synchronous history does not exist")
    if not devices_free():
        raise RuntimeError("Devices occupied; no server launched")
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("Port8000 occupied; no server launched")
    model = Path(__file__).resolve().parents[1]
    stage = model / "doc/optimized_vllm"
    assert "--shutdown-timeout" not in (stage / "run_server.sh").read_text()
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    evidence = args.output_dir.resolve()
    evidence.mkdir(parents=True)
    report = dict(
        passed=False,
        budget=args.budget,
        tracking=True,
        async_scheduling=bool(args.async_control),
        commands=[],
        cleanup_owner_decision=cleanup_owner_decision(stage),
    )
    manifest = evidence / "execution.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n")

    sources = sorted((model / "tt").glob("*.py")) + [
        stage / "run_server.sh",
        model / "doc/context_contract.json",
        model / "doc/datatype_sweep/selected_precision_config.json",
        repo / "models/common/sampling/generator.py",
        repo / "models/common/sampling/tt_sampling.py",
    ]
    report["source_sha256"] = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report["shared_runtime"] = runtime_identity(repo, stage)
    if args.async_control:
        control = json.loads((args.async_control.parent / "execution.json").read_text())
        assert control["passed"] and control["tracking"] and not control["async_scheduling"]
        for key in ("budget", "source_sha256", "shared_runtime"):
            assert report[key] == control[key], f"Control differs: {key}"
    save()
    try:
        command = ["bash", str(stage / "run_server.sh"), str(evidence)]
        env = dict(
            os.environ, TOKEN_BUDGET=str(args.budget), TRACKING="1", ASYNC_SCHEDULING=str(int(bool(args.async_control)))
        )
        with (evidence / "runner.log").open("w") as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        report.update(server_command=command, runner_pid=server.pid)
        save()
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Owned server exited during startup: {server.returncode}")
            try:
                get("/health")
                break
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Server health deadline; retained for diagnosis")
                time.sleep(2)
        owned = owned_processes(server.pid)
        report["owned_process_start_ticks"] = owned
        save()
        module = "models.autoports.qwen_qwen3_8_flash_next.tests."
        history = [
            sys.executable,
            "-m",
            module + "check_vllm_async_history",
            "--server-log",
            str(evidence / "server.log"),
            "--output",
            str(evidence / "history.json"),
            "--prompt-lengths",
            "5,7" if args.budget >= 17 else "3,5",
            "--output-tokens",
            "160",
        ]
        if args.async_control:
            history.extend(["--async-scheduling", "--control", str(args.async_control.resolve())])
        mixed = [
            sys.executable,
            "-m",
            module + "check_vllm_mixed",
            "--server-log",
            str(evidence / "server.log"),
            "--output",
            str(evidence / "mixed.json"),
            "--include-eval-sampling",
        ]
        audit = [
            sys.executable,
            "-m",
            module + "audit_serving_log",
            "--log",
            str(evidence / "server.log"),
            "--output",
            str(evidence / "route_audit.json"),
            "--require-device-only",
        ]
        pending = [
            sys.executable,
            "-m",
            module + "check_vllm_pending_prefill",
            "--server-log",
            str(evidence / "server.log"),
            "--output",
            str(evidence / "pending_prefill.json"),
        ]
        for name, command in (
            ("pending_prefill", pending),
            ("history", history),
            ("mixed", mixed),
            ("route_audit", audit),
        ):
            row = dict(name=name, command=command)
            report["commands"].append(row)
            save()
            with (evidence / f"{name}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            row["exit"] = result.returncode
            save()
            if result.returncode:
                raise RuntimeError(f"{name} failed; owned server retained for diagnosis")
        deadline = time.monotonic() + 120
        while True:
            idle = idle_metrics()
            if idle is not None:
                report["idle_before_shutdown"] = idle
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Requests did not drain; owned server retained")
            time.sleep(2)
        owned.update(owned_processes(server.pid))
        report["owned_process_start_ticks"] = owned
        server.send_signal(signal.SIGTERM)
        report["runner_shutdown_exit"] = server.wait(timeout=120)
        deadline = time.monotonic() + 30
        while True:
            processes = process_snapshot()
            report["remaining_owned_pids"] = [
                pid for pid, start in owned.items() if processes.get(pid, (None, None))[1] == start
            ]
            if not report["remaining_owned_pids"] or time.monotonic() >= deadline:
                break
            time.sleep(1)
        report["devices_free"] = devices_free()
        report["source_unchanged"] = all(
            hashlib.sha256((repo / path).read_bytes()).hexdigest() == value
            for path, value in report["source_sha256"].items()
        )
        report["shared_runtime_unchanged"] = runtime_identity(repo, stage) == report["shared_runtime"]
        assert not report["remaining_owned_pids"] and report["devices_free"]
        report["shutdown_audit"] = shutdown_audit(evidence / "server.log")
        assert all(
            report["shutdown_audit"][key] for key in ("no_teardown_error", "shutdown_recorded")
        ), "Unexpected teardown error or missing shutdown record"
        assert report["source_unchanged"] and report["shared_runtime_unchanged"]
        launch = json.loads((evidence / "vllm_run_summary.json").read_text())
        assert launch["stages"] == ["serve"] and launch["max_model_len"] == 262144
        assert launch["max_num_seqs"] == 2 and launch["mesh_device"] == "P300x2"
        actual = launch["server_command"]
        assert int(actual[actual.index("--max-num-batched-tokens") + 1]) == args.budget
        assert "--shutdown-timeout" not in actual
        expected_flag = "--async-scheduling" if args.async_control else "--no-async-scheduling"
        assert expected_flag in actual
        report["actual_launch_summary"] = launch
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
