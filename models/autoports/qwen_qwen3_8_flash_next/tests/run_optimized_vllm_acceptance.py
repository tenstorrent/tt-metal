# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run final serving acceptance on an owned server, preserving failures for diagnosis."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from .run_optimized_vllm_chunk_sweep import devices_free, get, idle_metrics, owned_processes, process_snapshot
from .run_optimized_vllm_measurement import cleanup_owner_decision, runtime_identity, sha, shutdown_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.budget != 9 or args.output_dir.exists():
        parser.error("Use selected budget9 and a fresh evidence directory")
    assert devices_free(), "Devices occupied; no server launched"
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", 8000)) != 0, "Port8000 occupied"
    model = Path(__file__).resolve().parents[1]
    stage = model / "doc/optimized_vllm"
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    evidence = args.output_dir.resolve()
    evidence.mkdir(parents=True)
    server_dir = evidence / "server"
    server_dir.mkdir()
    report = dict(passed=False, budget=args.budget, cleanup_owner_decision=cleanup_owner_decision(stage))
    manifest = evidence / "execution.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n")

    paths = sorted((model / "tt").glob("*.py")) + [
        stage / "run_server.sh",
        stage / "acceptance.sh",
        model / "doc/context_contract.json",
        model / "doc/datatype_sweep/selected_precision_config.json",
        repo / "models/common/sampling/generator.py",
        repo / "models/common/sampling/tt_sampling.py",
    ]
    report["source_sha256"] = {str(p.relative_to(repo)): sha(p) for p in paths}
    report["shared_runtime"] = runtime_identity(repo, stage)
    save()
    try:
        command = ["bash", str(stage / "run_server.sh"), str(server_dir)]
        env = dict(os.environ, TOKEN_BUDGET=str(args.budget), TRACKING="0", ASYNC_SCHEDULING="1")
        with (server_dir / "runner.log").open("w") as log:
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
        command = [
            "bash",
            str(stage / "acceptance.sh"),
            str(server_dir / "server.log"),
            str(evidence / "checks"),
            str(args.budget),
        ]
        report["acceptance_command"] = command
        save()
        with (evidence / "acceptance_driver.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=86400)
        report["acceptance_exit"] = result.returncode
        save()
        gate_exit = evidence / "checks/eval_readiness_check.exit"
        assert result.returncode in (0, 1) and gate_exit.is_file(), "Acceptance failed; owned server retained"
        assert int(gate_exit.read_text()) == result.returncode
        for phase in ("quality", "mixed", "timing", "load"):
            audit = json.loads((evidence / f"checks/{phase}_route_audit.json").read_text())
            assert audit["host_submissions"] == audit["out_of_vocabulary"] == 0 and audit["device_submissions"] > 0
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
            remaining = process_snapshot()
            report["remaining_owned_pids"] = [
                pid for pid, started in owned.items() if pid in remaining and remaining[pid][1] == started
            ]
            if not report["remaining_owned_pids"] or time.monotonic() >= deadline:
                break
            time.sleep(1)
        report["devices_free"] = devices_free()
        assert not report["remaining_owned_pids"] and report["devices_free"], "Owned cleanup incomplete"
        report["shutdown_audit"] = shutdown_audit(server_dir / "server.log")
        assert report["shutdown_audit"]["shutdown_recorded"] and report["shutdown_audit"]["no_teardown_error"]
        report["shared_runtime_unchanged"] = runtime_identity(repo, stage) == report["shared_runtime"]
        report["source_unchanged"] = all(sha(repo / p) == value for p, value in report["source_sha256"].items())
        assert report["shared_runtime_unchanged"] and report["source_unchanged"]
        launch = json.loads((server_dir / "vllm_run_summary.json").read_text())
        assert launch["max_model_len"] == 262144 and launch["max_num_seqs"] == 2
        actual = launch["server_command"]
        assert int(actual[actual.index("--max-num-batched-tokens") + 1]) == args.budget
        assert "--async-scheduling" in actual and "--shutdown-timeout" not in actual
        report["actual_launch_summary"] = launch
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
