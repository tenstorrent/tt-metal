# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure a real serving arm with explicit source provenance and owned cleanup."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .run_optimized_vllm_chunk_sweep import devices_free, get, idle_metrics, owned_processes, process_snapshot


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_identity(repo, stage):
    plugin = Path("/home/ttuser/qwen38-flash/vllm-tt-plugin")
    plugin_status = subprocess.check_output(["git", "-C", str(plugin), "status", "--porcelain"], text=True)
    assert not plugin_status, "Plugin has uncommitted changes"
    vllm_root = Path(importlib.metadata.distribution("vllm").locate_file("vllm")).resolve()
    tree = hashlib.sha256()
    for path in sorted(vllm_root.rglob("*.py")):
        tree.update(str(path.relative_to(vllm_root)).encode() + b"\0" + path.read_bytes() + b"\0")
    readiness_root = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check"
    readiness_tree = hashlib.sha256()
    for path in sorted(readiness_root.rglob("*.py")):
        readiness_tree.update(str(path.relative_to(readiness_root)).encode() + b"\0" + path.read_bytes() + b"\0")
    build = json.loads((stage / "wide_sampler_build.json").read_text())
    paths = {repo / name for name in build["sha256"] if name.startswith(("ttnn/", "build_Release/"))}
    paths.update(
        repo / "ttnn/cpp/ttnn/operations/ccl/all_gather/device" / name
        for name in ("all_gather_multicast_factory.cpp", "kernels/multicast_reader.cpp", "kernels/multicast_writer.cpp")
    )
    return dict(
        plugin_root=str(plugin),
        plugin_head=subprocess.check_output(["git", "-C", str(plugin), "rev-parse", "HEAD"], text=True).strip(),
        plugin_status=plugin_status,
        vllm_version=importlib.metadata.version("vllm"),
        vllm_root=str(vllm_root),
        vllm_python_tree_sha256=tree.hexdigest(),
        readiness_root=str(readiness_root),
        readiness_python_tree_sha256=readiness_tree.hexdigest(),
        native_sha256={str(path.relative_to(repo)): sha(path) for path in sorted(paths)},
    )


def cleanup_owner_decision(stage):
    path = stage / "owner_decision_shutdown_2026_11_17.json"
    decision = json.loads(path.read_text())
    assert decision["accepted_by_owner"] and not decision["plugin_change_authorized"]
    assert decision["shutdown_timeout"] == 0
    plugin = "/home/ttuser/qwen38-flash/vllm-tt-plugin"
    assert (
        subprocess.check_output(["git", "-C", plugin, "rev-parse", "HEAD"], text=True).strip()
        == decision["plugin_head"]
    )
    assert not subprocess.check_output(["git", "-C", plugin, "status", "--porcelain"], text=True)
    return dict(path=str(path), sha256=sha(path), accepted=True, plugin_unchanged=True)


def shutdown_audit(path):
    """Record orderly teardown separately from owner-accepted process/handle cleanup."""
    log = path.read_text()
    captured, released = Counter(), Counter()
    for line in log.splitlines():
        if "QWEN_VLLM_EVENT " not in line:
            continue
        event = json.loads(line.split("QWEN_VLLM_EVENT ", 1)[1])
        if event["kind"] == "capture":
            captured[event["trace_id"]] += 1
        elif event["kind"] == "shutdown_release":
            released[event["trace_id"]] += 1
    start = log.find("[shutdown] API server: shutdown triggered")
    tail = log[start:] if start >= 0 else log
    runner = path.parent / "runner.log"
    runner_log = runner.read_text() if runner.exists() else ""
    return dict(
        captured_traces=sum(captured.values()),
        released_traces=sum(released.values()),
        all_model_traces_released=bool(captured) and captured == released,
        no_forced_process_kill="force killing" not in tail and "sending SIGKILL" not in runner_log,
        no_teardown_error=not any(
            marker in tail for marker in ("ERROR", "Async decode drain failed", "Failed to release", "shutdown failed")
        ),
        shutdown_recorded="[shutdown] API server: shutdown triggered" in log,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-revision", help="Replace only generator.py and slot_state.py for this arm")
    parser.add_argument("--serial-reference", type=Path, help="Prior matching serial-prefill burst summary")
    args = parser.parse_args()
    if args.budget < 8 or args.output_dir.exists():
        parser.error("Use budget >=8 and a fresh output directory")
    if args.serial_reference is not None and not args.serial_reference.is_file():
        parser.error("Serial reference does not exist")
    if not devices_free():
        raise RuntimeError("Devices occupied; no source changes or server launch")
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("Port8000 is occupied; no source changes or server launch")
    model = Path(__file__).resolve().parents[1]
    stage = model / "doc/optimized_vllm"
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    args.output_dir.mkdir(parents=True)
    evidence = args.output_dir.resolve()
    source_paths = [model / "tt/generator.py", model / "tt/slot_state.py"]
    saved = {str(path): path.read_text() for path in source_paths}
    report = dict(passed=False, source_restored=False, baseline_revision=args.baseline_revision)
    manifest = evidence / "execution.json"

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n")

    save()
    # This recovery record remains available even if a failed server is retained.
    (evidence / "source_restore.json").write_text(json.dumps(saved, indent=2) + "\n")
    server = None
    cleaned = False
    measured_source = dict(saved)
    try:
        if args.baseline_revision:
            replacements = {
                path: subprocess.check_output(
                    ["git", "show", f"{args.baseline_revision}:{path.relative_to(repo)}"], text=True
                )
                for path in source_paths
            }
            for path, value in replacements.items():
                path.write_text(value)
                measured_source[str(path)] = value
        server_script = stage / "run_server.sh"
        assert "--shutdown-timeout" not in server_script.read_text(), "Preserve owner-selected existing cleanup"
        report["cleanup_owner_decision"] = cleanup_owner_decision(stage)
        tt = json.loads(re.search(r"--tt-config '([^']+)'", server_script.read_text())[1])
        selected = model / "doc/datatype_sweep/selected_precision_config.json"
        context = json.loads((model / "doc/context_contract.json").read_text())["supported_context"]
        assert context == 262144
        contract = dict(
            context=context,
            max_num_seqs=2,
            mesh=[1, 4],
            block_size=32,
            sample_on_device_mode="all",
            async_scheduling=True,
            tracking=False,
            device_profiler=False,
            shutdown_timeout=0,
            max_num_batched_tokens=args.budget,
            selected_config_sha256=sha(selected),
            **tt,
        )
        paths = (
            source_paths
            + [
                model / "tt/generator_vllm.py",
                repo / "models/common/sampling/generator.py",
                repo / "models/common/sampling/tt_sampling.py",
                server_script,
                stage / "benchmark.sh",
                selected,
            ]
            + sorted((model / "tt").glob("*.py"))
        )
        provenance = dict(
            server_contract=contract,
            checkout_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            baseline_generator_revision=args.baseline_revision,
            source_sha256={str(path.relative_to(repo)): sha(path) for path in paths},
            baseline_scope="Only generator.py and slot_state.py are replaced; repaired shared runtime is identical in both arms.",
            shared_runtime=runtime_identity(repo, stage),
        )
        (evidence / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        command = ["bash", str(server_script), str(evidence)]
        env = dict(os.environ, TOKEN_BUDGET=str(args.budget), TRACKING="0", ASYNC_SCHEDULING="1")
        with (evidence / "runner.log").open("w") as output:
            server = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        report.update(server_command=command, runner_pid=server.pid)
        save()
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {server.returncode}")
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
        loaded = set()
        launch_environments = set()
        titled_workers = []
        env_names = (
            "TT_METAL_TRACE_ALLOC_TRACKING",
            "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
            "TT_METAL_DEVICE_PROFILER",
            "TT_METAL_WATCHER",
            "TT_VISIBLE_DEVICES",
            "TT_MESH_GRAPH_DESC_PATH",
            "TT_MODEL_CLASS_OVERRIDES",
        )
        for pid in owned:
            maps = Path(f"/proc/{pid}/maps")
            if maps.exists():
                mapped = maps.read_text()
                if "/_ttnn.so" in mapped:
                    environment = dict(
                        entry.split("=", 1)
                        for entry in Path(f"/proc/{pid}/environ").read_bytes().decode().split("\0")
                        if "=" in entry
                    )
                    selected_env = {name: environment.get(name) for name in env_names}
                    command_line = Path(f"/proc/{pid}/cmdline").read_bytes().decode()
                    if command_line.startswith("VLLM::EngineCore"):
                        # setproctitle can overwrite /proc's initial environment region.
                        # It does not change Python/C runtime getenv values. Verify the
                        # untouched API parent and preserve this observation separately.
                        titled_workers.append(dict(pid=pid, initial_environment=selected_env))
                    else:
                        assert "vllm.entrypoints.openai.api_server" in command_line
                        assert selected_env["TT_METAL_TRACE_ALLOC_TRACKING"] == "0"
                        assert selected_env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
                        assert selected_env["TT_METAL_DEVICE_PROFILER"] in (None, "0")
                        assert selected_env["TT_METAL_WATCHER"] is None
                        launch_environments.add(json.dumps(selected_env, sort_keys=True))
                for line in mapped.splitlines():
                    path = Path(line.split()[-1])
                    if ".so" in path.name and any(
                        path.is_relative_to(root) for root in (repo / "build_Release/lib", repo / "ttnn/ttnn")
                    ):
                        loaded.add(path)
        assert any(path.name == "_ttnn.so" for path in loaded), "Owned worker has no TTNN binary mapping"
        assert len(launch_environments) == 1 and titled_workers, "Missing owned API or EngineCore evidence"
        provenance["launch_environment"] = json.loads(launch_environments.pop())
        provenance["titled_worker_initial_environments"] = titled_workers
        provenance["environment_verification_scope"] = (
            "Owned API process environment, inherited by spawned EngineCore; pinned launch source. "
            "EngineCore setproctitle overwrites /proc initial environment bytes, so missing keys there "
            "are recorded but cannot establish runtime getenv values. CPU reproduction: "
            "doc/optimized_vllm/proc_environment_reproducer.json."
        )
        provenance["loaded_binaries_sha256"] = {str(path.relative_to(repo)): sha(path) for path in sorted(loaded)}
        (evidence / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        lifecycle_command = [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_8_flash_next.tests.check_vllm_pending_prefill",
            "--server-log",
            str(evidence / "server.log"),
            "--output",
            str(evidence / "pending_prefill.json"),
        ]
        report["pending_prefill_command"] = lifecycle_command
        save()
        with (evidence / "pending_prefill.log").open("w") as output:
            result = subprocess.run(lifecycle_command, stdout=output, stderr=subprocess.STDOUT, timeout=1800)
        report["pending_prefill_exit"] = result.returncode
        assert result.returncode == 0, "Pending prefill correctness failed; source/server retained"
        command = [
            "bash",
            str(stage / "benchmark.sh"),
            str(evidence / "server.log"),
            str(evidence),
            "--stages",
            "benchmark,prefill_burst",
        ]
        if args.serial_reference is not None:
            command.extend(["--prefill-burst-serial-result", str(args.serial_reference.resolve())])
        report["benchmark_command"] = command
        save()
        with (evidence / "benchmark_driver.log").open("w") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=7500)
        report["benchmark_exit"] = result.returncode
        if result.returncode:
            raise RuntimeError("Benchmark failed; owned server and source retained for diagnosis")
        (evidence / "benchmark_run_summary.json").write_bytes((evidence / "vllm_run_summary.json").read_bytes())
        deadline = time.monotonic() + 120
        while True:
            idle = idle_metrics()
            if idle is not None:
                report["idle_before_shutdown"] = idle
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Requests did not drain; retained for diagnosis")
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
        if report["remaining_owned_pids"] or not report["devices_free"]:
            raise RuntimeError("Owned server cleanup incomplete; source retained")
        cleaned = True
        report["shutdown_audit"] = shutdown_audit(evidence / "server.log")
        assert all(
            report["shutdown_audit"][key] for key in ("no_teardown_error", "shutdown_recorded")
        ), "Unexpected teardown error or missing shutdown record"
        launch = json.loads((evidence / "vllm_run_summary.json").read_text())
        assert set(launch["stages"]) == {"serve", "benchmark", "prefill_burst"}
        assert launch["mesh_device"] == "P300x2"
        assert launch["max_num_seqs"] == contract["max_num_seqs"]
        assert launch["max_model_len"] == contract["context"]
        assert launch["block_size"] == contract["block_size"]
        assert launch["tt_config"] == dict(tt, sample_on_device_mode="all")
        launched_command = launch["server_command"]
        assert int(launched_command[launched_command.index("--max-num-batched-tokens") + 1]) == args.budget
        assert "--shutdown-timeout" not in launched_command
        assert "--async-scheduling" in launched_command and "--no-enable-prefix-caching" in launched_command
        provenance["actual_launch_summary"] = launch
        (evidence / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        report["shared_runtime_unchanged"] = runtime_identity(repo, stage) == provenance["shared_runtime"]
        report["loaded_binaries_unchanged"] = all(
            sha(repo / path) == value for path, value in provenance["loaded_binaries_sha256"].items()
        )
        assert report["shared_runtime_unchanged"] and report["loaded_binaries_unchanged"]
        report["measured_source_unchanged"] = all(
            sha(repo / path) == value for path, value in provenance["source_sha256"].items()
        )
        assert report["measured_source_unchanged"]
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if cleaned or server is None:
            conflicts = [path for path, value in measured_source.items() if Path(path).read_text() != value]
            report["restore_conflicts"] = conflicts
            if not conflicts:
                for path, value in saved.items():
                    Path(path).write_text(value)
            report["source_restored"] = all(Path(path).read_text() == value for path, value in saved.items())
            report["passed"] = report["passed"] and report["source_restored"]
        save()
    assert report["source_restored"], "Source changed during measurement; recovery backup retained"
    print(json.dumps(report))


if __name__ == "__main__":
    main()
