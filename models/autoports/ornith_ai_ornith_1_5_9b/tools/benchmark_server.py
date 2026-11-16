# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Select an owned, bounded Stage 10 server and retain observed configuration."""

import argparse
import ast
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[2]
ROOT = MODEL / "doc/benchmark"
STATE = ROOT / "owned_server.json"
REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def devices_free():
    for node in Path("/dev/tenstorrent").glob("[0-9]*"):
        result = subprocess.run(["fuser", str(node)], capture_output=True)
        if result.returncode != 1:
            raise RuntimeError(f"Device held or ownership check failed: {node}; {result.stdout!r}")


def idle(url):
    for observation in range(2):
        with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
            metrics = response.read().decode()
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            values = [
                match.group(1)
                for line in metrics.splitlines()
                if (match := re.fullmatch(re.escape(name) + r"(?:\{[^}]*\})?\s+(\S+)(?:\s+\d+)?", line))
            ]
            if not values or any(float(v) != 0 for v in values):
                raise RuntimeError(f"Server not idle: {name}={values}")
        if observation == 0:
            time.sleep(2)


def stop():
    if not STATE.exists():
        devices_free()
        return
    state = json.loads(STATE.read_text())
    pid = state["pid"]
    if process_start(pid) == state["process_start"]:
        idle(state["base_url"])
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and process_start(pid) == state["process_start"]:
            if Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z"):
                break
            time.sleep(1)
        else:
            if process_start(pid) == state["process_start"]:
                raise RuntimeError("Owned idle server did not exit within 120s; no next launch")
    log = Path(state["directory"]) / "runner.log"
    if "sending SIGKILL" in log.read_text(errors="replace"):
        raise RuntimeError("Server launcher required forced shutdown; inspect before next launch")
    devices_free()
    write(Path(state["directory"]) / "shutdown.json", {"idle_stop": True, "utc": time.time()})
    STATE.unlink()


def engine_process():
    # Restrict discovery to the owned timeout driver's descendants.
    state = json.loads(STATE.read_text())
    descendants = {state["pid"]}
    for _ in range(5):
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                status = (proc / "stat").read_text().split(") ", 1)[1].split()
                if int(status[1]) in descendants:
                    descendants.add(int(proc.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
    for pid in sorted(descendants):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except FileNotFoundError:
            continue
        if "vllm.entrypoints.openai.api_server" in argv:
            return pid, argv
    # vLLM may replace argv with its process title. Its owned launcher prints
    # the exact command before Popen; engine startup independently records args.
    import shlex

    lines = (Path(state["directory"]) / "runner.log").read_text().splitlines()
    command = next(shlex.split(line.strip()[5:]) for line in lines if line.strip().startswith("cmd: "))
    for pid in sorted(descendants):
        try:
            title = Path(f"/proc/{pid}/cmdline").read_bytes().decode()
        except FileNotFoundError:
            continue
        if "APIServer" in title or "api_server" in title:
            return pid, command
    raise RuntimeError("Could not prove owned API process launch command")


def select(capacity, url, output):
    began = time.monotonic()
    if STATE.exists():
        prior = json.loads(STATE.read_text())
        reuse = prior["capacity"] == capacity and prior["base_url"] == url
        reuse = reuse and process_start(prior["pid"]) == prior["process_start"]
        if reuse:
            snapshot = Path(prior["directory"]) / "timing.json"
            loaded = json.loads(snapshot.read_text())["metadata"] if snapshot.exists() else {}
            current = hashlib.sha256((MODEL / "tt/generator_vllm.py").read_bytes()).hexdigest()
            reuse = loaded.get("generator_module_sha256") == current
        if not reuse:
            stop()
    if not STATE.exists():
        devices_free()
        directory = ROOT / "servers" / f"b{capacity}-{time.time_ns()}"
        directory.mkdir(parents=True)
        baseline = json.loads(
            (MODEL / "readiness_vllm/stage10_guarded_after_b32_untracked/vllm_run_summary.json").read_text()
        )
        extras = baseline["server_command"][baseline["server_command"].index("--max-num-batched-tokens") :]
        # The launcher uses underscore aliases; the observed engine command is
        # retained verbatim and normalized only for the benchmark schema below.
        import shlex

        command = [
            "timeout",
            "21600",
            str(REPO / "python_env/bin/python"),
            "-u",
            "-m",
            "readiness_check.run_vllm_server",
            "--model-dir",
            str(MODEL),
            "--hf-model",
            baseline["hf_model"],
            "--mesh-device",
            "P300x2",
            "--max-model-len",
            "262144",
            "--block-size",
            "32",
            "--tt-config",
            json.dumps(baseline["tt_config"]),
            "--additional-server-args",
            shlex.join(extras),
            "--output-dir",
            str(directory),
            "--stages",
            "serve",
            "--port",
            url.rsplit(":", 1)[1],
            "--max-num-seqs",
            str(capacity),
        ]
        env = dict(os.environ)
        env.update(
            {
                "TT_MODEL_CLASS_OVERRIDES": "TTQwen3_5ForConditionalGeneration=models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm:OrnithForCausalLM",
                "VLLM_SERVER_DEV_MODE": "1",
                "TT_METAL_TRACE_ALLOC_TRACKING": "0",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
                "ORNITH_BENCHMARK_TIMING": str(directory / "timing.json"),
            }
        )
        for name in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_DEVICE_PROFILER_DISPATCH"):
            if env.get(name):
                raise RuntimeError(f"Refusing inherited {name}")
        write(
            directory / "launch.json",
            {
                "command": command,
                "environment": {k: v for k, v in env.items() if k.startswith(("TT_", "ORNITH_", "MESH_"))},
                "setup_start_unix": time.time(),
            },
        )
        with (directory / "runner.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write(
            STATE,
            {
                "pid": process.pid,
                "process_start": process_start(process.pid),
                "capacity": capacity,
                "base_url": url,
                "directory": str(directory),
            },
        )
        wait = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "scripts/wait_for"
        result = subprocess.run(
            [
                str(wait),
                "--pid",
                str(process.pid),
                "--file",
                str(directory / "runner.log"),
                "--pattern",
                "Server ready at",
                "--timeout",
                "1800",
                "--tail",
                "12",
            ]
        )
        if result.returncode or process.poll() is not None:
            raise RuntimeError(f"Bounded server startup failed; preserve {directory}")
    state = json.loads(STATE.read_text())
    directory = Path(state["directory"])
    idle(url)
    pid, argv = engine_process()
    log = (directory / "server.log").read_text(errors="replace")
    args_line = next(
        line.split("non-default args: ", 1)[1] for line in log.splitlines() if "non-default args: " in line
    )
    effective = ast.literal_eval(args_line)
    assert effective["max_num_seqs"] == capacity and effective["max_model_len"] == 262144
    assert effective["revision"] == effective["tokenizer_revision"] == REVISION
    assert "enable_prefix_caching=False" in log
    assert "models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm" in log
    metadata = json.loads((directory / "timing.json").read_text())["metadata"]
    assert metadata["layer_indices"] == list(range(32))
    assert metadata["layer_count"] == metadata["configured_layer_count"] == 32
    assert metadata["model_revision"] == REVISION
    assert metadata["precision"]["requested"]["id"] == "bfp8_all_lofi"
    assert metadata["max_num_seqs"] == capacity and metadata["max_model_len"] == 262144
    assert Path(metadata["generator_module_file"]).resolve() == MODEL / "tt/generator_vllm.py"
    # Source-level hyphen normalization is semantically identical to argparse's
    # underscore alias; preserve the actual process argv beside it.
    normalized = [arg.replace("_", "-") if arg.startswith("--") else arg for arg in argv]
    evidence = {
        "api_pid": pid,
        "process_argv": argv,
        "effective_args": effective,
        "engine_configuration": [line for line in log.splitlines() if "Initializing a V1" in line],
        "model_import": [line for line in log.splitlines() if "TT_MODEL_CLASS_OVERRIDES" in line],
        "server_log": str(directory / "server.log"),
        "timing_path": str(directory / "timing.json"),
        "launch_record": str(directory / "launch.json"),
        "loaded_model_metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.parent / f"configuration-b{capacity}.json"
    write(config_path, evidence)
    commits = {
        "tt-metal": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "vllm-tt-plugin": subprocess.check_output(
            ["git", "-C", "/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    import importlib.metadata

    commits["vllm"] = "db9527a46873454610df6dbedf79a36d6bf1a7f6 (upstream v0.31.0 tag; installed PyPI sdist)"
    timing_path = directory / "timing.json"
    identity = {
        "model": effective["model"],
        "implementation": str(MODEL.relative_to(REPO)),
        "generator_module": "models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm",
        "model_revision": effective["revision"],
        "tokenizer_revision": effective["tokenizer_revision"],
        "precision": metadata["precision"]["requested"]["id"],
        "layer_count": metadata["layer_count"],
        "configured_layer_count": metadata["configured_layer_count"],
        "source_commits": commits,
        "hardware": "4 Blackhole p300c chips; P300x2; TP4 mesh1x4; serial",
        "server_command": normalized,
        "max_num_seqs": capacity,
        "max_model_len": effective["max_model_len"],
        "prefix_caching": False,
        "base_url": url,
        "configuration_evidence": config_path.name,
        "configuration_evidence_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "timing_path": str(directory / "timing.json"),
        "timing_snapshot": str(timing_path),
        "vllm_distribution": importlib.metadata.version("vllm"),
        "owned_server_directory": str(directory),
        "selection_seconds": time.monotonic() - began,
    }
    write(output, identity)
    if capacity == 32:
        write(ROOT / "identity.json", identity)
    print(f"SERVER_READY b={capacity} pid={pid} configuration={config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        stop()
    else:
        select(args.max_num_seqs, args.base_url, args.output)
