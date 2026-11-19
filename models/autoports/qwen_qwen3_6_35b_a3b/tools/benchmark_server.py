# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Select an owned Stage 10 server and retain its observed configuration."""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import psutil

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[2]
ROOT = MODEL / "doc/benchmark"
SUPPORT = Path(os.environ.get("QWEN_BENCHMARK_SUPPORT_DIR", str(ROOT))).resolve()
STATE = SUPPORT / "server_state.json"
sys.path.insert(0, str(MODEL / "doc/vllm_integration"))
from run_final64_perf import server_identity


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop():
    if not STATE.exists():
        return
    state = json.loads(STATE.read_text())
    if state.get("stopped"):
        return
    pid = state.get("runner_pid")
    if pid and psutil.pid_exists(pid):
        if psutil.Process(pid).create_time() != state["runner_created"]:
            raise RuntimeError("Recorded runner PID was reused; refusing cleanup")
        subprocess.run(
            [
                str(REPO / "python_env/bin/python"),
                str(MODEL / "doc/optimized_vllm/stop_idle_server.py"),
                "--runner-pid",
                str(pid),
                "--server-log",
                state["server_log"],
            ],
            check=True,
            timeout=360,
        )
    holders = subprocess.run(["fuser", *map(str, Path("/dev/tenstorrent").glob("*"))], capture_output=True, text=True)
    if holders.returncode != 1:
        raise RuntimeError("Devices still held: " + holders.stdout)
    state["stopped"] = True
    save(STATE, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        stop()
        return
    if args.max_num_seqs not in (1, 8, 16, 32) or not args.output:
        parser.error("A supported capacity and output are required")
    started = time.monotonic()
    old = json.loads(STATE.read_text()) if STATE.exists() else {}
    reuse = (
        not old.get("stopped", True)
        and old.get("capacity") == args.max_num_seqs
        and psutil.pid_exists(old.get("runner_pid", -1))
    )
    if not reuse:
        stop()
        holders = subprocess.run(
            ["fuser", *map(str, Path("/dev/tenstorrent").glob("*"))], capture_output=True, text=True
        )
        if holders.returncode != 1:
            raise RuntimeError("Cannot launch over device holders: " + holders.stdout)
        directory = SUPPORT / "servers" / f"b{args.max_num_seqs}-{time.time_ns()}"
        directory.mkdir(parents=True)
        log = directory / "server.log"
        env = dict(
            os.environ,
            QWEN_VLLM_AUDIT="0",
            QWEN_VLLM_REDUCED="0",
            QWEN_VLLM_ASYNC="1",
            TT_METAL_TRACE_ALLOC_TRACKING="0",
            TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
            QWEN_BENCHMARK_PHASE_DIR=str(directory / "phases"),
            PYTEST_ADDOPTS="-p vllm",
        )
        for key in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_ENABLE_PROFILER", "TT_METAL_WATCHER"):
            env.pop(key, None)
        command = [
            "bash",
            str(MODEL / "tests/run_vllm_server.sh"),
            str(directory),
            str(args.max_num_seqs),
            "2048",
            "--stages",
            "serve",
        ]
        with (directory / "launcher.log").open("w") as stream:
            process = subprocess.Popen(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
        old = dict(
            stopped=False,
            capacity=args.max_num_seqs,
            launcher_pid=process.pid,
            command=command,
            server_log=str(log),
            phase_dir=str(directory / "phases"),
            launch_monotonic=started,
        )
        save(STATE, old)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server launcher exited {process.returncode}; see {directory}")
            children = psutil.Process(process.pid).children(recursive=True)
            for child in children:
                if "readiness_check.run_vllm_server" in child.cmdline():
                    old.update(runner_pid=child.pid, runner_created=child.create_time())
                    save(STATE, old)
            if log.exists() and "Application startup complete" in log.read_text(errors="replace"):
                with urlopen(args.base_url + "/health", timeout=5) as response:
                    if response.status == 200:
                        break
            time.sleep(2)
        else:
            raise RuntimeError("Startup deadline expired; bounded launcher retained for safe recovery")
    log = Path(old["server_log"])
    observed = server_identity(SimpleNamespace(server_log=log))
    initial = observed["initialization"]
    engine = psutil.Process(observed["engine_pid"])
    api = engine.parent()
    argv = api.cmdline()
    if "vllm.entrypoints.openai.api_server" not in argv:
        raise RuntimeError("Unexpected API process ancestry")
    text = log.read_text(errors="replace")
    lines = [
        line
        for line in text.splitlines()
        if any(
            s in line
            for s in (
                "non-default args:",
                "Initializing a V1",
                "QWEN_VLLM_EVENT",
                "QWEN_BENCHMARK_READY",
                "Applied TT_MODEL_CLASS_OVERRIDES",
            )
        )
    ]
    lines = [line for line in lines if '"event": "initialize"' in line or "QWEN_VLLM_EVENT" not in line]
    settings = ast.literal_eval(
        next(line.split("non-default args: ", 1)[1] for line in lines if "non-default args:" in line)
    )
    assert settings["max_num_seqs"] == args.max_num_seqs == initial["max_num_seqs"]
    assert settings["max_model_len"] == initial["max_model_len"] == 262144
    assert initial["layers"] == list(range(40)) and not initial["reduced"]
    assert initial["device_ids"] == [0, 1, 2, 3]
    assert initial["module_file"] == str(MODEL / "tt/generator_vllm.py")
    assert initial["precision"] == json.loads((MODEL / "doc/datatype_sweep/selected_precision_config.json").read_text())
    assert "enable_prefix_caching=False" in text
    assert initial["adapter"] == "models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm:QwenForCausalLM"
    with urlopen(args.base_url + "/v1/models", timeout=5) as response:
        api_models = json.load(response)
    assert api_models["data"][0]["id"] == "Qwen/Qwen3.6-35B-A3B"
    assert api_models["data"][0]["max_model_len"] == 262144
    output = args.output.resolve()
    evidence = output.parent / f"configuration-b{args.max_num_seqs}.json"
    save(
        evidence,
        dict(
            api_pid=api.pid,
            api_created=api.create_time(),
            api_command=argv,
            observed=observed,
            startup_lines=lines,
            api_models=api_models,
            phase_dir=old["phase_dir"],
            reused=reuse,
            select_seconds=time.monotonic() - started,
        ),
    )
    commits = json.loads((SUPPORT / "setup/source_commits.json").read_text())
    result = dict(
        model="Qwen/Qwen3.6-35B-A3B",
        implementation=str(MODEL.relative_to(REPO)),
        generator_module=initial["adapter"].split(":")[0],
        model_revision=settings["revision"],
        tokenizer_revision=settings["tokenizer_revision"],
        precision=initial["precision"],
        layer_count=40,
        configured_layer_count=40,
        source_commits=commits,
        server_lifetime_guard_seconds=int(os.environ.get("QWEN_VLLM_SERVER_LIFETIME_SECONDS", "28800")),
        hardware="P300x2: 4 Blackhole p300c chips, TP4/EP4, 1x4 mesh, FABRIC_1D; serial exclusive regime",
        server_command=argv,
        prefix_caching=False,
        max_num_seqs=args.max_num_seqs,
        max_model_len=262144,
        base_url=args.base_url,
        configuration_evidence=evidence.name,
        configuration_evidence_sha256=sha(evidence),
        phase_dir=old["phase_dir"],
        server_log=str(log),
        engine_pid=engine.pid,
        worker_grid=initial["worker_grid"],
        module_file=initial["module_file"],
        module_sha256=sha(Path(initial["module_file"])),
    )
    save(output, result)
    if args.max_num_seqs == 32:
        save(ROOT / "identity.json", result)
    print(f"READY b={args.max_num_seqs} reused={reuse} identity={output}")


if __name__ == "__main__":
    main()
