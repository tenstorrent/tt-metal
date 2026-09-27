# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Own only this benchmark's server; validate live configuration before returning."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from benchmark_control import query
from benchmark_server_config import MODEL, MODEL_DIR, REVISION, profile_plan, validate_observed_config

REPO = MODEL_DIR.parents[2]
BENCH = MODEL_DIR / "doc/benchmark"
STATE = BENCH / "owned_server.json"
CONTROL_ROOT = REPO / "bringup/benchmark-control/13a05473"


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def process_start(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    return text[text.rfind(")") + 2 :].split()[19]


def process_alive(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return text[text.rfind(")") + 2 :].split()[0] != "Z"
    except FileNotFoundError:
        return False


def stop_owned():
    if not STATE.exists():
        return
    state = json.loads(STATE.read_text())
    pid = state["pid"]
    if process_alive(pid):
        if process_start(pid) != state["process_start"] or os.getpgid(pid) != pid:
            raise RuntimeError("Owned server PID identity changed; refusing to signal")
        actual = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if b"vllm.entrypoints.openai.api_server" not in actual:
            raise RuntimeError("Owned PID is not the recorded serving process")
        os.killpg(pid, signal.SIGINT)
        end = time.monotonic() + 45
        while process_alive(pid) and time.monotonic() < end:
            time.sleep(0.25)
        # All children inherit the new process group created by this hook.
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    save(
        BENCH / f"stopped-b{state['slots']}-{pid}.json",
        {**state, "stopped_at": time.time(), "leader_alive": process_alive(pid)},
    )
    STATE.unlink()


def get_json(url):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-num-seqs", type=int, choices=(1, 32))
    parser.add_argument("--base-url")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    if args.stop:
        stop_owned()
        return
    if not args.output or not args.base_url or not args.max_num_seqs:
        parser.error("profile, base URL and output are required")
    parsed = urlparse(args.base_url)
    if parsed.hostname not in ("127.0.0.1", "localhost") or parsed.scheme != "http":
        raise ValueError("Only local owned serving is supported")
    # Reuse only a process with durable identity owned by this stage.
    state = json.loads(STATE.read_text()) if STATE.exists() else None
    reuse = state and state["slots"] == args.max_num_seqs and process_alive(state["pid"])
    if reuse and process_start(state["pid"]) != state["process_start"]:
        raise RuntimeError("Owned server PID was reused")
    if not reuse:
        stop_owned()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", parsed.port or 80))
        CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
        control = CONTROL_ROOT / f"b{args.max_num_seqs}.sock"
        if control.exists():
            control.unlink()
        plan = profile_plan(args.max_num_seqs)
        command = plan["command"]
        command[command.index("--port") + 1] = str(parsed.port or 80)
        log = args.output.parent / f"server-b{args.max_num_seqs}.log"
        environment = {**os.environ, **plan["environment_overrides"]}
        for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "GEMMA4_AUTOPORT_PROBE_LAYERS"):
            environment.pop(name, None)
        environment.update(
            {
                "GEMMA4_BENCHMARK_CONTROL": str(control),
                "TT_METAL_LOGS_PATH": str(args.output.parent / f"runtime-b{args.max_num_seqs}"),
                "VLLM_CACHE_ROOT": str(BENCH / "vllm_cache"),
            }
        )
        started = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True
            )
        state = {
            "pid": process.pid,
            "process_start": process_start(process.pid),
            "slots": args.max_num_seqs,
            "control": str(control),
            "command": command,
            "server_log": str(log),
            "launched_wall_time": time.time(),
            "environment": {
                key: environment.get(key)
                for key in (
                    "PYTHONPATH",
                    "MESH_DEVICE",
                    "OMP_NUM_THREADS",
                    "GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING",
                    "GEMMA4_BENCHMARK_CONTROL",
                    "TT_METAL_DEVICE_PROFILER",
                    "TT_METAL_WATCHER",
                    "TT_METAL_LOGS_PATH",
                    "VLLM_CACHE_ROOT",
                    "VLLM_SERVER_DEV_MODE",
                )
            },
        }
        save(STATE, state)
        ready = False
        try:
            while time.monotonic() - started < 900:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {log}")
                try:
                    with urlopen(args.base_url + "/health", timeout=2) as response:
                        ready = response.status == 200 and control.exists()
                    if ready:
                        break
                except OSError:
                    pass
                time.sleep(2)
            if not ready:
                raise TimeoutError("Server readiness exceeded 900 seconds")
            state["startup_seconds"] = time.monotonic() - started
            save(STATE, state)
        except BaseException:
            stop_owned()
            raise
    info = get_json(args.base_url + "/server_info?config_format=json")
    observed = validate_observed_config(info, args.max_num_seqs)
    snapshot = query(state["control"])
    actual = snapshot["identity"]
    if actual["layer_count"] != 30 or actual["layer_indices"] != list(range(30)) or actual["mesh"] != [1, 4]:
        raise ValueError("Observer does not attest full30-layer1x4 implementation")
    if actual["max_num_seqs"] != args.max_num_seqs or actual["max_model_len"] != 262144:
        raise ValueError("Loaded generator configuration differs from engine configuration")
    if (
        actual["precision"]["config_id"] != "head4_inner_all4_shared_down4"
        or not actual["precision"]["construction_verified"]
    ):
        raise ValueError("Loaded precision differs from selected construction policy")
    if Path(actual["generator_file"]) != MODEL_DIR / "tt/generator_vllm.py":
        raise ValueError("Loaded generator is outside this checkout")
    evidence = args.output.parent / f"configuration-b{args.max_num_seqs}.json"
    save(
        evidence,
        {"observed_configuration": observed, "raw_server_info": info, "phase_observer": actual, "launch": state},
    )
    # Inspect the wheel using the same interpreter as the actual launch; the
    # nested vLLM checkout is not the imported core and cannot supply its SHA.
    core_probe = """import ast, hashlib, importlib.metadata, importlib.util, json
from pathlib import Path
root = Path(importlib.util.find_spec('vllm').origin).parent
version_tree = ast.parse((root / '_version.py').read_text())
commit = next((ast.literal_eval(node.value) for node in version_tree.body
 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'commit_id'
 for target in node.targets)), None)
print(json.dumps({'version': importlib.metadata.version('vllm'), 'module_path': str(root),
 'commit_id': commit,
 'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
 for name in ('entrypoints/openai/api_server.py', 'benchmarks/serve.py')}}))
"""
    core = json.loads(subprocess.check_output([state["command"][0], "-c", core_probe], text=True))
    identity = {
        "installed_core": core,
        "model": MODEL,
        "implementation": str(MODEL_DIR.relative_to(REPO)),
        "generator_module": actual["generator_module"],
        "generator_file": actual["generator_file"],
        "model_revision": REVISION,
        "tokenizer_revision": observed["tokenizer_revision"],
        "precision": "head4_inner_all4_shared_down4",
        "layer_count": 30,
        "configured_layer_count": 30,
        "source_commits": {
            "tt-metal": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "tt-plugin-checkout": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO / "vllm/plugins/vllm-tt-plugin", text=True
            ).strip(),
            "vllm-core": core["commit_id"],
        },
        "hardware": "P300x2; four Blackhole ASICs; TP4/DP1; mesh1x4",
        "server_command": state["command"],
        "prefix_caching": False,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": 262144,
        "base_url": args.base_url,
        "configuration_evidence": evidence.name,
        "configuration_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "phase_control": state["control"],
        "startup_seconds": state["startup_seconds"],
    }
    save(args.output, identity)
    if args.max_num_seqs == 32:
        save(BENCH / "identity.json", identity)
    print(json.dumps({"ready": True, "slots": args.max_num_seqs, "pid": state["pid"]}))


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise InterruptedError(f"server hook interrupted by signal{signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException:
        stop_owned()
        raise
