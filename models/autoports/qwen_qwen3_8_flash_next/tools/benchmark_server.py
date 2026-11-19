#!/usr/bin/env python3
"""Owned Stage-11 server profiles; importing this module never touches devices.

The supervisor uses the Stage-10 readiness runner's lifecycle, without modifying
the plugin. Only its own idle server may be stopped. Setup evidence survives the
client subprocess and every profile keeps its own effective-config snapshot.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO = MODEL_DIR.parents[2]
LANE = REPO.parent
BENCH = MODEL_DIR / "doc/benchmark"
STATE = BENCH / "setup/owned-server.json"
PYTHON = LANE / "vllm-env/bin/python"
HF = LANE / "hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa"
MODULE = "models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm"
TT_CONFIG = {
    "sample_on_device_mode": "all",
    "trace_region_size": 0,
    "fabric_config": "FABRIC_1D_RING",
    "trace_mode": "all",
    "prefill_chunk_alignment": 8,
    "decode_interleave_prefill_steps": 1,
    "decode_interleave_decode_steps": 2,
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def process(pid):
    try:
        root = Path(f"/proc/{pid}")
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": int(pid),
            "ppid": int(fields[1]),
            "start_ticks": int(fields[19]),
            "state": fields[0],
            "command": (root / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
        }
    except (OSError, ValueError, IndexError):
        return None


def alive(record):
    current = process(record["pid"])
    return bool(current and current["start_ticks"] == record["start_ticks"] and current["state"] != "Z")


def descendants(root):
    records = [p for path in Path("/proc").glob("[0-9]*") if (p := process(int(path.name)))]
    ids = {root}
    while True:
        expanded = ids | {p["pid"] for p in records if p["ppid"] in ids}
        if expanded == ids:
            return [p for p in records if p["pid"] in ids]
        ids = expanded


def get_text(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


def health(base):
    try:
        get_text(base + "/health")
        return True
    except Exception:
        return False


def idle(base):
    metrics = get_text(base + "/metrics")
    values = {}
    for key in ("num_requests_running", "num_requests_waiting"):
        rows = re.findall(r"^vllm:" + key + r"(?:\{[^\n]*\})?\s+([\d.eE+-]+)$", metrics, re.M)
        if not rows:
            raise RuntimeError(f"Cannot prove idle: missing {key} metric")
        values[key] = sum(map(float, rows))
    if any(values.values()):
        raise RuntimeError(f"Refusing shutdown with outstanding requests: {values}")
    return values


def device_holders():
    result = subprocess.run(
        ["fuser", *[f"/dev/tenstorrent/{n}" for n in range(4)]], capture_output=True, text=True, timeout=10
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Cannot inspect device ownership: {result.stderr}")
    return sorted({int(p) for p in result.stdout.split() if p.isdigit()})


def stop_owned(state):
    supervisor = state.get("supervisor")
    if not supervisor or not alive(supervisor):
        remaining = [p for p in state.get("owned_processes", []) if alive(p)]
        outer = state.get("outer")
        if outer and alive(outer):
            remaining.append(outer)
        if remaining:
            raise RuntimeError(f"Supervisor absent but owned descendants remain: {remaining}")
        return
    observed_idle = idle(state["base_url"])
    owned = descendants(supervisor["pid"])
    state["owned_processes"] = owned
    write_json(STATE, state)
    # Existing readiness cleanup sends TERM to the API, waits 15s, then KILL.
    # The Stage-10 owner accepted interrupted trace release; retain that fact.
    os.kill(supervisor["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and any(alive(p) for p in owned):
        time.sleep(0.5)
    remaining = [p for p in owned if alive(p)]
    holders = device_holders()
    evidence = {
        "idle_before_shutdown": observed_idle,
        "owned_processes": owned,
        "remaining_owned_processes": remaining,
        "device_holders": holders,
        "passed": not remaining and not holders,
        "cleanup_policy": "readiness_check._shutdown; owner-accepted incomplete orderly trace release",
        "owner_decision": str(MODEL_DIR / "doc/optimized_vllm/owner_decision_shutdown_2026_11_17.json"),
    }
    write_json(Path(state["setup_dir"]) / "cleanup.json", evidence)
    if remaining or holders:
        raise RuntimeError("Owned cleanup incomplete; preserve evidence and investigate before another device job")
    state["stopped"] = True
    write_json(STATE, state)


def server_command(slots, port):
    return [
        str(PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(HF),
        "--served-model-name",
        "Qwen/Qwen3.8-Flash-Next",
        "--block-size",
        "32",
        "--max-num-seqs",
        str(slots),
        "--port",
        str(port),
        "--max-model-len",
        "262144",
        "--additional-config",
        json.dumps({"tt": TT_CONFIG}),
        "--max-num-batched-tokens",
        "9",
        "--no-enable-prefix-caching",
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--async-scheduling",
    ]


def supervise(setup_dir):
    from readiness_check.run_vllm_server import _hold_until_signal, _shutdown, _wait_for_server

    setup = Path(setup_dir)
    spec = json.loads((setup / "launch.json").read_text())
    env = dict(os.environ, MESH_DEVICE="P300x2", HF_MODEL=str(HF), VLLM_RPC_TIMEOUT="300000")
    with (setup / "server.log").open("wb") as log:
        proc = subprocess.Popen(spec["server_command"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    owner = {"supervisor": process(os.getpid()), "api": process(proc.pid)}
    write_json(setup / "processes.json", owner)
    try:
        _wait_for_server(port=spec["port"], proc=proc, log_file=setup / "server.log", timeout_seconds=1800)
        write_json(setup / "ready.json", {"ready_unix": time.time(), **owner})
        _hold_until_signal(proc, setup / "server.log")
    finally:
        _shutdown(proc, setup / "server.log")


def parse_configuration(log, slots):
    args_rows = [line.split("non-default args: ", 1)[1] for line in log.splitlines() if "non-default args: " in line]
    construction_rows = [
        line.split("QWEN_VLLM_CONSTRUCTION ", 1)[1] for line in log.splitlines() if "QWEN_VLLM_CONSTRUCTION " in line
    ]
    if not args_rows or not construction_rows:
        raise RuntimeError("Missing effective vLLM configuration or autoport construction evidence")
    args = ast.literal_eval(args_rows[-1])
    construction = json.loads(construction_rows[-1])
    expected = {
        "max_num_seqs": slots,
        "max_model_len": 262144,
        "block_size": 32,
        "enable_prefix_caching": False,
        "max_num_batched_tokens": 9,
        "async_scheduling": True,
        "model": str(HF),
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise RuntimeError(f"Observed config differs: {key}={args.get(key)!r}, expected {value!r}")
    if args.get("additional_config", {}).get("tt") != TT_CONFIG:
        raise RuntimeError("Observed TT config differs from accepted Stage 10")
    if (
        construction.get("layers") != list(range(48))
        or construction.get("max_seq_len") != 262144
        or construction.get("max_batch_size") != slots
        or construction.get("adapter") != MODULE + ":QwenFlashNextForCausalLM"
    ):
        raise RuntimeError("Observed autoport identity/layers/capacity differs from Stage 10")
    return args, construction


def snapshot(state, output, baseline):
    setup = Path(state["setup_dir"])
    log = (setup / "server.log").read_text(errors="replace")
    args, construction = parse_configuration(log, state["max_num_seqs"])
    api = state["api"]
    current = process(api["pid"])
    if not alive(api) or current["command"] != state["server_command"]:
        raise RuntimeError("API process command/start identity changed")
    if baseline["layer_count"] != 48 or baseline["configured_layer_count"] != 48:
        raise RuntimeError("Baseline must identify all 48 configured layers")
    observed_precision = construction["precision"]
    if baseline["precision"] not in (observed_precision, observed_precision["config_id"]):
        raise RuntimeError("Baseline precision differs from constructed precision")
    evidence = output.parent / f"configuration-b{state['max_num_seqs']}.json"
    selected_lines = [
        line
        for line in log.splitlines()
        if any(
            token in line
            for token in (
                "non-default args:",
                "QWEN_VLLM_CONSTRUCTION ",
                "QWEN_BENCHMARK",
                "Applied TT_MODEL_CLASS_OVERRIDES:",
                "Initializing a V1 LLM engine",
                "Automatic prefix caching",
                "Application startup complete",
            )
        )
    ]
    write_json(
        evidence,
        {
            "observed_process": current,
            "observed_args": args,
            "construction": construction,
            "server_log": str(setup / "server.log"),
            "startup_lines": selected_lines,
            "phase_log": state["phase_log"],
            "owner": state,
        },
    )
    result = dict(
        baseline,
        base_url=state["base_url"],
        max_num_seqs=args["max_num_seqs"],
        max_model_len=args["max_model_len"],
        prefix_caching=args["enable_prefix_caching"],
        server_command=current["command"],
        configuration_evidence=evidence.name,
        configuration_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        phase_log=state["phase_log"],
        server_log=str(setup / "server.log"),
        owner_state=str(STATE),
        server_pid=api["pid"],
        setup_dir=str(setup),
        generator_file=str(MODEL_DIR / "tt/generator_vllm.py"),
    )
    write_json(output, result)


def launch(slots, base_url, port):
    holders = device_holders()
    if holders:
        raise RuntimeError(f"Devices unavailable; will not stop processes absent from ownership state: {holders}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
    setup = BENCH / "setup" / f"b{slots}-{time.time_ns()}"
    setup.mkdir(parents=True)
    phase = Path(os.environ.get("QWEN_BENCHMARK_PHASE_LOG", str(setup / "phases.jsonl"))).resolve()
    spec = {
        "server_command": server_command(slots, port),
        "port": port,
        "max_num_seqs": slots,
        "base_url": base_url,
        "setup_dir": str(setup),
        "phase_log": str(phase),
        "started_unix": time.time(),
    }
    write_json(setup / "launch.json", spec)
    # The session is detached from benchmark_stage's short-lived control hook.
    shell = "\n".join(
        [
            "set -euo pipefail",
            f"source {shlex.quote(str(LANE / 'vllm-env/bin/activate'))}",
            f"source {shlex.quote(str(LANE / 'lane-env.sh'))}",
            f"export VLLM_TT_PLUGIN_ROOT={shlex.quote(str(LANE / 'vllm-tt-plugin'))}",
            f"export TT_MODEL_CLASS_OVERRIDES={shlex.quote('TTQwen4ExpForConditionalGeneration=' + MODULE + ':QwenFlashNextForCausalLM')}",
            "export TT_VISIBLE_DEVICES=0,1,2,3",
            "export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto",
            "export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0",
            "export VLLM_SERVER_DEV_MODE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1",
            f"export QWEN_BENCHMARK_PHASE_LOG={shlex.quote(str(phase))}",
            "unset QWEN_VLLM_LAYER_INDICES TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER",
            "exec timeout --kill-after=60 86400 "
            + shlex.join([str(PYTHON), str(Path(__file__).resolve()), "--supervise", str(setup)]),
        ]
    )
    (setup / "launch.sh").write_text(shell + "\n")
    with (setup / "supervisor.log").open("wb") as log:
        outer = subprocess.Popen(
            ["bash", str(setup / "launch.sh")],
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    spec["outer"] = process(outer.pid)
    write_json(STATE, spec)
    deadline = time.monotonic() + 1860
    while time.monotonic() < deadline:
        if (setup / "processes.json").exists():
            spec.update(json.loads((setup / "processes.json").read_text()))
            spec["owned_processes"] = descendants(spec["supervisor"]["pid"])
            write_json(STATE, spec)
        if (setup / "ready.json").exists() and health(base_url):
            spec["setup_seconds"] = time.time() - spec["started_unix"]
            write_json(STATE, spec)
            return spec
        if outer.poll() is not None:
            raise RuntimeError(f"Server setup exited {outer.returncode}; preserve {setup}")
        time.sleep(2)
    raise RuntimeError(f"Server setup exceeded bound; retain {setup} and owned process evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-num-seqs", type=int, choices=(1, 2))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--supervise", help=argparse.SUPPRESS)
    opts = parser.parse_args()
    if opts.supervise:
        supervise(opts.supervise)
        return
    parsed = urlparse(opts.base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1") or parsed.path:
        parser.error("base URL must be a local http endpoint without path")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent / "server-control.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text()) if STATE.exists() else None
        if opts.stop:
            if state:
                stop_owned(state)
            print("Owned benchmark server cleanup complete")
            return
        if opts.max_num_seqs is None or opts.output is None:
            parser.error("--max-num-seqs and --output are required")
        baseline = json.loads((BENCH / "identity.json").read_text())
        if state and state.get("supervisor") and alive(state["supervisor"]):
            if state["max_num_seqs"] != opts.max_num_seqs or state["base_url"] != opts.base_url:
                stop_owned(state)
                state = None
            elif not health(opts.base_url):
                raise RuntimeError("Owned server is not healthy; preserve evidence and diagnose before shutdown")
        else:
            if state:
                stop_owned(state)
            state = None
        if state is None:
            state = launch(opts.max_num_seqs, opts.base_url, parsed.port or 80)
        snapshot(state, opts.output.resolve(), baseline)
        print(f"Ready B{opts.max_num_seqs}: {opts.output}")


if __name__ == "__main__":
    main()
