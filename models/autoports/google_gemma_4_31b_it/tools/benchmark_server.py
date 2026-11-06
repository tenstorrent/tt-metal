#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Select a bounded, owned Stage 10 server and retain observed configuration."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL.parents[2]
STATE = MODEL / "doc/benchmark/owned_server.json"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
MODULE = "models.autoports.google_gemma_4_31b_it.tt.generator_vllm"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    # Includes the model-owned timing collector benchmark_phases.py.
    return {str(p.relative_to(ROOT)): sha(p) for p in (MODEL / "tt").glob("*.py")}


def clean_stale_phase_socket(phase_log):
    digest = hashlib.sha256(os.path.abspath(phase_log).encode()).hexdigest()[:16]
    path = Path(f"/tmp/gemma4-phase-{digest}.sock")
    if not path.exists():
        return
    metadata = path.lstat()
    if metadata.st_uid != os.getuid() or not stat.S_ISSOCK(metadata.st_mode):
        raise RuntimeError(f"Refusing to remove unowned/non-socket path {path}")
    if device_owners():
        raise RuntimeError("Cannot remove a phase socket while a device process is alive")
    # A live listener can exist before its engine opens hardware. Probe it,
    # and remove only a positively refused connection, never a timed-out one.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        try:
            probe.connect(str(path))
        except ConnectionRefusedError:
            if path.lstat().st_ino != metadata.st_ino:
                raise RuntimeError("Phase socket changed during stale-owner check")
            path.unlink()
        else:
            raise RuntimeError(f"Phase socket still has a listener: {path}")


def process(pid):
    try:
        directory = Path(f"/proc/{pid}")
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": int(pid),
            "start_ticks": fields[19],
            "ppid": int(fields[1]),
            "command": (directory / "cmdline").read_bytes().decode().strip("\0").split("\0"),
        }
    except (OSError, ValueError):
        return None


def processes():
    return [p for d in Path("/proc").glob("[0-9]*") if (p := process(d.name))]


def descendants(pid):
    entries = processes()
    ids = {pid}
    while True:
        more = {p["pid"] for p in entries if p["ppid"] in ids}
        if more <= ids:
            return [p for p in entries if p["pid"] in ids]
        ids |= more


def device_owners():
    owners = []
    for p in processes():
        for fd in Path(f"/proc/{p['pid']}/fd").glob("*"):
            try:
                if os.readlink(fd).startswith("/dev/tenstorrent/"):
                    owners.append(p)
                    break
            except OSError:
                pass
    return owners


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


def idle(base_url):
    metrics = fetch(base_url.rstrip("/") + "/metrics")
    values = []
    for line in metrics.splitlines():
        if re.match(r"vllm:num_requests_(running|waiting)(\{| )", line):
            values.append(float(line.split()[-1]))
    return len(values) >= 2 and all(value == 0 for value in values)


def stop_owned(state):
    current = process(state["pid"])
    if current is None:
        if device_owners():
            raise RuntimeError("Recorded launcher exited but device owners remain; preserve evidence for recovery")
        return
    if current["start_ticks"] != state["start_ticks"]:
        raise RuntimeError("PID reused; refusing to stop it")
    owned = {p["pid"] for p in descendants(current["pid"])}
    foreign = [p for p in device_owners() if p["pid"] not in owned]
    if foreign:
        raise RuntimeError(f"Unowned device processes: {foreign}")
    for _ in range(3):
        if not idle(state["base_url"]):
            raise RuntimeError("Server has outstanding requests; refusing shutdown")
        time.sleep(1)
    # Signal only the readiness driver. Its hold-open handler shuts down its
    # idle API process and waits for cleanup; never signal active device work.
    drivers = [p for p in descendants(current["pid"]) if "readiness_check.run_vllm_server" in p["command"]]
    drivers = [p for p in drivers if "-m" in p["command"] and Path(p["command"][0]).name != "timeout"]
    if len(drivers) != 1:
        raise RuntimeError(f"Cannot identify the unique owned readiness driver: {drivers}")
    os.kill(drivers[0]["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if not device_owners() and process(current["pid"]) is None:
            return
        time.sleep(1)
    raise RuntimeError("Owned server did not release devices; no forced termination attempted")


def events(log):
    found = {}
    for line in log.splitlines():
        pos = line.find('{"event":')
        if pos >= 0:
            try:
                event, _ = json.JSONDecoder().raw_decode(line[pos:])
                found[event["event"]] = event
            except (ValueError, KeyError):
                pass
    return found


def argument(command, flag):
    aliases = (flag, flag.replace("-", "_").replace("__", "--", 1))
    for alias in aliases:
        if alias in command:
            return command[command.index(alias) + 1]
    raise RuntimeError(f"Actual server command missing {flag}")


def observe(state, output, requested):
    api = [p for p in descendants(state["pid"]) if "vllm.entrypoints.openai.api_server" in p["command"]]
    if len(api) != 1:
        raise RuntimeError(f"Expected one owned API server, observed {api}")
    command = api[0]["command"]
    log_path = Path(state["server_dir"]) / "server.log"
    log = log_path.read_text(errors="replace")
    observed = events(log)
    registration = observed["autoport_registration"]
    pool = observed["serving_pool_budget"]
    assert registration["class"] == MODULE + ":Gemma4ForCausalLM", registration
    assert registration["layers"] == list(range(60)), registration
    assert registration["precision"]["config_id"] == "head_bfp4", registration
    assert int(argument(command, "--max-num-seqs")) == pool["max_num_seqs"] == requested
    assert int(argument(command, "--max-model-len")) == pool["max_model_len"] == 262144
    assert argument(command, "--revision") == argument(command, "--tokenizer-revision") == REVISION
    assert "--enable-prefix-caching" in command and "--async-scheduling" in command
    assert json.loads(argument(command, "--override-generation-config")) == {"top_k": 32}
    models = json.loads(fetch(state["base_url"] + "/v1/models"))
    assert any(m["id"] == "google/gemma-4-31B-it" and m.get("max_model_len") == 262144 for m in models["data"]), models
    snapshot = output.parent / f"configuration-b{requested}.json"
    current_hashes = source_hashes()
    if state.get("source_sha256_at_launch") != current_hashes:
        raise RuntimeError("Runtime source changed since server launch or launch hashes are absent; relaunch required")
    plugin = Path("/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09")

    def commit(path):
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

    commits = {"tt-metal": commit(ROOT), "vllm-tt-plugin": commit(plugin)}
    if state.get("tt_metal_commit_at_launch") != commits["tt-metal"]:
        raise RuntimeError("tt-metal HEAD changed since launch; preserve launch provenance and relaunch")
    vllm_package = json.loads(
        subprocess.check_output(
            [
                str(ROOT / "python_env/bin/python"),
                "-c",
                "import importlib.metadata as m,json; d=m.distribution('vllm'); "
                "print(json.dumps({'version':d.version,'direct_url':d.read_text('direct_url.json')}))",
            ],
            text=True,
        )
    )
    direct_url = json.loads(vllm_package["direct_url"]) if vllm_package["direct_url"] else {}
    commits["vllm"] = direct_url.get("vcs_info", {}).get("commit_id")
    write(
        snapshot,
        {
            "launcher": state,
            "api_process": api[0],
            "models_response": models,
            "registration": registration,
            "pool": pool,
            "source_commits": commits,
            "source_sha256": current_hashes,
            "startup_log": log,
            "device_owners": device_owners(),
            "vllm_package": vllm_package,
        },
    )
    identity = {
        "model": "google/gemma-4-31B-it",
        "implementation": str(MODEL.relative_to(ROOT)),
        "generator_module": MODULE,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "precision": registration["precision"],
        "layer_count": 60,
        "configured_layer_count": 60,
        "source_commits": commits,
        "hardware": "P300x2; Blackhole mesh 1x4; TP4",
        "server_command": command,
        "prefix_caching": True,
        "max_num_seqs": pool["max_num_seqs"],
        "max_model_len": pool["max_model_len"],
        "base_url": state["base_url"],
        "phase_log": state["phase_log"],
        "server_log": str(log_path),
        "configuration_evidence": snapshot.name,
        "configuration_evidence_sha256": sha(snapshot),
        "startup_seconds": state["startup_seconds"],
        "pid": api[0]["pid"],
        "module_file": registration.get("module_file"),
        "device_ids": registration.get("device_ids"),
        "observed_registration": registration,
        "vllm_package": vllm_package,
    }
    write(output, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-num-seqs", type=int, choices=(1, 8, 16, 32))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stop", action="store_true", help="Stop only the recorded idle server")
    args = parser.parse_args()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads(STATE.read_text()) if STATE.exists() else None
        if args.stop:
            if old:
                stop_owned(old)
                STATE.unlink()
            return
        if args.output is None or args.max_num_seqs is None:
            parser.error("--output and --max-num-seqs are required")
        output = args.output.resolve()
        if old and old["max_num_seqs"] == args.max_num_seqs and old["base_url"] == args.base_url:
            live = process(old["pid"])
            expected_phase_log = str(output.parent / f"phase-b{args.max_num_seqs}.jsonl")
            if (
                live
                and live["start_ticks"] == old["start_ticks"]
                and old.get("phase_log") == expected_phase_log
                and old.get("source_sha256_at_launch") == source_hashes()
            ):
                observe(old, output, args.max_num_seqs)
                return
        if old:
            stop_owned(old)
            STATE.unlink()
        owners = device_owners()
        if owners:
            raise RuntimeError(f"Devices already owned; refusing launch: {owners}")
        parsed = urlparse(args.base_url)
        if parsed.hostname not in ("127.0.0.1", "localhost"):
            raise RuntimeError("Only the local reserved device is supported")
        server_dir = output.parent / f"server-b{args.max_num_seqs}"
        server_dir.mkdir(parents=True, exist_ok=False)
        phase_log = output.parent / f"phase-b{args.max_num_seqs}.jsonl"
        clean_stale_phase_socket(phase_log)
        env = dict(os.environ)
        for key in ("GEMMA4_SERVING_TEST_LAYERS", "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
            if env.get(key):
                raise RuntimeError(f"Forbidden benchmark environment {key}")
        env["GEMMA4_BENCHMARK_PHASE_LOG"] = str(phase_log)
        cmd = [
            "bash",
            str(MODEL / "tests/run_serving.sh"),
            "--stages",
            "serve",
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--enable-prefix-caching",
            "--port",
            str(parsed.port or 8000),
            "--output-dir",
            str(server_dir),
        ]
        snapshot = "/home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/" + REVISION
        # Retain the upstream underscore spelling and add the equivalent
        # canonical CLI flag so the benchmark validator sees the actual argv.
        extras = [
            "--revision",
            REVISION,
            "--tokenizer-revision",
            REVISION,
            "--language-model-only",
            "--async-scheduling",
            "--override-generation-config",
            '{"top_k":32}',
            "--served-model-name",
            "google/gemma-4-31B-it",
            "--max-num-seqs",
            str(args.max_num_seqs),
            snapshot,
        ]
        cmd += ["--additional-server-args", shlex.join(extras)]
        launch_hashes = source_hashes()
        launch_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        start = time.monotonic()
        started_unix = time.time()
        with (server_dir / "runner.log").open("wb") as log:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        state = {
            **process(proc.pid),
            "max_num_seqs": args.max_num_seqs,
            "base_url": args.base_url,
            "server_dir": str(server_dir),
            "phase_log": str(phase_log),
            "launch_command": cmd,
            "started_unix": started_unix,
            "started_monotonic": start,
            "source_sha256_at_launch": launch_hashes,
            "tt_metal_commit_at_launch": launch_commit,
            "startup_seconds": None,
            "bounded_seconds": 7200,
        }
        write(STATE, state)
        while time.monotonic() - start < 1900:
            if proc.poll() is not None:
                raise RuntimeError(f"Server launcher exited {proc.returncode}; retained {server_dir}")
            try:
                fetch(args.base_url + "/health")
            except (OSError, ValueError):
                time.sleep(2)
                continue
            state["startup_seconds"] = time.monotonic() - start
            write(STATE, state)
            observe(state, output, args.max_num_seqs)
            print(json.dumps({"ready": str(output), "startup_seconds": state["startup_seconds"]}))
            return
        raise RuntimeError(
            f"Readiness timeout; bounded launcher and evidence retained at {server_dir}; "
            "do not terminate outstanding device work"
        )


if __name__ == "__main__":
    main()
