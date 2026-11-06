#!/usr/bin/env python3
"""Run upstream serving client with a quiescent model-observer evidence window."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CLIENT = ROOT / "bringup/artifacts/benchmark-perf-client/bin/vllm"
TOKENIZER = "/home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"


def exchange(path, command):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(30)
        sock.connect(str(path))
        sock.sendall((json.dumps({"command": command}) + "\n").encode())
        with sock.makefile("r") as stream:
            result = json.loads(stream.readline())
    if result.get("error") or result.get("errors"):
        raise RuntimeError(result)
    return result


def main():
    args = sys.argv[1:]

    def value(flag):
        return args[args.index(flag) + 1]

    output = Path(value("--result-dir"))
    name = value("--result-filename")
    batch = int(value("--max-concurrency"))
    measured = "warmup" not in name
    phase_path = os.path.abspath(output / f"phase-b{batch}.jsonl")
    endpoint = "/tmp/gemma4-phase-" + hashlib.sha256(phase_path.encode()).hexdigest()[:16] + ".sock"
    if measured:
        start = exchange(endpoint, "reset")
        (output / f"phase-b{batch}-window-start.json").write_text(json.dumps(start, indent=2) + "\n")
    command = [
        str(CLIENT),
        "bench",
        "serve",
        "--tokenizer",
        TOKENIZER,
        "--ready-check-timeout-sec",
        "0",
        "--num-warmups",
        "0",
        *args,
    ]
    env = dict(os.environ, VLLM_PLUGINS="")
    started = time.monotonic_ns()
    result = subprocess.run(command, env=env, check=False)
    ended = time.monotonic_ns()
    if measured:
        end = exchange(endpoint, "drain")
        end["client_process_start_ns"] = started
        end["client_process_end_ns"] = ended
        end["client_exit_code"] = result.returncode
        (output / f"phase-b{batch}-window-end.json").write_text(json.dumps(end, indent=2) + "\n")
    # Canonical trailing newline before downstream evidence hashes are computed.
    raw = output / name
    if raw.exists() and not raw.read_bytes().endswith(b"\n"):
        with raw.open("ab") as stream:
            stream.write(b"\n")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
