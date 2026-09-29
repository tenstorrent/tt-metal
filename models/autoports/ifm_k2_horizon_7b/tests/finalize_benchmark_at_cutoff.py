"""Close this owned benchmark and report before its original deadline."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL / "doc/benchmark"
RUN = ROOT / "run"


def main():
    invocation = json.loads((ROOT / "invocation.json").read_text())
    deadline = invocation["started_monotonic"] + 3600
    attempt = RUN / "accuracy-resume/attempt.json"
    while not attempt.exists():
        if time.monotonic() >= deadline - 45:
            raise TimeoutError("Accuracy continuation did not finish in its reserved interval")
        time.sleep(0.5)
    source = RUN / "accuracy-resume/ifeval"
    target = RUN / "ifeval"
    target.mkdir(exist_ok=True)
    for path in source.glob("*"):
        if path.is_file():
            destination = target / path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite raw evidence {destination}")
            shutil.copy2(path, destination)
    state = json.loads((ROOT / "server-state.json").read_text())
    runtime = json.loads(Path(state["runtime_identity"]).read_text())
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(10)
        connection.connect(runtime["snapshot_socket"])
        connection.sendall(b"snapshot\n")
        reply = b""
        while b"\n" not in reply:
            block = connection.recv(65536)
            if not block:
                break
            reply += block
    (RUN / "accuracy-resume/final-phase-status.json").write_bytes(reply)
    shutil.copy2(state["phase_path"], RUN / "accuracy-resume/final-phases.jsonl")
    try:
        metrics = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5).read()
        (RUN / "accuracy-resume/final-metrics.txt").write_bytes(metrics)
    except OSError as exc:
        (RUN / "accuracy-resume/final-metrics-error.txt").write_text(str(exc) + "\n")
    results = {}
    jobs = [
        ("owned-process-cleanup", [sys.executable, str(MODEL / "tests/benchmark_server.py"), "--stop"], 65),
        ("final-device-enumeration", ["tt-smi", "-ls", "--local"], 15),
        (
            "upstream-question-scoring",
            [sys.executable, str(MODEL / "tests/score_preserved_benchmark_questions.py")],
            40,
        ),
        ("final-report", [sys.executable, str(MODEL / "tests/finish_incomplete_benchmark_report.py")], 20),
    ]
    environment = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    for label, command, cap in jobs:
        remaining = deadline - time.monotonic() - 3
        if remaining <= 0:
            results[label] = {"status": "deadline exhausted"}
            continue
        try:
            with (RUN / f"{label}.log").open("w") as log:
                log.write(json.dumps(command) + "\n")
                log.flush()
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=min(cap, remaining)
                )
            results[label] = {"exit_code": result.returncode}
        except (OSError, subprocess.TimeoutExpired) as exc:
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
    results["original_elapsed_seconds"] = time.monotonic() - invocation["started_monotonic"]
    results["server_pid_owned"] = state["pid"]
    (RUN / "finalization.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
