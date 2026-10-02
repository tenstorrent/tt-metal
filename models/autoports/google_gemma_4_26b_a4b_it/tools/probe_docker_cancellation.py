# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Reproduce late Docker-exec writes and test explicitly owned cancellation.

Use only a disposable, exclusively owned, already-running container with the
candidate helper mounted at /guard.py. No model or device access is performed.
"""

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.container != "gemma4-eval-cancel-control":
        raise ValueError("Use the dedicated disposable cancellation-control container")
    docker = ["sudo", "-n", "docker", "exec", args.container]

    def execute(*command):
        return subprocess.check_output(docker + list(command), text=True, timeout=15).strip()

    def cancel_client(command):
        process = subprocess.Popen(docker + command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            ready = process.stdout.readline().strip()
            if ready != "READY":
                raise RuntimeError("Remote command did not publish readiness: " + ready)
            started = time.monotonic()
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            return time.monotonic() - started
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            process.stdout.close()
            process.stderr.close()

    baseline_program = (
        "import time; from pathlib import Path; print('READY',flush=True); "
        "time.sleep(2); Path('/tmp/baseline-late-write').touch()"
    )
    baseline_cancel_s = cancel_client(["python3", "-c", baseline_program])
    time.sleep(2.2)
    leaked = execute("python3", "-c", "from pathlib import Path; print(Path('/tmp/baseline-late-write').exists())")

    # This peer is not a descendant of the command under test and must survive.
    subprocess.check_call(
        [
            "sudo",
            "-n",
            "docker",
            "exec",
            "-d",
            args.container,
            "python3",
            "-c",
            "import os,time; from pathlib import Path; Path('/tmp/cancel-peer.pid').write_text(str(os.getpid())); time.sleep(120)",
        ]
    )
    child = "import time; from pathlib import Path; time.sleep(2); Path('/tmp/candidate-late-write').touch()"
    candidate_program = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c', {!r}], start_new_session=True); "
        "print('READY',flush=True); time.sleep(30)"
    ).format(child)
    base = "/tmp/harbor-owned-" + uuid.uuid4().hex
    candidate_cancel_s = cancel_client(["python3", "/guard.py", "launch", base, "python3", "-c", candidate_program])
    started = time.monotonic()
    cleanup = json.loads(execute("python3", "/guard.py", "cleanup", base))
    cleanup_s = time.monotonic() - started
    time.sleep(2.2)
    candidate_leaked = execute(
        "python3", "-c", "from pathlib import Path; print(Path('/tmp/candidate-late-write').exists())"
    )
    peer = execute(
        "python3",
        "-c",
        "import os; from pathlib import Path; os.kill(int(Path('/tmp/cancel-peer.pid').read_text()),0); print('alive')",
    )
    report = {
        "scope": "owned Docker cancellation microtest; not model speed or SWE reward",
        "container": args.container,
        "baseline_client_cancel_s": baseline_cancel_s,
        "baseline_late_write_observed": leaked == "True",
        "candidate_client_cancel_s": candidate_cancel_s,
        "candidate_cleanup_s": cleanup_s,
        "candidate_cleanup": cleanup,
        "candidate_late_write_observed": candidate_leaked == "True",
        "unrelated_peer_alive": peer == "alive",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if (
        not report["baseline_late_write_observed"]
        or report["candidate_late_write_observed"]
        or not report["unrelated_peer_alive"]
    ):
        raise AssertionError("Cancellation control did not satisfy the expected contrast")


if __name__ == "__main__":
    main()
