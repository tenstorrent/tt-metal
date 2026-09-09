# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run one hardware command; terminate its process group and reset on timeout.
Usage: python watchdog.py LOG TIMEOUT_SECONDS COMMAND [ARGS...]
The caller must have reserved device 0. A failed command stops the verification.
"""
import os
from pathlib import Path
import signal
import subprocess
import sys

log = Path(sys.argv[1])
timeout = int(sys.argv[2])
command = sys.argv[3:]
with log.open("w") as stream:
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        result = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        result = 124
contents = log.read_text(errors="replace")
if timed_out or (result and ("device timeout" in contents.lower() or "potential hang" in contents.lower())):
    with log.with_suffix(".reset.log").open("w") as stream:
        reset = subprocess.run(["tt-smi", "-r", "0"], stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    print(f"Watchdog recovery: reset exit={reset.returncode}; original exit={result}", flush=True)
print(f"exit={result}; log={log}", flush=True)
sys.exit(result)
