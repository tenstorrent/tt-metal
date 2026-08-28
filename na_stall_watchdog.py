#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Kill a NA test PID if the log stalls after the test body starts.

Device open is not a stall. The clock starts on ``warmup done`` (or the first
``probe N name: M ms``). First-launch JIT is not a stall -- those hangs wait for
run_na's ``-t``. After warmup, 40s with no new probe-time / PASSED / FAILED
line means the device deadlocked -- do not wait for the full timeout.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time

ARM = re.compile(r"warmup done|compile/warmup|probe \d+ \S+: compile")
PROGRESS = re.compile(r"probe \d+ \S+:\s+[0-9]|===== neighborhood_sdpa|=== PASSED|=== FAILED|PASSED |FAILED ")


def kill_tree(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--stall", type=int, default=40)
    args = parser.parse_args()

    # Wait for the log inode (run_na truncates in place before we start).
    for _ in range(50):
        if os.path.exists(args.log):
            break
        time.sleep(0.1)

    armed = False
    last_progress = time.time()
    with open(args.log, "r", encoding="utf-8", errors="replace") as log:
        log.seek(0, os.SEEK_END)
        while True:
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError:
                return 0
            line = log.readline()
            if line:
                if ARM.search(line):
                    armed = True
                    last_progress = time.time()
                if PROGRESS.search(line):
                    armed = True
                    last_progress = time.time()
                continue
            if armed and (time.time() - last_progress) >= args.stall:
                print(
                    f"!!! WATCHDOG_STALL: no progress for {args.stall}s after test body started "
                    f"-- killing pid {args.pid} (device deadlock, not waiting for the full timeout)",
                    flush=True,
                )
                kill_tree(args.pid)
                return 2
            time.sleep(0.25)


if __name__ == "__main__":
    sys.exit(main())
