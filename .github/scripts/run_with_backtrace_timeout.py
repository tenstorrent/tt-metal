#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run a command with a timeout; dump a gdb backtrace if it exceeds the limit.

Usage:
    run_with_backtrace_timeout.py [KEY=VAL ...] <executable> [args ...]

Leading KEY=VALUE arguments are added to the child's environment.
If the child is still running after TIMEOUT_SECS, all thread backtraces are
dumped via gdb, then the process is killed with SIGABRT followed by SIGKILL.
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time

TIMEOUT_SECS = 120

_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _dump_backtrace_and_kill(pid: int, cmd_display: str) -> None:
    print(f"::error::TIMEOUT after {TIMEOUT_SECS}s, dumping backtrace: {cmd_display}", flush=True)
    try:
        subprocess.run(
            ["gdb", "-batch", "-ex", "thread apply all bt full", "-p", str(pid)],
            timeout=30,
        )
    except Exception as exc:
        print(f"gdb failed: {exc}", flush=True)
    for sig, delay in [(signal.SIGABRT, 5), (signal.SIGKILL, 0)]:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        if delay:
            time.sleep(delay)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(f"Usage: {sys.argv[0]} [KEY=VAL ...] <executable> [args ...]", file=sys.stderr)
        sys.exit(1)

    extra_env: dict[str, str] = {}
    while args and _ENV_VAR_RE.match(args[0]):
        key, _, val = args.pop(0).partition("=")
        extra_env[key] = val

    if not args:
        print("No executable provided after environment variables.", file=sys.stderr)
        sys.exit(1)

    env = {**os.environ, **extra_env}

    cmd_display = " ".join(f"{k}={v}" for k, v in extra_env.items())
    if cmd_display:
        cmd_display += " "
    cmd_display += " ".join(args)

    proc = subprocess.Popen(args, env=env)
    timer = threading.Timer(TIMEOUT_SECS, _dump_backtrace_and_kill, args=[proc.pid, cmd_display])
    timer.start()
    try:
        proc.wait()
    finally:
        timer.cancel()

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
