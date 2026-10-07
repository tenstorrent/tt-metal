# SPDX-License-Identifier: Apache-2.0
"""Reproducer: the dispatcher startup NOC-counter race hangs device close.

Opens one Blackhole chip with fast dispatch and two command queues, then closes it. The open/close runs in a
child process; this parent enforces hard time limits, so the script always exits:

  0  PASS            the device opened and closed
  2  HANG_AT_CLOSE   the device opened but close did not finish within --close-timeout (or teardown reported that
                     the dispatch cores did not finish); the chip needs `tt-smi -r <id>` before it is used again
  3  HANG_AT_OPEN    the device did not open within --open-timeout
  1  ERROR           the child failed for another reason (see its output)

Run it on a tree with repro-only-forced-delay.patch applied; see README.md. Each run compiles firmware and dispatch
kernels into a fresh JIT cache (a new temporary TT_METAL_CACHE) unless --jit-cache is given, so the patched sources
are always the ones that run.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

CHILD = r"""
import sys, time
import ttnn

def mark(what):
    print("REPRO %.3f %s" % (time.monotonic(), what), flush=True)

device_id = int(sys.argv[1])
mark("OPENING")
device = ttnn.CreateDevice(device_id, num_command_queues=2)
mark("OPENED")
ttnn.CloseDevice(device)
mark("CLOSED")
"""

# Printed by device teardown when TT_METAL_OPERATION_TIMEOUT_SECONDS bounds the wait for the dispatch cores.
TEARDOWN_TIMEOUT_TEXT = "Exception waiting for dispatch cores to finish during teardown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--open-timeout", type=float, default=900.0, help="seconds; includes the JIT build of firmware")
    ap.add_argument("--close-timeout", type=float, default=60.0, help="seconds; a healthy close takes a few")
    ap.add_argument("--jit-cache", help="JIT cache directory to use (default: a new temporary directory)")
    a = ap.parse_args()

    env = dict(os.environ)
    if env.get("TT_METAL_SLOW_DISPATCH_MODE"):
        print("ERROR: slow dispatch mode is set; this reproducer needs fast dispatch", flush=True)
        return 1
    # Without an operation timeout a stuck teardown blocks, which this parent detects and bounds. With one, close
    # returns after logging TEARDOWN_TIMEOUT_TEXT; that is detected too, but unsetting it keeps the verdict simple.
    env.pop("TT_METAL_OPERATION_TIMEOUT_SECONDS", None)
    cache = a.jit_cache or tempfile.mkdtemp(prefix="dispatch-startup-race-jit-")
    env["TT_METAL_CACHE"] = cache
    print(f"device {a.device_id}; JIT cache {cache}", flush=True)

    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(a.device_id)],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    marks, teardown_timeout = {}, threading.Event()

    def reader():
        for line in child.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith("REPRO "):
                parts = line.split()
                if len(parts) >= 3:
                    marks.setdefault(parts[2], time.monotonic())
            if TEARDOWN_TIMEOUT_TEXT in line:
                teardown_timeout.set()

    threading.Thread(target=reader, daemon=True).start()
    start = time.monotonic()
    verdict = None
    while child.poll() is None:
        time.sleep(0.2)
        now = time.monotonic()
        if "OPENED" not in marks and now - start > a.open_timeout:
            verdict = "HANG_AT_OPEN"
        elif "OPENED" in marks and "CLOSED" not in marks and now - marks["OPENED"] > a.close_timeout:
            verdict = "HANG_AT_CLOSE"
        if verdict:
            for sig in (signal.SIGTERM, signal.SIGKILL):  # only the child's own process group
                if child.poll() is None:
                    os.killpg(child.pid, sig)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
            break
    time.sleep(0.5)  # let the reader drain the last lines
    if verdict is None:
        if teardown_timeout.is_set():
            verdict = "HANG_AT_CLOSE"
        elif child.returncode == 0 and "CLOSED" in marks:
            verdict = "PASS"
        else:
            verdict = "ERROR"

    opened = marks.get("OPENED")
    open_s = opened - marks["OPENING"] if opened and "OPENING" in marks else None
    close_s = marks["CLOSED"] - opened if opened and "CLOSED" in marks else None
    print(
        f"VERDICT {verdict} open_s={'%.2f' % open_s if open_s is not None else '-'} "
        f"close_s={'%.2f' % close_s if close_s is not None else '-'} child_rc={child.returncode}",
        flush=True,
    )
    if verdict == "HANG_AT_CLOSE":
        print(
            f"Device {a.device_id} close did not complete: its dispatch cores are stuck. "
            f"Reset the chip (tt-smi -r {a.device_id}) before using it again.",
            flush=True,
        )
    if not a.jit_cache:
        shutil.rmtree(cache, ignore_errors=True)
    return {"PASS": 0, "HANG_AT_CLOSE": 2, "HANG_AT_OPEN": 3}.get(verdict, 1)


if __name__ == "__main__":
    sys.exit(main())
