# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep one readiness launch's child processes owned through startup and exit."""

import argparse
import ctypes
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

LAUNCH_MARKER = "QWEN_VLLM_LAUNCH_ID"
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
LIBC.pidfd_send_signal.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)


def pidfd_open(pid):
    # The provided standalone Python omits os.pidfd_open; glibc exposes it.
    fd = LIBC.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    return fd


def pidfd_send_signal(fd, signum):
    if LIBC.pidfd_send_signal(fd, signum, None, 0) != 0:
        raise OSError(ctypes.get_errno(), "pidfd_send_signal failed")


@dataclass(frozen=True)
class OwnedProcess:
    pid: int
    start_ticks: int


def read_owned(pid, marker, proc_root=Path("/proc")):
    """Match a complete environment entry and retain the process birth identity."""
    path = proc_root / str(pid)
    try:
        if pid == os.getpid() or path.stat().st_uid != os.getuid():
            return None
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        entry = f"{LAUNCH_MARKER}={marker}".encode()
        if entry not in (path / "environ").read_bytes().split(b"\0"):
            return None
        return OwnedProcess(pid, int(fields[19]))
    except (OSError, IndexError, ValueError):
        # Processes can exit during the scan. Unreadable processes are never
        # assumed to belong to this launch and are never signalled.
        return None


def owned_processes(marker, proc_root=Path("/proc")):
    processes = []
    for path in proc_root.iterdir():
        if path.name.isdigit():
            process = read_owned(int(path.name), marker, proc_root)
            if process is not None:
                processes.append(process)
    return sorted(processes, key=lambda process: process.pid)


def signal_owned(process, marker, signum):
    """Use a pidfd so PID reuse after validation cannot signal another launch."""
    try:
        fd = pidfd_open(process.pid)
    except ProcessLookupError:
        return False
    try:
        if read_owned(process.pid, marker) != process:
            return False
        pidfd_send_signal(fd, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def reap_children():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def cleanup_owned(marker, timeout):
    deadline = time.monotonic() + timeout
    notified = set()
    while True:
        reap_children()
        remaining = owned_processes(marker)
        if not remaining:
            return []
        for process in remaining:
            if process not in notified:
                try:
                    if signal_owned(process, marker, signal.SIGTERM):
                        print(f"VLLM_STAGE_CLEANUP SIGTERM pid={process.pid} launch_id={marker}", flush=True)
                except OSError as error:
                    print(f"VLLM_STAGE_CLEANUP signal_failed pid={process.pid}: {error}", flush=True)
                notified.add(process)
        delay = deadline - time.monotonic()
        if delay <= 0:
            return owned_processes(marker)
        time.sleep(min(0.1, delay))


def run(command, *, runner_grace=20, shutdown_timeout=30):
    marker = uuid.uuid4().hex
    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = signum

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    proc = None
    returncode = 1
    try:
        # Adopt orphaned grandchildren so this guard can also reap their zombies.
        if LIBC.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "Cannot enable child process reaping")
        proc = subprocess.Popen(command, env={**os.environ, LAUNCH_MARKER: marker}, start_new_session=True)
        print(f"VLLM_STAGE_GUARD pid={os.getpid()} runner_pid={proc.pid} launch_id={marker}", flush=True)
        while proc.poll() is None and stop_signal is None:
            time.sleep(0.1)
        if stop_signal is not None:
            # SIGINT invokes the packaged runner's finally even before it has
            # installed its serve-only SIGTERM handler. Preserve that cleanup.
            process = read_owned(proc.pid, marker)
            if process is not None:
                signal_owned(process, marker, signal.SIGINT)
            try:
                proc.wait(timeout=runner_grace)
            except subprocess.TimeoutExpired:
                pass
            returncode = 128 + stop_signal
        else:
            returncode = proc.returncode
            if returncode < 0:
                returncode = 128 - returncode
    finally:
        remaining = cleanup_owned(marker, shutdown_timeout)
        reap_children()
        if remaining:
            pids = ",".join(str(process.pid) for process in remaining)
            print(
                f"VLLM_STAGE_CLEANUP remaining_pids={pids} launch_id={marker}; "
                "preserve logs and capture triage before further recovery",
                flush=True,
            )
            returncode = returncode or 1
        else:
            print(f"VLLM_STAGE_CLEANUP complete launch_id={marker}", flush=True)
        if proc is not None:
            proc.poll()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-grace", type=float, default=20)
    parser.add_argument("--shutdown-timeout", type=float, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or min(args.runner_grace, args.shutdown_timeout) <= 0:
        parser.error("a command and positive shutdown timeouts are required")
    return run(command, runner_grace=args.runner_grace, shutdown_timeout=args.shutdown_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
