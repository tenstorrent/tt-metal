#!/usr/bin/env python3
"""Bounded host-sudo metadata observer. No device opens or external signals."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time

import owned_seed_copy as owned
import slurm_pair_checks as checks

CHECKS_SHA256 = "a5d4cd04210c0df621f1d73fc49b60bbe50a10855605d7f83d7e937e618355a2"


def process_uid(pid):
    fields = next(
        line.split()[1:] for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("Uid:")
    )
    assert len(fields) == 4 and len(set(fields)) == 1, "Incomplete or changing worker UID"
    return int(fields[0])


def read_receipt(path, uid=0):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        assert stat.S_ISREG(info.st_mode) and info.st_uid == uid, "Unowned observer receipt"
        with os.fdopen(descriptor, "r", closefd=False) as stream:
            return json.load(stream)
    finally:
        os.close(descriptor)


def bindings(directory, role, seconds):
    return {
        "operation_dir": str(directory.resolve(strict=True)),
        "role": role,
        "helper_sha256": checks.sha(Path(__file__)),
        "checks_sha256": checks.sha(Path(checks.__file__)),
        "drivers_sha256": checks.sha(directory / "drivers.json"),
        "internal_deadline_seconds": seconds,
    }


def stop_self_on_input_end():
    # The privileged worker only signals itself. EOF also covers controller death.
    sys.stdin.readline()
    os.kill(os.getpid(), signal.SIGTERM)


def worker(directory, role, seconds):
    assert os.getuid() == os.geteuid() == 0, "Observer requires existing host sudo"
    assert 0 < seconds <= 75
    captured = owned.identity(os.getpid())
    assert captured and process_uid(captured["pid"]) == 0
    record = {**bindings(directory, role, seconds), "worker": captured, "uid": 0, "released": False}
    assert record["checks_sha256"] == CHECKS_SHA256, "Predicate changed"

    def interrupted(signum, frame):
        raise owned.Cancelled("Observer deadline" if signum == signal.SIGALRM else "Observer ordinary cancellation")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        owned.save(directory / (role + "-started.json"), record)
        assert sys.stdin.readline() == "GO\n", "Observer gate closed before admission"
        record["released"] = True
        threading.Thread(target=stop_self_on_input_end, daemon=True).start()
        result = checks.quiet(json.loads((directory / "drivers.json").read_text()))
        owned.save(directory / (role + ".json"), result)
        record["completed"] = True
        return 0 if result["quiet"] else 2
    except BaseException as error:
        record["completed"] = False
        record["failure"] = type(error).__name__
        return 3
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        record["observation_stopped"] = True
        owned.save(directory / (role + "-closed.json"), record)


def observe(adapter, directory, role, cancel=lambda: False, seconds=90, internal_seconds=75):
    """Capture root identity before GO; EOF closes only this owned worker."""
    assert re.fullmatch(r"[a-z][a-z0-9-]*", role)
    assert 0 < internal_seconds <= 75 and internal_seconds + 10 <= seconds <= 90
    adapter, directory = adapter.resolve(strict=True), directory.resolve(strict=True)
    assert adapter == Path(__file__).resolve().parent
    assert directory.is_dir() and directory.stat().st_uid == os.getuid()
    expected = bindings(directory, role, internal_seconds)
    assert expected["checks_sha256"] == CHECKS_SHA256
    paths = {
        name: directory / (role + suffix)
        for name, suffix in {
            "started": "-started.json",
            "closed": "-closed.json",
            "ownership": "-ownership.json",
            "log": ".log",
            "result": ".json",
        }.items()
    }
    assert not any(path.exists() or path.is_symlink() for path in paths.values()), "Observer role already exists"
    record = {**expected, "controller": owned.identity(os.getpid()), "released": False, "closed": False}
    assert record["controller"]
    process, failure, started = None, None, None
    end = time.monotonic() + seconds
    with paths["log"].open("x") as log:
        try:
            process = subprocess.Popen(
                [
                    "sudo",
                    "-n",
                    "/usr/bin/python3",
                    "-B",
                    str(adapter / "host_fd_probe.py"),
                    "--worker",
                    "--directory",
                    str(directory),
                    "--role",
                    role,
                    "--seconds",
                    str(internal_seconds),
                ],
                stdin=subprocess.PIPE,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["sudo_child"] = owned.identity(process.pid)
            assert record["sudo_child"] and owned.same_process(record["sudo_child"]), "No sudo child birth"
            owned.save(paths["ownership"], record)
            capture_end = min(end - 10, time.monotonic() + 10)
            while not paths["started"].exists():
                if cancel():
                    raise owned.Cancelled("Observer cancelled before root capture")
                assert process.poll() is None, "Sudo observer failed before identity capture"
                assert time.monotonic() < capture_end, "Root identity capture deadline"
                time.sleep(0.05)
            started = read_receipt(paths["started"])
            assert all(started.get(key) == value for key, value in expected.items()), "Observer binding mismatch"
            assert started["uid"] == 0 and not started["released"]
            record["worker"] = started["worker"]
            assert owned.same_process(record["worker"]) and process_uid(record["worker"]["pid"]) == 0
            owned.save(paths["ownership"], record)
            if cancel():
                raise owned.Cancelled("Observer cancelled before GO")
            process.stdin.write("GO\n")
            process.stdin.flush()
            record["released"] = True
            owned.save(paths["ownership"], record)
            while process.poll() is None:
                if cancel() or time.monotonic() >= end - 10:
                    raise owned.Cancelled("Observer cancellation or deadline")
                time.sleep(0.05)
            if process.returncode not in (0, 2):
                raise owned.Cancelled("Observer did not complete its metadata scan")
        except BaseException as error:
            failure = error
        finally:
            # Closing only our stdin requests ordinary root-worker termination.
            # There is no parent signal, sudo kill, PID reuse race or escalation.
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                try:
                    code = process.wait(timeout=min(10, max(0.1, end - time.monotonic())))
                    closed = read_receipt(paths["closed"])
                    if started is None and paths["started"].exists():
                        started = read_receipt(paths["started"])
                    assert started and closed["worker"] == started["worker"]
                    assert closed["uid"] == 0 and closed["observation_stopped"]
                    assert all(closed.get(key) == value for key, value in expected.items()), "Closure binding mismatch"
                    assert not owned.same_process(closed["worker"]), "Root observer still exists"
                    record["worker"] = closed["worker"]
                    record["close"] = {
                        "reaped": True,
                        "exit_code": code,
                        "escalated": False,
                        "privileged_worker_absent": True,
                    }
                    record["closed"] = True
                except BaseException as error:
                    record["close"] = {
                        "reaped": process.poll() is not None,
                        "escalated": False,
                        "privileged_worker_absent": False,
                    }
                    record["closure_failure"] = type(error).__name__
                    failure = failure or error
            if failure:
                record["incomplete"] = type(failure).__name__
            owned.save(paths["ownership"], record)
    if failure:
        raise failure
    result = read_receipt(paths["result"])
    assert code == (0 if result["quiet"] else 2) and closed["completed"]
    return code, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    options = parser.parse_args()
    assert re.fullmatch(r"[a-z][a-z0-9-]*", options.role)
    sys.exit(worker(options.directory.resolve(strict=True), options.role, options.seconds))
