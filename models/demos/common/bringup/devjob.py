# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F238: device jobs longer than an agent's tool call (about 10 minutes), run detached so nothing has to kill them.

An agent's shell call ends at its time limit, and a device run cut off there leaves every card marked dirty: the next
run then resets the whole box (DeepSeek-V4.1 FIX.16, P.22 and FIX.19, 2026-10-31 / 11-01, the last one needing an owner
reset). A job started here runs in its own session (setsid), survives the agent's call, the agent's end and an
overseer abort, and is never killed by this tool; the agent waits for it in short calls.

    python -m models.demos.common.bringup.devjob start NAME -- scripts/run_safe_pytest.sh ...   # returns at once
    python -m models.demos.common.bringup.devjob wait NAME [--timeout 540]   # rc of the job, or 75 = still running
    python -m models.demos.common.bringup.devjob status [NAME]               # every job, or one
    python -m models.demos.common.bringup.devjob tail NAME [-n 40]           # the end of its log

Jobs live in $BRINGUP_DEVJOB_DIR, else <repo>/generated/devjobs/<NAME>/ (cmd.json, log, rc when finished)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

RUNNING = 75  # EX_TEMPFAIL: wait again


def root() -> Path:
    env = os.environ.get("BRINGUP_DEVJOB_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[4] / "generated" / "devjobs"


def job_dir(name: str) -> Path:
    if not name or "/" in name or name.startswith("."):
        raise SystemExit(f"devjob: bad name {name!r}")
    return root() / name


def start(name: str, cmd: list[str]) -> int:
    d = job_dir(name)
    if (d / "cmd.json").exists() and not (d / "rc").exists():
        raise SystemExit(f"devjob {name}: already running (wait for it, or pick another name)")
    d.mkdir(parents=True, exist_ok=True)
    for f in ("rc", "log"):
        (d / f).unlink(missing_ok=True)
    line = " ".join(shlex.quote(c) for c in cmd)
    # the rc is written by the job's own shell, so it is there whoever ends the agent that started it
    script = f"{line}; echo $? > {shlex.quote(str(d / 'rc'))}"
    with open(d / "log", "wb") as log:
        p = subprocess.Popen(
            ["bash", "-c", script],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=os.getcwd(),
            start_new_session=True,
        )
    (d / "cmd.json").write_text(json.dumps({"cmd": line, "pid": p.pid, "cwd": os.getcwd(), "t": time.time()}))
    print(f"devjob {name}: started pid {p.pid}; log {d / 'log'}; then `devjob wait {name}`")
    return 0


def _rc(d: Path) -> int | None:
    try:
        return int((d / "rc").read_text().strip())
    except (OSError, ValueError):
        return None


def wait(name: str, timeout: float, poll: float = 2.0) -> int:
    d = job_dir(name)
    if not (d / "cmd.json").exists():
        raise SystemExit(f"devjob {name}: no such job")
    t0 = time.time()
    while True:
        rc = _rc(d)
        if rc is not None:
            print(f"devjob {name}: finished rc {rc}")
            return rc
        if time.time() - t0 >= timeout:
            started = json.loads((d / "cmd.json").read_text())["t"]
            print(f"devjob {name}: still running ({time.time() - started:.0f} s); wait again (never kill it)")
            return RUNNING
        time.sleep(poll)


def status(name: str | None) -> int:
    base = root()
    names = [name] if name else sorted(p.name for p in base.glob("*") if (p / "cmd.json").exists())
    for n in names:
        d = job_dir(n)
        meta = json.loads((d / "cmd.json").read_text())
        rc = _rc(d)
        state = f"finished rc {rc}" if rc is not None else f"running {time.time() - meta['t']:.0f} s"
        print(f"{n}: {state}: {meta['cmd'][:120]}")
    return 0


def tail(name: str, n: int) -> int:
    lines = (job_dir(name) / "log").read_text(errors="replace").splitlines()
    print("\n".join(lines[-n:]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="devjob", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="op", required=True)
    s = sub.add_parser("start")
    s.add_argument("name")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    w = sub.add_parser("wait")
    w.add_argument("name")
    w.add_argument("--timeout", type=float, default=540.0)
    st = sub.add_parser("status")
    st.add_argument("name", nargs="?")
    t = sub.add_parser("tail")
    t.add_argument("name")
    t.add_argument("-n", type=int, default=40)
    a = ap.parse_args(argv)
    if a.op == "start":
        cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not cmd:
            ap.error("start NAME -- COMMAND ...")
        return start(a.name, cmd)
    if a.op == "wait":
        return wait(a.name, a.timeout)
    if a.op == "status":
        return status(a.name)
    return tail(a.name, a.n)


if __name__ == "__main__":
    sys.exit(main())
