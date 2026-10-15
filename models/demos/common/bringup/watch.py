# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Block until the run needs the supervisor: the one wait a supervising session uses between events.

    python -m models.demos.common.bringup.watch --spec S [--timeout 6600] [--quiet-gates]

Reads ``runs/<run>/orchestrator.log`` from its current end and returns on the first line the supervisor acts on (a
gate result, a second or later attempt, STOPPED, WAITING FOR A PERSON, paused, DEFERRED, KNOWLEDGE CONTRADICTED, a
traceback, the run's end) or as soon as the orchestrator is not running (its pid file, written by ``run`` /
``resume``, is gone or names a dead process). An orchestrator that already exited, e.g. waiting for an approval,
returns at once with the log's last event (e.g. its WAITING FOR A PERSON line), so a wait can never start on a run that has stopped.

Exit codes: 0 an event (lines printed), 2 the timeout passed with nothing to act on (re-arm), 3 the orchestrator is not
running (the last event printed: act on it before relaunching).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

EVENT, TIMEOUT, NOT_RUNNING = 0, 2, 3
PATTERN = re.compile(
    r"gate PASS|gate FAIL|HANG|STOPPED|stopped tasks|WAITING FOR A PERSON|paused before|DEFERRED|deferral rejected"
    r"|KNOWLEDGE CONTRADICTED|Traceback|nothing left to run|complete with|ORCHESTRATOR_EXIT|attempt [2-9]\d*:"
)
GATES = re.compile(r"gate PASS|gate FAIL")
PID_FILE = "ORCHESTRATOR.pid"


def alive(run_dir: Path) -> bool:
    """The orchestrator of this run is running (its pid file names a live process)."""
    try:
        pid = int((run_dir / PID_FILE).read_text().split()[0])
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def events(lines: list[str], quiet_gates: bool = False) -> list[str]:
    """The lines a supervisor acts on. With quiet_gates a plain gate FAIL (the check before an agent) is dropped."""
    out = [ln for ln in lines if PATTERN.search(ln)]
    if quiet_gates:
        out = [ln for ln in out if not (GATES.search(ln) and "FAIL" in ln)]
    return out


def last_event(log: Path) -> str:
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    hits = [h for h in events(lines) if "ORCHESTRATOR_EXIT" not in h] or lines  # the launcher's exit line says less
    return hits[-1][:300] if hits else "(empty log)"


def watch(run_dir: Path, timeout: float, poll: float = 15.0, quiet_gates: bool = False, out=print) -> int:
    log = run_dir / "orchestrator.log"
    if not alive(run_dir):
        out(f"ORCHESTRATOR NOT RUNNING; last event: {last_event(log)}")
        return NOT_RUNNING
    pos = log.stat().st_size if log.exists() else 0
    end = time.monotonic() + timeout
    buf = ""
    while True:
        if log.exists():
            with open(log, errors="replace") as f:
                f.seek(pos)
                buf += f.read()
                pos = f.tell()
            *done, buf = buf.split("\n")
            hits = events(done, quiet_gates)
            if hits:
                for h in hits:
                    out(h[:300])
                return EVENT
        if not alive(run_dir):
            out(f"ORCHESTRATOR NOT RUNNING; last event: {last_event(log)}")
            return NOT_RUNNING
        if time.monotonic() >= end:
            out(f"no event in {timeout:.0f} s; the orchestrator is running (re-arm the watch)")
            return TIMEOUT
        time.sleep(poll)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
    ap.add_argument("--run-dir", help="instead of --spec: runs/<run>")
    ap.add_argument("--timeout", type=float, default=6600.0, help="seconds (below the harness's 2 h limit)")
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--quiet-gates", action="store_true", help="skip a plain gate FAIL (the check before an agent)")
    a = ap.parse_args(argv)
    if a.run_dir:
        run_dir = Path(a.run_dir)
    else:
        from models.demos.common.bringup.core.gate import run_name
        from models.demos.common.bringup.core.ledger import Ledger
        from models.demos.common.bringup.core.spec import Spec

        spec = Spec.load(a.spec)
        run_dir = spec.run_dir(run_name(Ledger(spec.bringup_dir)))
    return watch(run_dir, a.timeout, a.poll, a.quiet_gates)


if __name__ == "__main__":
    sys.exit(main())
