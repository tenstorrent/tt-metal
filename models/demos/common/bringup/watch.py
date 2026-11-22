# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Block until the run needs the supervisor: the one wait a supervising session uses between events.

    python -m models.demos.common.bringup.watch --spec S [--timeout 1800] [--quiet-gates]

Reads ``runs/<run>/orchestrator.log`` from its current end and returns on the first line the supervisor acts on (a
gate result, a second or later attempt, STOPPED, WAITING FOR A PERSON, paused, DEFERRED, KNOWLEDGE CONTRADICTED, a
traceback, the run's end) or as soon as the orchestrator is not running (its pid file, written by ``run`` /
``resume``, is gone or names a dead process). An orchestrator that already exited, e.g. waiting for an approval,
returns at once with the log's last event (e.g. its WAITING FOR A PERSON line), so a wait can never start on a run that has stopped.

Exit codes: 0 an event (lines printed), 2 the timeout passed with nothing to act on (re-arm), 3 the orchestrator is not
running (the last event printed: act on it before relaunching).

FIX.31: a gate, a safe-runner run or a chain script holding the device is live too, whoever started it: a gate
(core/gate.run_gate: `gate --continue`, `gate --check`) and every scripts/run_safe_pytest.sh run of a bring-up spec
(core/runrec.py) name themselves in runs/<run>/live/ while they run, and a chain script registers its own pid once:

    python -m models.demos.common.bringup.watch live --spec S [--pid PID] [--what "chain: X.3 units"]

While any of them is alive the watch waits (no NOT RUNNING), and it wakes on their results in runs/<run>/gates.log
(a gate's PASS / FAIL, a failed unit; an agent's `gate --check` lines never wake it). After the last one ends it
waits ``--grace`` seconds for the next (a chain's gap between two gates) before it reports NOT RUNNING.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

EVENT, TIMEOUT, NOT_RUNNING = 0, 2, 3
PATTERN = re.compile(
    r"gate PASS|gate FAIL|HANG|STOPPED|stopped tasks|WAITING FOR A PERSON|paused before|DEFERRED|deferral rejected"
    r"|KNOWLEDGE CONTRADICTED|Traceback|nothing left to run|complete with|ORCHESTRATOR_EXIT|attempt [2-9]\d*:"
    r"|fix attempt \d+:"  # a scripted step (golden, box, ladder) failed its gate: never routine
    r"|ABORTED by the overseer|OVERSEER NOTE"  # F69
    r"|COMMIT PENDING"  # F149
    r"|gate unit \d+/\d+ \S+ (?:FAIL|STOPPED)"  # F169: the failed unit, as soon as it fails (not the whole gate)
)
GATES = re.compile(r"gate PASS|gate FAIL")
PID_FILE = "ORCHESTRATOR.pid"
LIVE_DIR = "live"  # FIX.31: <pid>.json / gate-<pid>-<task>.json / chain-<pid>.json while a run holds the device
GATES_LOG = "gates.log"  # FIX.31: results of gates and safe-runner runs started outside the orchestrator too
_PID_FIELD = re.compile(r"\bpid=(\d+)\b")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def orchestrator_pid(run_dir: Path) -> int | None:
    try:
        return int((run_dir / PID_FILE).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def live_runs(run_dir: Path) -> list[dict]:
    """FIX.31: the gates, safe-runner runs and chain scripts registered in runs/<run>/live/ whose process is alive."""
    out = []
    d = Path(run_dir) / LIVE_DIR
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            info = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if _pid_alive(info.get("pid")):
            out.append(info)
    return out


def register(run_dir: Path, pid: int, what: str) -> Path:
    """FIX.31: a chain script (or any long device job) names itself live until its pid ends."""
    d = Path(run_dir) / LIVE_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"chain-{pid}.json"
    p.write_text(json.dumps({"pid": pid, "what": what, "t": time.time()}))
    return p


def alive(run_dir: Path) -> bool:
    """The orchestrator of this run is running (its pid file names a live process)."""
    try:
        pid = int((run_dir / PID_FILE).read_text().split()[0])
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def events(lines: list[str], quiet_gates: bool = False, quiet_passes: bool = False) -> list[str]:
    """The lines a supervisor acts on. With quiet_gates a plain gate FAIL (the check before an agent) is dropped."""
    out = [ln for ln in lines if PATTERN.search(ln)]
    if quiet_gates:
        out = [ln for ln in out if not (GATES.search(ln) and "FAIL" in ln)]
    if quiet_passes:
        out = [ln for ln in out if "gate PASS" not in ln]
    return out


def last_event(log: Path) -> str:
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    hits = [h for h in events(lines) if "ORCHESTRATOR_EXIT" not in h] or lines  # the launcher's exit line says less
    return hits[-1][:300] if hits else "(empty log)"


REPORTED = "watch_reported.json"  # F170: dashboard lines a watch already returned on (once per change, not per call)


def first_report(run_dir: Path, line: str) -> bool:
    """F170 (audit item 20): a DASHBOARD RED / STALE line wakes the overseer once per change (RED: its content; STALE:
    per publish), not on every watch call (a red dashboard made every re-armed watch return at once)."""
    if line.startswith("DASHBOARD STALE"):
        from models.demos.common.bringup.dashboard import links as LK

        key = f"STALE {LK.published_at(run_dir)}"
    else:
        key = line
    p = Path(run_dir) / REPORTED
    try:
        seen = json.loads(p.read_text()) if p.exists() else []
    except ValueError:
        seen = []
    if key in seen:
        return False
    p.write_text(json.dumps((seen + [key])[-20:]) + "\n")
    return True


def _live_text(runs: list[dict]) -> str:
    return "; ".join(
        f"{r.get('what', '?')} (pid {r.get('pid')}{', ' + r['unit'] if r.get('unit') else ''})" for r in runs
    )


def watch(
    run_dir: Path,
    timeout: float,
    poll: float = 15.0,
    quiet_gates: bool = False,
    out=print,
    quiet_passes=False,
    stale=lambda: None,
    grace: float = 60.0,
) -> int:
    """stale: returns a line when the published dashboard lags the local one (dashboard/links.py, F71).
    FIX.31: a live gate / run / chain (``live_runs``) keeps the watch waiting while the orchestrator is not running;
    their result lines in gates.log are events like the orchestrator's (``grace``: the wait for the next live run after
    the last one ended)."""
    log = run_dir / "orchestrator.log"
    glog = run_dir / GATES_LOG
    if not alive(run_dir) and not live_runs(run_dir):
        out(f"ORCHESTRATOR NOT RUNNING; last event: {last_event(log)}")
        return NOT_RUNNING
    pos = {p: (p.stat().st_size if p.exists() else 0) for p in (log, glog)}
    bufs = {log: "", glog: ""}
    end = time.monotonic() + timeout
    idle_since = None
    while True:
        orch = orchestrator_pid(run_dir) if alive(run_dir) else None
        hits = []
        for p in (log, glog):
            if not p.exists():
                continue
            with open(p, errors="replace") as fh:
                fh.seek(pos[p])
                bufs[p] += fh.read()
                pos[p] = fh.tell()
            *done, bufs[p] = bufs[p].split("\n")
            if p == glog:  # the orchestrator's own gates are in its log already
                done = [ln for ln in done if not (orch and (m := _PID_FIELD.search(ln)) and int(m.group(1)) == orch)]
            hits += events(done, quiet_gates, quiet_passes)
        if hits:
            for h in hits:
                out(h[:300])
            return EVENT
        if not alive(run_dir):
            runs = live_runs(run_dir)
            if runs:
                idle_since = None
            else:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since >= grace:
                    last = last_event(glog) if glog.exists() and glog.stat().st_mtime > _mtime(log) else last_event(log)
                    out(f"ORCHESTRATOR NOT RUNNING; no live gate or run either; last event: {last}")
                    return NOT_RUNNING
        s = stale()
        if s and first_report(run_dir, s):
            out(s)
            return EVENT
        if time.monotonic() >= end:
            runs = [] if alive(run_dir) else live_runs(run_dir)
            who = f"live without the orchestrator: {_live_text(runs)}" if runs else "the orchestrator is running"
            out(f"no event in {timeout:.0f} s; {who} (re-arm the watch)")
            return TIMEOUT
        time.sleep(poll)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _run_dir_of(a):
    from models.demos.common.bringup.core.gate import run_name
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    spec = Spec.load(a.spec)
    return spec, spec.run_dir(run_name(Ledger(spec.ledger_dir)))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["live"]:  # FIX.31: a chain script registers itself
        lp = argparse.ArgumentParser(prog="watch live")
        lp.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
        lp.add_argument("--run-dir")
        lp.add_argument("--pid", type=int, default=os.getppid(), help="the process to follow (default: the caller)")
        lp.add_argument("--what", default="chain script")
        la = lp.parse_args(argv[1:])
        run_dir = Path(la.run_dir) if la.run_dir else _run_dir_of(la)[1]
        print(f"live: {register(run_dir, la.pid, la.what)} (until pid {la.pid} ends)")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
    ap.add_argument("--run-dir", help="instead of --spec: runs/<run>")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds: the heartbeat, at most 30 min (owner)")
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--grace", type=float, default=60.0, help="FIX.31: wait this long for a chain's next gate")
    ap.add_argument("--quiet-gates", action="store_true", help="skip a plain gate FAIL (the check before an agent)")
    ap.add_argument(
        "--quiet-passes", action="store_true", help="skip gate PASS lines (problems only; the timeout reports progress)"
    )
    a = ap.parse_args(argv)
    stale = lambda: None  # noqa: E731
    if a.run_dir:
        run_dir = Path(a.run_dir)
    else:
        spec, run_dir = _run_dir_of(a)
        from models.demos.common.bringup.dashboard import links as LK

        stale = lambda: LK.stale(spec.ledger_dir, run_dir=run_dir)  # noqa: E731
    rc = watch(run_dir, a.timeout, a.poll, a.quiet_gates, quiet_passes=a.quiet_passes, stale=stale, grace=a.grace)
    if rc == TIMEOUT:
        log = (
            (run_dir / "orchestrator.log").read_text(errors="replace")
            if (run_dir / "orchestrator.log").exists()
            else ""
        )
        print(
            f"progress: {log.count('gate PASS')} gates passed in this log; last: {last_event(run_dir / 'orchestrator.log')}"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
