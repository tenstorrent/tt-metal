#!/usr/bin/env python3
"""Silence-based hang watchdog for a device job (replaces tools/hangwatch.sh).

    hangwatch.py --pid <pid> --log <logfile> [--stall-min 45] [--rules 'engram=60,build=45']

Every 10 s: if the job's log has been silent longer than the stall limit for the CURRENT stage (decided by the last log line matching a
rule regex; default 45 min everywhere, 60 for Engram: the Engram load alone takes ~9 min and an 8 min default killed 2 DeepSeek runs), then:
  1. python stacks: py-spy dump if installed, else SIGUSR1 if the process registered a handler (faulthandler.register(SIGUSR1) in the demo;
     checked in /proc/<pid>/status SigCgt, because an unhandled SIGUSR1 would kill it before triage),
  2. tt-triage against the LIVE hung process (it needs the running runtime), report saved,
  3. kill the WHOLE process tree + process group (the old watcher killed only the timeout wrapper and left python holding the device),
  4. device reset (default 'tt-smi -glx_reset', fallback 'tt-smi -r'), rc recorded in the log as '[hangwatch] reset rc=N',
  5. exit 3 (so the caller can retry). Exit 0 when the job ended by itself.
Everything is appended to the log with the prefix [hangwatch] and a JSON event is written to <log>.hang.json.
Use --triage-cmd/--reset-cmd 'none' to disable a step. The reset takes /tmp/dsv4_dev.lock unless this process already inherited it.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

DEFAULT_RULES = "engram=60,build=45,built layer=45,compile=45,trace=45,decode=45"
FLOOR_MIN = 45


def parse_rules(s):
    out = []
    for item in [x for x in s.split(",") if x.strip()]:
        pat, _, mins = item.rpartition("=")
        out.append((re.compile(pat, re.I), float(mins)))
    return out


def last_line(log, nbytes=8192):
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            lines = [l for l in f.read().decode(errors="ignore").splitlines() if l.strip() and "[hangwatch]" not in l]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def stall_limit_min(line, rules, default):
    for pat, mins in rules:
        if pat.search(line):
            return mins
    return default


def children_of(pid):
    """All descendants of pid (via /proc/*/stat ppid)."""
    ppid = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                s = open(f"/proc/{d}/stat").read()
                ppid[int(d)] = int(s[s.rindex(")") + 2 :].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        for c, pp in ppid.items():
            if pp == p and c not in out:
                out.append(c)
                todo.append(c)
    return out


def alive(pid):
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def kill_tree(pid, grace=5.0):
    """SIGTERM then SIGKILL the process, its descendants and (if it leads one) its process group. Returns the pids signalled."""
    pids = [pid] + children_of(pid)
    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if pgid and pgid == pid:
            try:
                os.killpg(pgid, sig)
            except OSError:
                pass
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        t0 = time.time()
        while time.time() - t0 < (grace if sig == signal.SIGTERM else 2) and any(alive(p) for p in pids):
            time.sleep(0.2)
        if not any(alive(p) for p in pids):
            break
    return pids


def has_usr1_handler(pid):
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("SigCgt:"):
                return bool(int(l.split()[1], 16) & (1 << (signal.SIGUSR1 - 1)))
    except OSError:
        pass
    return False


def holds_lock(lock):
    try:
        return any(lock in os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd"))
    except OSError:
        return False


def run(cmd, log, timeout):
    if not cmd or cmd == "none":
        return None
    with open(log, "a") as f:
        try:
            return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return 124


def say(log, msg):
    line = f"[hangwatch] {msg}"
    if sys.stdout.isatty():
        print(line, flush=True)
    with open(log, "a") as f:
        f.write(line + "\n")


def handle_hang(a, pid, silent_s, line):
    host = os.uname().nodename.split(".")[0]
    os.makedirs(a.report_dir, exist_ok=True)
    rep = os.path.join(a.report_dir, f"hang_{host}_{pid}_{time.strftime('%H%M')}.txt")
    ev = {
        "time": time.strftime("%FT%T"),
        "host": host,
        "pid": pid,
        "silent_min": round(silent_s / 60, 1),
        "last_line": line[:300],
        "report": rep,
    }
    say(a.log, f"pid {pid} log silent {silent_s/60:.1f} min (last line: {line[:120]!r}) -> stacks, triage -> {rep}")
    if subprocess.run("command -v py-spy", shell=True, capture_output=True).returncode == 0:
        ev["pyspy_rc"] = run(f"py-spy dump --pid {pid} > {rep}.stacks 2>&1", a.log, 60)
    elif has_usr1_handler(pid):
        os.kill(pid, signal.SIGUSR1)
        ev["sigusr1"] = True
        time.sleep(3)  # faulthandler writes to the process's stderr = the log
    else:
        ev["stacks"] = "unavailable (no py-spy, no SIGUSR1 handler registered)"
    ev["triage_rc"] = run(a.triage_cmd.replace("{report}", rep), a.log, 300)
    killed = kill_tree(pid)
    ev["killed"] = killed
    say(a.log, f"killed process tree {killed}")
    time.sleep(2)
    lock = "/tmp/dsv4_dev.lock"
    pre = "" if holds_lock(lock) else f"flock -w 1200 {lock} "
    rc = None
    for cmd in [c for c in a.reset_cmd.split(";;") if c.strip()]:
        rc = run(pre + cmd, a.log, 600) if cmd.strip() != "none" else None
        say(a.log, f"reset ({cmd.strip()}) rc={rc}" if rc is not None else "reset skipped")
        ev.setdefault("reset_rcs", []).append(rc)
        if rc == 0:
            break
    say(a.log, f"reset rc={rc}")
    ev["reset_rc"] = rc
    with open(a.log + ".hang.json", "w") as f:
        json.dump(ev, f, indent=1)
    return 3


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--log", required=True)
    p.add_argument(
        "--stall-min", type=float, default=FLOOR_MIN, help="default stall limit in minutes (>=45 recommended)"
    )
    p.add_argument(
        "--rules", default=DEFAULT_RULES, help="'regex=minutes,...' on the LAST log line selects the stage limit"
    )
    p.add_argument("--stall-seconds", type=float, help="testing: override all limits, in seconds")
    p.add_argument("--poll-s", type=float, default=10)
    p.add_argument("--report-dir", default=os.environ.get("HANGWATCH_REPORT_DIR", "triage"))
    p.add_argument(
        "--triage-cmd",
        default="cd ${TT_METAL_HOME:-.} && timeout 300 python tools/tt-triage.py --skip-version-check --disable-progress --disable-colors --llm-output-path={report}",
    )
    p.add_argument(
        "--reset-cmd", default="tt-smi -glx_reset;;tt-smi -r", help="';;'-separated, tried in order until one returns 0"
    )
    a = p.parse_args(argv)
    if a.stall_seconds is None and a.stall_min < FLOOR_MIN:
        print(
            f"[hangwatch] WARNING: stall {a.stall_min} min < {FLOOR_MIN}: the Engram load alone is ~9 min and builds are silent for many minutes",
            file=sys.stderr,
        )
    rules = parse_rules(a.rules)
    while alive(a.pid):
        try:
            age = time.time() - os.stat(a.log).st_mtime
        except OSError:
            age = 0.0
        line = last_line(a.log)
        limit = a.stall_seconds if a.stall_seconds is not None else stall_limit_min(line, rules, a.stall_min) * 60
        if age >= limit:
            return handle_hang(a, a.pid, age, line)
        time.sleep(a.poll_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
