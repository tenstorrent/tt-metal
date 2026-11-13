#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Broker collision gate: finds device-broker jobs that this series did not start.

A tray test's quiet and burst are only valid if nothing else used the device in between. Before every run, this reads
the broker's job list (`tt-device-mcp status N`: RUNNING, QUEUED and RECENT; read-only) and reports every tenant job
(owner icon 👤; the broker's own 🔧 fabric-check/broker rows are not collisions) that is not ours, running now or ended
at or after --since. Ours: a job id in --own (and --own-file), the job this process runs in (its ttdev-job-<id>.scope
cgroup), a command matching --own-re, or a job of the calling user (--user) that has ended: a run started right after
our own earlier one must not wait for it. A running or queued job of the calling user still counts: other
projects may run as the same user. It never cancels or kills a job.

  check  report foreign jobs once: one COLLISION JSON row per job not yet in --seen, appended to --rows and printed.
         Inside a broker job, queued jobs are ignored (they cannot start before ours ends).
         exit 0 none (or only ended jobs already in --seen), 10 foreign job(s) all ended, 11 one still running (or queued, without --ignore-queued),
         2 the broker could not be read.
  wait   check, then poll until no foreign job runs (or is queued, without --ignore-queued) and /health reads healthy,
         unheld, nothing off the bus, for --settle s in a row; new foreign jobs seen meanwhile get rows too.
         exit 0 no foreign job (returns at once), 10 settled after a collision (the caller restarts its quiet from zero),
         12 --max-wait passed, 2 the broker could not be read.
  queued another job waits: exit 0 when a foreign tenant job is queued (run.sh --hold ends its hold then),
         1 none, 2 the broker could not be read. Rows and --seen are not written.

Times in rows are the box's local time. End times of RECENT jobs are estimated as
submit + wait + runtime (the status table has no end column).
env (tests only): BROKER_STATUS_CMD, BROKER_HEALTH_CMD (shell commands printing the status text / health JSON),
BROKER_CGROUP (default /proc/self/cgroup).
"""
import argparse
import json
import os
import pwd
import re
import subprocess
import sys
import time

STATUS_CMD = "tt-device-mcp status 50"
HEALTH_CMD = "curl -s -m 10 --unix-socket /run/tt-device-broker/broker.sock http://x/health"
LINE = re.compile(r"^\s*(\d{1,2}) (\d\d:\d\d:\d\d)\s+(\d+|--)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$")
ACTIVE = {"running", "queued", "started"}


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout


def secs(v):
    m = re.fullmatch(r"([\d.]+)([smh]?)", v)
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)] if m else 0.0


def epoch(day, hms, now):
    """The status table shows day-of-month and local time only: the latest such time not after now (+1 min skew)."""
    t = time.localtime(now)
    y, mo = t.tm_year, t.tm_mon
    for _ in range(2):
        try:
            e = time.mktime(time.strptime(f"{y}-{mo:02d}-{int(day):02d} {hms}", "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            e = None
        if e is not None and e <= now + 60:
            return e
        y, mo = (y - 1, 12) if mo == 1 else (y, mo - 1)
    return now


def local(e):
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(e)) if e is not None else None


def parse(text, now):
    jobs = {}
    for ln in text.splitlines():
        m = LINE.match(ln)
        if not m or m.group(3) == "--":
            continue
        day, hms, jid, icon, owner, rt, wt, status, ex, cmd = m.groups()
        sub = epoch(day, hms, now)
        st = sub + (secs(wt) if wt != "-" else 0)
        end = None if status in ACTIVE else st + (secs(rt) if rt != "-" else 0)
        kill = re.match(r"\[([^\]]*)\]", cmd)
        chips = re.search(r"chips?\s*\[?([\d][\d, ]*)\]?", cmd)
        j = dict(
            job=jid,
            owner=owner,
            tenant="🔧" not in icon,
            status=status,
            exit=ex,
            submit=sub,
            start=st,
            end=end,
            kill_reason=kill.group(1) if kill else None,
            chips=chips.group(1).strip(" ,") if chips else None,
            command=cmd[:200],
        )
        old = jobs.get(jid)  # a job listed twice (RUNNING and RECENT): the active row wins
        if old is None or status in ACTIVE:
            jobs[jid] = j
    return list(jobs.values())


def self_job():
    try:
        with open(os.environ.get("BROKER_CGROUP", "/proc/self/cgroup")) as f:
            m = re.search(r"ttdev-job-(\d+)\.scope", f.read())
            return m.group(1) if m else None
    except OSError:
        return None


def own_ids(a):
    ids = {x for x in re.split(r"[,\s]+", a.own or "") if x}
    if a.own_file and os.path.exists(a.own_file):
        with open(a.own_file) as f:
            ids |= set(f.read().split())
    s = self_job()
    if s:
        ids.add(s)
    return ids


def me():
    """The calling user: under sudo the user who ran sudo, else the uid's name."""
    u = os.environ.get("SUDO_USER") if os.geteuid() == 0 else None
    return u or pwd.getpwuid(os.getuid()).pw_name


def seen_ids(a):
    if a.seen and os.path.exists(a.seen):
        with open(a.seen) as f:
            return set(f.read().split())
    return set()


def foreign(a, now):
    rc, out = sh(os.environ.get("BROKER_STATUS_CMD", STATUS_CMD))
    if rc != 0 or not re.search(r"^(RUNNING|RECENT)", out, re.M):
        return None
    own, rx = own_ids(a), re.compile(a.own_re) if a.own_re else None
    res = []
    for j in parse(out, now):
        if not j["tenant"] or j["job"] in own or (rx and rx.search(j["command"])):
            continue
        if j["owner"] == a.user and j["end"] is not None:
            continue  # our own job that has ended
        if j["status"] == "queued" and a.ignore_queued:
            continue
        if j["end"] is None or j["end"] >= a.since:
            res.append(j)
    return res


def record(a, jobs, now):
    """Row per job not in --seen; returns how many were new."""
    seen, new = seen_ids(a), 0
    for j in jobs:
        if j["job"] in seen:
            continue
        row = dict(
            key=f"{a.key}-collision-j{j['job']}" if a.key else f"collision-j{j['job']}",
            verdict="COLLISION",
            box=os.uname().nodename.split(".")[0],
            job=j["job"],
            owner=j["owner"],
            status=j["status"],
            exit=j["exit"],
            submit_local=local(j["submit"]),
            start_local=local(j["start"]),
            end_local=local(j["end"]) if j["end"] is not None else "running",
            end_estimated=j["end"] is not None,
            kill_reason=j["kill_reason"],
            chips=j["chips"],
            command=j["command"],
            seen_local=local(now),
            since_local=local(a.since),
            note="foreign broker job: quiet restarts from zero; nothing cancelled",
        )
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        if a.rows:
            with open(a.rows, "a") as f:
                f.write(line + "\n")
        if a.seen:
            with open(a.seen, "a") as f:
                f.write(j["job"] + "\n")
        seen.add(j["job"])
        new += 1
    return new


def healthy():
    rc, out = sh(os.environ.get("BROKER_HEALTH_CMD", HEALTH_CMD))
    try:
        h = json.loads(out)
    except ValueError:
        return False, "health unreadable"
    off = h.get("off_bus")
    ok = h.get("fsm_state") == "healthy" and not h.get("held") and not off
    return (
        ok,
        f"fsm={h.get('fsm_state')} held={h.get('held')} off_bus={off} running={h.get('running')} queued={h.get('queued')}",
    )


def busy(jobs, a):
    return any(j["end"] is None and (j["status"] != "queued" or not a.ignore_queued) for j in jobs)


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} collision gate: {msg}", file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["check", "wait", "queued"])
    p.add_argument("--since", type=float, default=None, help="epoch: jobs ended at or after it count (default: now)")
    p.add_argument("--own", default="", help="our broker job ids, comma separated")
    p.add_argument("--own-file", help="file of our job ids (whitespace separated)")
    p.add_argument("--own-re", help="regex: a job whose command matches it is ours")
    p.add_argument("--user", default=None, help="broker owner whose ended jobs are ours (default: the calling user)")
    p.add_argument(
        "--ignore-queued",
        action="store_true",
        help="queued jobs are no collision (we run inside a broker job: they cannot start before we end)",
    )
    p.add_argument("--rows", help="append COLLISION JSON rows here")
    p.add_argument("--seen", help="file of job ids already recorded (no second row)")
    p.add_argument("--key", default="", help="row key prefix")
    p.add_argument("--settle", type=int, default=300, help="s of healthy, unheld, 0 off-bus /health in a row")
    p.add_argument("--poll", type=int, default=30)
    p.add_argument("--max-wait", type=int, default=4 * 3600)
    a = p.parse_args()
    now = time.time()
    a.since = now if a.since is None else a.since
    a.user = a.user or me()
    if a.mode == "queued":  # a queued job waits for our job to end: that is what this mode looks for
        a.ignore_queued = False
        jobs = foreign(a, now)
        if jobs is None:
            log("broker status unreadable")
            return 2
        q = [j for j in jobs if j["status"] == "queued"]
        if q:
            print(", ".join(f"{j['job']} ({j['owner']})" for j in q))
        return 0 if q else 1
    a.ignore_queued = a.ignore_queued or bool(self_job())  # inside a broker job a queued job waits for us
    jobs = foreign(a, now)
    if jobs is None:
        log("broker status unreadable")
        return 2
    hit = record(a, jobs, now) > 0 or busy(jobs, a)  # a job already in --seen that has ended is no new collision
    if a.mode == "check":
        return 11 if busy(jobs, a) else 10 if hit else 0
    if not hit:
        return 0  # no foreign job: nothing to wait for (the caller's own chip and holder checks still run)
    t0, ok_since = now, None
    while True:
        if busy(jobs, a):
            ok_since = None
            log(
                "foreign job(s) active: "
                + ", ".join(f"{j['job']} ({j['owner']}, {j['status']})" for j in jobs if j["end"] is None)
                + "; waiting, nothing cancelled"
            )
        else:
            ok, why = healthy()
            if not ok:
                ok_since = None
                log(f"broker not settled: {why}")
            else:
                ok_since = ok_since or time.time()
                if time.time() - ok_since >= a.settle:
                    if hit:
                        log(f"settled {a.settle} s after the collision ({why}); restart the quiet from zero")
                    return 10 if hit else 0
        if time.time() - t0 > a.max_wait:
            log(f"not settled within {a.max_wait} s")
            return 12
        time.sleep(a.poll)
        jobs = foreign(a, time.time())
        if jobs is None:
            log("broker status unreadable")
            return 2
        hit = record(a, jobs, time.time()) > 0 or hit


if __name__ == "__main__":
    sys.exit(main())
