#!/usr/bin/env python3
"""One-screen coordinator status (generalizes /mnt/tt-data/ssinghal/stallcheck.sh). Read-only.

    status_report.py [--hosts 45 46 ..|--all] [--log-dir DIR] [--logs bringup/m/LOG.md ...] [--branches ssinghal/a ssinghal/b] [--repo .]

Sections: (1) per host: each running pytest/demo job (pid, age, CPU%, the log file it writes to, minutes that log has been silent) and
a SILENT flag when the log is quiet for > --silent-min (default 20; a build or Engram load can legitimately be quiet ~10 min, hangwatch uses 45);
(2) build-slot occupancy (probes the slot lock files with a non-blocking flock and releases at once; a waiting builder may retry 10 s later);
(3) the 5 newest entries of each LOG.md with their FRESH/STALE state; (4) tip commit of each branch. Hosts are probed in parallel.
"""
import argparse
import concurrent.futures as cf
import fcntl
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host_health  # noqa: E402
import kitlib  # noqa: E402
import log_tool  # noqa: E402

REMOTE = r"""
now=$(date +%s)
ps -eo pid,etimes,pcpu,args | grep -E 'python.*(pytest|text_demo|/demo/)|[^a-z]pytest ' | grep -v -E 'grep|flock|bash -c|timeout|status_report' | while read pid et cpu rest; do
  log=$(ls -l /proc/$pid/fd 2>/dev/null | grep -E '\.log' | awk '{print $NF}' | head -1)
  [ -z "$log" ] && log=$(cat /proc/$(pgrep -P $pid | head -1)/environ 2>/dev/null | tr '\0' '\n' | grep -m1 '^GRID_LOG=' | cut -d= -f2)
  if [ -n "$log" ] && [ -e "$log" ]; then silent=$(( (now - $(stat -c %Y "$log")) / 60 )); else silent=-1; fi
  echo "JOB|$pid|$et|$cpu|$log|$silent|$(echo $rest | cut -c1-110)"
done
echo "LOCK|$(flock -n /tmp/dsv4_dev.lock true 2>/dev/null && echo free || echo held)"
"""


def parse_jobs(text):
    jobs, lock = [], None
    for l in text.splitlines():
        if l.startswith("JOB|"):
            _, pid, et, cpu, log, silent, cmd = (l.split("|", 6) + [""] * 7)[:7]
            jobs.append(
                {
                    "pid": pid,
                    "age_min": int(et or 0) // 60,
                    "cpu": cpu,
                    "log": log,
                    "silent_min": int(silent),
                    "cmd": cmd,
                }
            )
        elif l.startswith("LOCK|"):
            lock = l.split("|")[1]
    return jobs, lock


def host_status(h):
    addr = h if "." in h else f"10.82.97.{h}"
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", addr, "bash -s"],
            input=REMOTE,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return h, None, None
    return (h, *parse_jobs(r.stdout)) if r.returncode == 0 else (h, None, None)


def slot_occupancy(d, n=10):
    """-> (busy, total) by probing slot<i>.lock files; None if the dir is absent."""
    if not os.path.isdir(d):
        return None
    files = sorted(glob.glob(os.path.join(d, "slot*.lock")))
    busy = 0
    for f in files:
        fd = os.open(f, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            busy += 1
        finally:
            os.close(fd)
    return busy, len(files)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hosts", nargs="*", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--silent-min", type=int, default=20)
    p.add_argument("--slot-dir", default="/mnt/tt-data/ssinghal/buildslots")
    p.add_argument("--logs", nargs="*", default=[], help="LOG.md files")
    p.add_argument("--branches", nargs="*", default=[])
    p.add_argument("--repo", default=".")
    a = p.parse_args(argv)
    hosts = a.hosts or (host_health.DEFAULT_HOSTS if a.all else [])
    bad = host_health.load_bad(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bad_hosts.txt"))
    print(f"=== status {time.strftime('%F %T')} ===")
    if hosts:
        with cf.ThreadPoolExecutor(8) as ex:
            res = list(ex.map(host_status, hosts))
        for h, jobs, lock in res:
            if jobs is None:
                print(f".{h:<3} UNREACHABLE" + ("  (known bad)" if h in bad else ""))
                continue
            if not jobs:
                print(f".{h:<3} idle" + ("  lock HELD" if lock == "held" else ""))
            for j in jobs:
                flag = "  <-- SILENT" if j["silent_min"] >= a.silent_min else ""
                print(
                    f".{h:<3} pid {j['pid']:<7} age {j['age_min']:>4}m cpu {j['cpu']:>5}% silent {j['silent_min']:>4}m  {os.path.basename(j['log']) or '(log unknown)'}  {j['cmd'][:60]}{flag}"
                )
    occ = slot_occupancy(a.slot_dir)
    print(f"build slots: {'n/a' if occ is None else f'{occ[0]}/{occ[1]} busy'}  ({a.slot_dir})")
    for lg in a.logs:
        entries = log_tool.load(lg)
        print(f"--- {lg}: last {min(5, len(entries))} of {len(entries)} entries")
        for e in entries[-5:]:
            s, why = log_tool.entry_state(e, a.repo, entries)
            print(f"  {e['id']} {s:<10} {e.get('status','?'):<8} {e.get('track','-'):<10} {e.get('claim','')[:80]}")
    for b in a.branches:
        tip = kitlib.git(a.repo, "log", "-1", "--format=%h %cd %s", "--date=format:%m-%d %H:%M", b, check=False)
        print(f"branch {b}: {tip[:100] if tip else 'NOT FOUND'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
