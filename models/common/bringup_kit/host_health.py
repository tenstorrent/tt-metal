#!/usr/bin/env python3
"""Pre-flight health of device hosts.

    host_health.py 45 46 47            (last octets of 10.82.97.N, or full addresses)
    host_health.py --all [--json] [--smi] [--canary]

Per host (one ssh round trip, read-only): reachability, /dev/tenstorrent device count, stray python/pytest processes, state of the device
flock (/tmp/dsv4_dev.lock; probed with flock -n, released immediately), load average, free disk on --weight-dir and --log-dir, optional
tt-smi (--smi; it queries the devices, skip it on a host that runs a job), and membership in bad_hosts.txt (next to this file, or --bad-hosts).
Verdict: BAD (in bad_hosts.txt) > UNREACHABLE > BUSY (stray python or lock held) > LOW_DISK > OK. Exit 1 if any requested host is not OK.
--canary (needs hardware validation, see canary.py): on an OK host takes the device lock, runs a tiny deterministic matmul on every chip and
compares each chip's output hash with the others and with canary_ref.json; any mismatch -> verdict CANARY_FAIL. Catches a host like .31.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = "/tmp/dsv4_dev.lock"
DEFAULT_HOSTS = [
    "30",
    "32",
    "33",
    "34",
    "35",
    "40",
    "41",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
]  # 31 and 42 deliberately absent

PROBE = r"""
echo host=$(hostname -s)
echo devices=$(ls /dev/tenstorrent 2>/dev/null | wc -l)
echo stray=$(ps -eo pid,etime,args | grep -E 'python.*(pytest|text_demo|/demo/)|pytest ' | grep -v -E 'grep|host_health|flock|bash -c' | wc -l)
echo stray_cmd=$(ps -eo pid,etime,args | grep -E 'python.*(pytest|text_demo|/demo/)|pytest ' | grep -v -E 'grep|host_health|flock|bash -c' | head -1 | cut -c1-160)
if flock -n __LOCK__ true 2>/dev/null; then echo lock=free; else echo lock=held; fi
echo load=$(cut -d' ' -f1-3 /proc/loadavg)
echo disk_weights_gb=$(df -P -BG __WD__ 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
echo disk_logs_gb=$(df -P -BG __LD__ 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
__SMI__
"""


def parse_probe(text):
    d = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith(" "):
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def load_bad(path):
    bad = {}
    if os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if l and not l.startswith("#"):
                h, _, why = l.partition(" ")
                bad[h.split(".")[-1]] = why.strip()
    return bad


def verdict(h, info, bad, min_disk_gb):
    if h in bad:
        return "BAD", f"in bad_hosts.txt: {bad[h]}"
    if info is None:
        return "UNREACHABLE", "ssh failed"
    why = []
    if info.get("devices", "0") == "0":
        return "BAD", "no /dev/tenstorrent devices"
    if info.get("stray", "0") not in ("0", ""):
        why.append(f"{info['stray']} python/pytest process(es): {info.get('stray_cmd','')}")
    if info.get("lock") == "held":
        why.append("device lock held")
    if why:
        return "BUSY", "; ".join(why)
    for k in ("disk_weights_gb", "disk_logs_gb"):
        try:
            if float(info.get(k, "0") or 0) < min_disk_gb:
                return "LOW_DISK", f"{k}={info.get(k)} < {min_disk_gb} GB"
        except ValueError:
            return "LOW_DISK", f"{k} unreadable"
    return "OK", ""


def probe(host, a):
    addr = host if "." in host else f"10.82.97.{host}"
    smi = "echo smi=$(timeout 60 tt-smi -s 2>&1 | head -c 200 | tr '\\n' ' ')" if a.smi else ""
    script = (
        PROBE.replace("__LOCK__", LOCK)
        .replace("__WD__", a.weight_dir)
        .replace("__LD__", a.log_dir)
        .replace("__SMI__", smi)
    )
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={a.timeout}", addr, "bash -s"],
            input=script,
            text=True,
            capture_output=True,
            timeout=a.timeout + 90,
        )
    except subprocess.TimeoutExpired:
        return None
    return parse_probe(r.stdout) if r.returncode == 0 else None


def run_canary(host, a):
    addr = host if "." in host else f"10.82.97.{host}"
    cmd = f"flock -n {LOCK} bash -c 'cd {a.worktree} && source python_env/bin/activate && TT_METAL_HOME=$PWD PYTHONPATH=$PWD python3 {os.path.join(HERE,'canary.py')}' 2>&1 | tail -5"
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", addr, cmd], capture_output=True, text=True, timeout=900)
    return ("CANARY_OK" in r.stdout), r.stdout.strip()[-300:]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hosts", nargs="*")
    p.add_argument("--all", action="store_true", help=f"default list {DEFAULT_HOSTS}")
    p.add_argument("--bad-hosts", default=os.path.join(HERE, "bad_hosts.txt"))
    p.add_argument("--weight-dir", default="/mnt/tt-data/ssinghal/tt-metal-cache")
    p.add_argument("--log-dir", default="/mnt/tt-data/ssinghal/dsv4-logs")
    p.add_argument("--min-disk-gb", type=float, default=50)
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--smi", action="store_true")
    p.add_argument("--canary", action="store_true")
    p.add_argument("--worktree", default=os.getcwd(), help="tree with python_env for the canary")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    hosts = [
        h.split(".")[-1] if h.startswith("10.82.97.") else h for h in (a.hosts or (DEFAULT_HOSTS if a.all else []))
    ]
    if not hosts:
        p.error("give hosts or --all")
    bad = load_bad(a.bad_hosts)
    rows, ok_all = [], True
    for h in hosts:
        info = None if h in bad else probe(h, a)
        v, why = verdict(h, info, bad, a.min_disk_gb)
        if a.canary and v == "OK":
            good, out = run_canary(h, a)
            if not good:
                v, why = "CANARY_FAIL", out
        rows.append({**(info or {}), "host": h, "verdict": v, "why": why})
        ok_all &= v == "OK"
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(f"{'host':<5} {'verdict':<12} dev stray lock  load             wgtGB logGB  note")
        for r in rows:
            print(
                f"{r['host']:<5} {r['verdict']:<12} {r.get('devices','-'):<3} {r.get('stray','-'):<5} {r.get('lock','-'):<5} {r.get('load','-'):<16} {r.get('disk_weights_gb','-'):<5} {r.get('disk_logs_gb','-'):<5}  {r['why']}"
            )
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
