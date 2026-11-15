# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.30 (owner 2026-11-13): host memory around every device process.

X.3's one-process gate (FIX.29) was SIGKILLed 151 s in (pytest exit 137) by the kernel's OOM killer: the Slurm job's
cgroup has no memory limit (memory.max = max, events max 0) yet counted oom_kill 2, and the safe runner's own kill path
("Caught SIG...") never ran. The node (566 GB) had ~3 GB free with swap full while ~310 GB were held by no process:
kernel / driver memory leaked by earlier killed device processes (tt-kmd 2.11.0, IOMMU on). The framework cannot free
it (a driver fix, infra's), but it must not start a device process into it: this module measures.

- ``snapshot()``: the node's memory now, and ``leaked`` = used (total - free - buffers - page cache - reclaimable
  slab) minus every process's anonymous RSS, hugetlb, unreclaimable slab, page tables and kernel stacks: memory no
  process and no known kernel pool holds.
- ``watch``: samples a process tree (the safe runner's child) every second: its peak RSS, its peak anonymous RSS (what
  it really needs: file-backed pages, the memory-mapped engram tables, are reclaimable) and the lowest MemAvailable
  while it ran; written to a JSON file each sample, so a SIGKILLed process still leaves its record.
- ``preflight``: MemAvailable against the measured need (``BRINGUP_HOST_MEM_NEED_GB``, the gate sets it from the
  records of the same configuration): below it, an INFRA line (free, needed, leaked) and exit 10, never a start.
- ``report``: the run's peak, the memory it left behind, whether the kernel's OOM killer fired (/proc/vmstat oom_kill),
  and its record appended to ``BRINGUP_HOST_MEM_FILE`` under ``BRINGUP_HOST_MEM_KEY``.

Standard library only: scripts/lib/tt_safe_run.sh runs it with ``python3 -I`` (no repo import path).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

GB = 1024**3
NEED_ENV = "BRINGUP_HOST_MEM_NEED_GB"  # the gate: the recorded need of this configuration (GB); unset: FLOOR only
FILE_ENV = "BRINGUP_HOST_MEM_FILE"  # the gate: where each process's record goes (jsonl)
KEY_ENV = "BRINGUP_HOST_MEM_KEY"  # the gate: the configuration a record belongs to (a unit or a shared group)
FLOOR_ENV = "BRINGUP_HOST_MEM_FLOOR_GB"  # the least MemAvailable any device process starts with (default FLOOR_GB)
FLOOR_GB = 16.0
MARGIN = 1.15  # a recorded need is a measurement of earlier runs: start only with 15 % above it, plus MARGIN_GB
MARGIN_GB = 8.0
INFRA_RC = 10  # preflight: not enough host memory (the safe runner exits 3 with a SAFE_PYTEST_INFRA line)
INFRA_TAG = "SAFE_PYTEST_INFRA: host memory"  # the gate and the orchestrator stop on this, never count a failure


def meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    """/proc/meminfo in bytes."""
    out = {}
    for line in Path(path).read_text().splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        if parts:
            out[k.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return out


def _status(pid: int | str) -> dict[str, str]:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return {}
    return {k: v.strip() for k, _, v in (ln.partition(":") for ln in text.splitlines())}


def _kb(st: dict, key: str) -> int:
    v = st.get(key, "0 kB").split()
    return int(v[0]) * 1024 if v else 0


def processes() -> dict[int, dict]:
    """{pid: {ppid, rss, anon}} of every process readable now (bytes)."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        st = _status(d)
        if not st:
            continue
        out[int(d)] = {
            "ppid": int(st.get("PPid", "0") or 0),
            "rss": _kb(st, "VmRSS"),
            "anon": _kb(st, "RssAnon") + _kb(st, "RssShmem"),
            "hwm": _kb(st, "VmHWM"),
        }
    return out


def snapshot(mi: dict | None = None, procs: dict | None = None) -> dict:
    """The node's memory now, in GB: total, free, avail, used, rss / anon (all processes), hugetlb, kernel, leaked."""
    mi = mi if mi is not None else meminfo()
    procs = procs if procs is not None else processes()
    g = lambda k: mi.get(k, 0)  # noqa: E731
    used = g("MemTotal") - g("MemFree") - g("Buffers") - g("Cached") - g("SReclaimable")
    anon = sum(p["anon"] for p in procs.values())
    kernel = g("SUnreclaim") + g("PageTables") + g("KernelStack")
    leaked = max(0, used - anon - g("Hugetlb") - kernel)
    vals = {
        "total": g("MemTotal"),
        "free": g("MemFree"),
        "avail": g("MemAvailable"),
        "used": used,
        "rss": sum(p["rss"] for p in procs.values()),
        "anon": anon,
        "hugetlb": g("Hugetlb"),
        "kernel": kernel,
        "leaked": leaked,
        "swap_used": g("SwapTotal") - g("SwapFree"),
    }
    return {k: round(v / GB, 1) for k, v in vals.items()}


def oom_kills() -> int:
    """The kernel OOM killer's count since boot (/proc/vmstat oom_kill), -1 if unreadable."""
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def describe(s: dict) -> str:
    return (
        f"available {s['avail']:.1f} GB, free {s['free']:.1f} GB, used {s['used']:.1f} GB of {s['total']:.0f} GB "
        f"(processes' anonymous RSS {s['anon']:.1f} GB, hugetlb {s['hugetlb']:.1f} GB, kernel {s['kernel']:.1f} GB, "
        f"leaked {s['leaked']:.1f} GB: used by no process), swap used {s['swap_used']:.1f} GB"
    )


def need_gb(env: dict | None = None) -> float:
    """What a device process needs available to start: the recorded need of its configuration, at least the floor."""
    env = os.environ if env is None else env
    floor = float(env.get(FLOOR_ENV) or FLOOR_GB)
    try:
        need = float(env.get(NEED_ENV) or 0)
    except ValueError:
        need = 0.0
    return max(floor, need)


def preflight(need: float, s: dict | None = None) -> tuple[bool, str]:
    """(ok, line): MemAvailable against ``need`` GB."""
    s = s or snapshot()
    if s["avail"] >= need:
        return True, f"SAFE_PYTEST_HOST_MEM: before: {describe(s)}; this run needs {need:.1f} GB: ok"
    return False, (
        f"{INFRA_TAG}: only {s['avail']:.1f} GB available, this run needs {need:.1f} GB (measured for its "
        f"configuration); {s['leaked']:.1f} GB leaked (used by no process: kernel / driver memory of earlier device "
        f"processes, tt-kmd). Not started (it would be OOM-killed). Free the memory (infra: #exabox-infra; never "
        f"reset or reload the driver from a run), then resume. Now: {describe(s)}"
    )


def tree(root: int, procs: dict) -> list[int]:
    kids: dict[int, list[int]] = {}
    for pid, p in procs.items():
        kids.setdefault(p["ppid"], []).append(pid)
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        if pid in procs:
            out.append(pid)
            todo += kids.get(pid, [])
    return out


def watch(pid: int, out: Path, interval: float = 1.0) -> dict:
    """Sample the process tree of ``pid`` until it ends; the record is rewritten each sample (survives a SIGKILL of
    the watched process: the watcher is not in its tree)."""
    s0 = snapshot()
    rec = {
        "start": s0,
        "oom_kills_before": oom_kills(),
        "peak_rss_gb": 0.0,
        "peak_anon_gb": 0.0,
        "min_avail_gb": s0["avail"],
        "samples": 0,
        "t0": time.time(),
    }
    while True:
        procs = processes()
        if pid not in procs:
            break
        mine = tree(pid, procs)
        rss = sum(procs[p]["rss"] for p in mine) / GB
        anon = sum(procs[p]["anon"] for p in mine) / GB
        hwm = procs[pid]["hwm"] / GB
        avail = meminfo().get("MemAvailable", 0) / GB
        rec["peak_rss_gb"] = round(max(rec["peak_rss_gb"], rss, hwm), 1)
        rec["peak_anon_gb"] = round(max(rec["peak_anon_gb"], anon), 1)
        rec["min_avail_gb"] = round(min(rec["min_avail_gb"], avail), 1)
        rec["samples"] += 1
        rec["s"] = round(time.time() - rec["t0"], 1)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, out)
        time.sleep(interval)
    return rec


def report(watch_file: Path, rc: int, env: dict | None = None) -> tuple[list[str], dict]:
    """The run's host-memory lines for its log and the record kept for its configuration."""
    env = os.environ if env is None else env
    try:
        rec = json.loads(watch_file.read_text())
    except (OSError, ValueError):
        rec = {}
    end = snapshot()
    start = rec.get("start") or end
    kills = oom_kills()
    before = rec.get("oom_kills_before", -1)
    drop = round(max(0.0, start["avail"] - rec.get("min_avail_gb", start["avail"])), 1)
    out = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "key": env.get(KEY_ENV, ""),
        "task": env.get("BRINGUP_TASK", ""),
        "rc": rc,
        "s": rec.get("s", 0.0),
        "peak_rss_gb": rec.get("peak_rss_gb", 0.0),
        "peak_anon_gb": rec.get("peak_anon_gb", 0.0),
        "avail_drop_gb": drop,
        "need_gb": round(max(rec.get("peak_anon_gb", 0.0), drop), 1),
        "start": start,
        "end": end,
        "left_behind_gb": round(max(0.0, end["leaked"] - start["leaked"]), 1),
        "oom_kills": (kills - before) if kills >= 0 and before >= 0 else None,
    }
    lines = [
        f"SAFE_PYTEST_HOST_MEM: this run: peak RSS {out['peak_rss_gb']:.1f} GB (anonymous {out['peak_anon_gb']:.1f} "
        f"GB), MemAvailable fell by up to {drop:.1f} GB while it ran; its need: {out['need_gb']:.1f} GB"
        + (f" (recorded as {out['key']})" if out["key"] else ""),
        f"SAFE_PYTEST_HOST_MEM: after: {describe(end)}; left behind {out['left_behind_gb']:.1f} GB",
    ]
    if out["oom_kills"]:
        lines.append(
            f"SAFE_PYTEST_HOST_MEM: the kernel's OOM killer fired {out['oom_kills']} time(s) during this run "
            "(/proc/vmstat oom_kill)" + (": the test process was SIGKILLed for host memory" if rc == 137 else "")
        )
    elif rc == 137:
        lines.append(
            "SAFE_PYTEST_HOST_MEM: the test process was SIGKILLed (exit 137) without an OOM kill counted "
            "(/proc/vmstat): not this runner (it logs 'Caught SIG...'), look for another killer"
        )
    f = env.get(FILE_ENV)
    if f and rec:
        p = Path(f)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            fh.write(json.dumps(out, sort_keys=True) + "\n")
    return lines, out


def records(path: Path) -> list[dict]:
    try:
        text = path.read_text()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def recorded_need(path: Path, keys: list[str]) -> float | None:
    """The need of a configuration: the largest measured need of its records (its first key that has any), with the
    margin; None when none of ``keys`` was measured. A killed run's record counts (it measured at least that much)."""
    recs = records(path)
    for key in keys:
        got = [float(r.get("need_gb") or 0) for r in recs if r.get("key") == key]
        if got:
            return round(max(got) * MARGIN + MARGIN_GB, 1)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    pf = sub.add_parser("preflight")
    pf.add_argument("--need-gb", type=float, default=None)
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--interval", type=float, default=1.0)
    r = sub.add_parser("report")
    r.add_argument("--watch", type=Path, required=True)
    r.add_argument("--rc", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        print(f"SAFE_PYTEST_HOST_MEM: {describe(snapshot())}")
        return 0
    if a.cmd == "preflight":
        ok, line = preflight(a.need_gb if a.need_gb is not None else need_gb())
        print(line, flush=True)
        return 0 if ok else INFRA_RC
    if a.cmd == "watch":
        watch(a.pid, a.out, a.interval)
        return 0
    lines, _ = report(a.watch, a.rc)
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
