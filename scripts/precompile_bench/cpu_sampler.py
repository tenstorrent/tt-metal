#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight /proc-based CPU/utilization sampler for the precompile benchmark.

Samples a process *subtree* (everything descended from a given root PID — i.e. the
benchmark orchestrator) every `interval` seconds and attributes the CPU-time burned
in each tick to one of three buckets by process name:

  * compiler  — the kernel JIT toolchain (riscv-tt-elf-g++/gcc/as/ld, cc1plus, ccache,
                sfpi, collect2, lto-wrapper). This is "CPU that went to JIT".
  * python    — the pytest / ttnn host process(es).
  * other     — anything else in the subtree.

Why /proc and not psutil-per-process: kernel JIT spawns a swarm of *short-lived*
compiler processes; walking /proc and differencing cumulative utime+stime catches
any process seen in at least one tick, and the per-tick CPU-seconds integrate cleanly
per phase in the summarizer (segmented by the marks file). The authoritative total
CPU per phase is still `bash time` (reaps every child); this sampler adds the
*shape over time* and the compiler-vs-python *split* that `time` can't give.

CSV columns (one row per tick):
  epoch,dt,sys_util_pct,sys_cores_busy,tree_cpu_s,compiler_cpu_s,python_cpu_s,other_cpu_s,nproc_tree,tree_rss_gb

  epoch        wall clock (float seconds)
  dt           seconds since previous tick (the integration width)
  sys_util_pct system-wide busy% from /proc/stat (0..100, all host cores) — contention signal
  sys_cores_busy  sys_util_pct/100 * host-cpu-count
  *_cpu_s      CPU-seconds consumed *in this tick* by the subtree / bucket
  nproc_tree   number of live processes in the subtree this tick
  tree_rss_gb  resident memory of the subtree (GiB)

Usage: cpu_sampler.py <out_csv> <root_pid> [interval_seconds=0.25]
Stops on SIGTERM/SIGINT (flushes the file).
"""
import os
import sys
import time
import signal

CLK = os.sysconf("SC_CLK_TCK")  # jiffies per second (usually 100)
PAGESIZE = os.sysconf("SC_PAGE_SIZE")

COMPILER_HINTS = ("riscv", "cc1", "ccache", "sfpi", "collect2", "lto", "g++", "gcc")
PYTHON_HINTS = ("python", "pytest")

_running = True


def _stop(*_a):
    global _running
    _running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def read_proc_stat_idle_total():
    """Return (idle+iowait, total) jiffies from /proc/stat aggregate cpu line."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        return idle, sum(vals)
    except Exception:
        return 0, 0


def scan_procs():
    """Return {pid: (ppid, comm, utime+stime jiffies, rss_pages)} for all live procs."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                data = f.read()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        except Exception:
            continue
        # robust parse: comm is between first '(' and last ')'
        try:
            lp = data.index("(")
            rp = data.rindex(")")
            comm = data[lp + 1 : rp]
            rest = data[rp + 2 :].split()
            # rest[0]=state(3) ... ppid is field4 -> rest[1]; utime f14->rest[11]; stime f15->rest[12]
            ppid = int(rest[1])
            cpu = int(rest[11]) + int(rest[12])
            rss = int(rest[21])  # field 24 (rss in pages) -> rest index 24-3=21
            out[int(name)] = (ppid, comm, cpu, rss)
        except Exception:
            continue
    return out


def subtree_pids(procs, root):
    """All pids descended from root (inclusive of root's children; root itself excluded)."""
    children = {}
    for pid, (ppid, *_rest) in procs.items():
        children.setdefault(ppid, []).append(pid)
    seen = set()
    stack = list(children.get(root, []))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(children.get(p, []))
    return seen


def bucket(comm):
    c = comm.lower()
    for h in COMPILER_HINTS:
        if h in c:
            return "compiler"
    for h in PYTHON_HINTS:
        if h in c:
            return "python"
    return "other"


def main():
    out_path = sys.argv[1]
    root = int(sys.argv[2])
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25

    prev_cpu = {}  # pid -> last cumulative cpu jiffies
    prev_idle, prev_total = read_proc_stat_idle_total()
    host_ncpu = os.cpu_count() or 1
    last_t = time.time()

    with open(out_path, "w") as f:
        f.write(
            "epoch,dt,sys_util_pct,sys_cores_busy,tree_cpu_s,compiler_cpu_s,"
            "python_cpu_s,other_cpu_s,nproc_tree,tree_rss_gb\n"
        )
        f.flush()
        while _running:
            time.sleep(interval)
            now = time.time()
            dt = now - last_t
            last_t = now

            procs = scan_procs()
            kids = subtree_pids(procs, root)

            idle, total = read_proc_stat_idle_total()
            d_idle = idle - prev_idle
            d_total = total - prev_total
            prev_idle, prev_total = idle, total
            sys_util = 100.0 * (1.0 - d_idle / d_total) if d_total > 0 else 0.0
            sys_cores = sys_util / 100.0 * host_ncpu

            buckets = {"compiler": 0, "python": 0, "other": 0}
            rss_pages = 0
            cur_cpu = {}
            for pid in kids:
                ppid, comm, cpu, rss = procs[pid]
                cur_cpu[pid] = cpu
                # First sighting: attribute the process's CPU-so-far to THIS interval. Kernel JIT
                # spawns swarms of short-lived gcc processes that are born and die between samples;
                # seeding the baseline at `cpu` (delta 0) would drop all of their CPU. Seeding at 0
                # captures it (slight over-attribution of a long-lived proc's pre-history to its first
                # tick, negligible for the short compiler procs this is meant to catch).
                d = cpu - prev_cpu.get(pid, 0)
                if d < 0:
                    d = 0
                buckets[bucket(comm)] += d
                rss_pages += rss
            prev_cpu = cur_cpu

            tree_cpu_s = sum(buckets.values()) / CLK
            comp_s = buckets["compiler"] / CLK
            py_s = buckets["python"] / CLK
            oth_s = buckets["other"] / CLK
            rss_gb = rss_pages * PAGESIZE / (1024**3)

            f.write(
                f"{now:.3f},{dt:.3f},{sys_util:.1f},{sys_cores:.2f},"
                f"{tree_cpu_s:.3f},{comp_s:.3f},{py_s:.3f},{oth_s:.3f},"
                f"{len(kids)},{rss_gb:.3f}\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
