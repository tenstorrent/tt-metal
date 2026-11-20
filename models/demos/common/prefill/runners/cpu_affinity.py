# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Give each runner rank its own physical cores, away from busy-polled CPUs.

tt-run's rankfile binds each rank to one slot, i.e. one physical core (2 CPUs with SMT), so the main thread, dispatch,
one worker thread per device and the real-time profiler all share it. On Weka hosts that core can be one wekanode
busy-polls (e.g. CPUs 1-2,33-34), which a container cannot see as a cgroup or process.

pin_runner_cpus(), called before the device opens (tt-metal pins its device workers to the CPUs the main thread may
run on), moves this process's threads to:
- this rank's slice of the pool's physical cores, split evenly between the ranks on this host
  (OMPI_COMM_WORLD_LOCAL_RANK/_SIZE). The pool is the same for every rank, so the slices never overlap: the CPUs of
  this cgroup (else the online CPUs), or the CPU list in PREFILL_RUNNER_CPUS;
- minus the slice's CPUs busy for >= 90% of a short sample, and their SMT siblings (auto mode only).
If the slice is empty or all busy, or the kernel refuses it for the main thread, the launcher's binding is kept (it may
overlap other ranks' slices); PREFILL_RUNNER_CPUS=off keeps it too. Without OMPI_COMM_WORLD_LOCAL_* the process is
rank 0 of 1 and takes the whole pool.
"""

import os
import time

_BUSY_FRACTION = 0.9
_SAMPLE_S = 0.25


def parse_cpu_list(text):
    """'0,3-5' -> {0, 3, 4, 5}."""
    cpus = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _host_cpus():
    """CPUs of this process's cgroup (v2: the nearest cgroup up the path with a cpuset), else the online CPUs."""
    group = ""
    for line in (_read("/proc/self/cgroup") or "").splitlines():
        if line.startswith("0::"):
            group = line[3:].strip("/")
    parts = group.split("/") if group else []
    for depth in range(len(parts), -1, -1):
        text = _read(os.path.join("/sys/fs/cgroup", *parts[:depth], "cpuset.cpus.effective"))
        if text:
            return parse_cpu_list(text)
    text = _read("/sys/devices/system/cpu/online")
    return parse_cpu_list(text) if text else set(range(os.cpu_count() or 1))


def _cpu_times():
    """{cpu: (busy, total)} jiffies from /proc/stat."""
    times = {}
    with open("/proc/stat") as f:
        for line in f:
            name, *fields = line.split()
            if name.startswith("cpu") and name != "cpu":
                values = [int(v) for v in fields[:8]]
                idle = values[3] + values[4]  # idle + iowait
                times[int(name[3:])] = (sum(values) - idle, sum(values))
    return times


def busy_cpus(sample_s=_SAMPLE_S, fraction=_BUSY_FRACTION):
    """CPUs busy for at least `fraction` of a `sample_s` window (the calling thread sleeps through it)."""
    try:
        before = _cpu_times()
        time.sleep(sample_s)
        after = _cpu_times()
    except (OSError, ValueError):
        return set()
    busy = set()
    for cpu, (b1, t1) in after.items():
        b0, t0 = before.get(cpu, (b1, t1))
        if t1 > t0 and (b1 - b0) / (t1 - t0) >= fraction:
            busy.add(cpu)
    return busy


def _core(cpu):
    """The CPUs of `cpu`'s physical core (its SMT siblings), or just `cpu`."""
    text = _read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    try:
        return frozenset(parse_cpu_list(text)) if text else frozenset({cpu})
    except ValueError:
        return frozenset({cpu})


def _rank_slice(pool, rank, size):
    """This rank's contiguous slice of the pool's physical cores, SMT siblings kept together."""
    cores = sorted({_core(cpu) & pool for cpu in pool}, key=min)
    return {cpu for core in cores[rank * len(cores) // size : (rank + 1) * len(cores) // size] for cpu in core}


def _pin_all_threads(cpus):
    """Pins the main thread (raises OSError if refused), then every other thread until no new one appears.

    Returns the number of other threads the kernel refused."""
    os.sched_setaffinity(0, cpus)
    refused = 0
    done = {os.getpid()}
    while True:
        try:
            tids = {int(t) for t in os.listdir("/proc/self/task")} - done
        except OSError:
            return refused
        if not tids:
            return refused
        for tid in tids:
            try:
                os.sched_setaffinity(tid, cpus)
            except ProcessLookupError:
                pass
            except OSError:
                refused += 1
        done |= tids


def pin_runner_cpus():
    """Pin this process's threads to its rank's CPUs; returns a one-line status."""
    setting = os.environ.get("PREFILL_RUNNER_CPUS", "auto").strip()
    kept = f"kept launcher binding ({_ranges(os.sched_getaffinity(0))})"
    if setting == "off":
        return kept
    try:
        pool = _host_cpus() if setting == "auto" else parse_cpu_list(setting)
        rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        size = max(int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "1")), 1)
    except ValueError as e:
        return f"{kept}: bad setting ({e})"
    share = _rank_slice(pool, rank, size)
    note = ""
    if setting == "auto" and share:
        avoided = {cpu for busy in busy_cpus() for cpu in _core(busy)} & share
        if avoided:
            share -= avoided
            note = f", avoiding busy {_ranges(avoided)}"
    if not share:
        return f"{kept}: no free cores for local rank {rank}/{size}"
    try:
        refused = _pin_all_threads(share)
    except OSError as e:
        return f"{kept}: pinning to {_ranges(share)} failed ({e.strerror})"
    if refused:
        note += f", {refused} threads refused"
    return f"local rank {rank}/{size}: {len(share)} CPUs {_ranges(share)}{note}"


def _ranges(cpus):
    out, run = [], []
    for cpu in sorted(cpus):
        if run and cpu != run[-1] + 1:
            out.append(run)
            run = []
        run.append(cpu)
    if run:
        out.append(run)
    return ",".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in out)
