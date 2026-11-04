# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experiment probe: per test phase times and runner state, recorded into the JUnit properties."""
import os
import time

import pytest


def _proc_stats():
    procs = zombies = pyth = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
        except OSError:
            continue
        procs += 1
        name, state = s[s.index("(") + 1 : s.rindex(")")], s[s.rindex(")") + 2]
        zombies += state == "Z"
        pyth += name.startswith("python") or name.startswith("pytest")
    return procs, zombies, pyth


def _rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _used_mb(path):
    try:
        st = os.statvfs(path)
        return (st.f_blocks - st.f_bfree) * st.f_frsize // (1 << 20)
    except OSError:
        return -1


def _fds(pid):
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return -1


def _mem_avail_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    from helpers import test_config as tc

    t = {"load": 0.0, "wait": 0.0, "polls": 0, "rt": 0.0, "run": 0.0}
    run0, wait0 = tc.TestConfig.run_elf_files, tc.TestConfig.wait_for_tensix_operations_finished
    rt0, whole0 = tc.TestConfig.write_runtimes_to_L1, tc.TestConfig.run

    def timed(key, fn):
        def wrapper(self, *a, **k):
            s = time.perf_counter()
            try:
                return fn(self, *a, **k)
            finally:
                t[key] += time.perf_counter() - s

        return wrapper
    read0 = tc.read_from_device

    def run_elf_files(self, *a, **k):
        s = time.perf_counter()
        try:
            return run0(self, *a, **k)
        finally:
            t["load"] += time.perf_counter() - s

    def wait(self, *a, **k):
        s = time.perf_counter()
        try:
            return wait0(self, *a, **k)
        finally:
            t["wait"] += time.perf_counter() - s

    def read(*a, **k):
        t["polls"] += 1
        return read0(*a, **k)

    tc.TestConfig.run_elf_files, tc.TestConfig.wait_for_tensix_operations_finished = run_elf_files, wait
    tc.TestConfig.write_runtimes_to_L1, tc.TestConfig.run = timed("rt", rt0), timed("run", whole0)
    tc.read_from_device = read
    procs, zombies, pyth = _proc_stats()
    env0 = (_rss_mb(os.getpid()), _rss_mb(os.getppid()), _used_mb("/dev/shm"), _used_mb("/tmp"), _fds(os.getppid()), _fds(os.getpid()))
    s = time.perf_counter()
    yield
    item.user_properties += [
        ("probe_call", round(time.perf_counter() - s, 3)),
        ("probe_load", round(t["load"], 3)),
        ("probe_wait", round(t["wait"], 3)),
        ("probe_runtimes", round(t["rt"], 3)),
        ("probe_run", round(t["run"], 3)),
        ("probe_reads", t["polls"]),
        ("probe_procs", procs),
        ("probe_zombies", zombies),
        ("probe_python", pyth),
        ("probe_mem_mb", _mem_avail_mb()),
        ("probe_load1", round(os.getloadavg()[0], 1)),
        ("probe_rss_start", env0[0]),
        ("probe_rss_end", _rss_mb(os.getpid())),
        ("probe_parent_rss", env0[1]),
        ("probe_shm_mb", env0[2]),
        ("probe_tmp_mb", env0[3]),
        ("probe_parent_fds", env0[4]),
        ("probe_fds", env0[5]),
    ]
