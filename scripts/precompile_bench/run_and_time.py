#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run a command, capture wall + whole-subtree CPU (user/sys) + peak RSS + return code.

Stand-in for `/usr/bin/time -v` (absent on this host). Uses getrusage(RUSAGE_CHILDREN),
which aggregates user+system CPU over the entire *reaped* descendant tree — so a pytest
run plus every JIT compiler process it forks and waits on is counted. ru_maxrss is the
largest RSS of any single child (peak working set), in KiB on Linux.

Usage: run_and_time.py <timing_out> <log_out> <cmd> [args...]
Writes one line to <timing_out>:  wall=<s> user=<s> sys=<s> maxrss_kb=<n> rc=<n>
The command's stdout+stderr go to <log_out>. Always exits 0 (rc is recorded in the file).
"""
import resource
import subprocess
import sys
import time

timing_out = sys.argv[1]
log_out = sys.argv[2]
cmd = sys.argv[3:]

t0 = time.time()
with open(log_out, "wb") as logf:
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    rc = p.wait()
t1 = time.time()

ru = resource.getrusage(resource.RUSAGE_CHILDREN)
with open(timing_out, "w") as f:
    f.write(f"wall={t1 - t0:.3f} user={ru.ru_utime:.3f} sys={ru.ru_stime:.3f} " f"maxrss_kb={ru.ru_maxrss} rc={rc}\n")
