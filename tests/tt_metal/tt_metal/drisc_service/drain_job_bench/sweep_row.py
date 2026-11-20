# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One row of sweep.sh: the zones run's drainer stalls, and the copy load's GB/s and 64 B probe latency in the window
before the profiler attached and in the profiler's workload window (from the zones log's timestamps)."""
import re, sys, os

tag, zf, lf = sys.argv[1:4]
active = done = None
wl_ms = None
stalls = 0
missing = 0
fatal = 0
for line in open(zf, errors="replace"):
    ts, _, rest = line.partition(" ")
    t = int(ts) / 1e9 if ts.isdigit() else None
    if "[streaming profiler] active" in rest:
        active = t
    m = re.search(r"workload done in ([0-9.]+) ms", rest)
    if m:
        wl_ms = float(m.group(1))
        done = t
    m = re.search(r"Device 0: (\d+) profiler stalls", rest)
    if m:
        stalls = int(m.group(1))
    if "not in the capture" in rest:
        missing += 1
    if "TT_FATAL" in rest or "TT_THROW" in rest:
        fatal += 1


def window(lo, hi):
    rows = []
    if not os.path.exists(lf):
        return None
    for l in open(lf):
        p = l.split()
        t = int(p[0]) / 1e9
        if lo <= t - 0.05 and t <= hi:
            rows.append((float(p[1]), float(p[2]), float(p[3])))
    if not rows:
        return None
    return (
        sum(r[0] for r in rows) / len(rows),
        sorted(r[1] for r in rows)[len(rows) // 2],
        max(r[2] for r in rows),
        len(rows),
    )


fmt = lambda w: "-" if w is None else f"{w[0]:6.1f}GB/s p50 {w[1]:6.1f}us max {w[2]:7.1f}us ({w[3]})"
pre = att = wl = None
if done and wl_ms and active:
    start = done - wl_ms / 1000
    pre = window(0, active - 1.0)
    att = window(active, start - 0.05)
    wl = window(start, done + 0.05)
print(
    f"{tag:22s} wl={wl_ms}ms stalls={stalls} missing={missing} fatal={fatal} | before: {fmt(pre)} | attached: {fmt(att)} | workload: {fmt(wl)}",
    flush=True,
)
