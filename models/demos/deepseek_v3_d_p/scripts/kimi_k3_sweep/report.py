#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
"""Per-phase throughput from a drive.sh run: report.py <rundir>

A phase is the slice of every rank's timing CSV between the row counts drive.sh recorded before it.
- wall: first rank-0 chunk start -> last rank-3 chunk end (pipeline fill and drain included)
- tokens: users x ISL; the padding of a partial last chunk is not counted
- steady: chunk_size x users_share / last-rank completion interval (fill and drain excluded)
- TTFT: per user, phase start -> that user's last chunk done on rank 3 (round-robin order)
- bottleneck: per-rank mean compute, max over ranks
"""
import statistics
import sys
from pathlib import Path

CHUNK = 5120
RANKS = 4


def load(rundir):
    rows = {}
    for rank in range(RANKS):
        lines = (line.strip().split(",") for line in open(rundir / "timing" / f"rank{rank}.csv"))
        rows[rank] = [(float(p[2]), float(p[3])) for p in lines if len(p) == 4]
    return rows


def main(rundir):
    rundir = Path(rundir)
    rows = load(rundir)
    phases = [line.split() for line in open(rundir / "phases.txt")]
    print(
        "| phase | users | ISL | prefix | chunks | wall s | tok/s | tok/s/user | steady tok/s/user | TTFT mean s | bottleneck ms |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cur, nxt in zip(phases, phases[1:]):
        tag, users, isl, prefix = cur[0], int(cur[2]), int(cur[3]), int(cur[4])
        starts = [int(x) for x in cur[1].split(",")]
        ends = [int(x) for x in nxt[1].split(",")]
        seg = {r: rows[r][starts[r] : ends[r]] for r in range(RANKS)}
        last = seg[RANKS - 1]
        if not seg[0] or not last:
            continue
        t0 = seg[0][0][0]
        done = [start + ms / 1e3 for start, ms in last]
        wall = done[-1] - t0
        tokens = users * isl
        steady = "-"
        if len(done) > 1:
            steady = f"{tokens * (len(done) - 1) / len(done) / (done[-1] - done[0]) / users:,.0f}"
        per_user = -(-isl // CHUNK)
        ttft = "-"
        if len(done) == per_user * users:
            ttft = f"{statistics.mean(done[(per_user - 1) * users + u] - t0 for u in range(users)):.1f}"
        bottleneck = max(statistics.mean(ms for _, ms in seg[r]) for r in range(RANKS))
        print(
            f"| {tag} | {users} | {isl:,} | {prefix:,} | {len(last)} | {wall:.1f} | {tokens / wall:,.0f} | "
            f"{tokens / wall / users:,.0f} | {steady} | {ttft} | {bottleneck:.0f} |"
        )


if __name__ == "__main__":
    main(sys.argv[1])
