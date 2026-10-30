# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Follow-ups: designed pipelines on every MGD, and a cheap degree check against the gap."""

import collections
import math
import random
import signal
import sys
import time

import run
from tierb import Pipeline, full_check, need, slim_check


def _alarm(*_):
    raise TimeoutError()


def degree_check(fab, pipes):
    """A stage that fills every mesh it can bind to sits alone, so each distinct neighbour
    needs its own connected mesh: deg(stage) <= max over its meshes of connected-mesh count."""
    adj = collections.Counter()
    for (a, b), c in fab.cap.items():
        if c > 0:
            adj[a] += 1
            adj[b] += 1
    for p in pipes:
        nb = collections.defaultdict(set)
        for a, b in p.edges:
            if a != b:
                nb[a].add(b)
                nb[b].add(a)
        for s, sh in enumerate(p.shapes):
            ms = [(m, math.prod(r)) for m in fab.meshes if (r := need(sh, m)) is not None]
            if ms and all(n == m.nslices for m, n in ms):
                if len(nb[s]) > max(adj[m.idx] for m, _ in ms):
                    return False
    return True


def timed(fn, secs=run.CASE_TIMEOUT):
    signal.alarm(secs)
    t0 = time.perf_counter()
    try:
        r = fn()
    except TimeoutError:
        r = ("TIMEOUT", "", -1)
    signal.alarm(0)
    return r, (time.perf_counter() - t0) * 1e3


def main():
    signal.signal(signal.SIGALRM, _alarm)
    fabs = run.fabrics()

    print("== Designed pipeline: ring of one 1-slice stage per slice (as blaze pads) ==")
    res = collections.Counter()
    slow = []
    for path, fab in fabs:
        stages = []
        for m in fab.meshes:
            stages += [m.slice] * m.nslices
        n = len(stages)
        if n < 2:
            continue
        p = Pipeline("ring", stages, [(i, (i + 1) % n) for i in range(n)])
        s_ok, _ = slim_check(fab, [p])
        (v, _, nodes), ms = timed(lambda: full_check(fab, [p], 200_000))
        res[(("slim PASS" if s_ok else "slim REJECT"), v)] += 1
        slow.append((ms, v, len(fab.meshes), n, path.split("/")[-1]))
    for k, c in sorted(res.items()):
        print(f"  {k[0]:12s} full {k[1]:9s}: {c}")
    print("  slowest:")
    for row in sorted(slow, reverse=True)[:6]:
        print(f"    {row[0]:8.1f} ms  {row[1]:9s} meshes {row[2]:3d} stages {row[3]:3d}  {row[4]}")
    rej = [r for r in slow if r[1] == "REJECT"]
    print("  full REJECT on designed ring:", [r[4] for r in rej][:15])

    print("\n== Random pipelines on fabrics with <= 8 meshes: does a degree check close the gap? ==")
    random.seed(1)
    small = [(p, f) for p, f in fabs if len(f.meshes) <= 8]
    tot = gap = gap_deg = 0
    causes = collections.Counter()
    for path, fab in small:
        total = sum(m.nslices for m in fab.meshes)
        for _ in range(2 * run.PER_MGD):
            p, kind = run.gen_pipeline(fab, "p0", total)
            s_ok, _ = slim_check(fab, [p])
            if not s_ok:
                continue
            tot += 1
            (v, _, _), _ = timed(lambda: full_check(fab, [p], 200_000))
            if v != "REJECT":
                continue
            gap += 1
            (v2, _, _), _ = timed(lambda: full_check(fab, [p], 200_000, carve=False))
            cause = "channels/split" if v2 == "REJECT" else "layout-in-mesh"
            if not degree_check(fab, [p]):
                gap_deg += 1
                causes[(cause, "caught by degree")] += 1
            else:
                causes[(cause, "missed")] += 1
    print(
        f"  slim passes: {tot}   gap: {gap} ({100*gap/max(1,tot):.1f}%)   "
        f"caught by degree check: {gap_deg} ({100*gap_deg/max(1,gap):.1f}% of gap)"
    )
    for k, c in sorted(causes.items()):
        print(f"    {k[0]:15s} {k[1]:18s}: {c}")


if __name__ == "__main__":
    main()
