# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Experiment: how much does the full Tier B search catch beyond the slim checks?"""

import collections
import math
import os
import random
import signal
import sys
import time

from tierb import Pipeline, full_check, load_mgd, need, slim_check

REPO = os.environ["TT_METAL_HOME"].rstrip("/") + "/"
MGD_LIST = os.environ.get("MGD_LIST", "mgd_files.txt")
MAX_MESHES = int(os.environ.get("MAX_MESHES", "8"))  # skip larger fabrics (blitz rings reach 144 meshes)
CASE_TIMEOUT = int(os.environ.get("CASE_TIMEOUT", "3"))
SEED = int(os.environ.get("SEED", "0"))
PER_MGD = int(os.environ.get("PER_MGD", "3"))  # generated pipelines per fabric
random.seed(SEED)
LIMIT = 200_000


def fabrics():
    seen = {}
    for line in open(MGD_LIST):
        path = line.strip()
        try:
            fab = load_mgd(REPO + path)
        except Exception as e:  # noqa: BLE001
            print("skip", path, str(e)[:80], file=sys.stderr)
            continue
        if not fab.meshes or len(fab.meshes) > MAX_MESHES:
            continue
        sig = (tuple(m.key() for m in fab.meshes), tuple(sorted(fab.cap.items())))
        seen.setdefault(sig, (path, fab))
    return list(seen.values())


def stage_shape(mesh, max_slices):
    sl, host = mesh.slice, mesh.host
    opts = []
    for a in range(1, host[0] + 1):
        for b in range(1, (host[1] if len(host) > 1 else 1) + 1):
            if a * b in (1, 2, 4) and a * b <= max_slices:
                opts.append((a, b))
    weights = [4 if a * b == 1 else 2 if a * b == 2 else 1 for a, b in opts]
    a, b = random.choices(opts, weights)[0]
    if len(sl) == 1:
        return (sl[0] * a * b,)
    return (sl[0] * a, sl[1] * b) + tuple(sl[2:])


def gen_pipeline(fab, name, total):
    kind = random.choices(["ring", "chain", "fork"], [6, 2, 2])[0]
    # Target slice count: around the system size, sometimes above.
    target = max(2, int(total * random.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.15])))
    shapes, used = [], 0
    while used < target and len(shapes) < 48:
        mesh = random.choice(fab.meshes)
        sh = stage_shape(mesh, max(1, target - used))
        n = min((math.prod(r) for m in fab.meshes if (r := need(sh, m)) is not None), default=1)
        shapes.append(sh)
        used += n
    if len(shapes) < 2:
        shapes.append(shapes[0])
    n = len(shapes)
    if kind == "ring":
        edges = [(i, (i + 1) % n) for i in range(n)]
    elif kind == "chain":
        edges = [(i, i + 1) for i in range(n - 1)]
    else:
        k = random.randint(2, max(2, min(3, n - 1)))
        branches = [[] for _ in range(k)]
        for i in range(1, n):
            branches[(i - 1) % k].append(i)
        edges = []
        for br in branches:
            if not br:
                continue
            prev = 0
            for s in br:
                edges.append((prev, s))
                prev = s
            if random.random() < 0.6:
                edges.append((prev, 0))  # return to the router
    return Pipeline(name, shapes, edges), kind


def _alarm(*_):
    raise TimeoutError()


def main():
    signal.signal(signal.SIGALRM, _alarm)
    fabs = fabrics()
    print(f"distinct fabrics: {len(fabs)}  (from {sum(1 for _ in open(MGD_LIST))} files)")
    hist = collections.Counter(m.nslices for _, f in fabs for m in f.meshes)
    print("slices per mesh:", dict(sorted(hist.items())))
    rows = []
    for fi, (path, fab) in enumerate(fabs):
        print(f"[{fi+1}/{len(fabs)}] {len(fab.meshes)} meshes  {path.split('/')[-1]}", file=sys.stderr, flush=True)
        total = sum(m.nslices for m in fab.meshes)
        for i in range(PER_MGD):
            npipes = 1 if random.random() < 0.8 else 2
            pipes, kinds = [], []
            for j in range(npipes):
                p, k = gen_pipeline(fab, f"p{j}", total // npipes or 1)
                pipes.append(p)
                kinds.append(k)
            s_ok, s_why = slim_check(fab, pipes)
            t0 = time.perf_counter()
            signal.alarm(CASE_TIMEOUT)
            try:
                verdict, why, nodes = full_check(fab, pipes, LIMIT)
            except TimeoutError:
                verdict, why, nodes = "TIMEOUT", "3s", -1
            signal.alarm(0)
            dt = time.perf_counter() - t0
            cause = ""
            if s_ok and verdict == "REJECT":
                signal.alarm(CASE_TIMEOUT)
                try:
                    v2, _, _ = full_check(fab, pipes, LIMIT, carve=False)
                except TimeoutError:
                    v2 = "TIMEOUT"
                signal.alarm(0)
                cause = "channels/split" if v2 == "REJECT" else "layout-in-mesh"
            rows.append(
                dict(
                    path=path,
                    meshes=len(fab.meshes),
                    slices=total,
                    kinds="+".join(kinds),
                    stages=sum(len(p.shapes) for p in pipes),
                    slim="PASS" if s_ok else "REJECT:" + s_why,
                    full=verdict,
                    cause=cause,
                    nodes=nodes,
                    ms=dt * 1e3,
                )
            )
    n = len(rows)
    slim_rej = sum(r["slim"] != "PASS" for r in rows)
    full_rej = sum(r["full"] == "REJECT" for r in rows)
    undec = sum(r["full"] == "UNDECIDED" for r in rows)
    tmo = [r for r in rows if r["full"] == "TIMEOUT"]
    print(
        f"full TIMEOUT (>3 s in Python): {len(tmo)}  by mesh count:",
        dict(sorted(collections.Counter(r["meshes"] for r in tmo).items())),
        " slim verdicts on them:",
        dict(collections.Counter(r["slim"] for r in tmo)),
    )
    gap = [r for r in rows if r["slim"] == "PASS" and r["full"] == "REJECT"]
    print(f"\ncases: {n}")
    print(
        f"slim rejects: {slim_rej} ({100*slim_rej/n:.1f}%)  by reason:",
        dict(collections.Counter(r["slim"] for r in rows if r["slim"] != "PASS")),
    )
    print(f"full rejects: {full_rej} ({100*full_rej/n:.1f}%)   full undecided (budget {LIMIT}): {undec}")
    print(
        f"GAP (slim pass, full reject): {len(gap)} ({100*len(gap)/n:.1f}% of all, "
        f"{100*len(gap)/max(1,n-slim_rej):.1f}% of slim passes)"
    )
    print("  gap by cause:", dict(collections.Counter(r["cause"] for r in gap)))
    print("  gap by pipeline kind:", dict(collections.Counter(r["kinds"] for r in gap)))
    print("  gap by mesh count:", dict(sorted(collections.Counter(r["meshes"] for r in gap).items())))
    kinds_all = collections.Counter(r["kinds"] for r in rows if r["slim"] == "PASS")
    print("  slim passes by kind:", dict(kinds_all))
    ms = sorted(r["ms"] for r in rows if r["full"] != "TIMEOUT")
    nodes = sorted(r["nodes"] for r in rows if r["full"] != "TIMEOUT")
    small = sorted(r["ms"] for r in rows if r["full"] != "TIMEOUT" and r["meshes"] <= 8)
    pct = lambda a, q: a[min(len(a) - 1, int(q * len(a)))]
    print(
        f"\nfull search time, fabrics with <= 8 meshes, ms: p50 {pct(small,.5):.2f}  p99 {pct(small,.99):.1f}  max {small[-1]:.1f}"
    )
    print(
        f"\nfull search time (Python) ms: p50 {pct(ms,.5):.2f}  p90 {pct(ms,.9):.2f}  "
        f"p99 {pct(ms,.99):.1f}  max {ms[-1]:.1f}"
    )
    print(f"full search nodes: p50 {pct(nodes,.5)}  p99 {pct(nodes,.99)}  max {nodes[-1]}")
    print("\nslowest 5:")
    for r in sorted(rows, key=lambda r: -r["ms"])[:5]:
        print(
            f"  {r['ms']:8.1f} ms  nodes {r['nodes']:7d}  {r['full']:9s} meshes {r['meshes']:3d} "
            f"slices {r['slices']:4d} stages {r['stages']:3d} {r['kinds']:10s} {r['path'].split('/')[-1]}"
        )
    print("\nsample gap cases:")
    for r in gap[:12]:
        print(
            f"  {r['cause']:15s} {r['kinds']:10s} meshes {r['meshes']:3d} slices {r['slices']:4d} "
            f"stages {r['stages']:3d}  {r['path'].split('/')[-1]}"
        )


if __name__ == "__main__":
    main()
