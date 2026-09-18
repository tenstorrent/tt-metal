# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Summarize results/*.log step times (skips first --warmup steps) into one table + CSV."""

import argparse, glob, os, re, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "results"))
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.dir, "*.log"))):
    txt = open(f, errors="ignore").read()
    steps = re.findall(
        r"Step: (\d+), Loss: ([\d.]+), Time: ([\d.]+) ms, TPS: (\d+)(?:, TFLOPS: ([\d.e+]+), MFU: ([\d.e+-]+)%)?", txt
    )
    if not steps:
        continue
    st = [s for s in steps if int(s[0]) > a.warmup]
    if not st:
        continue
    mesh = re.search(r"mesh\s+(\S+)", txt)
    batch = re.search(r"size (\d+) ;", txt)
    naive = {}
    for k, s, e in (
        ("fwd", "dataloader_step_done", "forward_pass_done"),
        ("bwd", "forward_pass_done", "backward_pass_done"),
        ("sync", "backward_pass_done", "gradient_sync_done"),
        ("opt", "gradient_sync_done", "optimizer_step_done"),
    ):
        ts = dict()
        vals = []
        for m in re.finditer(r"\[NAIVE_PROFILER\]\s+(\S+)\s+timestamp_us=(\d+)", txt):
            name, t = m.group(1), int(m.group(2))
            if name.startswith("iteration_"):
                if s in ts and e in ts and int(name.split("_")[1]) > a.warmup:
                    vals.append((ts[e] - ts[s]) / 1e3)
                ts = {}
            else:
                ts[name] = t
        naive[k] = statistics.mean(vals) if vals else float("nan")
    rows.append(
        dict(
            run=os.path.basename(f)[:-4],
            mesh=mesh.group(1) if mesh else "?",
            batch=batch.group(1) if batch else "?",
            steps=len(st),
            step_ms=statistics.mean(float(s[2]) for s in st),
            tps=statistics.mean(float(s[3]) for s in st),
            mfu=statistics.mean(float(s[5]) for s in st if s[5]) if st[0][5] else float("nan"),
            **{f"{k}_ms": v for k, v in naive.items()},
        )
    )
cols = ["run", "mesh", "batch", "steps", "step_ms", "tps", "mfu", "fwd_ms", "bwd_ms", "sync_ms", "opt_ms"]
print("".join(f"{c:>{max(10,len(c)+2) if c!='run' else 28}}" for c in cols))
for r in rows:
    print(
        f"{r['run']:>28}"
        + "".join(f"{(r[c] if not isinstance(r[c], float) else round(r[c],1)):>{max(10,len(c)+2)}}" for c in cols[1:])
    )
with open(os.path.join(a.dir, "summary.csv"), "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows:
        fh.write(",".join(str(r[c]) for c in cols) + "\n")
