# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-step device-time analysis of Tracy ``ops_perf_results*.csv`` files from ``train.py`` runs at several depths.

For each depth: steady-state steps (after ``--warmup``), device time per step, op count per step, per-op-class and
per-op-code totals. Across depths: least-squares fit ``t(L) = fixed + L * per_layer`` with residuals, projection to
``--full-depth`` layers, and the ranked cost table at projected full depth with the op class of every row.

    python analyze_tracy.py --full-depth 40 --steps 4 --warmup 2 1:dir1/ops.csv 2:dir2/ops.csv 4:... 8:...
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys

csv.field_size_limit(10**9)

# Op class by op code (doc/plan.md section 3 and the optimisation-loop skill): useful math, layout, glue, wasted.
LAYOUT = (
    "createheads",
    "createqkvheads",
    "concatheads",
    "transpose",
    "permute",
    "reshape",
    "concat",
    "slice",
    "untilize",
    "tilize",
    "typecast",
    "repeat",
    "pad",
)
GLUE = ("binary", "unary", "multiply", "add", "subtract", "fill", "zeros", "full", "morehmean")
MATH = ("matmul", "sdpa", "rmsnorm", "swiglu", "crossentropy", "softmax", "rotaryembedding", "adamw", "embedding")


def classify(op_code: str) -> str:
    code = op_code.lower()
    if any(k in code for k in MATH):
        return "math"
    if any(k in code for k in LAYOUT):
        return "layout"
    if any(k in code for k in GLUE):
        return "glue"
    return "other"


def shape(row, prefix):
    dims = []
    for d in ("W", "Z", "Y", "X"):
        v = row.get(f"{prefix}_{d}_PAD[LOGICAL]", "")
        m = re.match(r"\s*(\d+)\s*(?:\[(\d+)\])?", v)
        if m:
            dims.append(m.group(1) + (f"[{m.group(2)}]" if m.group(2) and m.group(2) != m.group(1) else ""))
    return "x".join(dims)


def load(path: str, device: str = "0"):
    rows = [r for r in csv.DictReader(open(path)) if r["DEVICE ID"] == device]
    rows.sort(key=lambda r: int(r.get("GLOBAL CALL COUNT", 0) or 0))
    return rows


def split_steps(rows):
    """Steps are delimited by the optimizer: the last AdamW op of a step closes it."""
    steps, cur = [], []
    for i, r in enumerate(rows):
        cur.append(r)
        nxt = rows[i + 1]["OP CODE"] if i + 1 < len(rows) else ""
        if "AdamW" in r["OP CODE"] and "AdamW" not in nxt:
            steps.append(cur)
            cur = []
    if cur:
        steps.append(cur)
    return steps


def summarize(path, warmup, nsteps):
    rows = load(path)
    steps = split_steps(rows)
    use = steps[warmup : warmup + nsteps] if nsteps else steps[warmup:]
    if not use:
        sys.exit(f"{path}: {len(steps)} steps found, none after warmup {warmup}")
    per_step_us = [sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in s) / 1e3 for s in use]
    n = len(use)
    by_code = collections.defaultdict(lambda: [0.0, 0])
    by_class = collections.defaultdict(float)
    by_item = collections.defaultdict(lambda: [0.0, 0])
    for s in use:
        for r in s:
            us = float(r["DEVICE KERNEL DURATION [ns]"]) / 1e3
            code = r["OP CODE"]
            by_code[code][0] += us / n
            by_code[code][1] += 1 / n
            by_class[classify(code)] += us / n
            key = (code, shape(r, "INPUT_0"), shape(r, "INPUT_1"), r.get("CORE COUNT", ""))
            by_item[key][0] += us / n
            by_item[key][1] += 1 / n
    return {
        "steps_found": len(steps),
        "steps_used": n,
        "ops_per_step": sum(len(s) for s in use) / n,
        "device_ms_per_step": sum(per_step_us) / n / 1e3,
        "per_step_ms": [u / 1e3 for u in per_step_us],
        "by_code": dict(by_code),
        "by_class": dict(by_class),
        "by_item": {" | ".join(k): v for k, v in by_item.items()},
    }


def fit(points):
    """Least squares t = fixed + L * per_layer over (L, t)."""
    n = len(points)
    sx = sum(l for l, _ in points)
    sy = sum(t for _, t in points)
    sxx = sum(l * l for l, _ in points)
    sxy = sum(l * t for l, t in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return intercept, slope


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="depth:csv")
    p.add_argument("--full-depth", type=int, default=40)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--json", default=None)
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()

    runs = {}
    for spec in args.runs:
        depth, path = spec.split(":", 1)
        runs[int(depth)] = summarize(path, args.warmup, args.steps)
    depths = sorted(runs)
    print("depth | steps used | ops/step | device ms/step | per-step ms")
    for d in depths:
        s = runs[d]
        print(
            f"{d:5d} | {s['steps_used']:4d} | {s['ops_per_step']:8.0f} | {s['device_ms_per_step']:9.2f} | "
            + ", ".join(f"{x:.2f}" for x in s["per_step_ms"])
        )
    fixed, per_layer = fit([(d, runs[d]["device_ms_per_step"]) for d in depths])
    print(f"\nfit: device ms = {fixed:.2f} + L x {per_layer:.3f}")
    for d in depths:
        pred = fixed + d * per_layer
        res = runs[d]["device_ms_per_step"] - pred
        print(
            f"  L={d:2d}: measured {runs[d]['device_ms_per_step']:.2f}, fit {pred:.2f}, residual {res:+.2f} ms ({100 * res / runs[d]['device_ms_per_step']:+.1f} %)"
        )
    proj = fixed + args.full_depth * per_layer
    print(f"projection to {args.full_depth} layers: {proj:.1f} ms device per step")
    ops_fixed, ops_layer = fit([(d, runs[d]["ops_per_step"]) for d in depths])
    print(
        f"ops per step: {ops_fixed:.0f} + L x {ops_layer:.1f} -> {ops_fixed + args.full_depth * ops_layer:.0f} at full depth"
    )

    # Per-class and per-item fits, projected to full depth.
    classes = sorted({c for d in depths for c in runs[d]["by_class"]})
    print("\nclass | fixed ms | per-layer ms | projected ms at full depth | share")
    class_proj = {}
    for c in classes:
        f, s = fit([(d, runs[d]["by_class"].get(c, 0.0) / 1e3) for d in depths])
        class_proj[c] = f + args.full_depth * s
    for c in classes:
        f, s = fit([(d, runs[d]["by_class"].get(c, 0.0) / 1e3) for d in depths])
        print(f"{c:7s} | {f:8.2f} | {s:12.3f} | {class_proj[c]:10.1f} | {100 * class_proj[c] / proj:5.1f} %")

    items = sorted({k for d in depths for k in runs[d]["by_item"]})
    ranked = []
    for k in items:
        pts = [(d, runs[d]["by_item"].get(k, [0.0, 0])[0] / 1e3) for d in depths]
        cnt = [(d, runs[d]["by_item"].get(k, [0.0, 0])[1]) for d in depths]
        f, s = fit(pts)
        cf, cs = fit(cnt)
        ranked.append((f + args.full_depth * s, k, cf + args.full_depth * cs, s, f))
    ranked.sort(reverse=True)
    print(
        f"\nranked cost table, projected to {args.full_depth} layers (ms/step | ops/step | class | op | in0 | in1 | cores)"
    )
    for cost, k, cnt, s, f in ranked[: args.top]:
        code = k.split(" | ")[0]
        print(f"{cost:8.2f} | {cnt:7.1f} | {classify(code):6s} | {k}")
    if args.json:
        json.dump(
            {
                "runs": runs,
                "fit": {"fixed": fixed, "per_layer": per_layer, "projection": proj, "full_depth": args.full_depth},
                "class_projection": class_proj,
                "ranked": ranked[: args.top],
            },
            open(args.json, "w"),
            indent=1,
        )


if __name__ == "__main__":
    main()
