#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Analysis of the MinimalMatmulStridedReduceScatterAsync sweep results.

    python tools/mmrs_sweep/analyze.py <shape> <stage> [--md OUT.md]

Safe to run while a sweep is still going -- it reads whatever has been appended so far.

Configs whose recorded PCC is below the gate are dropped rather than ranked. A blocking config that
is not a subblock multiple can corrupt output silently, and the fastest wrong config would otherwise
win the table.
"""

import argparse
import json
import os
from collections import defaultdict

OUT = os.path.expanduser("~/.tt-buddy/mmrs_sweep")
PCC_MIN = 0.99

AXES = ("gx", "gy", "mb", "kb", "nb", "chunk", "links", "workers", "packet")


def load(shape, stage):
    path = os.path.join(OUT, f"results_{shape}_{stage}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no results at {path}")
    best, dropped, untimed = {}, 0, 0
    for ln in open(path):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if not r.get("ok"):
            continue
        if r.get("us") is None:
            untimed += 1
            continue
        if r.get("pcc") is not None and r["pcc"] < PCC_MIN:
            dropped += 1
            continue
        key = tuple(r[a] for a in AXES) + (r["mode"],)
        if key not in best or r["us"] < best[key]["us"]:
            best[key] = r
    return list(best.values()), dropped, untimed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape")
    ap.add_argument("stage")
    ap.add_argument("--md")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    recs, dropped, untimed = load(args.shape, args.stage)
    if not recs:
        raise SystemExit("no timed, PCC-passing configs yet")

    fused = sorted([r for r in recs if r["mode"] == "fused"], key=lambda r: r["us"])
    unfused = {tuple(r[a] for a in AXES): r["us"] for r in recs if r["mode"] != "fused"}

    lines = [f"# mmrs sweep — {args.shape} / {args.stage}", ""]
    lines.append(f"{len(recs)} timed configs ({len(fused)} fused), {dropped} PCC-failed, {untimed} untimed")
    lines.append("")
    lines.append(f"## Top {args.top} fused by device time")
    lines.append("")
    lines.append("| us | grid | blk m/k/n | sb | chunk | links | workers | packet | pcc | vs unfused |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---|---:|")
    for r in fused[: args.top]:
        base = unfused.get(tuple(r[a] for a in AXES))
        speedup = f"{base / r['us']:.2f}x" if base else "-"
        lines.append(
            f"| {r['us']:.2f} | {r['gx']}x{r['gy']} | {r['mb']}/{r['kb']}/{r['nb']} | "
            f"{r['sbh']}x{r['sbw']} | {r['chunk']} | {r['links']} | {r['workers']} | "
            f"{r['packet']} | {r['pcc'] if r['pcc'] is not None else '-'} | {speedup} |"
        )

    # Per-axis marginals: the best time achieved at each value of each axis. Reads as "how much does
    # this axis matter", which is what decides whether it stays in the next stage.
    lines += ["", "## Best fused time per axis value", ""]
    for axis in AXES:
        by_val = defaultdict(list)
        for r in fused:
            by_val[r[axis]].append(r["us"])
        cells = " ".join(f"{v}={min(ts):.1f}" for v, ts in sorted(by_val.items()))
        spread = max(min(ts) for ts in by_val.values()) / min(min(ts) for ts in by_val.values())
        lines.append(f"- `{axis}` (spread {spread:.2f}x): {cells}")

    best = fused[0]
    lines += ["", f"**Best:** {best['us']:.2f} us — " + " ".join(f"{a}={best[a]}" for a in AXES)]

    text = "\n".join(lines)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
