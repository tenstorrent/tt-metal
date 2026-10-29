# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Diff two ``analyze_tracy.py --json`` outputs: per-depth device time, per-class projection, and the ranked items
that disappeared, appeared or changed, so a change can show "targeted ops gone and nothing new".

    python diff_tracy.py before.json after.json [--min-ms 1.0]
"""

from __future__ import annotations

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--min-ms", type=float, default=1.0, help="hide item deltas below this (projected ms/step)")
    args = p.parse_args()
    a, b = json.load(open(args.before)), json.load(open(args.after))
    print("depth | device ms/step before -> after | ops/step before -> after")
    for d in sorted(a["runs"], key=int):
        ra, rb = a["runs"][d], b["runs"].get(d)
        if rb is None:
            continue
        print(
            f"{int(d):5d} | {ra['device_ms_per_step']:8.2f} -> {rb['device_ms_per_step']:8.2f} ({rb['device_ms_per_step'] - ra['device_ms_per_step']:+7.2f}) | {ra['ops_per_step']:5.0f} -> {rb['ops_per_step']:5.0f}"
        )
    fa, fb = a["fit"], b["fit"]
    print(
        f"fit: fixed {fa['fixed']:.2f} -> {fb['fixed']:.2f}, per layer {fa['per_layer']:.3f} -> {fb['per_layer']:.3f}, projection {fa['projection']:.1f} -> {fb['projection']:.1f} ms ({fb['projection'] - fa['projection']:+.1f})"
    )
    print(
        "class projection: "
        + ", ".join(
            f"{c} {a['class_projection'].get(c, 0):.1f} -> {b['class_projection'].get(c, 0):.1f}"
            for c in sorted(set(a["class_projection"]) | set(b["class_projection"]))
        )
    )

    def items(x):
        return {row[1]: (row[0], row[2]) for row in x["ranked"]}

    ia, ib = items(a), items(b)
    print(f"\nitems (projected ms/step, ops/step), |delta| >= {args.min_ms} ms:")
    for key in sorted(set(ia) | set(ib), key=lambda k: -abs(ib.get(k, (0, 0))[0] - ia.get(k, (0, 0))[0])):
        ma, na = ia.get(key, (0.0, 0.0))
        mb, nb = ib.get(key, (0.0, 0.0))
        if abs(mb - ma) < args.min_ms:
            continue
        tag = "GONE" if key not in ib else ("NEW" if key not in ia else "")
        print(f"  {mb - ma:+8.2f} ms  {ma:8.2f} -> {mb:8.2f}  ops {na:6.1f} -> {nb:6.1f}  {tag:4s} {key}")


if __name__ == "__main__":
    main()
