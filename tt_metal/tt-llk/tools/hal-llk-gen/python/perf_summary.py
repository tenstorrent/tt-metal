#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Cycle summary of one run: original vs generated LLKs, from the perf stage output.

Reads <perf dir>/{orig,hal}_r<i>/perf_data/<module>/<module>.post.csv (written by stages/perf.sh).
For every variant and run type the value is the median over the repetitions of the framework's
mean(); the ratio is generated / original (> 1 = slower). Per run type the summary gives the
median, minimum and maximum of the per-variant ratios.

  l1      TILE_LOOP, L1_TO_L1: cycles per tile of the whole kernel (unpack + math + pack)
  l1_cov  the same, only variants whose formats the functional test also checks (eltwise binary)
  unp, math, pack   TILE_LOOP, UNPACK/MATH/PACK_ISOLATE: one thread measured alone
  kern    KERNEL zone, L1_TO_L1: cycles of the whole kernel run

Usage: perf_summary.py <perf dir> [--reps N] [--json out.json] [--md out.md]
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_compare import table  # noqa: E402

# Formats of test_eltwise_binary.py's main parameter set that the perf variants can have
# (Bfp4_b and Float32 are not perf formats); dest_acc Yes only with Float32 input there.
FUNCTIONAL_FORMATS = {"Bfp8_b", "Float16_b"}
ENTRIES = [
    ("l1", "TILE_LOOP", "L1_TO_L1", False),
    ("l1_cov", "TILE_LOOP", "L1_TO_L1", True),
    ("unp", "TILE_LOOP", "UNPACK_ISOLATE", False),
    ("math", "TILE_LOOP", "MATH_ISOLATE", False),
    ("pack", "TILE_LOOP", "PACK_ISOLATE", False),
    ("kern", "KERNEL", "L1_TO_L1", False),
]
LABELS = {
    "l1": "L1->L1 per tile",
    "l1_cov": "L1->L1 per tile, functionally checked formats",
    "unp": "unpack isolate per tile",
    "math": "math isolate per tile",
    "pack": "pack isolate per tile",
    "kern": "kernel zone, L1->L1",
}


def covered(variant):
    a, _, out, acc = variant.split(" | ")[:4]
    return (
        a.split(".")[-1] in FUNCTIONAL_FORMATS
        and out.split(".")[-1] in FUNCTIONAL_FORMATS
        and acc.endswith("No")
    )


def ratios(o, n, marker, rt, only_covered=False):
    res = []
    for (v, m, r), ov in o.items():
        if m != marker or r != rt or (only_covered and not covered(v)):
            continue
        nv = n.get((v, m, r))
        if nv:
            res.append((v, st.median(ov), st.median(nv), st.median(nv) / st.median(ov)))
    return res


def summarize(perf_dir, reps):
    ol = [f"orig_r{i}" for i in range(1, reps + 1)]
    nl = [f"hal_r{i}" for i in range(1, reps + 1)]
    missing = [
        x for x in ol + nl if not os.path.isdir(os.path.join(perf_dir, x, "perf_data"))
    ]
    if missing:
        raise SystemExit(f"perf data missing: {' '.join(missing)}")
    o, n = table(perf_dir, ol), table(perf_dir, nl)
    res = {"reps": reps, "original_labels": ol, "hal_labels": nl}
    for key, marker, rt, cov in ENTRIES:
        rr = ratios(o, n, marker, rt, cov)
        if not rr:
            continue
        res[key] = dict(
            n=len(rr),
            orig=st.median(x[1] for x in rr),
            new=st.median(x[2] for x in rr),
            ratio_median=st.median(x[3] for x in rr),
            ratio_min=min(x[3] for x in rr),
            ratio_max=max(x[3] for x in rr),
            faster_variants=sum(x[3] < 1 for x in rr),
        )
    res["rep_spread_max_pct"] = max(
        100 * (max(x) - min(x)) / st.median(x)
        for d in (o, n)
        for (_v, m, _rt), x in d.items()
        if m == "TILE_LOOP"
    )
    res["per_variant_l1"] = [
        dict(variant=v, orig=a, new=b, ratio=r)
        for v, a, b, r in sorted(ratios(o, n, "TILE_LOOP", "L1_TO_L1"))
    ]
    return res


def markdown(res):
    lines = [
        "# Perf cycles: original vs generated LLKs",
        "",
        f"{res['reps']} repetitions per side, alternating, each after tt-smi -r, LLK asserts off. "
        "Value per variant = median over repetitions of the framework's mean(); "
        "ratio = generated / original (> 1 = slower).",
        "",
        "| measure | variants | original (median) | generated (median) | ratio median | ratio min | ratio max | faster variants |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, *_ in ENTRIES:
        if key in res:
            r = res[key]
            lines.append(
                f"| {LABELS[key]} | {r['n']} | {r['orig']:.2f} | {r['new']:.2f} | {r['ratio_median']:.3f} | "
                f"{r['ratio_min']:.3f} | {r['ratio_max']:.3f} | {r['faster_variants']} |"
            )
    lines += [
        "",
        f"Largest spread between repetitions (TILE_LOOP, any variant): {res['rep_spread_max_pct']:.2f} %",
        "",
        "## Per variant, L1->L1 cycles per tile",
        "",
        "| variant | original | generated | ratio |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {x['variant']} | {x['orig']:.2f} | {x['new']:.2f} | {x['ratio']:.3f} |"
        for x in res["per_variant_l1"]
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("perf_dir")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args()
    res = summarize(a.perf_dir, a.reps)
    md = markdown(res)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if a.md:
        open(a.md, "w").write(md)
    if "l1" in res:
        r = res["l1"]
        print(
            f"L1->L1 per tile: original {r['orig']:.2f}, generated {r['new']:.2f}, ratio {r['ratio_median']:.3f} "
            f"({r['ratio_min']:.3f}-{r['ratio_max']:.3f}), faster in {r['faster_variants']} of {r['n']} variants"
        )


if __name__ == "__main__":
    main()
