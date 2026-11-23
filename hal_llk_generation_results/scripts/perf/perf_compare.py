#!/usr/bin/env python3
"""Compare LLK perf-test cycles: original vs generated code (eltwise binary, Elwadd).

Reads <out>/<label>/perf_data/<module>/<module>.post.csv (TILE_LOOP rows are already
divided by loop_factor * tile_cnt by the framework = cycles per tile; KERNEL/INIT
rows are total cycles per zone).

Usage: perf_compare.py <out dir> --orig orig_r1 orig_r2 ... --new hal8a_r1 ... [--md out.md] [--json out.json]
"""
import argparse
import glob
import json
import os
import statistics as st

import pandas as pd

KEY = ["formats.input_A", "formats.input_B", "formats.output", "dest_acc", "mathop", "math_fidelity"]
RUN_TYPES = ["L1_TO_L1", "UNPACK_ISOLATE", "MATH_ISOLATE", "PACK_ISOLATE",
             "L1_CONGESTION[UNPACK]", "L1_CONGESTION[PACK]"]
MARKERS = ["TILE_LOOP", "KERNEL", "INIT"]


def load(out, label):
    files = glob.glob(os.path.join(out, label, "perf_data", "*", "*.post.csv"))
    if len(files) != 1:
        raise SystemExit(f"{label}: expected one *.post.csv, found {files}")
    d = pd.read_csv(files[0])
    d["variant"] = d[KEY].astype(str).agg(" | ".join, axis=1)
    return d


def table(out, labels):
    """variant x marker -> run_type -> list of values over repetitions."""
    res = {}
    for lab in labels:
        d = load(out, lab)
        for _, r in d.iterrows():
            for rt in RUN_TYPES:
                col = f"mean({rt})"
                if col in d.columns and pd.notna(r[col]):
                    res.setdefault((r["variant"], r["marker"], rt), []).append(float(r[col]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--orig", nargs="+", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--name", default="new")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()

    o, n = table(a.out, a.orig), table(a.out, a.new)
    variants = sorted({k[0] for k in o})
    if variants != sorted({k[0] for k in n}):
        raise SystemExit("variant sets differ between original and new runs")

    lines = [f"# Perf cycles: original vs {a.name}", "",
             f"original runs: {' '.join(a.orig)}; {a.name} runs: {' '.join(a.new)}",
             "Values = median over runs of the framework's mean(); TILE_LOOP = cycles per tile, "
             "KERNEL/INIT = cycles per zone. ratio = new / original (> 1 = slower).", ""]
    summary = {}
    for marker in MARKERS:
        lines += [f"## {marker}", "",
                  "| run type | orig median over variants | new median over variants | ratio median | ratio min | ratio max | orig rep spread max % | new rep spread max % |",
                  "|---|---|---|---|---|---|---|---|"]
        for rt in RUN_TYPES:
            ov, nv, ratios, osp, nsp = [], [], [], [], []
            for v in variants:
                ok, nk = o.get((v, marker, rt)), n.get((v, marker, rt))
                if not ok or not nk:
                    continue
                om, nm = st.median(ok), st.median(nk)
                ov.append(om); nv.append(nm); ratios.append(nm / om)
                osp.append(100 * (max(ok) - min(ok)) / om); nsp.append(100 * (max(nk) - min(nk)) / nm)
            if not ratios:
                continue
            row = dict(orig=st.median(ov), new=st.median(nv), ratio_median=st.median(ratios),
                       ratio_min=min(ratios), ratio_max=max(ratios),
                       orig_spread_max_pct=max(osp), new_spread_max_pct=max(nsp), n_variants=len(ratios))
            summary[f"{marker}/{rt}"] = row
            lines.append(f"| {rt} | {row['orig']:.2f} | {row['new']:.2f} | {row['ratio_median']:.3f} | "
                         f"{row['ratio_min']:.3f} | {row['ratio_max']:.3f} | {row['orig_spread_max_pct']:.1f} | "
                         f"{row['new_spread_max_pct']:.1f} |")
        lines.append("")

    lines += ["## Per variant, TILE_LOOP L1_TO_L1 (cycles per tile)", "",
              "| variant | orig | new | ratio |", "|---|---|---|---|"]
    per_variant = []
    for v in variants:
        ok, nk = o.get((v, "TILE_LOOP", "L1_TO_L1")), n.get((v, "TILE_LOOP", "L1_TO_L1"))
        if ok and nk:
            om, nm = st.median(ok), st.median(nk)
            per_variant.append(dict(variant=v, orig=om, new=nm, ratio=nm / om))
            lines.append(f"| {v} | {om:.2f} | {nm:.2f} | {nm / om:.3f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.md:
        open(a.md, "w").write(text)
    if a.json:
        json.dump(dict(orig=a.orig, new=a.new, summary=summary, per_variant=per_variant), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
