#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join load_skew.csv (real per-chip routing) with per_op.csv's experts rows (zone profile): how much of the
worst chip's experts time is load imbalance and how much is the kernel.

  load_skew_join.py [--skew load_skew.csv] [--per-op per_op.csv] [--fit bench/experts_fit.txt] [--out load_skew_join.csv]

Rows match on (W, h, input, layer) for single requests, and on (W, B, layer) with the per_op row's h label
for packed forwards (the packed h strings are the same '+'-joined segment depths).
model(T, A) = the bench fit (path=hybrid, additive+c): a * A * weight_MB_per_expert + b * T + c, T = the chip's
routed token-expert assignments, A = its active experts.
  imbalance_ms   = worst_ms - mean_ms (measured, zone profile)
  model_worst_ms = model at the worst chip's (T, A); model_mean_ms at the mean (T, A)
  kernel_gap_ms  = worst_ms - model_worst_ms (what the kernel loses vs the bench at that load)
  skew_model_ms  = model_worst_ms - model_mean_ms (imbalance the bench model predicts from routing alone)
"""

import argparse
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERT_MB = 3 * 6144 * 3072 * 0.5625 / 1e6  # one bf4 routed expert (gate, up, down), bench_experts.py's W


def load_fit(path, which="hybrid", model="additive+c"):
    block = open(path).read().split(f"=== path={which}")[1].split("\n===")[0]
    m = re.search(rf"^{re.escape(model)}\s+a=([\d.]+) ms/MB.*?b=([\d.]+) us/token\s+c=([\d.]+) us", block, re.M)
    a, b, c = (float(x) for x in m.groups())
    return lambda T, A: a * A * EXPERT_MB + b * T / 1e3 + c / 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skew", default=str(HERE.parent / "load_skew.csv"))
    ap.add_argument("--per-op", default=str(HERE.parent / "per_op.csv"))
    ap.add_argument("--fit", default=str(HERE.parent / "bench" / "experts_fit.txt"))
    ap.add_argument("--out", default=str(HERE.parent / "load_skew_join.csv"))
    args = ap.parse_args()
    model = load_fit(args.fit)

    prof = {}
    for r in csv.DictReader(open(args.per_op)):
        if r["op"] == "experts":
            prof[(r["W"], r["h"], r["input"], r["B"], r["layer"])] = r  # later runs win
    out = []
    for s in csv.DictReader(open(args.skew)):
        p = prof.get((s["W"], s["h"], s["input"], s["B"], s["layer"]))
        if p is None:
            continue
        tok = [int(x) for x in s["per_chip_tokens"].split(";")]
        t_w, t_m = max(tok), sum(tok) / len(tok)
        a_w, a_m = int(s["worst_chip_active"]), float(s["experts_active_per_chip_mean"])
        w, m = float(p["worst_ms"]), float(p["mean_ms"])
        mw, mm = model(t_w, a_w), model(t_m, a_m)
        out.append(
            {
                "case": s["case"],
                "layer": s["layer"],
                "W": s["W"],
                "h": s["h"],
                "B": s["B"],
                "input": s["input"],
                "run_id": p["run_id"],
                "tok_worst": t_w,
                "tok_mean": round(t_m, 1),
                "max_over_mean": s["max_over_mean"],
                "act_worst": a_w,
                "act_mean": a_m,
                "worst_ms": w,
                "mean_ms": m,
                "imbalance_ms": round(w - m, 4),
                "model_worst_ms": round(mw, 4),
                "model_mean_ms": round(mm, 4),
                "skew_model_ms": round(mw - mm, 4),
                "kernel_gap_ms": round(w - mw, 4),
                "roof_ms": p["roof_ms"],
            }
        )
    with open(args.out, "w", newline="") as f:
        if out:
            wr = csv.DictWriter(f, fieldnames=list(out[0]))
            wr.writeheader()
            wr.writerows(out)
    for r in out:
        print(
            f"{r['case']:28s} L{r['layer']} tok {r['tok_worst']}/{r['tok_mean']} worst {r['worst_ms']:.3f} mean "
            f"{r['mean_ms']:.3f} imb {r['imbalance_ms']:.3f} model_w {r['model_worst_ms']:.3f} gap {r['kernel_gap_ms']:.3f}"
        )
    print(f"[join] {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
