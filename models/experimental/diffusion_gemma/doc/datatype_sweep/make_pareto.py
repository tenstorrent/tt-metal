# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render the two dg-07 datatype-sweep Pareto charts (argmax-agreement vs latency,
accept-agreement vs latency), marking the selected point and the minimum-allowed line.

Ranking axis is TRACED per-block throughput (t/s), read from the perf sweep JSONs. Decision
agreement (vs the bf16-experts reference) is read from the decision-agreement JSON. The bf16
reference is agreement=1.0 by construction (it is the reference).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_ARGMAX = 0.95
MIN_ACCEPT = 0.90


def _load_perf(dir_path):
    pts = []
    for f in sorted(glob.glob(os.path.join(dir_path, "traced_tuned_s*.json"))):
        d = json.load(open(f))
        pts.append((d["steps"], d["tokens_per_block_per_s"], d["steady_block_latency_s"]))
    return sorted(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-bf16", default="/home/zni/dg-agent-runs/dtsweep/perf_bf16")
    ap.add_argument("--perf-bfp8", default="/home/zni/dg-agent-runs/dtsweep/perf_bfp8")
    ap.add_argument("--agreement", default="/home/zni/dg-agent-runs/dtsweep/agreement.json")
    ap.add_argument("--out-dir", default="/home/zni/tt-metal/models/experimental/diffusion_gemma/doc/datatype_sweep")
    args = ap.parse_args()

    bf16 = _load_perf(args.perf_bf16)
    bfp8 = _load_perf(args.perf_bfp8)
    agr = json.load(open(args.agreement))
    argmax_bfp8 = agr["committed_match"]  # committed clean-argmax agreement vs bf16 reference
    accept_bfp8 = agr["mean_accept_iou"]

    for metric, y_bfp8, minline, title, fname in [
        (
            "argmax",
            argmax_bfp8,
            MIN_ARGMAX,
            "Committed-argmax agreement vs bf16 reference",
            "pareto_argmax_vs_latency.png",
        ),
        (
            "accept",
            accept_bfp8,
            MIN_ACCEPT,
            "Accept/renoise agreement (IoU) vs bf16 reference",
            "pareto_accept_vs_latency.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        # bf16 = the selected/reference config (agreement == 1.0 at every step count)
        xs = [p[1] for p in bf16]
        ax.scatter(
            xs,
            [1.0] * len(xs),
            s=140,
            marker="*",
            color="#1f6feb",
            zorder=5,
            label="bf16 experts (SELECTED, reference)",
        )
        for steps, tps, _lat in bf16:
            ax.annotate(
                f"{steps} steps", (tps, 1.0), textcoords="offset points", xytext=(4, 8), fontsize=8, color="#1f6feb"
            )
        # bfp8 candidate (single agreement number, plotted at each step count's latency)
        xs2 = [p[1] for p in bfp8]
        ax.scatter(
            xs2,
            [y_bfp8] * len(xs2),
            s=90,
            marker="o",
            color="#d1242f",
            zorder=5,
            label=f"bfp8 experts (REJECTED, agreement={y_bfp8:.3f})",
        )
        for steps, tps, _lat in bfp8:
            ax.annotate(
                f"{steps} steps",
                (tps, y_bfp8),
                textcoords="offset points",
                xytext=(4, -12),
                fontsize=8,
                color="#d1242f",
            )
        ax.axhline(minline, ls="--", color="#6e7781", label=f"minimum-allowed agreement = {minline:.2f}")
        ax.axvline(100.0, ls=":", color="#8250df", alpha=0.6, label="100 t/s target")
        ax.set_xlabel("TRACED throughput (tokens / s), higher = faster")
        ax.set_ylabel(f"{metric} decision agreement vs bf16 reference")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        out = os.path.join(args.out_dir, fname)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print("WROTE", out)


if __name__ == "__main__":
    main()
