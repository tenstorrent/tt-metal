# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Plot LTX-2.3 TP x SP per-layer latency tradeoff from bucketed results.

Reads ltx_tpsp_results.json (written by bucket.py), emits one stacked-bar +
total-line PNG per stage. Primary curve = SP-on-axis1 configs ordered by TP;
altaxis points overlaid as annotated markers.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "ltx_tpsp_results.json"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."

# Clean, colorblind-safe 3-bucket palette.
C = {"matmul_tp_ccl": "#4C78A8", "ring_attention": "#F58518", "overhead": "#BAB0AC"}
LABEL = {"matmul_tp_ccl": "Matmul + TP-CCL", "ring_attention": "Ring attention (SP)", "overhead": "Norm/RoPE/misc"}
PRIMARY = ["tp1_sp32", "tp2_sp16", "tp4_sp8", "tp8_sp4", "tp16_sp2", "tp32_sp1"]

with open(RESULTS) as f:
    data = json.load(f)


def stage_records(stage):
    return {v["config"]: v for k, v in data.items() if v["stage"] == stage}


for stage in ("stage_1", "stage_2"):
    recs = stage_records(stage)
    xs = [c for c in PRIMARY if c in recs]
    if not xs:
        continue
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bottom = np.zeros(len(xs))
    for bk in ("matmul_tp_ccl", "ring_attention", "overhead"):
        vals = np.array([recs[c][bk] for c in xs])
        ax.bar(range(len(xs)), vals, bottom=bottom, color=C[bk], label=LABEL[bk], width=0.62, zorder=2)
        bottom += vals
    totals = [recs[c]["total"] for c in xs]
    ax.plot(range(len(xs)), totals, "k--o", lw=1.5, ms=5, zorder=3, label="Per-layer total")
    # Mark shipped + untuned.
    for i, c in enumerate(xs):
        tag = []
        if c == "tp4_sp8":
            tag.append("SHIPPED")
        if c != "tp4_sp8":
            tag.append("untuned*")
        if tag:
            ax.annotate(" ".join(tag), (i, totals[i]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8,
                        color="#2a7d2a" if c == "tp4_sp8" else "#999")
    labels = [f"TP={recs[c]['TP']}\nSP={recs[c]['SP']}" for c in xs]
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-layer device time (µs)")
    n = {"stage_1": "9 728", "stage_2": "38 912"}[stage]
    ax.set_title(f"LTX-2.3 denoiser layer: TP×SP tradeoff — {stage} (N={n} tokens), BH Galaxy 4×8")
    ax.legend(loc="upper center", ncol=4, fontsize=9, framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.margins(y=0.15)
    fig.tight_layout()
    out = f"{OUTDIR}/ltx_tpsp_{stage}.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)

    # Also emit altaxis comparison if present.
    alt = {c: recs[c] for c in ("tp4_sp8_altaxis", "tp8_sp4_altaxis") if c in recs}
    if alt:
        print(f"[{stage}] axis-placement variants:")
        for c, r in alt.items():
            base = c.replace("_altaxis", "")
            b = recs.get(base)
            print(f"  {c}: total={r['total']}  vs {base}: total={b['total'] if b else '?'}"
                  f"  (ring {r['ring_attention']} vs {b['ring_attention'] if b else '?'})")
