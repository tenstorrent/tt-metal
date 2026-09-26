#!/usr/bin/env python3
"""Plots for the SP=2 follow-up (results_sp2/plots/*.png).

  plots_sp2.py            (run from m3_budget_study/)

1 layer ms vs h, SP=2 vs SP=4 (dense / sparse panels)   2 chip-us per token-layer vs h
3 full-model e2e tok/s, cold vs hot, 4 x (2,4) vs 2 x (4,4)   4 simulated tok/s vs hot-latency p99 per layout
"""
import csv, glob, json, os, re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots import GRID, INK, INK2, SERIES, SURF, style

OUT = "results_sp2/plots"
CHIPS = {"sp2": 8, "sp4": 16}


def grid(path, exp, dense, sparse):
    pts = {}
    for r in csv.DictReader(open(path)):
        if r["status"] != "OK" or r["exp"] != exp or int(r["W"]) != 2048 or not r["wall_ms_median"]:
            continue
        s = json.loads(r["segments_json"])[0]
        kind = {dense: "dense", sparse: "sparse"}.get(r["layer_set"])
        if kind and s["n"] == 2048:
            pts[(kind, s["h"])] = float(r["wall_ms_median"]) / (3 if kind == "dense" else 8)
    return pts


def main():
    os.makedirs(OUT, exist_ok=True)
    g4 = grid("results/runs.csv", "E2", "D", "S8")
    g2 = grid("results_sp2/runs.csv", "A2", "D2", "S8P")

    # 1 + 2: per-layer ms and chip-us per token-layer vs h
    for fname, ylab, conv, title in (
        ("1_layer_ms_vs_h_sp2_vs_sp4.png", "ms per layer", lambda ms, lay: ms, "ms per layer (W=2048, n=2048)"),
        (
            "2_chip_us_per_token_layer.png",
            "chip-µs per token-layer",
            lambda ms, lay: CHIPS[lay] * ms * 1000 / 2048,
            "chip-µs per token-layer (W=2048)",
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
        for ax, kind in zip(axes, ("dense", "sparse")):
            for i, (lay, g, label) in enumerate(
                (("sp4", g4, "SP=4 (4,4), 16 chips"), ("sp2", g2, "SP=2 (2,4), 8 chips"))
            ):
                hs = sorted(h for k, h in g if k == kind)
                if not hs:
                    continue
                ys = [conv(g[(kind, h)], lay) for h in hs]
                ax.plot([h / 1000 for h in hs], ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=label)
                ax.annotate(
                    f"{ys[-1]:.1f}",
                    (hs[-1] / 1000, ys[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    color=INK,
                    fontsize=9,
                )
            style(ax, f"{kind.capitalize()} layer: {title}", "history h (k tokens)", ylab)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
            ax.set_xlim(-20, 640)
            ax.set_ylim(0)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)

    # 3: e2e tok/s (open loop) per stream, per layout and W
    summ = "results_sp2/e2e/summary.txt"
    if os.path.exists(summ):
        vals, sess = {}, None
        for line in open(summ):
            if line.startswith("## "):
                sess = line[3:].strip()
            elif (
                sess
                and "sync0" in sess
                and "split" not in sess
                and (m := re.match(r"\s+(\w+)\s+chunks.*tok/s\s+(\d+)", line))
            ):
                if m[1].endswith("_open"):
                    vals[(sess, m[1][:-5])] = int(m[2]) / 1000
        cats = ["cold", "h141", "h549"]
        groups = [
            ("r4_w4096_sync0", "4 × (2,4) SP=2, W=4096"),
            ("r4_w8192_sync0", "4 × (2,4) SP=2, W=8192"),
            ("r2_w4096_sync0", "2 × (4,4) SP=4, W=4096"),
            ("r2_w8192_sync0", "2 × (4,4) SP=4, W=8192"),
        ]
        fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURF)
        pal = [SERIES[0], "#6fa8e8", SERIES[1], "#f2a07a"]
        wbar = 0.2
        for gi, (s, label) in enumerate(groups):
            xs = [ci + (gi - 1.5) * wbar for ci in range(len(cats))]
            ys = [vals.get((s, c), 0) for c in cats]
            ax.bar(xs, ys, width=wbar, color=pal[gi], edgecolor=SURF, linewidth=2, label=label)
            for x, y in zip(xs, ys):
                if y:
                    ax.annotate(
                        f"{y:.0f}",
                        (x, y),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        color=INK,
                        fontsize=8,
                    )
        ax.set_xticks(range(len(cats)), ["cold", "hot, 139k history", "hot, 549k history"])
        style(ax, "Full model on one galaxy: throughput by layout (open loop)", "", "k tokens / s")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "3_e2e_tok_s.png"), dpi=150)

    # 4: sim tok/s vs hot latency p99, best split per scenario / W / policy
    files = sorted(glob.glob("results_sp2/sim_layouts/L-*_measured.txt"))
    if files:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURF)
        markers = {"fcfs": "o", "cost": "s"}
        for i, f in enumerate(files):
            scen = os.path.basename(f).split("_")[0]
            best = {}
            for line in open(f):
                p = line.split()
                if len(p) > 8 and p[0].isdigit() and p[1] in markers:
                    key = (int(p[0]), p[1])
                    tok, p99 = float(p[2]), float(p[7])
                    if key not in best or tok > best[key][0]:
                        best[key] = (tok, p99)
            color = (SERIES + ["#eda100"])[i]
            done = set()
            for (W, pol), (tok, p99) in sorted(best.items()):
                ax.scatter(
                    [p99 / 1000],
                    [tok / 1000],
                    s=60,
                    marker=markers[pol],
                    color=color,
                    edgecolor=SURF,
                    linewidth=1.5,
                    zorder=3,
                )
                twins = [
                    w
                    for (w2, p2), (t2, q2) in best.items()
                    for w in [w2]
                    if p2 == pol and abs(q2 - p99) < 60 and abs(t2 - tok) < 400
                ]
                key = (pol, round(p99, -2), round(tok, -3))
                if key in done:
                    continue
                done.add(key)
                ws = "/".join(f"{w // 1024}k" for w in sorted(set(twins)))
                ax.annotate(
                    f"{scen} {pol} W={ws}",
                    (p99 / 1000, tok / 1000),
                    xytext=(7, 3),
                    textcoords="offset points",
                    color=INK,
                    fontsize=8,
                )
        ax.axvline(1.5, color=INK2, linewidth=1, linestyle="--")
        ax.annotate(
            "1.5 s target",
            (1.5, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            color=INK2,
            fontsize=8,
        )
        for pol, mk in markers.items():
            ax.scatter([], [], marker=mk, color=INK2, label=pol)
        ax.legend(
            frameon=False, fontsize=8, labelcolor=INK, loc="lower right", title="marker = policy", title_fontsize=8
        )
        style(
            ax,
            "4 galaxies: simulated throughput vs hot-request latency p99 (best split each)",
            "hot-request latency p99 (s)",
            "k useful tokens / s",
        )
        ax.set_ylim(0)
        ax.set_xlim(0)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "4_sim_tok_s_vs_hot_latency.png"), dpi=150)
    print(f"[plots_sp2] wrote {OUT}")


if __name__ == "__main__":
    main()
