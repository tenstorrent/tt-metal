#!/usr/bin/env python3
"""Plots for the budget study report (results/plots/*.png) from results/runs.csv.

  plots.py results
"""
import csv, json, os, sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3 (validated all-pairs)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LAYERS = {"D": 3, "S8": 8, "S0": 8}
SKIP = {"e1_d_w2048"}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def label_end(ax, xs, ys, text, color_idx, direct=True):
    ax.plot(xs, ys, color=SERIES[color_idx], linewidth=2, marker="o", markersize=5, label=text)
    if direct:
        ax.annotate(
            text, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9
        )


def load(res):
    pts = {}
    for r in csv.DictReader(open(os.path.join(res, "runs.csv"))):
        if r["status"] != "OK" or r["run_id"] in SKIP or not r["wall_ms_median"]:
            continue
        if r["exp"] == "E2c":
            continue
        seg = json.loads(r["segments_json"])[0]
        pts.setdefault((r["layer_set"], int(r["W"]), seg["h"], seg["n"]), float(r["wall_ms_median"]))
    return pts


def main(res):
    out = os.path.join(res, "plots")
    os.makedirs(out, exist_ok=True)
    pts = load(res)

    # 1. stage ms vs W, cold
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURF)
    for i, ls in enumerate(("S8", "S0", "D")):
        xs = sorted(W for (l, W, h, n) in pts if l == ls and h == 0 and n == W)
        label_end(
            ax,
            xs,
            [pts[(ls, W, 0, W)] for W in xs],
            {"S8": "S8 (8 sparse)", "S0": "S0 (3 dense + 5 sparse)", "D": "D (3 dense)"}[ls],
            i,
        )
    style(ax, "Cold forward time vs width, one (4,4) SP=4 stage", "forward width W (tokens)", "ms per forward")
    ax.set_xlim(1500, 12500)
    ax.set_ylim(0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "1_stage_ms_vs_W_cold.png"), dpi=150)

    # 2. layer ms vs h at n = 256 / 1024 / 2048, dense vs sparse
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    for ax, ls, name in ((axes[0], "D", "Dense layer"), (axes[1], "S8", "Sparse (MSA) layer")):
        for i, n in enumerate((2048, 1024, 256)):
            xs = sorted(h for (l, W, h, nn) in pts if l == ls and W == 2048 and nn == n)
            ys = [pts[(ls, 2048, h, n)] / LAYERS[ls] for h in xs]
            label_end(ax, [x / 1000 for x in xs], ys, f"n={n}", i, direct=ls != "D")
        if ls == "D":
            ax.annotate(
                "n = 256 / 1024 / 2048\n(lines coincide)",
                (xs[-1] / 1000, ys[-1]),
                xytext=(-150, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=9,
            )
        style(ax, f"{name}, W=2048: ms per layer vs history", "history h (k tokens)", "ms per layer")
        ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
        ax.set_xlim(-20, 640)
        ax.set_ylim(0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "2_layer_ms_vs_h.png"), dpi=150)

    # 2b. dense attention row floor: per-row depth cost vs segment width
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURF)
    for i, ls in enumerate(("D", "S8")):
        xs, ys = [], []
        for W in (2048, 4096, 8192):
            if (ls, W, 548864, W) in pts and (ls, W, 0, W) in pts:
                xs.append(W)
                ys.append((pts[(ls, W, 548864, W)] - pts[(ls, W, 0, W)]) / LAYERS[ls])
        label_end(ax, xs, ys, {"D": "dense", "S8": "sparse"}[ls], i)
    style(
        ax,
        "Extra ms per layer at h=549k vs cold, by segment rows",
        "rows in the segment (= W, B=1)",
        "ms per layer (h=549k minus h=0)",
    )
    ax.set_xlim(1500, 9500)
    ax.set_ylim(0)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "2b_depth_cost_vs_rows.png"), dpi=150)
    # 3. stacked op breakdown at h = 0 / 141k / 549k (n=2048), from the E3b zone profiles
    ops = {}
    for r in csv.DictReader(open(os.path.join(res, "ops.csv"))):
        if r["run_id"].startswith("e3b_") and r["run_id"].endswith("_n2048"):
            h = int(r["run_id"].split("_")[1][1:])
            ops[(h, r["layer_type"], r["zone"])] = float(r["device_ms_worst_chip"])
    groups = {
        "dense": [
            ("ring_joint_sdpa", ["attn/ring_joint_sdpa"]),
            ("other attention", ["attn", "-attn/ring_joint_sdpa"]),
            ("MLP", ["mlp"]),
            ("norms + CCL", ["(layer total)", "-attn", "-mlp"]),
        ],
        "sparse": [
            ("ag_kv", ["attn/ag_kv"]),
            ("ag_index_k", ["attn/ag_index_k"]),
            ("indexer", ["attn/indexer"]),
            ("sparse_sdpa", ["attn/sparse_sdpa"]),
            ("other attention", ["attn", "-attn/ag_kv", "-attn/ag_index_k", "-attn/indexer", "-attn/sparse_sdpa"]),
            ("MoE", ["mlp"]),
            ("norms + CCL", ["(layer total)", "-attn", "-mlp"]),
        ],
    }
    hs = sorted({h for h, _, _ in ops})
    if hs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
        pal = SERIES + ["#eda100", "#e87ba4", "#008300", "#4a3aa7"]
        for ax, lt in zip(axes, ("dense", "sparse")):
            bottoms = [0.0] * len(hs)
            for gi, (name, terms) in enumerate(groups[lt]):
                vals = [
                    max(0.0, sum((-1 if t[0] == "-" else 1) * ops.get((h, lt, t.lstrip("-")), 0.0) for t in terms))
                    for h in hs
                ]
                ax.bar(
                    [f"{h / 1000:.0f}k" for h in hs],
                    vals,
                    bottom=bottoms,
                    color=pal[gi],
                    width=0.55,
                    edgecolor=SURF,
                    linewidth=2,
                    label=name,
                )
                bottoms = [b + v for b, v in zip(bottoms, vals)]
            for x, b in enumerate(bottoms):
                ax.annotate(
                    f"{b:.1f} ms", (x, b), xytext=(0, 4), textcoords="offset points", ha="center", color=INK, fontsize=9
                )
            style(
                ax,
                f"{lt.capitalize()} layer device time by zone (worst chip, W=2048, n=2048)",
                "history h",
                "device ms per layer",
            )
            ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "3_op_breakdown_vs_h.png"), dpi=150)

    # 4. additivity: measured vs predicted packed forwards (additivity.py output)
    add = os.path.join(res, "additivity.csv")
    if os.path.exists(add):
        rows = list(csv.DictReader(open(add)))
        fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=SURF)
        lim = max(max(float(r["meas_ms"]), float(r["pred_ms"])) for r in rows) * 1.08
        ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, linestyle="--", label="y = x")
        for x in (0.9, 1.1):
            ax.plot([0, lim], [0, lim * x], color=GRID, linewidth=1)
        for i, ls in enumerate(("S8", "S0", "D")):
            pts_ = [(float(r["pred_ms"]), float(r["meas_ms"])) for r in rows if r["layer_set"] == ls]
            ax.scatter(
                [p for p, _ in pts_],
                [m for _, m in pts_],
                s=40,
                color=SERIES[i],
                edgecolor=SURF,
                linewidth=1.5,
                label=f"{ls} ({len(pts_)} forwards)",
                zorder=3,
            )
        worst = max(rows, key=lambda r: abs(float(r["resid_pct"])))
        ax.annotate(
            f"worst {worst['layer_set']} {worst['compo']} {float(worst['resid_pct']):+.1f}%",
            (float(worst["pred_ms"]), float(worst["meas_ms"])),
            xytext=(8, -14),
            textcoords="offset points",
            color=INK,
            fontsize=9,
        )
        style(
            ax, "Packed forwards: measured vs additive prediction", "predicted ms (cold packed + Σ ΔT1)", "measured ms"
        )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(
            frameon=False, fontsize=9, labelcolor=INK, loc="upper left", title="grey lines: ±10%", title_fontsize=8
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out, "4_additivity.png"), dpi=150)

    # 5. simulator: tok/s vs W per policy (default split); tok/s per layer split (W=8192, fcfs)
    def sim_rows(path):
        rows = []
        for line in open(path):
            f = line.split()
            if len(f) > 6 and f[0].isdigit() and f[1] in ("fcfs", "bucket", "cost"):
                rows.append((int(f[0]), f[1], float(f[2]), [int(u) for u in f[6:]]))
        return rows

    sim = os.path.join(res, "sim")
    if os.path.exists(os.path.join(sim, "1_widths_policies.txt")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF, gridspec_kw={"width_ratios": [1, 1.2]})
        rows = sim_rows(os.path.join(sim, "1_widths_policies.txt"))
        for i, pol in enumerate(("fcfs", "bucket", "cost")):
            xs = [W for W, p, _, _ in rows if p == pol]
            ys = [t / 1000 for W, p, t, _ in rows if p == pol]
            axes[0].plot(
                xs,
                ys,
                color=SERIES[i],
                linewidth=2,
                marker="o",
                markersize=5,
                label=pol,
                linestyle="--" if pol == "bucket" else "-",
            )
        style(
            axes[0],
            "Simulated throughput vs forward width (split 8,8,8,8,7,7,7,7)",
            "forward width W (tokens)",
            "k tokens / s",
        )
        axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, title="fcfs and bucket overlap", title_fontsize=8)
        axes[0].set_ylim(0)
        splits = ["8,8,8,8,7,7,7,7", "6,8,8,8,8,8,7,7", "5,8,8,8,8,8,8,7", "4,8,8,8,8,8,8,8", "3,9,8,8,8,8,8,8"]
        vals, utils = [], []
        for sp in splits:
            r = [
                x
                for x in sim_rows(os.path.join(sim, f"2_split_{sp.replace(',', '-')}.txt"))
                if x[0] == 8192 and x[1] == "fcfs"
            ][0]
            vals.append(r[2] / 1000)
            utils.append(r[3])
        y = list(range(len(splits)))
        axes[1].barh(y, vals, color=SERIES[0], height=0.55)
        for yi, v, u in zip(y, vals, utils):
            axes[1].annotate(
                f"{v:.1f}k  (stage util {min(u)}–{max(u)}%)",
                (v, yi),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=9,
            )
        axes[1].set_yticks(y, splits)
        axes[1].invert_yaxis()
        style(axes[1], "Simulated throughput by layer split (W=8192, fcfs)", "k tokens / s", "")
        axes[1].set_xlim(0, max(vals) * 1.55)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "5_sim_throughput.png"), dpi=150)
    print(f"[plots] wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
