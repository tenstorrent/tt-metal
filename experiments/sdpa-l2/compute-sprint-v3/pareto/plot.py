"""Render the six selected v3 variants from recorded measurements only."""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
ORDER = "DCBAEG"
COLORS = dict(D="#3658a2", C="#8155a6", B="#258657", A="#64748b", E="#009caf", G="#d87d24")
LABELS = dict(D="D · HiFi4 / FP32", C="C · QK4/PV2 / FP32", B="B · HiFi2 / compensated BF16",
              A="A · stock BF16 streaming", E="E · LoFi / BFP8 KV", G="G · LoFi / BFP4 KV")


def read(path):
    return json.loads(path.read_text())


def main():
    data = read(HERE / "matched-v1.json")
    assert data["complete"] and data["selected_sources_immutable"]
    assert len(data["records"]) == 126
    perf_sources = [V3 / "fp32/integrity-early-sustained-v2.json",
                    V3 / "review/B-valid-perf-final.json", V3 / "unchanged-A-resident-v2.json",
                    V3 / "compensated/e-valid-perf-v1.json", V3 / "compensated/g-valid-perf-v1.json"]
    fp = read(perf_sources[0])
    speed = {}
    for v in "DC":
        rows = [r for r in fp if r["variant"] == v and r["algorithm"] == "l1_early"]
        speed[v] = rows[0]["useful_flops"] / (statistics.median(r["median_ms"] for r in rows) * 1e9)
    speed["B"] = read(perf_sources[1])["results"]["group2_valid"]["tflops_per_core"]
    speed["A"] = read(perf_sources[2])["tflops_per_core"]
    for v, source in zip("EG", perf_sources[3:]):
        speed[v] = next(r for r in read(source) if r["case"] == "resident-v2")["candidate"]["tflops_per_core"]
    summary = {"throughput_tflops_per_core": speed, "panels": {}, "input_sha256": {
        str(p.relative_to(V3)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in perf_sources + [HERE / "matched-v1.json"]}}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#f8fafc")
    fig.subplots_adjust(left=.075, right=.97, bottom=.29, top=.79, wspace=.22)
    fig.text(.075, .935, "SDPA frontier", fontsize=27, fontweight="bold", color="#17243a")
    fig.text(.075, .889, "Accuracy across matched inputs vs. compute-isolated throughput", fontsize=16, color="#475569")
    fig.text(.97, .934, "BLACKHOLE · Q256 / K512 / D128", ha="right", fontsize=11, color="#475569")
    core_markers = {4096: "o", 32768: "s", 262144: "^"}
    stress_markers = {"common_q": "^", "common_k": "s", "common_v": "o"}
    for ax, suite in zip(axes, ("core", "stress")):
        ax.set_facecolor("white")
        ax.set_yscale("log")
        ax.set_xlim(.76, 2.27)
        ax.set_xticks([.9, 1.2, 1.5, 1.8, 2.1])
        ax.grid(axis="y", which="major", color="#e2e8f0", linewidth=.9)
        ax.grid(axis="x", color="#f1f5f9", linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Resident useful TFLOP/s per core  →  faster", labelpad=11)
        ax.axhline(.5, color="#9aa7b5", linestyle=(0, (4, 4)), linewidth=1)
        ax.text(.785, .54, "0.5% reference", fontsize=9, color="#718096")
        summary["panels"][suite] = {}
        for v in ORDER:
            rows = [r for r in data["records"] if r["variant"] == v and r["suite"] == suite]
            assert len(rows) == (18 if suite == "core" else 3)
            assert all(r["raw_trace_equal"] and r["inputs_immutable"] for r in rows)
            ys = np.array([r["metrics"]["l2_pct"] for r in rows])
            assert np.isfinite(ys).all() and (ys > 0).all()
            stats = dict(med=float(np.median(ys)), q1=float(np.quantile(ys, .25)),
                         q3=float(np.quantile(ys, .75)), whislo=float(ys.min()), whishi=float(ys.max()), fliers=[])
            summary["panels"][suite][v] = dict(n=len(rows), **stats)
            color, x = COLORS[v], speed[v]
            if suite == "core":
                ax.bxp([stats], positions=[x], widths=.047, manage_ticks=False, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=color + "35", edgecolor=color, linewidth=1.5),
                       medianprops=dict(color=color, linewidth=2.8),
                       whiskerprops=dict(color=color, linewidth=1.3), capprops=dict(color=color, linewidth=1.3))
            else:
                ax.vlines(x, ys.min(), ys.max(), color=color, alpha=.5, linewidth=1.2)
                ax.hlines(np.median(ys), x-.021, x+.021, color=color, linewidth=2.4)
            for r in rows:
                marker = core_markers[r["k_length"]] if suite == "core" else stress_markers[r["distribution"]]
                ax.scatter([x], [r["metrics"]["l2_pct"]], color=color, marker=marker,
                           s=23 if suite == "core" else 50, edgecolors="white", linewidths=.35,
                           alpha=.7 if suite == "core" else .95, zorder=4)
            offsets = dict(D=(13, -24), C=(13, -24), B=(-28, 24), A=(-30, 48), E=(19, -22), G=(19, 20))
            ax.annotate(v, (x, stats["med"]), xytext=offsets[v], textcoords="offset points",
                        fontsize=13, fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=.8))
        ax.set_ylim((.07, 120) if suite == "core" else (.002, 160))
        ax.set_ylabel("Relative L2 error (%) · log scale  ←  lower is better", labelpad=10)
    axes[0].set_title("Broad input suite  |  18 cases per variant", loc="left", fontsize=14, pad=18, fontweight="bold")
    axes[1].set_title("Common-mode stress  |  3 cases per variant", loc="left", fontsize=14, pad=18, fontweight="bold")
    axes[0].legend(handles=[Line2D([], [], marker=m, linestyle="none", color="#64748b", label=f"{n//1024}K KV")
                             for n, m in core_markers.items()], loc="upper left", frameon=False, fontsize=9)
    axes[1].legend(handles=[Line2D([], [], marker=m, linestyle="none", color="#64748b", label=k.replace("common_", "").upper() + " +32")
                             for k, m in stress_markers.items()], loc="upper left", frameon=False, fontsize=9)
    a_uniform = next(r for r in data["records"] if r["variant"] == "A" and
                     r["distribution"] == "uniform" and r["k_length"] == 262144)
    axes[0].annotate("uniform attention · 256K", (speed["A"], a_uniform["metrics"]["l2_pct"]),
                     xytext=(-130, -5), textcoords="offset points", color=COLORS["A"], fontsize=9,
                     arrowprops=dict(arrowstyle="-", color=COLORS["A"], lw=.7))
    fig.legend(handles=[Patch(facecolor=COLORS[v], label=f"{LABELS[v]}   ({speed[v]:.3f})") for v in ORDER],
               loc="lower center", bbox_to_anchor=(.52, .158), ncol=3, frameon=False, fontsize=11,
               columnspacing=2.5, labelspacing=1.15)
    fig.text(.075, .13, "Box: middle 50% of case errors • bar: median • whiskers: min–max • dots: individual cases (no x jitter).",
             fontsize=10, color="#475569")
    fig.text(.075, .102, "Broad suite: normal, clipped ±2, Q/K ×0.25, Q/K ×2, sparse outliers, uniform attention; each at 4K / 32K / 256K KV.",
             fontsize=10, color="#475569")
    fig.text(.075, .075, "Stress: common Q, K, or V +32 at 32K KV. All use identical BF16 inputs across variants, 256 query rows, 1 head, FP64 reference.",
             fontsize=10, color="#475569")
    fig.text(.075, .047, "Case spread is not statistical uncertainty. Throughput is measured separately with resident repeated KV; preprocessing excluded.",
             fontsize=10, color="#475569")
    fig.text(.075, .021, "All six choices shown, not necessarily nondominated on these two axes. G additionally reduces KV bytes for ring scaling.",
             fontsize=10, color="#475569")
    output = HERE / "sdpa_pareto_l2_distribution.png"
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor())
    (HERE / "plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
