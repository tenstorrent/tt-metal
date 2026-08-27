"""Render the before/after precision figures for SFPU_LUT_RETUNE_WORMHOLE.md.

Input:  sfpu_lut_data/curves_{main,retuned}.json   (measured on n300, fp32 -> fp32)
Output: sfpu_lut_plots/*.svg   (set PLOT_PNG_DIR to also drop rasters for eyeballing)

Palette is the dataviz reference instance, light surface. Before/after is one hue in two
shades (blue ramp steps 400 and 650), checked by computation rather than by eye:
OKLab dE 24.1 between the shades (>= 15 floor for a same-hue pair) and 3.54:1 / 9.66:1
against the surface (>= 3:1). The exact-function reference curve wears ink, not a series
colour, so it never reads as a third series.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_MAIN, C_NEW, C_EXACT = "#3987e5", "#104281", "#52514e"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "sfpu_lut_plots")
PNG = os.environ.get("PLOT_PNG_DIR")
os.makedirs(OUT, exist_ok=True)
if PNG:
    os.makedirs(PNG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
})

D = {t: json.load(open(os.path.join(HERE, f"curves_{t}.json"))) for t in ("main", "retuned")}
SEGS = D["main"]["segments"]
OPS = ("tanh", "sigmoid_appx", "gelu_appx")
TITLES = {"tanh": "tanh, APPROXIMATION_MODE=true",
          "sigmoid_appx": "sigmoid_appx", "gelu_appx": "gelu_appx"}
LEGEND = [Line2D([], [], color=C_MAIN, lw=2.2, label="main"),
          Line2D([], [], color=C_NEW, lw=2.2, label="retuned constants")]
LEGEND_EX = [Line2D([], [], color=C_EXACT, lw=1.4, label="exact")] + LEGEND


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    if PNG:
        fig.savefig(os.path.join(PNG, name.replace(".svg", ".png")), dpi=140,
                    bbox_inches="tight")
    plt.close(fig)


def series(tag, op):
    """x, kernel output, signed error, exact."""
    r = D[tag]["data"][op]
    return [p[0] for p in r], [p[1] for p in r], [p[1] - p[2] for p in r], [p[2] for p in r]


def window(tag, op, lo, hi, which=2):
    x, y, e, ex = series(tag, op)
    v = (y, e, ex)[which - 1] if which else y
    v = {1: y, 2: e, 3: ex}[which]
    return [(a, b) for a, b in zip(x, v) if lo <= a <= hi]


def seg_max(tag, op, i):
    """Worst |error| on segment i. The final segment is closed at the sweep's top end."""
    edges = SEGS[op]
    lo, hi = edges[i], edges[i + 1]
    last = i == len(edges) - 2
    x, _, e, _ = series(tag, op)
    return max(abs(v) for a, v in zip(x, e) if (lo <= a <= hi) if (a < hi or last))


def dress(ax, xlabel=None, ylabel=None, title=None, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=9.5, pad=6)


def knees(ax, op, lo, hi, label=True):
    ks = [k for k in SEGS[op][1:-1] if lo < k < hi]
    for k in ks:
        ax.axvline(k, color=AXIS, lw=0.8)
    if label and len(ks) <= 3:
        for k in ks:
            ax.annotate(f"{k:g}", (k, 0.0), xycoords=("data", "axes fraction"),
                        xytext=(3, 4), textcoords="offset points", fontsize=7.5, color=MUTED)


# ------------------------------------------------------------------ fig 1: dumbbell
rows = []
for op in OPS:
    edges = SEGS[op]
    for i in range(len(edges) - 1):
        lo = edges[i]
        lbl = (f"[{lo:g}, {edges[i+1]:g})" if i < len(edges) - 2 else f"[{lo:g}, ∞)")
        rows.append((op, lbl, seg_max("main", op, i), seg_max("retuned", op, i)))

fig, ax = plt.subplots(figsize=(9.0, 5.0))
y, yticks, ylabels, groups = 0, [], [], {}
prev = None
for op, lbl, a, b in rows:
    if prev is not None and op != prev:
        y += 0.8
    groups.setdefault(op, []).append(y)
    ax.plot([a, b], [y, y], color=GRID, lw=3.5, solid_capstyle="round", zorder=1)
    same = abs(a - b) <= 1e-9 * max(a, b)
    ax.plot([a], [y], "o", ms=8, color=MUTED if same else C_MAIN, mec=SURFACE, mew=1.5, zorder=3)
    if not same:
        ax.plot([b], [y], "o", ms=8, color=C_NEW, mec=SURFACE, mew=1.5, zorder=4)
    # Values live in a fixed column outside the right spine, so no label can ever
    # collide with a dot, a tick label or its neighbour.
    txt = f"{a:.4f}  unchanged" if same else f"{a:.4f}  →  {b:.4f}"
    ax.annotate(txt, (1.015, y), xycoords=("axes fraction", "data"), va="center",
                fontsize=8.5, color=MUTED if same else INK2, annotation_clip=False)
    if not same:
        ax.annotate(f"{a / b:.1f}×", (1.235, y), xycoords=("axes fraction", "data"),
                    va="center", fontsize=8.5, color=C_NEW, weight="bold",
                    annotation_clip=False)
    yticks.append(y)
    ylabels.append(lbl)
    prev, y = op, y + 1

ax.set_xscale("log")
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_ylim(y - 0.4, -1.5)
ax.set_xlim(4e-4, 0.4)
dress(ax, xlabel="max |result − exact| on the segment   (log scale)", axis="x")
for op, ys in groups.items():
    ax.annotate(TITLES[op], (0.0, min(ys) - 0.75), xycoords=("axes fraction", "data"),
                fontsize=9.5, color=INK, weight="bold", annotation_clip=False)
ax.set_title("Worst error per LUT segment — main vs retuned constants", loc="left",
             color=INK, fontsize=11.5, weight="bold", pad=26)
ax.annotate("n300 (Wormhole B0), Float32→Float32, dest_acc=Yes; 250 samples per segment",
            (0.0, 1.0), xycoords="axes fraction", xytext=(0, 12),
            textcoords="offset points", ha="left", color=INK2, fontsize=9)
fig.legend(handles=LEGEND, ncol=2, loc="lower left", frameon=False,
           bbox_to_anchor=(0.09, -0.03), fontsize=9, labelcolor=INK2, handlelength=1.8)
fig.tight_layout(rect=[0, 0.02, 1, 1])
save(fig, "fig1-segment-max-error.svg")

# --------------------------------------------- figs 2-3: value + error, tanh & sigmoid
for op, fname, exact_name in (("sigmoid_appx", "fig2-sigmoid-appx.svg", "sigmoid"),
                              ("tanh", "fig3-tanh.svg", "tanh")):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 3.5))
    lo, hi = 0.0, 4.0
    knees(axl, op, lo, hi)
    pts = window("main", op, lo, hi, 3)
    axl.plot([p[0] for p in pts], [p[1] for p in pts], color=C_EXACT, lw=1.3, zorder=2)
    for tag, c, lw in (("main", C_MAIN, 1.9), ("retuned", C_NEW, 2.1)):
        pts = window(tag, op, lo, hi, 1)
        axl.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=lw, zorder=3,
                 solid_capstyle="round")
    axl.set_xlim(lo, hi)
    dress(axl, xlabel="|x|   (rules = LUT breakpoints)", ylabel="kernel output",
          title=f"Output against exact {exact_name}")

    knees(axr, op, lo, hi)
    axr.axhline(0, color=AXIS, lw=1.0, zorder=2)
    for tag, c, lw in (("main", C_MAIN, 1.9), ("retuned", C_NEW, 2.1)):
        pts = window(tag, op, lo, hi, 2)
        axr.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=lw, zorder=3,
                 solid_capstyle="round")
        w = max(pts, key=lambda p: abs(p[1]))
        axr.plot([w[0]], [w[1]], "o", ms=5.5, color=c, mec=SURFACE, mew=1.5, zorder=5)
        axr.annotate(f"{abs(w[1]):.4f}", w, xytext=(7, 3 if w[1] >= 0 else -12),
                     textcoords="offset points", fontsize=8.5, color=c, weight="bold")
    axr.set_xlim(lo, hi)
    dress(axr, xlabel="|x|", ylabel="result − exact", title="Signed error")

    fig.suptitle(TITLES[op], x=0.0, y=1.03, ha="left", color=INK, fontsize=11.5,
                 weight="bold")
    fig.legend(handles=LEGEND_EX, ncol=3, loc="lower left", frameon=False,
               bbox_to_anchor=(0.0, -0.07), fontsize=9, labelcolor=INK2, handlelength=1.8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save(fig, fname)

# ------------------------------------- fig 4: gelu, one small multiple per LUT segment
edges = SEGS["gelu_appx"]
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0))
for i, ax in enumerate(axes.ravel()):
    lo, hi = edges[i], edges[i + 1]
    last = i == len(edges) - 2
    ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
    for tag, c, lw in (("main", C_MAIN, 1.9), ("retuned", C_NEW, 2.1)):
        pts = [p for p in window(tag, "gelu_appx", lo, hi, 2) if p[0] < hi or last]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=lw, zorder=3,
                solid_capstyle="round")
        w = max(pts, key=lambda p: abs(p[1]))
        ax.plot([w[0]], [w[1]], "o", ms=5, color=c, mec=SURFACE, mew=1.5, zorder=5)
    ax.set_xlim(lo, hi if not last else 4.0)
    ttl = f"[{lo:g}, {hi:g})" if not last else f"[{lo:g}, ∞)"
    m, n = seg_max("main", "gelu_appx", i), seg_max("retuned", "gelu_appx", i)
    sub = "unchanged" if abs(m - n) <= 1e-9 * max(m, n) else f"{m:.4f} → {n:.4f}  ({m/n:.1f}×)"
    dress(ax, xlabel="|x|" if i >= 3 else None,
          ylabel="result − exact" if i % 3 == 0 else None, title=f"{ttl}\n{sub}")
fig.suptitle("gelu_appx — signed error per LUT segment (each panel on its own scale)",
             x=0.0, y=1.02, ha="left", color=INK, fontsize=11.5, weight="bold")
fig.legend(handles=LEGEND, ncol=2, loc="lower left", frameon=False,
           bbox_to_anchor=(0.0, -0.045), fontsize=9, labelcolor=INK2, handlelength=1.8)
fig.tight_layout(rect=[0, 0.015, 1, 0.975])
save(fig, "fig4-gelu-appx.svg")

print("wrote:", sorted(os.listdir(OUT)))
