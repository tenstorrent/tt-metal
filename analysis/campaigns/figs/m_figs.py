#!/usr/bin/env python
"""Page B figures (m_ prefix): compute floor against the counters, the wall law drawn as blocks, measured
against model per-step terms, every wall of the campaign predicted against measured, the hold-out errors,
the attribute flow table and the decode law against the T2.8 sweep.

Read-only on every input (data/, bh/, model/ and the polaris working tree, imported in place); writes PNGs
to figs/ only. Run with the polaris venv python:
    $POLARIS/.venv/bin/python m_figs.py
"""
import csv
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

# Paths per PORTABLE_CONTRACT.md: ROOT is handoff/revamp (the directory above this file's),
# WORK the SDPA root above that; both are environment overrides.
ROOT = os.environ.get("HANDOFF", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("SDPA_WORK", os.path.dirname(os.path.dirname(ROOT)))
POLARIS = os.environ.get("POLARIS", os.path.join(WORK, "polaris"))
OUT = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, BG = "#0b0b0b", "#fcfcfb"
GRAY = "#b9b9b5"
MUTED = "#5c5c58"
DARKGRAY = "#7a7a76"
BAND10 = "#ececea"
BAND5 = "#dcdcd8"
LIGHTBLUE, LIGHTORANGE = "#9dc1ea", "#f5b79c"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.color": "#e4e4e0",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
        "hatch.linewidth": 0.9,
    }
)
W_IN = 1600 / 150  # 1600 px at 150 dpi
CLK = 1350.0  # cycles per us at 1.35 GHz

ANCHOR = "causal S4096 q128 k128 nh32 nkv8 bfp8 HiFi2, 110 cores, p100a, firmware 19.9.0 (today)"
GRID1_DESC = "S4096 nh32 nkv8 bfp8 HiFi2, 110 cores, p100a, firmware 19.9.0 (today), T2.1 zones-off walls"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def heading(fig, title, subtitle=None, y=0.975, title_w=92, sub_w=148, sub_size=9.5):
    """Bold title and muted subtitle at the top left; wrap widths chosen for 1600 px at these sizes."""
    t = "\n".join(textwrap.wrap(title, title_w))
    fig.text(0.012, y, t, fontsize=13, fontweight="bold", va="top", ha="left", color=INK)
    if subtitle:
        n = t.count("\n") + 1
        h = fig.get_size_inches()[1]
        st = "\n".join(textwrap.wrap(subtitle, sub_w))
        fig.text(0.012, y - (0.24 / h) * n - 0.008, st, fontsize=sub_size, va="top", ha="left", color=MUTED)


def footer(fig, text, w=165):
    fig.text(0.01, 0.006, "\n".join(textwrap.wrap(text, w)), fontsize=8, color=MUTED, ha="left", va="bottom")


def fz(v, nd=1):
    """Signed percent string without a negative zero."""
    if abs(v) < 0.5 * 10 ** (-nd):
        v = 0.0
    return f"{v:+.{nd}f}"


def text_px(txt, fontsize, dpi=150):
    """Rough rendered width of a label in pixels (DejaVu Sans, mixed digits and letters)."""
    return len(txt) * fontsize * dpi / 72 * 0.56


# ---------------------------------------------------------------------------------------------
# Model snapshots (git show, read-only) and config sets
# ---------------------------------------------------------------------------------------------
MODEL_LABEL = "branch mvlahovic/sdpa_revamp"  # how the figures name the model in titles and footers


def load_model():
    """Import ttsim/perf/roofline_sdpa.py from the polaris working tree (the branch head)."""
    sys.path.insert(0, POLARIS)
    import ttsim.perf.roofline_sdpa as m

    return m


HEAD = load_model()
sys.path.insert(0, os.path.join(ROOT, "model"))
import refit_r2_fit as RF  # the campaign wall tables and their sets (model/refit_r2_fit.py)


def mk(m, **kw):
    """SdpaConfig of module m from a superset of fields (fields the module does not know are dropped)."""
    fields = m.SdpaConfig.__dataclass_fields__
    cores = kw.pop("num_cores", 110)
    kw = {k: v for k, v in kw.items() if k in fields}
    return m.SdpaConfig(num_cores=cores, arch=m.ARCH_BH, **kw)


GQA = dict(num_heads=32, num_kv_heads=8)
MLA = dict(
    head_dim=576, v_head_dim=512, num_kv_heads=1, fidelity="HiFi4", input_dtype="bfloat16", exp_approx_mode=False
)
PROD = dict(
    num_heads=32, num_kv_heads=8, fidelity="HiFi4", exp_approx_mode=False, fp32_dest_acc=True, accum_dtype="float32"
)

# Grid 1: (short label, tag in decomp_summary_t21_t22.csv, config kwargs on top of S4096 + GQA)
GRID1 = [
    ("causal q128 k128", "t21_causal_q128k128", dict()),
    ("causal q128 k256", "t21_causal_q128k256", dict(k_chunk=256)),
    ("causal q128 k512", "t21_causal_q128k512", dict(k_chunk=512)),
    ("causal q64 k128", "t21_causal_q64k128", dict(q_chunk=64)),
    ("causal q256 k128", "t21_causal_q256k128", dict(q_chunk=256)),
    ("causal q512 k128", "t21_causal_q512k128", dict(q_chunk=512)),
    ("causal q512 k512", "t21_causal_q512k512", dict(q_chunk=512, k_chunk=512)),
    ("non-causal q128 k128", "t21_noncausal_q128k128", dict(is_causal=False)),
    ("non-causal q128 k256", "t21_noncausal_q128k256", dict(is_causal=False, k_chunk=256)),
    ("non-causal q128 k512", "t21_noncausal_q128k512", dict(is_causal=False, k_chunk=512)),
    ("non-causal q64 k128", "t21_noncausal_q64k128", dict(is_causal=False, q_chunk=64)),
    ("non-causal q256 k128", "t21_noncausal_q256k128", dict(is_causal=False, q_chunk=256)),
    ("non-causal q512 k128", "t21_noncausal_q512k128", dict(is_causal=False, q_chunk=512)),
    ("non-causal q512 k512", "t21_noncausal_q512k512", dict(is_causal=False, q_chunk=512, k_chunk=512)),
]

# Production Llama 3.1 8B SDPA (bh/production_config.md section 2): 64 cores, HiFi4, accurate exp, fp32 DEST,
# bfp8, causal; q = k = 64 at S1024, 256 above. Zones-off device walls, mean of 2 invocations, cycles.
PRODUCTION = [
    ("production S1024 q64 k64 g64", dict(S=1024, q_chunk=64, k_chunk=64, num_cores=64, **PROD), 468469),
    ("production S2048 q256 k256 g64", dict(S=2048, q_chunk=256, k_chunk=256, num_cores=64, **PROD), 1072058),
    ("production S4096 q256 k256 g64", dict(S=4096, q_chunk=256, k_chunk=256, num_cores=64, **PROD), 3983028),
    ("production S8192 q256 k256 g64", dict(S=8192, q_chunk=256, k_chunk=256, num_cores=64, **PROD), 15392503),
]

# T2.3r regime walls on today's card (bh/regimes.md section 1, zones off, mean of iterations 1 and 2, cycles).
REGIMES = [
    ("windowed S8192 W1024 nh16", dict(S=8192, num_heads=16, sliding_window=1024), 1411340),
    ("chunked S2048 start4096 nh16", dict(S=2048, kv_seq=6144, chunk_start_idx=4096, num_heads=16), 2026025),
    ("MLA nh16 S2048 q32", dict(S=2048, q_chunk=32, num_heads=16, **MLA), 9574958),
    ("cross 2k/8k nh16", dict(S=2048, kv_seq=8192, num_heads=16, is_causal=False), 1649702),
    ("dense mask S8192 nh16", dict(S=8192, num_heads=16, is_causal=False, has_attn_mask=True), 7717703),
    ("causal S16384 q128 k128", dict(S=16384, **GQA), 38629694),
    ("causal S1024 q128 k128", dict(S=1024, **GQA), 258568),
]

# T2.8 decode: the op as tt_transformers issues it (HiFi2, fp32 accumulation, accurate exp, Q bf16 in DRAM,
# KV bfp8 in 32-row pages, k_chunk 0 = the kernel's dynamic rule, max_cores_per_head_batch 16), nh32 nkv8 d128.
DECODE_KW = dict(
    cache_len=8192,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    k_chunk=0,
    fidelity="HiFi2",
    input_dtype="bfloat16",
    accum_dtype="float32",
    paged=True,
    page_block_size=32,
    max_cores_per_head_batch=16,
)
# T2.5 validation rows (data/model_level/validation/llama8b_attn_decode_b32_pos128_1024_4096_signed_errors.csv):
# batch 32, 64 grid, query height sharded in L1; DEVICE KERNEL DURATION ns of the three invocations per position.
DECODE_VALIDATION = {128: [71345, 71559, 72378], 1024: [282083, 281932, 282039], 4096: [1006497, 1004604, 1005454]}


def read_r1f_counters():
    """R1f unmodified-kernel counters (mean over the active cores, run_idx 1 and 2 averaged), keyed by the grid 1 tag."""
    path = os.path.join(ROOT, "data", "bh_zones", "r1f_counters_table.csv")
    with open(path) as f:
        rows = [r for r in csv.DictReader([l for l in f if not l.startswith("#")]) if int(r["run_idx"]) > 0]
    acc = {}
    for r in rows:
        tag = r["tag"].replace("r1f_", "t21_").replace("_zoff_mp", "")
        acc.setdefault(tag, []).append(r)
    out = {}
    for tag, rs in acc.items():
        out[tag] = {
            c: sum(float(r[f"{c}_mean_active"]) for r in rs) / len(rs)
            for c in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER")
        }
    return out


def read_decomp_summary():
    """The 26-row T2.1 / T2.2 summary preserved before the T2.4 rewrite (PROVENANCE line skipped)."""
    path = os.path.join(ROOT, "data", "t42_scripts", "decomp_summary_t21_t22.csv")
    with open(path) as f:
        rows = [l for l in f if not l.startswith("#")]
    return {r["tag"]: r for r in csv.DictReader(rows)}


def grid1_walls():
    """Measured zones-off device wall per grid 1 tag (mean of iterations 1 and 2)."""
    ds = read_decomp_summary()
    return {tag: float(ds[tag]["wall_zoff_mean"]) for _, tag, _ in GRID1}


def read_decode_sweep():
    path = os.path.join(ROOT, "data", "bh_zones", "decode_sweep_table.csv")
    with open(path) as f:
        rows = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(rows))


def wall_of(m, kw):
    return m.predict(mk(m, **kw)).wall_clock_cycles


def decode_wall(m, batch, cores, kv, pos, q_in_dram=True):
    """predict_decode of snapshot m for one T2.8 or T2.5 point; the q_in_dram flag exists from f0cfad7 on."""
    kw = dict(DECODE_KW, batch=batch, num_cores=cores, cur_pos=pos, kv_input_dtype=kv, arch=m.ARCH_BH)
    if "q_in_dram" in m.predict_decode.__code__.co_varnames:
        kw["q_in_dram"] = q_in_dram
    return m.predict_decode(**kw)


def place_labels(
    ax, pts, texts, avoid=(), obstacles=(), fontsize=8.0, color=INK, pad_pt=5, marker_pt=6.5, leader_from_pt=9, **kw
):
    """Collision-free point labels, greedy: for each point try positions around it (right, left, above,
    below, the diagonals, then the same ring farther out) and keep the first whose text box clears every
    marker (pts and avoid), every text artist in obstacles, every label placed before it and the axes frame;
    a leader line is drawn when the label had to move away. Works in display pixels, so the layout must be
    final (draw() is called here)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72
    P = ax.transData.transform(np.array(pts, dtype=float))
    A = ax.transData.transform(np.array(avoid, dtype=float)) if len(avoid) else np.zeros((0, 2))
    markers = np.vstack([P, A])
    axbb = ax.get_window_extent(renderer)
    r = marker_pt * px
    boxes = [o.get_window_extent(renderer).expanded(1.05, 1.1) for o in obstacles]
    cands = []
    for k in (1.0, 2.0, 3.2, 4.6):
        d = pad_pt * px * k
        cands += [
            (d, 0, "left", "center"),
            (-d, 0, "right", "center"),
            (0, d, "center", "bottom"),
            (0, -d, "center", "top"),
            (d, d, "left", "bottom"),
            (d, -d, "left", "top"),
            (-d, d, "right", "bottom"),
            (-d, -d, "right", "top"),
        ]
    inv = ax.transData.inverted()

    def clear(bb, i):
        for j, (mx, my) in enumerate(markers):
            if j != i and bb.x0 - r < mx < bb.x1 + r and bb.y0 - r < my < bb.y1 + r:
                return False
        if any(bb.overlaps(ob) for ob in boxes):
            return False
        return axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1

    out = []
    for i, txt in enumerate(texts):
        chosen = None
        for dx, dy, ha, va in cands:
            xd, yd = inv.transform((P[i, 0] + dx, P[i, 1] + dy))
            t = ax.text(xd, yd, txt, fontsize=fontsize, color=color, ha=ha, va=va, zorder=8, **kw)
            bb = t.get_window_extent(renderer).expanded(1.12, 1.2)
            if clear(bb, i):
                chosen = (t, bb, dx, dy)
                break
            t.remove()
        if chosen is None:
            dx, dy, ha, va = cands[0]
            xd, yd = inv.transform((P[i, 0] + dx, P[i, 1] + dy))
            t = ax.text(xd, yd, txt, fontsize=fontsize, color=color, ha=ha, va=va, zorder=8, **kw)
            chosen = (t, t.get_window_extent(renderer).expanded(1.12, 1.2), dx, dy)
        t, bb, dx, dy = chosen
        boxes.append(bb)
        if max(abs(dx), abs(dy)) > leader_from_pt * px:
            lx = min(max(P[i, 0], bb.x0), bb.x1)
            ly = min(max(P[i, 1], bb.y0), bb.y1)
            x0, y0 = inv.transform((P[i, 0], P[i, 1]))
            x1, y1 = inv.transform((lx, ly))
            ax.plot([x0, x1], [y0, y1], color=GRAY, lw=0.6, zorder=1)
        out.append(t)
    return out


# ---------------------------------------------------------------------------------------------
# 1. m_floor_vs_counters_today.png
# ---------------------------------------------------------------------------------------------
def fig_floor_vs_counters():
    ds = read_r1f_counters()
    zoned = read_decomp_summary()
    panels = [
        ("FPU", "fpu_model", "FPU_COUNTER", "fpu_cycles", "FPU_COUNTER"),
        ("SFPU", "sfpu_model", "SFPU_COUNTER", "sfpu_cycles", "SFPU_COUNTER"),
        ("MATH union", "floor_model", "MATH_COUNTER", "math_active_cycles", "MATH_COUNTER"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(W_IN, 10.4))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.835, bottom=0.16, wspace=0.3, hspace=0.45)
    summary = []
    fmt = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v / 1e6:.2f} M" if v >= 1e6 else f"{v / 1e3:.0f}k")
    for row, want_causal in enumerate([True, False]):
        for col, (name, mcol, ccol, hattr, cname) in enumerate(panels):
            ax = axes[row][col]
            xs, ys, hs, labs = [], [], [], []
            for lab, tag, kw in GRID1:
                r = ds[tag]
                if kw.get("is_causal", True) != want_causal:
                    continue
                xs.append(float(r[ccol]))
                ys.append(getattr(HEAD.predict(mk(HEAD, S=4096, **kw, **GQA)), hattr))
                hs.append(ys[-1])
                labs.append(lab.replace("non-causal ", "").replace("causal ", ""))
            xs, ys, hs = np.array(xs), np.array(ys), np.array(hs, dtype=float)
            lo, hi = min(xs.min(), ys.min(), hs.min()) * 0.93, max(xs.max(), ys.max(), hs.max()) * 1.07
            g = np.array([lo, hi])
            ax.fill_between(g, g * 0.90, g * 1.10, color=BAND10, lw=0, zorder=0)
            ax.fill_between(g, g * 0.95, g * 1.05, color=BAND5, lw=0, zorder=0)
            ax.plot(g, g, color=INK, lw=1.0, zorder=1)
            crosses = []
            for i in range(len(xs)):
                if want_causal:
                    ax.plot(xs[i], ys[i], "o", ms=8.5, color=BLUE, markeredgecolor=BG, markeredgewidth=1.2, zorder=4)
                else:
                    ax.plot(
                        xs[i],
                        ys[i],
                        "s",
                        ms=8.5,
                        markerfacecolor=BG,
                        markeredgecolor=ORANGE,
                        markeredgewidth=1.8,
                        zorder=4,
                    )
                if abs(hs[i] - ys[i]) / ys[i] > 0.005:
                    ax.annotate(
                        "",
                        xy=(xs[i], hs[i]),
                        xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.2, shrinkA=4, shrinkB=0),
                        zorder=5,
                    )
                    ax.plot(xs[i], hs[i], "x", ms=7.5, color=AQUA, markeredgewidth=1.8, zorder=6)
                    crosses.append((xs[i], hs[i]))
            err = 100 * (ys / xs - 1)
            summary.append((("causal " if want_causal else "non-causal ") + name, err.mean(), err.min(), err.max()))
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            ax.tick_params(labelsize=8.5)
            ax.set_title(f"{'causal' if want_causal else 'non-causal'}: model {name}", fontsize=10.5)
            ax.set_xlabel(f"{cname}, mean of 110 cores, cycles", fontsize=9)
            if col == 0:
                ax.set_ylabel("model, cycles (mean core)", fontsize=9)
            stats = ax.text(
                0.97,
                0.04,
                f"mean {fz(err.mean())} %, range {fz(err.min())} to {fz(err.max())} %",
                transform=ax.transAxes,
                fontsize=8.4,
                va="bottom",
                ha="right",
                color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=GRAY, lw=0.6),
                zorder=9,
            )
            obstacles = [stats]
            if want_causal and col == 2:
                # the anchor's MATH counter on the zoned kernel sits above the unmodified one (RISC-V zone time on TRISC1)
                anchor_union = ys[labs.index("q128 k128")]
                zm = float(zoned["t21_causal_q128k128"]["ctr_MATH_COUNTER_mean"])
                ax.plot([zm], [anchor_union], marker="|", ms=13, color=INK, markeredgewidth=2.0, ls="", zorder=7)
                ann = ax.annotate(
                    f"anchor, zoned kernel: MATH {zm:,.0f}\n(the unmodified kernel reads {xs[labs.index('q128 k128')]:,.0f})",
                    xy=(zm, anchor_union),
                    xytext=(0.03, 0.97),
                    textcoords="axes fraction",
                    fontsize=7.9,
                    color=INK,
                    ha="left",
                    va="top",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=7),
                    zorder=9,
                )
                crosses.append((zm, anchor_union))
                obstacles.append(ann)
            place_labels(ax, list(zip(xs, ys)), labs, avoid=crosses, obstacles=obstacles, fontsize=8.0, color=MUTED)
    handles = [
        Line2D([], [], marker="o", color=BLUE, ls="", ms=8, label="causal configs (top row)"),
        Line2D(
            [],
            [],
            marker="s",
            markerfacecolor=BG,
            markeredgecolor=ORANGE,
            markeredgewidth=1.8,
            ls="",
            ms=8,
            label="non-causal configs (bottom row)",
        ),
        Line2D(
            [],
            [],
            marker="|",
            color=INK,
            ls="",
            ms=11,
            markeredgewidth=2.0,
            label="anchor MATH counter of the zoned kernel",
        ),
        Patch(facecolor=BAND5, label="plus or minus 5 percent"),
        Patch(facecolor=BAND10, label="plus or minus 10 percent"),
        Line2D([], [], color=INK, lw=1.0, label="y = x"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.038), ncol=2, fontsize=8.6)
    heading(
        fig,
        "Compute floor against the R1f counters: model FPU, SFPU and MATH union for the 14 grid 1 configs",
        "S4096 nh32 nkv8 bfp8 HiFi2, 110 cores, p100a firmware 19.9.0; R1f perf-counter multipass captures on the unmodified kernel (zones compiled out), "
        "mean over the 110 cores, run_idx 1 and 2 averaged. Model values are predict() "
        f"of polaris ttsim/perf/roofline_sdpa.py ({MODEL_LABEL}, mean core): the FPU and SFPU laws and the overlap law are fit on these counters "
        "(the unmodified-kernel MATH basis for the overlap). Top row causal, bottom row non-causal, each panel zoomed to its own range.",
    )
    footer(
        fig,
        "MODEL vs MEASURED. Data: data/bh_zones/r1f_counters_table.csv (FPU_COUNTER, SFPU_COUNTER, MATH_COUNTER mean_active; bh/campaign_r1.md s9); the zoned anchor "
        f"MATH from data/t42_scripts/decomp_summary_t21_t22.csv; model values from polaris ttsim/perf/roofline_sdpa.py predict() ({MODEL_LABEL}).",
    )
    save(fig, "m_floor_vs_counters_today.png")
    for name, mn, lo, hi in summary:
        print(f"  {name}: mean {mn:+.1f} %, range {lo:+.1f} to {hi:+.1f} %")


# ---------------------------------------------------------------------------------------------
# 2. m_terms_blocks.png
# ---------------------------------------------------------------------------------------------
TERM_COL = {
    "compute_floor": AQUA,
    "control": DARKGRAY,
    "mask_bracket": RED,
    "fe_issue": ORANGE,
    "reader_wait": BLUE,
    "sfpu_issue": BG,
    "dest_roundtrip": ORANGE,
    "init": GRAY,
    "straggler": GRAY,
}
TERM_HATCH = {"sfpu_issue": "xx", "dest_roundtrip": "//", "straggler": "\\\\"}


def step_lanes(m, kw):
    """Per-step lanes and booked components of predict(kw) at the refit snapshot, all in cycles per step of the
    mean core (Q x K_eff steps); the wall core adds its extra steps as the straggler term."""
    r = m.predict(mk(m, **kw))
    a = m.ARCH_BH
    t = m.WALL_TERMS_BH[r.wall_regime]
    cfg = mk(m, **kw)
    qct, kct = cfg.q_chunk // 32, cfg.k_chunk // 32
    dct_qk = -(-cfg.head_dim // 32)
    dct_v = -(-(cfg.v_head_dim or cfg.head_dim) // 32)
    kv_dtype = cfg.kv_input_dtype or cfg.input_dtype
    kvbpt = m.BYTES_PER_TILE[kv_dtype]
    steps = r.q_chunks_per_core * r.k_eff
    bytes_step = kct * (dct_qk + dct_v) * kvbpt
    rate = (
        (a.kv_injector_rate_bpc + a.kv_injector_rate_bpc_per_ktile * kct)
        if t.dram_law == "injector"
        else m._kv_stream_rate(a, r.active_cores)
    )
    dram_fixed = kct * a.kv_stream_fixed_per_ktile * t.stream_scale if t.dram_law != "none" else 0.0
    dram_bytes = bytes_step / rate * t.stream_scale if t.dram_law != "none" else 0.0
    dram = m._kv_stream_step(a, t, kct=kct, kv_bytes_per_step=bytes_step, reading_cores=r.active_cores)
    legacy = r.kernel_path == "legacy"
    pack = 0.0 if legacy else t.pack_per_step + qct * (t.pack_per_qtile + t.pack_per_qktile * kct)
    per_step = {k: v / steps for k, v in r.components.items()}
    return dict(
        r=r,
        steps=steps,
        qct=qct,
        kct=kct,
        bytes_step=bytes_step,
        rate=rate,
        dram=dram,
        dram_fixed=dram_fixed,
        dram_bytes=dram_bytes,
        dram_scale=t.stream_scale,
        pack=pack,
        pack_parts=(t.pack_per_step, qct * t.pack_per_qtile, qct * t.pack_per_qktile * kct),
        per_step=per_step,
        legacy=legacy,
        active=r.active_cores,
        t=t,
        a=a,
    )


def draw_segments(ax, y, segs, h=0.62, fontsize=8.6, px_per_unit=None):
    """Stacked horizontal segments [(value, colour, hatch, text)] from x = 0. A text is drawn inside its
    segment only when it fits; skipped texts are returned so the caller can place them elsewhere."""
    left = 0.0
    skipped = []
    for v, col, hatch, txt in segs:
        if v <= 0:
            continue
        ax.barh(
            y,
            v,
            left=left,
            height=h,
            color=col,
            edgecolor=INK if hatch else BG,
            hatch=hatch,
            linewidth=0.9 if hatch else 1.2,
            zorder=3,
        )
        if txt:
            fits = px_per_unit is None or v * px_per_unit > text_px(txt, fontsize) + 14
            if fits:
                dark = col in (BLUE, DARKGRAY, MUTED) and not hatch
                ax.text(
                    left + v / 2,
                    y,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=BG if dark else INK,
                    zorder=4,
                )
            else:
                skipped.append(txt)
        left += v
    return left, skipped


def lane_label(ax, x, y, head, formula, px_per_unit, xmax, fontsize=8.8, bold=False):
    """Lane total plus its formula to the right of a lane; two lines when one line would run off the axes."""
    room = (xmax - x) * px_per_unit - 10
    one = f"{head} = {formula}" if formula else head
    if text_px(one, fontsize) <= room:
        ax.text(x, y, one, va="center", fontsize=fontsize, color=INK, fontweight="bold" if bold else "normal")
    else:
        ax.text(
            x,
            y,
            f"{head}\n= {formula}",
            va="center",
            fontsize=fontsize,
            color=INK,
            linespacing=1.15,
            fontweight="bold" if bold else "normal",
        )


def lane_block(ax, L, title, note):
    """Three lanes (stream, PACK, compute) and the booked step for one config."""
    fig = ax.figure
    fig.canvas.draw()
    xmax = ax.get_xlim()[1]
    px_per_unit = ax.get_window_extent(fig.canvas.get_renderer()).width / xmax
    ps = L["per_step"]
    a, t = L["a"], L["t"]
    ys = {"stream": 3, "pack": 2, "compute": 1, "booked": -0.1}
    ax.set_ylim(-0.75, 4.05)
    pad = 0.006 * xmax
    # stream lane
    if L["dram"] > 0:
        segs = [
            (L["dram_fixed"], LIGHTBLUE, "", f"{L['dram_fixed']:,.0f}"),
            (L["dram_bytes"], BLUE, "", f"{L['dram_bytes']:,.0f}"),
        ]
        formula = f"kct x {a.kv_stream_fixed_per_ktile:,.0f} + {L['bytes_step']:,.0f} B / {L['rate']:.3f} B per cycle"
        if L["dram_scale"] != 1.0:
            formula = f"({formula}) x {L['dram_scale']:g}"
        draw_segments(ax, ys["stream"], segs, px_per_unit=px_per_unit)
        lane_label(ax, L["dram"] + pad, ys["stream"], f"stream lane {L['dram']:,.0f}", formula, px_per_unit, xmax)
    else:
        ax.text(pad, ys["stream"], "no DRAM stream law for this regime", va="center", fontsize=8.8, color=MUTED)
    # PACK lane
    if L["legacy"]:
        ax.text(
            pad,
            ys["pack"],
            "no PACK issue lane: the exp runs on the MATH thread (legacy kernel path)",
            va="center",
            fontsize=8.8,
            color=MUTED,
        )
    else:
        p0, p1, p2 = L["pack_parts"]
        segs = [
            (p0, LIGHTORANGE, "", f"{p0:,.0f}"),
            (p1, LIGHTORANGE, "", f"{p1:,.0f}"),
            (p2, ORANGE, "", f"{p2:,.0f}"),
        ]
        formula = (f"{p0:,.0f} + " if p0 else "") + f"qct x {t.pack_per_qtile:g} + qct x kct x {t.pack_per_qktile:g}"
        draw_segments(ax, ys["pack"], segs, px_per_unit=px_per_unit)
        lane_label(ax, L["pack"] + pad, ys["pack"], f"PACK issue lane {L['pack']:,.0f}", formula, px_per_unit, xmax)
    # compute lane
    segs = [
        (ps["compute_floor"], AQUA, "", f"{ps['compute_floor']:,.0f}"),
        (ps["mask_bracket"], RED, "", ""),
        (ps["control"], DARKGRAY, "", f"{ps['control']:,.0f}"),
    ]
    formula = f"floor {ps['compute_floor']:,.0f} + mask {ps['mask_bracket']:,.0f} + control {ps['control']:,.0f}"
    if L["legacy"]:
        segs.append((ps["dest_roundtrip"], ORANGE, "//", f"{ps['dest_roundtrip']:,.0f}"))
        formula += f" + DEST {ps['dest_roundtrip']:,.0f}"
    comp, _ = draw_segments(ax, ys["compute"], segs, px_per_unit=px_per_unit)
    lane_label(ax, comp + pad, ys["compute"], f"compute lane {comp:,.0f}", formula, px_per_unit, xmax)
    # booked step
    order = ["compute_floor", "control", "mask_bracket", "fe_issue", "reader_wait", "sfpu_issue", "dest_roundtrip"]
    names = {
        "compute_floor": "compute_floor",
        "control": "control",
        "mask_bracket": "mask",
        "fe_issue": "fe_issue",
        "reader_wait": "reader_wait",
        "sfpu_issue": "sfpu_issue",
        "dest_roundtrip": "dest_roundtrip",
    }
    segs = [(ps[k], TERM_COL[k], TERM_HATCH.get(k, ""), f"{names[k]} {ps[k]:,.0f}") for k in order]
    tot, skipped = draw_segments(ax, ys["booked"], segs, h=0.7, px_per_unit=px_per_unit)
    small = ", ".join(skipped)
    lane_label(
        ax,
        tot + pad,
        ys["booked"],
        f"booked step {tot:,.0f} = the longest lane",
        f"({small})" if small else "",
        px_per_unit,
        xmax,
        bold=True,
    )
    ax.axvline(tot, color=INK, lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.set_yticks([ys["stream"], ys["pack"], ys["compute"], ys["booked"]])
    ax.set_yticklabels(["DRAM K/V\nstream lane", "PACK issue\nlane", "compute\nlane", "booked\ncomponents"], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("cycles per k-chunk step", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.text(xmax * 0.995, 3.82, note, ha="right", va="center", fontsize=8.6, color=MUTED)


def fig_terms_blocks():
    m = HEAD
    a = m.ARCH_BH
    tc = m.WALL_TERMS_BH["prefill_causal"]
    tn = m.WALL_TERMS_BH["prefill_noncausal"]
    anchor = step_lanes(m, dict(S=4096, **GQA))
    packb = step_lanes(m, dict(S=4096, q_chunk=256, **GQA))
    prod = step_lanes(m, dict(S=4096, q_chunk=256, k_chunk=256, num_cores=64, **PROD))
    ra = anchor["r"]
    fig = plt.figure(figsize=(W_IN, 15.0))
    gs = fig.add_gridspec(
        4, 1, height_ratios=[0.66, 1, 1, 1], hspace=0.6, left=0.10, right=0.985, top=0.842, bottom=0.14
    )
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    # top strip: the wall
    steps_mean = anchor["steps"]
    step_cyc = sum(v for k, v in anchor["per_step"].items() if k not in ("init", "straggler"))
    init = ra.components["init"]
    body = steps_mean * step_cyc
    strag = ra.components["straggler"]
    total = init + body + strag
    ax0.set_xlim(0, total * 1.36)
    ax0.set_ylim(-1.4, 1.35)
    draw_segments(
        ax0,
        0,
        [
            (init, GRAY, "", ""),
            (
                body,
                "#e8e8e4",
                "",
                f"Q x K_eff = {steps_mean:.1f} steps of {step_cyc:,.0f} cycles = {body:,.0f}   (mean core: Q = {ra.q_chunks_per_core:.2f} chunks, K_eff = {ra.k_eff})",
            ),
            (strag, GRAY, "\\\\", ""),
        ],
        h=0.8,
        fontsize=9,
    )
    ax0.annotate(
        f"init {init:,.0f} per launch: prologue 171 + tail 1,706 + between q chunks 379 + kernel start skew 630 to 720",
        xy=(init, 0.4),
        xytext=(init + 0.02 * total, 1.05),
        fontsize=8.4,
        color=INK,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7),
    )
    ax0.text(
        init + body + strag / 2,
        -0.52,
        f"straggler {strag:,.0f}: the wall core owns {ra.q_chunks_wall_core} chunks (pair rule) = {ra.steps_wall_core:.0f} steps,\n"
        f"{ra.steps_wall_core - steps_mean:.1f} more than the mean core, at the same per-step cost",
        ha="center",
        va="top",
        fontsize=8.4,
        color=INK,
    )
    err = 100 * (ra.wall_clock_cycles / 2562642 - 1)
    ax0.text(
        total + 0.012 * total,
        0,
        f"wall {ra.wall_clock_cycles:,.0f} cycles = {ra.wall_clock_cycles / CLK:,.1f} us\nmeasured 2,562,642 ({fz(err, 2)} %)",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )
    ax0.set_yticks([])
    ax0.grid(False)
    ax0.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v / 1e6:.1f} M"))
    ax0.tick_params(axis="x", labelsize=8.5)
    ax0.set_title(
        "wall = init + Q x K_eff x step + straggler, at the anchor (every k-chunk step of a config costs the same)",
        fontsize=10.5,
    )
    # three step examples
    ax1.set_xlim(0, anchor["dram"] * 1.5)
    lane_block(
        ax1,
        anchor,
        "anchor, causal q128 k128: the stream lane binds; reader_wait = stream lane minus PACK lane",
        f"model {ra.wall_clock_cycles:,.0f} vs measured 2,562,642 ({fz(err, 2)} %)",
    )
    rb = packb["r"]
    ax2.set_xlim(0, max(packb["dram"], packb["pack"]) * 1.5)
    lane_block(
        ax2,
        packb,
        "causal q256 k128, same bytes per step: the PACK lane binds; fe_issue = PACK lane minus compute lane",
        f"model {rb.wall_clock_cycles:,.0f} vs measured 1,826,180 ({fz(100 * (rb.wall_clock_cycles / 1826180 - 1))} %); wall core {rb.q_chunks_wall_core} chunks x {rb.k_eff} visits = {rb.steps_wall_core:.0f} steps",
    )
    rp = prod["r"]
    comp_p = (
        prod["per_step"]["compute_floor"]
        + prod["per_step"]["dest_roundtrip"]
        + prod["per_step"]["control"]
        + prod["per_step"]["mask_bracket"]
    )
    ax3.set_xlim(0, comp_p * 1.55)
    lane_block(
        ax3,
        prod,
        "production Llama 8B, S4096 q256 k256 g64 (legacy path): the compute lane with its DEST round trips binds",
        f"model {rp.wall_clock_cycles:,.0f} vs measured 3,983,028 ({fz(100 * (rp.wall_clock_cycles / 3983028 - 1), 2)} %); 64 cores stream at {prod['rate']:.3f} B per cycle per core",
    )
    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v / 1e3:.0f}k" if v else "0"))
    handles = [
        Patch(facecolor=AQUA, label="compute_floor: FPU + SFPU union (overlap law), or the L1 floor"),
        Patch(
            facecolor=DARKGRAY,
            label=f"control: un-zoned control flow, {tc.control_per_kchunk:g} + {tc.control_per_qtile:g} qct (causal) or {tn.control_per_kchunk:g} + {tn.control_per_qtile:g} qct",
        ),
        Patch(facecolor=RED, label=f"mask_bracket: {tc.mask_branch_per_kchunk:g} per step on causal-like regimes (A2)"),
        Patch(facecolor=ORANGE, label="fe_issue: PACK lane minus compute lane, when positive (+ fe_flat)"),
        Patch(facecolor=BLUE, label="reader_wait: stream lane minus compute lane, when positive"),
        Patch(
            facecolor=BG,
            edgecolor=INK,
            hatch="xx",
            label="sfpu_issue: exp issue beyond max(PACK, FPU, stream); 0 on every point (A6)",
        ),
        Patch(
            facecolor=ORANGE,
            edgecolor=INK,
            hatch="//",
            label=f"dest_roundtrip (legacy path): {a.legacy_step_cycles:,.0f} + {a.dest_roundtrip_cycles:,.0f} per 4-tile DEST window",
        ),
        Patch(
            facecolor=GRAY,
            edgecolor=INK,
            hatch="\\\\",
            label=f"straggler: the wall core's extra steps (pair rule); init {a.wall_fixed_cycles:,.0f} per launch",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.036), ncol=2, fontsize=8.2)
    heading(
        fig,
        "The wall as named terms: per k-chunk step the core runs the longest of three lanes, and the components book that step",
        f"Law of model/refit_r2_notes.md s1 as implemented in polaris ttsim/perf/roofline_sdpa.py ({MODEL_LABEL}, _wall_components): step = max(stream lane, PACK lane, "
        "compute lane); reader_wait and fe_issue are the exposed parts of the two longer lanes; the nine components (init, compute_floor, fe_issue, reader_wait, "
        "mask_bracket, sfpu_issue, dest_roundtrip, control, straggler) sum to wall_clock_cycles; the per-q-chunk normalization and the writer's W_WRITE sit inside "
        "control and the PACK lane. All numbers are predict() output; block widths are proportional within each row.",
    )
    rates = a.kv_stream_rate_bpc
    footer(
        fig,
        f"MODEL (predict() on {MODEL_LABEL}, constants of refit_r2_notes.md s2: {a.kv_stream_fixed_per_ktile:,.0f} per k-tile, {rates[110]:.3f} / {rates[64]:.3f} B per cycle per core "
        f"at 110 / 64 cores, PACK {tc.pack_per_qtile:g} + {tc.pack_per_qktile:g} per q-tile per k-tile, control {tc.control_per_kchunk:g} + {tc.control_per_qtile:g} qct, mask "
        f"{tc.mask_branch_per_kchunk:g}, legacy {a.legacy_step_cycles:,.0f} + {a.dest_roundtrip_cycles:,.0f} per window). Measured walls quoted in the notes: "
        "data/bh_zones/report_tables.md T1 (grid 1), bh/production_config.md s2.",
    )
    save(fig, "m_terms_blocks.png")


# ---------------------------------------------------------------------------------------------
# 4. m_terms_measured_vs_default.png
# ---------------------------------------------------------------------------------------------
def fig_terms_measured_vs_default():
    """Per-step wall terms at the anchor: the measured value on the wall core against the model's booked terms."""
    L = step_lanes(HEAD, dict(S=4096, **GQA))
    new = L["per_step"]
    STEPS_WC = 165.0  # wall-core steps at the anchor (report_tables.md T1 it1_steps_wc)
    wall, floor_wc, prologue = 2562642.0, 915586.0, 171.0  # T1, T3
    a4_wall, a2_wall = 1466798.0, 2553244.0  # T6
    unpack_waits = 979638.0  # T2: K_WAIT + V_WAIT + Q_WAIT
    control_meas = 81963.0  # T3 un-zoned control flow
    stream_a4 = wall - a4_wall  # 1,095,844
    fe_free = a4_wall - floor_wc - prologue  # 551,041 (includes control and mask)
    mask_meas = wall - a2_wall  # 9,398
    fe_excl = fe_free - control_meas - mask_meas  # 459,680
    resid_meas = wall - floor_wc - prologue  # 1,646,885
    meas = {
        "compute_floor": floor_wc / STEPS_WC,
        "fe_issue": fe_excl / STEPS_WC,
        "reader_wait": unpack_waits / STEPS_WC,
        "mask_bracket": mask_meas / STEPS_WC,
        "sfpu_issue": 0.0,
        "control": control_meas / STEPS_WC,
    }
    meas_hi = {"reader_wait": stream_a4 / STEPS_WC, "fe_issue": fe_free / STEPS_WC}
    terms = ["fe_issue", "reader_wait", "mask_bracket", "sfpu_issue", "control"]
    meas_vals = [meas[t] for t in terms]
    new_vals = [new[t] for t in terms]
    resid_new = sum(v for k, v in new.items() if k not in ("init", "compute_floor", "straggler"))
    resid_ms = resid_meas / STEPS_WC
    fig = plt.figure(figsize=(W_IN, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.15], wspace=0.3, left=0.085, right=0.985, top=0.775, bottom=0.235)
    ax = fig.add_subplot(gs[0])
    axi = fig.add_subplot(gs[1])
    bw = 0.34
    series = [
        ("measured per step, wall core (zones, ablations)", ORANGE, meas_vals, -0.5),
        ("model: predict() components over the mean-core steps", BLUE, new_vals, 0.5),
    ]
    for name, col, vals, off in series:
        for i, v in enumerate(vals):
            x = i + off * bw
            ax.bar(
                x, v, width=bw * 0.92, color=col, edgecolor=BG, linewidth=0.8, zorder=3, label=name if i == 0 else None
            )
            has_whisker = off < 0 and terms[i] in ("fe_issue", "reader_wait")
            if has_whisker:
                ax.text(x, v / 2, f"{v:,.0f}", ha="center", va="center", fontsize=8.6, color=BG, zorder=5)
            else:
                ax.text(x, v + 130, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.6, color=INK, zorder=5)
    # measured ranges: reader_wait from the UNPACK waits to the A4 subtraction; fe_issue with and without control + mask
    for i, key in [(1, "reader_wait"), (0, "fe_issue")]:
        x = i - 0.5 * bw
        lo, hi = meas[key], meas_hi[key]
        ax.plot([x, x], [lo, hi], color=INK, lw=1.3, zorder=6)
        ax.plot([x - 0.06, x + 0.06], [hi, hi], color=INK, lw=1.3, zorder=6)
        txt = f"{hi:,.0f} (A4 stream subtraction)" if key == "reader_wait" else f"{hi:,.0f} (+ control, mask)"
        if key == "reader_wait":
            ax.text(x - 0.05, hi + 120, txt, fontsize=7.9, color=INK, va="bottom", ha="right")
        else:
            ax.text(x, hi + 120, txt, fontsize=7.9, color=INK, va="bottom", ha="center")
    ax.axhline(resid_ms, color=ORANGE, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ax.axhline(resid_new, color=BLUE, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ax.text(
        4.55,
        max(resid_ms, resid_new) + 90,
        f"measured residual per step {resid_ms:,.0f} = (wall - floor - prologue) / 165",
        ha="right",
        va="bottom",
        fontsize=8.4,
        color=ORANGE,
    )
    ax.text(
        4.55,
        min(resid_ms, resid_new) - 130,
        f"model residual per step {resid_new:,.0f} = sum of the five terms",
        ha="right",
        va="top",
        fontsize=8.4,
        color=BLUE,
    )
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, fontsize=9.5)
    ax.set_xlim(-0.8, 4.6)
    ax.set_ylim(0, 11000)
    ax.set_ylabel("cycles per k-chunk step (165 steps on the wall core)")
    ax.set_title(
        f"residual terms per step (the compute floor, {new['compute_floor']:,.0f} per step, is common and not drawn)",
        fontsize=10,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.8)
    # init inset, per launch
    vals = [(2900.0, ORANGE, "2,900"), (HEAD.ARCH_BH.wall_fixed_cycles, BLUE, f"{HEAD.ARCH_BH.wall_fixed_cycles:,.0f}")]
    for i, (v, col, txt) in enumerate(vals):
        axi.bar(i, v, width=0.7, color=col, edgecolor=BG)
        axi.text(i, v + 60, txt, ha="center", va="bottom", fontsize=8, color=INK)
    axi.set_xticks(range(2))
    axi.set_xticklabels(["measured", "model"], fontsize=8, rotation=25, ha="right")
    axi.set_ylim(0, 4500)
    axi.set_ylabel("cycles per launch", fontsize=9)
    axi.set_title("init, per launch", fontsize=10.5)
    axi.text(
        0.5,
        0.95,
        "measured fixed part:\nprologue 171 + tail 1,706\n+ between q chunks 379\n+ start skew 630 to 720",
        transform=axi.transAxes,
        ha="center",
        va="top",
        fontsize=7.8,
        color=INK,
    )
    axi.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v / 1e3:.1f}k" if v else "0"))
    axi.tick_params(labelsize=8.5)
    heading(
        fig,
        "Per-step wall terms at the anchor: the measured value on the wall core against the model's booked terms",
        ANCHOR
        + ". Per step = per k-chunk step of the wall core (165 steps). Measured: UNPACK K + V + Q waits 979,638 (whisker to the A4 stream subtraction 1,095,844), "
        "front end with a free reader 551,041 minus control 81,963 and mask 9,398 (whisker with them inside), mask from A2, exp 0 on the wall from A6, control from the TRISC1 un-zoned time. "
        f"Model: predict() components divided by the mean-core steps (153.6), which equals the wall core's per-step cost; the five terms book {resid_new:,.0f} of the "
        f"{resid_ms:,.0f} measured residual per step, and both init values are the measured 2,900 fixed cycles per launch.",
    )
    footer(
        fig,
        "MEASURED bars: data/bh_zones/report_tables.md T1 (wall, steps), T2 (K_WAIT, V_WAIT, Q_WAIT), T3 (floor on wall core, prologue, un-zoned control), T6 (A4, A2 walls); "
        f"bh/zone_decomposition.md s5.3. MODEL bars: polaris ttsim/perf/roofline_sdpa.py predict() at the anchor config ({MODEL_LABEL}). Init 2,900 decomposition: refit_r2_notes.md s2.",
    )
    save(fig, "m_terms_measured_vs_default.png")
    print(f"  model per step {dict((k, round(v)) for k, v in new.items())}")
    print(
        f"  measured per step {dict((k, round(v)) for k, v in meas.items())}, hi {dict((k, round(v)) for k, v in meas_hi.items())}, residual {resid_ms:.0f}"
    )


# ---------------------------------------------------------------------------------------------
# 5. m_all_walls_pred_vs_meas.png
# ---------------------------------------------------------------------------------------------
REGIME_STYLE = {
    "causal": ("o", BLUE, "causal prefill (grid 1, S, dtype, grid, hold-outs)"),
    "noncausal": ("s", ORANGE, "non-causal prefill"),
    "cross": ("D", ORANGE, "cross attention"),
    "windowed": ("^", AQUA, "windowed"),
    "chunked": ("v", AQUA, "chunked (paged prefix)"),
    "mla": ("P", AQUA, "MLA prefill"),
    "masked": ("X", RED, "dense mask"),
    "sparse": ("*", RED, "sparse"),
    "joint": ("h", RED, "joint"),
    "production": ("p", DARKGRAY, "production Llama 8B (legacy path)"),
    "decode": ("<", BLUE, "decode (paged, non-paged, MLA)"),
}


def campaign_rows():
    """Every wall of the campaign with its set, priced on the working tree (model/refit_r2_fit.py tables)."""
    rows = []
    for lab, g, b, st, kw, meas in RF.WALLS:
        r = HEAD.predict(RF.cfg(**kw))
        rows.append(dict(label=lab, regime=g, block=b, set=st, meas=float(meas), model=float(r.wall_clock_cycles)))
    for lab, b, st, kw, meas in RF.DECODE:
        r = HEAD.predict_decode(**kw)
        rows.append(
            dict(label=lab, regime="decode", block=b, set=st, meas=meas * CLK, model=float(r.wall_clock_cycles))
        )
    for r in rows:
        r["err"] = 100 * (r["model"] / r["meas"] - 1)
    return rows


def fig_all_walls_pred_vs_meas():
    rows = campaign_rows()
    fig = plt.figure(figsize=(W_IN, 9.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.9], wspace=0.05, left=0.065, right=0.99, top=0.80, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])
    axt.axis("off")
    lo, hi = 8e4, 6.0e7
    g = np.array([lo, hi])
    ax.fill_between(g, g * 0.90, g * 1.10, color=BAND10, lw=0, zorder=0)
    ax.fill_between(g, g * 0.95, g * 1.05, color=BAND5, lw=0, zorder=0)
    ax.plot(g, g, color=INK, lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    for r in rows:
        mk_, col, _ = REGIME_STYLE[r["regime"]]
        ms = 11 if mk_ in "*P" else 8.5
        if r["set"] in ("holdout", "fresh"):
            ax.plot(r["meas"], r["model"], marker=mk_, ls="", ms=ms, mfc=BG, mec=col, mew=1.7, zorder=5)
        elif r["set"] == "pred":
            ax.plot(r["meas"], r["model"], marker=mk_, ls="", ms=ms, mfc=col, mec=INK, mew=0.9, alpha=0.85, zorder=4)
        else:
            ax.plot(r["meas"], r["model"], marker=mk_, ls="", ms=ms, mfc=col, mec=BG, mew=1.0, zorder=4)
    beyond = [r for r in rows if abs(r["err"]) > 5]
    place_labels(
        ax,
        [(r["meas"], r["model"]) for r in beyond],
        [f"{fz(r['err'])}" for r in beyond],
        fontsize=7.8,
        color=INK,
        pad_pt=6,
    )
    fmt = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g}k")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("measured device wall, cycles (zones off; decode: DEVICE KERNEL DURATION)")
    ax.set_ylabel("predicted wall, cycles (predict(), predict_decode())")
    ax.tick_params(labelsize=8.5)
    ax.set_title(
        "predicted against measured, every wall of the campaign (log axes); labels on the points beyond 5 percent",
        fontsize=10,
    )
    handles = [Line2D([], [], marker=m, color=c, ls="", ms=8, label=lab) for m, c, lab in REGIME_STYLE.values()]
    handles += [
        Line2D([], [], marker="o", color=INK, ls="", ms=8, mfc=INK, mec=BG, label="fit point"),
        Line2D(
            [],
            [],
            marker="o",
            color=INK,
            ls="",
            ms=8,
            mfc=INK,
            mec=INK,
            alpha=0.85,
            label="prediction (no constant fit on it)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=INK,
            ls="",
            ms=8,
            mfc=BG,
            mec=INK,
            mew=1.7,
            label="hold-out or fresh-build row (validation only)",
        ),
        Patch(facecolor=BAND5, label="plus or minus 5 percent"),
        Patch(facecolor=BAND10, label="plus or minus 10 percent"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.6, ncol=1, borderaxespad=0.5)
    # right: summary per set and the rows beyond 5 percent
    y = 0.99
    dy = 0.0265
    mono = dict(fontsize=8.0, family="DejaVu Sans Mono", va="top", color=INK, transform=axt.transAxes)
    axt.text(0.0, y, f"{'set':9s}{'n':>4s}{'<5%':>5s}{'<10%':>6s}{'mean':>8s}{'mabs':>7s}", fontweight="bold", **mono)
    y -= dy
    names = {"fit": "fit", "pred": "predict.", "holdout": "hold-out", "repeat": "repeat", "fresh": "fresh"}
    for st in ("fit", "pred", "holdout", "repeat", "fresh"):
        e = np.array([r["err"] for r in rows if r["set"] == st])
        axt.text(
            0.0,
            y,
            f"{names[st]:9s}{len(e):4d}{int((abs(e) <= 5).sum()):5d}{int((abs(e) <= 10).sum()):6d}{e.mean():+8.2f}{abs(e).mean():7.2f}",
            **mono,
        )
        y -= dy
    e = np.array([r["err"] for r in rows if r["set"] in ("fit", "pred")])
    axt.text(
        0.0,
        y,
        f"{'fit+pred':9s}{len(e):4d}{int((abs(e) <= 5).sum()):5d}{int((abs(e) <= 10).sum()):6d}{e.mean():+8.2f}{abs(e).mean():7.2f}",
        **mono,
    )
    y -= dy * 1.6
    axt.text(
        0.0,
        y,
        "walls beyond 5 percent (signed error, model minus measured):",
        fontsize=8.4,
        va="top",
        color=INK,
        fontweight="bold",
        transform=axt.transAxes,
    )
    y -= dy
    for r in sorted(beyond, key=lambda r: -abs(r["err"])):
        axt.text(0.0, y, f"{r['err']:+6.1f}  {r['label']} ({r['block']}, {names[r['set']]})", **mono)
        y -= dy
    y -= dy * 0.6
    causes = (
        "Causes. Causal S1024 / S2048, MLA S1024, windowed W1024 S8192 and MHA nh16: the wall core's steps beyond the light cores' "
        "count run on fewer DRAM readers at a faster rate than the single-rate stream lane carries (the two windowed misses are "
        "the two ends of that tail effect on the four windowed walls the factor is fit on). MLA decode at position 1024: the two "
        "batches differ by 15 us for twice the bytes, which one fixed cost per head slice plus a byte rate cannot hold. Hold-outs: "
        "HiFi4 on the streaming kernel adds an FPU-bound front end the lane maximum does not carry; S512 and MHA nh16 are short "
        "tails on 18 and 36 heavy cores; the R1g causal q256 k128 head_dim 64 wall wants a larger head_dim share of the causal "
        "PACK lane than the one shared value fit on the non-causal R1c wall gives it."
    )
    axt.text(0.0, y, "\n".join(textwrap.wrap(causes, 78)), fontsize=7.8, va="top", color=INK, transform=axt.transAxes)
    n = len(rows)
    heading(
        fig,
        f"Predicted against measured device wall: the {n} walls of the campaign, fit points filled, hold-outs hollow, regimes by marker",
        f"Model = predict() and predict_decode() of polaris ttsim/perf/roofline_sdpa.py ({MODEL_LABEL}); measured = zones-off device walls of the T2.1, T2.3, T2.4, "
        "R1a, R1c, R1d, R1e and R1g blocks (mean of invocations 1 and 2) and the DEVICE KERNEL DURATION medians of the T2.8 and R1b decode blocks, p100a firmware 19.9.0. "
        "Fit points are the walls a constant was fit on (model/refit_r2_notes.md s2 names them per constant; the R1c head_dim 64 pair fits the head_dim term); predictions "
        "are walls no constant rests on; hold-outs (R1c, the sparse T 32768 row, the R1g head_dim 64 block) and the fresh-build rows (R1e) were never used for any fit.",
    )
    footer(
        fig,
        "MODEL vs MEASURED. Rows and sets: data/refit_r2_walls.csv (model/refit_r2_fit.py); measured walls from data/bh_zones/r1_walls.csv, r1b_decode_table.csv, "
        "decode_sweep_table.csv, data/t42_scripts/decomp_summary_t21_t22.csv, bh/regimes.md s1, bh/production_config.md s2, bh/zone_decomposition.md s5.2.",
    )
    save(fig, "m_all_walls_pred_vs_meas.png")
    for r in rows:
        print(
            f"  {r['label']:42s} {r['block']:5s} {r['set']:8s} {r['meas']:12,.0f} {r['model']:12,.0f} {fz(r['err']):>7}"
        )


# ---------------------------------------------------------------------------------------------
# 5b. m_holdout_errors.png
# ---------------------------------------------------------------------------------------------
def read_validation_rows():
    """T2.5 model-level rows: DEVICE KERNEL DURATION and the harness prediction per op (medians of the three invocations)."""
    out = []
    base = os.path.join(ROOT, "data", "model_level", "validation")
    for run, lab in (
        ("llama8b_attn_prefill_S1024", "prefill S1024 (T2.5)"),
        ("llama8b_attn_prefill_S4096", "prefill S4096 (T2.5)"),
        ("llama8b_attn_prefill_S8192", "prefill S8192 (T2.5)"),
    ):
        with open(os.path.join(base, f"{run}_signed_errors.csv")) as f:
            rows = list(csv.DictReader([l for l in f if not l.startswith("#")]))
        meas = sorted(float(r["meas_ns"]) for r in rows)[len(rows) // 2]
        pred = float(rows[0]["pred_kernel_ns"])
        out.append((lab, meas, pred))
    with open(os.path.join(base, "llama8b_attn_decode_b32_pos128_1024_4096_signed_errors.csv")) as f:
        rows = list(csv.DictReader([l for l in f if not l.startswith("#")]))
    for pos in (128, 1024, 4096):
        sel = [r for r in rows if int(float(r["cur_pos"])) == pos]
        meas = sorted(float(r["meas_ns"]) for r in sel)[len(sel) // 2]
        out.append((f"decode b32 position {pos} (T2.5)", meas, float(sel[0]["pred_kernel_ns"])))
    return out


def fig_holdout_errors():
    rows = [r for r in campaign_rows() if r["set"] == "holdout"]
    pts = [(r["label"].replace("hold-out ", "") + f" ({r['block']})", r["err"], r["block"]) for r in rows]
    for lab, meas, pred in read_validation_rows():
        pts.append((lab, 100 * (pred / meas - 1), "T2.5"))
    fig = plt.figure(figsize=(W_IN, 7.4))
    ax = fig.add_axes([0.30, 0.13, 0.67, 0.66])
    ax.axvspan(-10, 10, color=BAND10, lw=0, zorder=0)
    ax.axvspan(-5, 5, color=BAND5, lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    col = {"R1c": ORANGE, "R1a": RED, "R1g": AQUA, "T2.5": BLUE}
    for i, (lab, e, blk) in enumerate(pts):
        y = len(pts) - 1 - i
        ax.barh(y, e, height=0.62, color=col[blk], edgecolor=BG, zorder=3)
        ax.text(
            e + (0.5 if e >= 0 else -0.5),
            y,
            fz(e),
            va="center",
            ha="left" if e >= 0 else "right",
            fontsize=8.6,
            color=INK,
            zorder=5,
        )
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([lab for lab, _, _ in reversed(pts)], fontsize=9)
    ax.set_xlim(-14, 14)
    ax.set_xlabel("signed error, percent (model minus measured over measured)")
    ax.set_title("validation only: no constant was fit on any of these rows", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    handles = [
        Patch(facecolor=ORANGE, label="R1c hold-out block: the causal S4096 q128 k128 anchor with one axis changed"),
        Patch(facecolor=RED, label="R1a sparse T 32768 (the sparse law is fit on T 8192 and 16384)"),
        Patch(
            facecolor=AQUA,
            label="R1g head_dim 64 block: hold-outs of the head_dim term (fit on the R1c head_dim 64 pair)",
        ),
        Patch(facecolor=BLUE, label="T2.5 model-level rows (Llama 3.1 8B attention layer, medians of 3 invocations)"),
        Patch(facecolor=BAND5, label="plus or minus 5 percent"),
        Patch(facecolor=BAND10, label="plus or minus 10 percent"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=8.4)
    e_hold = np.array([e for _, e, blk in pts if blk != "T2.5"])
    e_val = np.array([e for _, e, blk in pts if blk == "T2.5"])
    heading(
        fig,
        "Hold-out and model-level rows: signed error of the model on walls it was never fit to",
        f"Hold-outs (R1c, R1a sparse T 32768, R1g head_dim 64): {int((abs(e_hold) <= 10).sum())} of {len(e_hold)} within 10 percent, {int((abs(e_hold) <= 5).sum())} within 5, mean {fz(e_hold.mean())} percent, "
        f"mean abs {abs(e_hold).mean():.1f}; beyond 5: HiFi4, MHA nh16, S512 (R1c) and the causal q256 k128 head_dim 64 wall (R1g). T2.5 model-level rows: all {len(e_val)} within 1 percent "
        f"(mean {fz(e_val.mean())}), scored by tools/sdpa_validate.py on the tracy ops CSVs. Measured: zones-off device walls (R1c, R1a) and DEVICE KERNEL DURATION (T2.5), "
        f"p100a firmware 19.9.0; model predict() and predict_decode() of ttsim/perf/roofline_sdpa.py ({MODEL_LABEL}).",
    )
    footer(
        fig,
        "MEASURED: data/bh_zones/r1_walls.csv (R1c, R1a, R1g), data/model_level/validation/*_signed_errors.csv (T2.5, meas_ns and pred_kernel_ns). "
        "MODEL: predict() rows of data/refit_r2_walls.csv. Errors DERIVED.",
    )
    save(fig, "m_holdout_errors.png")
    for lab, e, blk in pts:
        print(f"  {lab:48s} {blk:5s} {fz(e):>7}")


# ---------------------------------------------------------------------------------------------
# 6. m_attr_flow.png
# ---------------------------------------------------------------------------------------------
FAM_COL = {
    "stream": "#cfe0f6",
    "floor": "#c9ecdd",
    "pack": "#fadcce",
    "decode": "#e2e2df",
    "flag": "#f1f1ee",
    "regime": "#f6cfcf",
}
FAM_NAME = {
    "stream": "DRAM stream lane (reader_wait)",
    "floor": "compute floor",
    "pack": "PACK lane and front end (fe_issue, control, dest_roundtrip)",
    "decode": "decode path (predict_decode)",
    "flag": "recorded, flagged or echoed only",
    "regime": "regime routing and geometry",
}


def attr_rows():
    """(ttnn call field, shim attr recorded by op.py, SdpaConfig field or predict_decode argument, term fed,
    fallback when absent, family); constants read from the refit snapshot."""
    a = HEAD.ARCH_BH
    tc = HEAD.WALL_TERMS_BH["prefill_causal"]
    rates = a.kv_stream_rate_bpc
    drate = a.decode_kv_stream_gbps_paged
    mla_scale = HEAD.WALL_TERMS_BH["mla"].stream_scale
    ch_scale = HEAD.WALL_TERMS_BH["chunked"].stream_scale
    mask_tile = HEAD.WALL_TERMS_BH["masked"].mask_per_tile
    fx = a.kv_stream_fixed_per_ktile + 8 * a.kv_stream_fixed_per_kv_tile  # 691 per k tile at head_dim 128
    pk = tc.pack_per_qktile + 8 * tc.pack_per_qk_dtile  # 348 per q tile per k tile at head_dim 128
    return [
        (
            "program_config.q_chunk_size",
            "q_chunk_size",
            "q_chunk\n(qct = q_chunk / 32)",
            f"compute_floor (overlap r (1 - 4 / qct^2), {a.fpu_overhead_tile_macs_per_qtile_step:g} tile MACs and {a.sfpu_overhead_per_qtile_step:g} SFPU cycles per q tile per step); fe_issue (PACK lane per q tile); control ({tc.control_per_qtile:g} x qct); straggler (chunk and pair count)",
            "32 [program_config_absent]; off_calibration_chunk outside {64, 128, 256, 512}",
            "pack",
        ),
        (
            "program_config.k_chunk_size",
            "k_chunk_size",
            "k_chunk\n(kct = k_chunk / 32)",
            f"reader_wait (stream lane kct x {fx:,.0f} + K/V bytes per step at head_dim 128, {a.kv_stream_fixed_per_kv_tile:g} of it per K or V tile of the head); fe_issue ({pk:g} per q tile per k tile, {tc.pack_per_qk_dtile:g} of it per head tile); compute_floor (visit count; the tiles follow the truncated diagonal)",
            "32 [program_config_absent]; decode 0 = the kernel's dynamic chunk rule [k_chunk_auto]",
            "stream",
        ),
        (
            "program_config.\ncompute_with_storage_grid_size\nor sub_core_grids",
            "num_cores\n(x * y, or the\nsub_core_grids core count)",
            "num_cores",
            f"Q = chunks per core; active cores; stream rate {rates[110]:.3f} (110 cores) or {rates[64]:.3f} (64) B per cycle per core; straggler; decode rate by grid",
            "whole grid 110 [grid_absent]; off_calibration_cores outside {110, 64}; decode_grid_uncalibrated",
            "stream",
        ),
        (
            "program_config.exp_approx_mode",
            "exp_approx_mode",
            "exp_approx_mode",
            f"compute_floor (SFPU exp {a.exp_tile_cycles:g} approx or {a.exp_tile_cycles_accurate:g} accurate cycles per tile)",
            "True [exp_mode_absent]",
            "floor",
        ),
        (
            "program_config.\nmax_cores_per_head_batch",
            "max_cores_per_head_batch",
            "predict_decode(\nmax_cores_per_head_batch)",
            f"decode core split (factory rule): cores per KV head, KV head groups per core in sequence (init {a.decode_fixed_overhead_cycles:,.0f} + {a.decode_fixed_per_head_group_cycles:,.0f} per group), active cores",
            "16 when a program config is given",
            "decode",
        ),
        (
            "compute_kernel_config.\nmath_fidelity",
            "fidelity",
            "fidelity",
            "compute_floor (cycles per tile MAC 16 / 32 / 48 / 64 at LoFi / HiFi2 / HiFi3 / HiFi4)",
            "HiFi2 [fidelity_absent]; off_calibration_fidelity",
            "floor",
        ),
        (
            "compute_kernel_config.\nfp32_dest_acc_en",
            "fp32_dest_acc_en",
            "fp32_dest_acc,\naccum_dtype float32",
            f"kernel path legacy: dest_roundtrip ({a.legacy_step_cycles:,.0f} + {a.dest_roundtrip_cycles:,.0f} per 4-tile window per step), overlap 0, no PACK lane; decode chunk cap 4 tiles",
            "False [fp32_acc_absent]; True flags legacy_path_production_family",
            "pack",
        ),
        (
            "compute_kernel_config.\npacker_l1_acc",
            "packer_l1_acc",
            "(recorded in attrs only)",
            "no term: the SDPA program factory does not read it",
            "listed in sdpa_defaulted",
            "flag",
        ),
        (
            "compute_kernel_config.\nmath_approx_mode",
            "math_approx_mode",
            "(recorded in attrs only)",
            "no term: the exp mode comes from program_config.exp_approx_mode",
            "listed in sdpa_defaulted",
            "flag",
        ),
        (
            "K tensor dtype",
            "kv_element_size",
            "kv_input_dtype",
            "reader_wait (K + V bytes per step: 1,088 bfp8 or 2,048 bf16 per tile); decode kv_stream_wait (KV bytes streamed)",
            "Q dtype [kv_dtype_absent]",
            "stream",
        ),
        (
            "Q tensor memory config (decode)",
            "q_in_l1",
            "predict_decode(q_in_dram)",
            "decode kv_stream_wait: one padded head-row tile block per active core from DRAM (64 x 8 KB at nh32 d128 bf16), or 0 B when L1 sharded",
            "DRAM (q_in_dram True)",
            "decode",
        ),
        (
            "attn_mask (prefill)",
            "has_attn_mask",
            "has_attn_mask",
            f"masked regime: every k chunk visited, {mask_tile:g} cycles per mask tile per step on the non-causal PACK lane (density independent)",
            "False; a decode mask flags decode_mask_unmodelled",
            "regime",
        ),
        (
            "sliding_window_size",
            "sliding_window_size",
            "sliding_window",
            "windowed regime: banded K_eff, wall core = the band with the most visits, stream lane",
            "0 = full attention",
            "regime",
        ),
        (
            "attention_sink",
            "attention_sink",
            "attention_sink",
            "compute_floor (one extra SFPU reduce + recip pass per q chunk)",
            "False",
            "floor",
        ),
        (
            "cu_window_seqlens",
            "is_windowed",
            "is_windowed",
            "no term: priced as plain prefill",
            "flag\nwindowed_mode_unmodelled",
            "flag",
        ),
        (
            "chunk_start_idx (int) or chunk_start_idx_tensor\n+ page_table_tensor (chunked prefill)",
            "chunk_start_idx or\nchunk_start_unknown;\nsdpa_variant = chunked",
            "chunk_start_idx,\nis_chunked, paged;\nkv_seq = start + S",
            f"chunked regime: K_eff = prefix + local causal ramp, stream lane x {ch_scale:g}, straggler (pair rule)",
            "tensor-form start bounded by page table width x block size [chunk_start_unknown]",
            "regime",
        ),
        (
            "head_dim_v (MLA prefill and decode)",
            "head_dim_v",
            "v_head_dim",
            f"compute_floor (P.V tiles dct_v); MLA regime: stream lane 278,528 B per step x {mla_scale:g}; MLA decode: {a.decode_fixed_overhead_cycles_mla:,.0f} + {a.decode_fixed_per_qhead_slice_cycles_mla:,.0f} per q-head slice + latent bytes at {a.decode_kv_stream_gbps_mla:.1f} GB/s",
            "V tensor dim (the attr first)",
            "floor",
        ),
        (
            "Q memory config and V tensor\n(MLA decode)",
            "q_in_l1, q_shard_cores,\nmla_v_read",
            "predict_decode(q_shard_cores,\nmla_v_read)",
            "MLA decode: head slices of the shard height each stream the cache as a user (ceil(heads / shard rows)); V is streamed only when a V tensor is passed",
            "Q in DRAM: one slice; no V tensor: K reused as V",
            "decode",
        ),
        (
            "cur_pos (list) or cur_pos_tensor (decode)",
            "cur_pos = max(list)\nor cur_pos_unknown",
            "predict_decode(cur_pos):\nattended = cur_pos + 1",
            "decode kv_stream_wait: whole k chunks streamed (paged: tiles of cur_pos + 1 rounded to a power of two, capped at the DEST size; 256 / 1,152 / 4,224 rows at 128 / 1,024 / 4,096)",
            "attended = cache capacity [cur_pos_unknown]",
            "decode",
        ),
        (
            "page_table_tensor + paged_cache_geometry\n(decode)",
            "paged, page_block_size",
            "paged, page_block_size",
            f"decode: paged stream rate {drate[64]:.1f} GB/s at 64 cores, {drate[110]:.1f} at 110 (log-log between); capacity per user = min(table width, blocks / batch) x block size",
            f"non-paged: {a.decode_kv_stream_gbps_nonpaged:.1f} GB/s at 110 cores (R1b), scaled by the paged grid ratio elsewhere [decode_nonpaged_grid_inferred]; block other than 32 flagged",
            "decode",
        ),
        (
            "is_causal (decode entry points)",
            "is_causal",
            "predict_decode(is_causal)",
            "no term: non-causal decode is priced as causal single-token decode",
            "True; False flags decode_mask_unmodelled",
            "flag",
        ),
        (
            "shapes: heads, S, q_chunk, cores\n(non-causal and cross, derived)",
            "(no attr; derived from the shapes\nand num_cores)",
            "_noncausal_chains(\nq chunks per head, chunks, cores)",
            f"non-causal and cross: one K/V forwarding chain per head that spans cores; with chains the injector lane ({a.kv_injector_rate_bpc:g} {a.kv_injector_rate_bpc_per_ktile:+g} x kct B per cycle), with none the all-cores stream lane",
            f"chain_geometry_off_calibration when the count is 0 or differs from the {HEAD.CALIBRATED_INJECTORS} injectors of the fit",
            "regime",
        ),
        (
            "fields absent at the call",
            "sdpa_defaulted\n(comma list)",
            "defaulted",
            "no term: echoed as perf_stats sdpa_defaulted_attrs next to sdpa_low_confidence_reasons",
            "(the echo itself)",
            "flag",
        ),
    ]


def fig_attr_flow():
    rows = attr_rows()
    cols = [
        ("ttnn call field", 0.0, 0.185),
        ("shim attr recorded\n(ttsim/front/ttnn/op.py)", 0.195, 0.175),
        ("SdpaConfig field or\npredict_decode argument", 0.38, 0.17),
        ("wall term or path it feeds\n(roofline_sdpa.py)", 0.56, 0.25),
        ("fallback when absent\n[reason flag]", 0.82, 0.18),
    ]
    n = len(rows)
    fig = plt.figure(figsize=(W_IN, 18.0))
    ax = fig.add_axes([0.012, 0.09, 0.976, 0.795])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rh = 1.0 / (n + 1.25)
    top = 1.0
    for name, x, w in cols:
        ax.text(
            x + w / 2,
            top - rh * 0.55,
            name,
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=INK,
            linespacing=1.15,
        )
    ax.plot([0, 1], [top - rh * 1.05, top - rh * 1.05], color=INK, lw=0.9)
    fig_w_in = fig.get_size_inches()[0] * 0.976
    for i, (f0, f1, f2, f3, f4, fam) in enumerate(rows):
        yc = top - rh * (i + 1.75)
        texts = [f0, f1, f2, f3, f4]
        for j, ((name, x, w), txt) in enumerate(zip(cols, texts)):
            fill = FAM_COL[fam] if j == 3 else ("#f7f7f5" if i % 2 == 0 else BG)
            ec = INK if j == 3 else GRAY
            box = FancyBboxPatch(
                (x + 0.003, yc - rh * 0.45),
                w - 0.006,
                rh * 0.9,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                facecolor=fill,
                edgecolor=ec,
                linewidth=0.7 if j == 3 else 0.5,
                zorder=2,
            )
            ax.add_patch(box)
            mono = j in (1, 2)
            fs = 7.5
            char_in = fs / 72 * (0.60 if mono else 0.55)
            width_chars = max(8, int((w * fig_w_in - 0.16) / char_in))
            wrapped = "\n".join(
                sum([textwrap.wrap(part, width_chars, break_long_words=False) or [""] for part in txt.split("\n")], [])
            )
            ax.text(
                x + 0.009,
                yc,
                wrapped,
                ha="left",
                va="center",
                fontsize=fs,
                color=INK,
                zorder=3,
                family="DejaVu Sans Mono" if mono else "DejaVu Sans",
                linespacing=1.12,
            )
            if j < len(cols) - 1:
                x_end = x + w - 0.003
                x_next = cols[j + 1][1] + 0.003
                if j == 3:
                    ax.plot([x_end, x_next], [yc, yc], color=GRAY, lw=0.6, ls=(0, (2, 2)), zorder=1)
                else:
                    ax.annotate(
                        "",
                        xy=(x_next, yc),
                        xytext=(x_end, yc),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0),
                        zorder=4,
                    )
    handles = [
        Patch(facecolor=FAM_COL[k], edgecolor=INK, lw=0.6, label=FAM_NAME[k])
        for k in ("stream", "pack", "floor", "regime", "decode", "flag")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.034),
        ncol=2,
        fontsize=8.6,
        title="colour of the term column: the family the attribute feeds first",
        title_fontsize=8.8,
    )
    heading(
        fig,
        "Which attribute travels where: from the ttnn call through the shim and SdpaConfig to the wall term it feeds",
        f"One row per attribute of attr_plumbing_spec.md s1.3 as plumbed in ttsim/front/ttnn/op.py (restructure_notes.md s10) and priced on {MODEL_LABEL}, plus the decode query placement, "
        "the MLA decode head slices and the derived non-causal chain count. Fallbacks are the kernel's own defaults "
        "(sdpa_program_factory.cpp:279-282, :103-109, :438-439; sdpa.cpp:53-54; rt_args_common.hpp:57-107) and each leaves a reason in sdpa_low_confidence_reasons; "
        "absent fields are echoed in sdpa_defaulted_attrs. scale and memory_config are consumed but price nothing.",
        y=0.982,
    )
    footer(
        fig,
        "STATED from source: ttsim/front/ttnn/op.py _sdpa_config_attrs, _sdpa_decode_attrs, _tensor_in_l1; ttsim/perf/roofline_sdpa.py sdpa_config_from_shapes, decode_config_from_shapes, "
        f"_wall_components, _noncausal_chains, _decode_core_split, predict_decode ({MODEL_LABEL}); model/attr_plumbing_spec.md s1.3; model/restructure_notes.md s10.1 to s10.3; "
        "model/refit_r2_notes.md s2 and s4.",
    )
    save(fig, "m_attr_flow.png")


# ---------------------------------------------------------------------------------------------
# 7. m_decode_law.png
# ---------------------------------------------------------------------------------------------
def fig_decode_law():
    m = HEAD
    a = m.ARCH_BH
    rows = read_decode_sweep()
    pts = []  # dict per measured point
    for r in rows:
        batch, cores, pos = int(r["batch"]), int(r["cores"]), int(r["position"])
        kv = r["kv_dtype"]
        meas = float(r["dur_ns_median_excl_first"]) / 1000
        rn = decode_wall(m, batch, cores, kv, pos)
        key = (
            "b32_g64"
            if (batch == 32 and cores == 64 and kv == "bfp8_b")
            else "g110"
            if cores == 110
            else "bf16"
            if kv == "bfloat16"
            else f"b{batch}"
        )
        pts.append(
            dict(
                key=key,
                batch=batch,
                cores=cores,
                kv=kv,
                pos=pos,
                meas=meas,
                new=rn.wall_clock_cycles / CLK,
                hpc=rn.config_echo["heads_per_core"],
                kv_mb=rn.config_echo["kv_bytes"] / 1e6,
                rate=rn.config_echo["kv_stream_gbps"],
            )
        )
    r1b = decode_wall(m, 32, 110, "bfp8_b", 8192)
    pts.append(
        dict(
            key="g110_r1b",
            batch=32,
            cores=110,
            kv="bfp8_b",
            pos=8192,
            meas=1759.3,
            new=r1b.wall_clock_cycles / CLK,
            hpc=r1b.config_echo["heads_per_core"],
            kv_mb=r1b.config_echo["kv_bytes"] / 1e6,
            rate=r1b.config_echo["kv_stream_gbps"],
        )
    )
    vals = []
    for pos, ns in DECODE_VALIDATION.items():
        med = sorted(ns)[1] / 1000
        rn = decode_wall(m, 32, 64, "bfp8_b", pos, q_in_dram=False)
        vals.append(dict(pos=pos, meas=med, new=rn.wall_clock_cycles / CLK))
    # law curves (exact model, including the whole-chunk staircase)
    P = np.arange(96, 9100, 4)
    law64 = np.array([decode_wall(m, 32, 64, "bfp8_b", int(p)).wall_clock_cycles / CLK for p in P])
    law110 = np.array([decode_wall(m, 32, 110, "bfp8_b", int(p)).wall_clock_cycles / CLK for p in P])
    fig = plt.figure(figsize=(W_IN, 9.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.24, left=0.065, right=0.985, top=0.78, bottom=0.26)
    ax = fig.add_subplot(gs[0])
    axe = fig.add_subplot(gs[1])
    style = {
        "b32_g64": dict(marker="o", color=BLUE, ms=8.5, mec=BG, mew=1.1),
        "g110": dict(marker="^", color=AQUA, ms=9.5, mec=BG, mew=1.1),
        "b8": dict(marker="s", color=ORANGE, ms=8.5, mec=BG, mew=1.1),
        "b16": dict(marker="s", color=ORANGE, ms=8.5, mec=BG, mew=1.1),
        "bf16": dict(marker="D", color=RED, ms=8.0, mec=BG, mew=1.1),
        "g110_r1b": dict(marker="^", color=AQUA, ms=9.5, mfc=BG, mec=AQUA, mew=1.8),
    }
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.plot(
        P,
        law64,
        color=BLUE,
        lw=1.6,
        zorder=2,
        label=f"model law, 64 cores, batch 32 (rate {a.decode_kv_stream_gbps_paged[64]:.1f} GB/s, 4 head groups per core)",
    )
    ax.plot(
        P,
        law110,
        color=AQUA,
        lw=1.6,
        zorder=2,
        label=f"model law, 110 cores, batch 32 (rate {a.decode_kv_stream_gbps_paged[110]:.1f} GB/s, 4 groups per core)",
    )
    for p in pts:
        s = style[p["key"]]
        ax.plot(p["pos"], p["meas"], ls="", zorder=5, **s)
        if p["key"] in ("b8", "b16", "bf16"):
            ax.plot(p["pos"], p["new"], marker="_", ms=16, color=s["color"], markeredgewidth=2.2, ls="", zorder=4)
    for v in vals:
        ax.plot(v["pos"], v["meas"], marker="o", ms=9.5, mfc=BG, mec=BLUE, mew=1.8, ls="", zorder=6)
    ax.set_xlim(90, 10500)
    ax.set_ylim(38, 2600)
    ax.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["128", "256", "512", "1024", "2048", "4096", "8192"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([50, 100, 200, 500, 1000, 2000])
    ax.set_xlabel("decode position (cur_pos); attended = cur_pos + 1")
    ax.set_ylabel("DEVICE KERNEL DURATION, us")
    ax.set_title("wall against position, log axes", fontsize=10.5)
    labels = [p for p in pts if p["key"] in ("b8", "b16", "bf16")]
    place_labels(
        ax,
        [(p["pos"], p["meas"]) for p in labels],
        [{"b8": "batch 8", "b16": "batch 16", "bf16": "bf16 K/V, batch 32"}[p["key"]] for p in labels],
        avoid=[(p["pos"], p["meas"]) for p in pts if p["key"] not in ("b8", "b16", "bf16")]
        + [(v["pos"], v["meas"]) for v in vals],
        fontsize=8.2,
        color=INK,
        pad_pt=7,
    )
    # error panel
    axe.axhspan(-10, 10, color=BAND10, lw=0, zorder=0)
    axe.axhspan(-3, 3, color=BAND5, lw=0, zorder=0)
    axe.axhline(0, color=INK, lw=1.0, zorder=1)
    axe.set_xscale("log", base=2)
    for p in pts:
        s = style[p["key"]]
        e_new = 100 * (p["new"] / p["meas"] - 1)
        axe.plot(p["pos"], e_new, ls="", zorder=5, **s)
    b32 = sorted([p for p in pts if p["key"] == "b32_g64"], key=lambda p: p["pos"])
    axe.plot([p["pos"] for p in b32], [100 * (p["new"] / p["meas"] - 1) for p in b32], color=BLUE, lw=1.2, zorder=3)
    for v in vals:
        axe.plot(
            v["pos"], 100 * (v["new"] / v["meas"] - 1), marker="o", ms=9.5, mfc=BG, mec=BLUE, mew=1.8, ls="", zorder=6
        )
    all_pts = [(p["pos"], 100 * (p["new"] / p["meas"] - 1), fz(100 * (p["new"] / p["meas"] - 1))) for p in pts] + [
        (v["pos"], 100 * (v["new"] / v["meas"] - 1), fz(100 * (v["new"] / v["meas"] - 1))) for v in vals
    ]
    place_labels(
        axe, [(x, y) for x, y, _ in all_pts], [t for _, _, t in all_pts], avoid=[], fontsize=7.6, color=MUTED, pad_pt=5
    )
    axe.set_xlim(90, 10500)
    axe.set_ylim(-12, 8)
    axe.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192])
    axe.set_xticklabels(["128", "256", "512", "1024", "2048", "4096", "8192"])
    axe.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axe.set_xlabel("decode position (cur_pos)")
    axe.set_ylabel("signed error, percent (model minus measured)")
    axe.set_title("signed error per point (bands: 3 and 10 percent)", fontsize=10.5)
    e_new = np.array([100 * (p["new"] / p["meas"] - 1) for p in pts if p["key"] != "g110_r1b"])
    e_val = np.array([100 * (v["new"] / v["meas"] - 1) for v in vals])
    e_r1b = [100 * (p["new"] / p["meas"] - 1) for p in pts if p["key"] == "g110_r1b"][0]
    axe.text(
        0.97,
        0.04,
        f"model law: 11 sweep points within {abs(e_new).max():.1f} percent\n(mean abs {abs(e_new).mean():.2f}), 3 validation rows within {abs(e_val).max():.1f},\n"
        f"R1b 110 cores position 8192 {fz(e_r1b)}",
        transform=axe.transAxes,
        fontsize=8.2,
        va="bottom",
        ha="right",
        color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=GRAY, lw=0.6),
        zorder=9,
    )
    handles = [
        Line2D([], [], ls="", label="T2.8 sweep, batch 32, 64 cores, bfp8 K/V (6 positions)", **style["b32_g64"]),
        Line2D([], [], ls="", label="T2.8, batch 32, 110 cores (2 positions)", **style["g110"]),
        Line2D([], [], ls="", label="T2.8, batch 8 and 16, 64 cores, position 1024", **style["b8"]),
        Line2D([], [], ls="", label="T2.8, batch 32, 64 cores, bf16 K/V, position 1024", **style["bf16"]),
        Line2D([], [], ls="", label="R1b, batch 32, 110 cores, position 8192 (prediction)", **style["g110_r1b"]),
        Line2D(
            [],
            [],
            marker="_",
            ms=14,
            color=INK,
            markeredgewidth=2.0,
            ls="",
            label="law value for a single point (left panel)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ms=9,
            mfc=BG,
            mec=BLUE,
            mew=1.8,
            ls="",
            label="T2.5 validation rows, batch 32, 64 cores, Q in L1 (medians of 3 invocations, not fit)",
        ),
        Line2D(
            [],
            [],
            color=BLUE,
            lw=1.6,
            label="model law, 64 cores: 13,000 + 930 x head groups per core + (KV + Q) bytes x 1.35 GHz / 298.9 GB/s",
        ),
        Line2D([], [], color=AQUA, lw=1.6, label="model law, 110 cores: the same with 330.5 GB/s"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, fontsize=8.3)
    heading(
        fig,
        "Decode wall against the T2.8 paged sweep: the model law and the T2.5 validation rows",
        "tt_transformers decode SDPA as issued (HiFi2, fp32 accumulation, accurate exp, Q bf16 in DRAM, KV bfp8 in 32-row pages, k_chunk 0), nh32 nkv8 d128, p100a firmware 19.9.0, "
        "DEVICE KERNEL DURATION median of invocations 1 and 2. Law (refit_r2_notes.md s2): wall = 13,000 + 930 x head groups per core + (KV + Q bytes) x 1.35 GHz / rate, rate 298.9 GB/s "
        "at 64 cores and 330.5 at 110 (64 active cores and 4 sequential groups per core at batch 32 on both grids; 1 and 2 groups at batch 8 and 16). Bytes are whole 128-row chunks "
        "(the kernel's dynamic chunk rule): 256 / 640 / 1,152 / 2,176 / 4,224 / 8,320 rows per user and head at positions 128 to 8192, so the law is a staircase. The T2.5 rows hold "
        "Q in L1 (0.52 MB less traffic, 1.7 us); the bf16 point is the only one beyond 1.3 percent (bf16 tiles stream about 7 percent faster per byte).",
    )
    footer(
        fig,
        "MEASURED: data/bh_zones/decode_sweep_table.csv (dur_ns_median_excl_first, kv_len_read; bh/decode_sweep.md s1), data/model_level/validation/"
        "llama8b_attn_decode_b32_pos128_1024_4096_signed_errors.csv (meas_ns, three invocations per position). MODEL: predict_decode() of polaris "
        f"ttsim/perf/roofline_sdpa.py ({MODEL_LABEL}; law, staircase). R1b point: data/bh_zones/r1b_decode_table.csv.",
    )
    save(fig, "m_decode_law.png")
    for p in pts:
        print(
            f"  {p['key']:8s} b{p['batch']:<3d} g{p['cores']:<4d} {p['kv']:9s} pos {p['pos']:5d} meas {p['meas']:8.1f} model {p['new']:8.1f} ({fz(100 * (p['new'] / p['meas'] - 1), 2)}) "
            f"kv {p['kv_mb']:6.1f} MB hpc {p['hpc']} rate {p['rate']}"
        )
    for v in vals:
        print(
            f"  validation pos {v['pos']:5d} meas {v['meas']:8.1f} model {v['new']:8.1f} ({fz(100 * (v['new'] / v['meas'] - 1), 2)})"
        )


if __name__ == "__main__":
    fig_floor_vs_counters()
    fig_terms_blocks()
    fig_terms_measured_vs_default()
    fig_all_walls_pred_vs_meas()
    fig_holdout_errors()
    fig_attr_flow()
    fig_decode_law()
