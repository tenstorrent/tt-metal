#!/usr/bin/env python
"""Page A figures, part 1 (a_ prefix): zone tax, anchor decomposition, ablations, residual composition,
counters vs zones, per-k-tile composition, utilization ladder.

Read-only on inputs (data/bh_zones, bh/, data/targets_table.csv); writes PNGs to figs/ only.
Run with the polaris venv python: $POLARIS/.venv/bin/python a_figs_part1.py
"""
import csv
import os
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import Patch

# Paths per PORTABLE_CONTRACT.md: ROOT is handoff/revamp (the directory above this file's),
# WORK the SDPA root above that; both are environment overrides.
ROOT = os.environ.get("HANDOFF", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "bh_zones")
OUT = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, BG = "#0b0b0b", "#fcfcfb"
GRAY = "#b9b9b5"
MUTED = "#5c5c58"
DARKGRAY = "#7a7a76"

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

ANCHOR = "causal S4096 q128 k128 nh32 nkv8 bfp8 HiFi2, 110 cores, p100a, firmware 19.9.0 (today)"
GRID1 = "causal and non-causal S4096 nh32 nkv8 bfp8 HiFi2, 110 cores, p100a, firmware 19.9.0 (today)"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def heading(fig, title, subtitle=None, y=0.975, title_w=100, sub_w=150):
    t = "\n".join(textwrap.wrap(title, title_w))
    fig.text(0.012, y, t, fontsize=13, fontweight="bold", va="top", ha="left", color=INK)
    if subtitle:
        n = t.count("\n") + 1
        st = "\n".join(textwrap.wrap(subtitle, sub_w))
        fig.text(0.012, y - 0.033 * n - 0.01, st, fontsize=9.5, va="top", ha="left", color=MUTED)


def footer(fig, text, w=165):
    fig.text(0.01, 0.006, "\n".join(textwrap.wrap(text, w)), fontsize=8, color=MUTED, ha="left", va="bottom")


def fnum(s):
    s = s.strip().replace(",", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return s


def fz(v, nd=1):
    """Format avoiding a negative zero."""
    if abs(v) < 0.5 * 10 ** (-nd):
        v = 0.0
    return f"{v:.{nd}f}"


# ---------------------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------------------------
def md_table(path, heading_prefix):
    """Return (header, rows) of the first markdown table under a '### <heading_prefix>' line."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("### " + heading_prefix))
    j = i + 1
    while not lines[j].startswith("|"):
        j += 1
    tbl = []
    while j < len(lines) and lines[j].startswith("|"):
        tbl.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
        j += 1
    return tbl[0], tbl[2:]  # header, rows (skip the separator)


RT = os.path.join(D, "report_tables.md")


def rt_T1():
    h, rows = md_table(RT, "T1 walls")
    out = {}
    for r in rows:
        d = {k: fnum(v) for k, v in zip(h, r)}
        out[d["tag"]] = d
    return out


def rt_T3_anchor():
    h, rows = md_table(RT, "T3 t21_causal_q128k128:")
    return {r[0]: fnum(r[1]) for r in rows}


def rt_T4():
    h, rows = md_table(RT, "T4 residual per k-tile")
    return {r[0]: {k: fnum(v) for k, v in zip(h[1:], r[1:])} for r in rows}


def rt_T6():
    h, rows = md_table(RT, "T6 ablations")
    return [{k: fnum(v) for k, v in zip(h, r)} for r in rows]


def rt_T7():
    h, rows = md_table(RT, "T7 perf counters")
    tags = h[1:]
    out = defaultdict(dict)
    for r in rows:
        for tag, v in zip(tags, r[1:]):
            out[tag][r[0]] = fnum(v)
    return out


def decomp(tag):
    """Mean over iterations 1 and 2 of cycles_corr on the wall core per (risc, part), plus kernel span means."""
    acc = defaultdict(list)
    dur = defaultdict(dict)
    with open(os.path.join(D, f"decomp_{tag}.csv")) as f:
        for r in csv.DictReader(f):
            acc[(r["risc"], r["part"])].append(float(r["cycles_corr"]))
            dur[r["risc"]][r["iter"]] = float(r["kernel_dur"])
    parts = {k: sum(v) / len(v) for k, v in acc.items()}
    spans = {k: sum(v.values()) / len(v) for k, v in dur.items()}
    return parts, spans


def runs_trisc1_mean(tag, kind):
    """Mean over iterations 1 and 2 (run_idx 1, 2) of the per-run mean TRISC1 kernel span over the 110 cores."""
    vals = []
    with open(os.path.join(D, f"{tag}_{kind}_runs.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["run_idx"]) in (1, 2):
                vals.append(float(r["trisc1_mean"]))
    return sum(vals) / len(vals)


def targets_anchor_rows():
    out = {}
    with open(os.path.join(ROOT, "data", "targets_table.csv")) as f:
        rows = [r for r in csv.reader(f) if not r[0].startswith("#")]
    h = rows[0]
    for r in rows[1:]:
        if r[0].startswith("prefill_causal_S4096_q128_k128_nh32_nkv8_"):
            d = dict(zip(h, r))
            out[d["source"]] = d
    return out


def wallcore_counters_anchor():
    """Wall-core counter values of the anchor multipass run (iteration 1 core (2,3), iteration 2 core (1,4) per
    t21_causal_q128k128_zon_mp_runs.csv), mean of the two iterations."""
    wall = {}
    with open(os.path.join(D, "t21_causal_q128k128_zon_mp_runs.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["run_idx"]) in (1, 2):
                wall[r["run_id"]] = (r["wall_core_x"], r["wall_core_y"])
    acc = defaultdict(list)
    with open(os.path.join(D, "t21_causal_q128k128_zon_mp_counters.csv")) as f:
        for r in csv.DictReader(f):
            if r["run_id"] in wall and (r["core_x"], r["core_y"]) == wall[r["run_id"]]:
                acc[r["counter"]].append(float(r["value"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


# ---------------------------------------------------------------------------------------------
# 1. a_zone_tax.png
# ---------------------------------------------------------------------------------------------
def fig_zone_tax():
    rows = list(csv.DictReader(open(os.path.join(D, "t02_zone_tax_summary.csv"))))
    riscs = ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
    rlabel = ["BRISC\n(writer)", "NCRISC\n(reader)", "TRISC_0\n(UNPACK)", "TRISC_1\n(MATH)", "TRISC_2\n(PACK)"]

    def get(tag, risc, loop, mode):
        for r in rows:
            if r["tag"] == tag and r["risc"] == risc and int(r["loop"]) == loop and int(r["mode"]) == mode:
                return r
        raise KeyError((tag, risc, loop, mode))

    def kinds(tag, loop):
        out = {}
        for risc in riscs:
            m1, m2, m3 = (get(tag, risc, loop, m) for m in (1, 2, 3))
            acc_in = float(m3["acc_sum"]) / float(m3["acc_n"]) - float(m3["base_cyc"]) / float(m3["n"])
            out[risc] = [float(m1["cost_per_zone"]), float(m2["cost_per_zone"]), float(m3["cost_per_zone"]), acc_in]
        return out

    k16 = kinds("t02_zone_tax_1x1", 16)
    k64 = kinds("t02_zone_tax_1x1", 64)
    k4x4 = kinds("t02_zone_tax_4x4", 16)
    names = [
        "raw DeviceZoneScopedN, per open + close (recorded)",
        "native DeviceZoneScopedSumN1, per occurrence",
        "custom SDPA_ZACC, whole-thread cost per occurrence (ACC_OUT; dashed line = 26.0 used in the decomposition)",
        "custom SDPA_ZACC, in-window inflation (ACC_IN; correction 2.0 per occurrence, 0.0 on TRISC_0)",
    ]
    cols = [RED, AQUA, BLUE, ORANGE]
    n = 4
    gw = 0.8
    bw = gw / n
    fig, ax = plt.subplots(figsize=(W_IN, 7.6))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.835, bottom=0.31)
    for gi, risc in enumerate(riscs):
        for si in range(n):
            x = gi - gw / 2 + bw * (si + 0.5)
            v = k16[risc][si]
            ax.bar(
                x, v, width=bw * 0.9, color=cols[si], edgecolor=BG, linewidth=0.8, label=names[si] if gi == 0 else None
            )
            ax.text(x, v + 0.8, fz(v), ha="center", va="bottom", fontsize=8.8, color=INK)
            ax.plot(
                [x],
                [k64[risc][si]],
                marker="D",
                ms=5.5,
                markerfacecolor=BG,
                markeredgecolor=INK,
                markeredgewidth=1.0,
                ls="",
                label="same zone with a 64-nop body (1x1)" if (gi == 0 and si == 0) else None,
                zorder=5,
            )
            ax.plot(
                [x],
                [k4x4[risc][si]],
                marker="_",
                ms=9,
                color=INK,
                markeredgewidth=1.6,
                ls="",
                label="same zone on a 4x4 grid, 16-nop body" if (gi == 0 and si == 0) else None,
                zorder=6,
            )
    ax.set_xticks(range(len(riscs)))
    ax.set_xticklabels(rlabel)
    ax.set_ylabel("cycles of zone tax per occurrence")
    ax.set_ylim(0, 66)
    ax.axhline(26.0, color=BLUE, lw=0.9, ls=(0, (4, 3)))
    heading(
        fig,
        "T0.2 zone tax per zone kind and RISC: what one profiler zone costs the thread that carries it",
        "nop kernels through ttnn.generic_op on today's card (p100a, firmware 19.9.0); n = 100 raw zones or 2,000 accumulate zones per RISC per run, "
        "16-nop body (bars) or 64-nop body (diamonds), single core (1x1) with the 4x4-grid values as ticks. ACC_IN = accumulated sum per occurrence "
        "minus the zones-off baseline per iteration. The SDPA decomposition subtracts ACC_IN per occurrence from every zone sum and ACC_OUT minus ACC_IN "
        "per occurrence from the un-zoned remainder (ZONE_TAX_IN_STEP).",
    )
    ax.legend(ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.8)
    footer(
        fig,
        "MEASURED. Data: data/bh_zones/t02_zone_tax_summary.csv (modes 1, 2, 3; raw t02_zone_tax_1x1.csv, t02_zone_tax_4x4.csv). "
        "Not in the CSV: a raw zone dropped once the 125-pair buffer is full still costs 11.5 cycles on every RISC (zone_decomposition.md s2.1, n = 1000 run, derived).",
    )
    save(fig, "a_zone_tax.png")


# ---------------------------------------------------------------------------------------------
# 2. a_anchor_threads.png
# ---------------------------------------------------------------------------------------------
WAIT_PARTS = ["K_WAIT", "V_WAIT", "Q_WAIT", "QKTIM_WAIT", "RESERVE_QKT", "OUT_RESERVE", "PACK_DONE"]
FE_PARTS = ["RECONFIG", "MASK", "MASK_DIAG", "EXP_INIT", "PUSHES", "PUSH_HOLD", "POPS"]
MATH_PARTS = ["QK_MM", "PV_MM", "SUBEXP", "REDUCE", "SALAD_EXP", "SALAD_CORR", "NORM"]
CTRL_PARTS = ["UNZONED_IN_STEP", "OUTSIDE_STEP_IN_QCHUNK", "INIT_PRE_FIRST_QCHUNK", "TAIL_AFTER_LAST_QCHUNK"]


def thread_classes(parts, spans):
    """Five classes per RISC in cycles (waits, front end / issue, math / work, control, tax or remainder) with short names."""
    out = {}
    for t in ["TRISC_0", "TRISC_1", "TRISC_2"]:
        g = lambda names: sum(parts.get((t, p), 0.0) for p in names)
        out[t] = (
            [g(WAIT_PARTS), g(FE_PARTS), g(MATH_PARTS), g(CTRL_PARTS), parts[(t, "ZONE_TAX_IN_STEP")]],
            ["waits", "front end", "math zones", "control", "tax"],
        )
    r = lambda p: parts.get(("NCRISC", p), 0.0)
    waits = r("R_BARRIER") + r("R_RESERVE")
    issue = r("R_ISSUE_AND_PUSH")
    ctrl = r("R_CONTROL_IN_KCHUNK") + r("INIT_PRE_FIRST_QCHUNK") + r("TAIL_AFTER_LAST_QCHUNK") + r("BETWEEN_QCHUNKS")
    out["NCRISC"] = (
        [waits, issue, 0.0, ctrl, spans["NCRISC"] - waits - issue - ctrl],
        ["barrier + reserve", "issue and push", "", "control", "remainder to span"],
    )
    w = lambda p: parts.get(("BRISC", p), 0.0)
    waits = w("W_WAIT")
    work = w("W_WRITE")
    ctrl = w("INIT_PRE_FIRST_QCHUNK") + w("TAIL_AFTER_LAST_QCHUNK") + w("BETWEEN_QCHUNKS")
    out["BRISC"] = (
        [waits, 0.0, work, ctrl, spans["BRISC"] - waits - work - ctrl],
        ["wait for output", "", "write", "control", "remainder to span"],
    )
    return out


def fig_anchor_threads():
    parts, spans = decomp("t21_causal_q128k128")
    wall = rt_T1()["t21_causal_q128k128"]["wall_zoff_mean"]
    cls = thread_classes(parts, spans)
    order = [
        ("BRISC", "BRISC\nwriter"),
        ("NCRISC", "NCRISC\nreader"),
        ("TRISC_0", "TRISC_0\nUNPACK"),
        ("TRISC_1", "TRISC_1\nMATH"),
        ("TRISC_2", "TRISC_2\nPACK"),
    ]
    labels = [
        "waits: cb_wait_front K / V / Q, qkt_im, reserves, pack_done; reader: NoC read barrier + cb_reserve_back; writer: cb_wait_front(out)",
        "front end: reconfig, mask bracket, exp_init, pushes, pops; reader: read issue and push",
        "math zones: QK_MM, PV_MM, sub-exp, reduce, SALAD, norm (floor + in-phase excess); writer: write",
        "un-zoned control flow (in step, between steps, prologue, tail)",
        "zone tax removed from the parts (compute: ZONE_TAX_IN_STEP); reader and writer: remainder to the kernel span",
    ]
    cols = [BLUE, ORANGE, AQUA, GRAY, None]
    fig, ax = plt.subplots(figsize=(W_IN, 7.0))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.80, bottom=0.27)
    for yi, (risc, lab) in enumerate(order):
        left = 0.0
        vals, short = cls[risc]
        for ci, v in enumerate(vals):
            pct = 100 * v / wall
            if ci == 4:
                ax.barh(
                    yi,
                    pct,
                    left=left,
                    color=BG,
                    edgecolor=RED,
                    hatch="////",
                    linewidth=1.0,
                    height=0.62,
                    label=labels[ci] if yi == 0 else None,
                )
            else:
                ax.barh(
                    yi,
                    pct,
                    left=left,
                    color=cols[ci],
                    edgecolor=BG,
                    linewidth=1.2,
                    height=0.62,
                    label=labels[ci] if yi == 0 else None,
                )
            if pct >= 4.5 and ci != 4:
                ax.text(
                    left + pct / 2,
                    yi,
                    f"{pct:.1f} %\n{v:,.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=BG if ci in (0, 1, 2) else INK,
                )
            elif pct >= 2.2:
                ax.text(
                    left + pct / 2,
                    yi,
                    f"{pct:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=BG if ci in (0, 1, 2) else INK,
                    bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none") if ci == 4 else None,
                )
            left += pct
        small = [f"{short[i]} {100 * v / wall:.1f} %" for i, v in enumerate(vals) if 0.05 <= 100 * v / wall < 2.2]
        ax.text(left + 0.6, yi, "; ".join(small), va="center", fontsize=8.4, color=MUTED)
    ax.axvline(100, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.text(100, -0.62, "zones-off wall 2,562,642 = 100 %", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([l for _, l in order], fontsize=10)
    ax.set_xlim(0, 140)
    ax.set_ylim(len(order) - 0.5, -0.75)
    ax.set_xlabel(
        "percent of the zones-off device wall (each bar sums to that RISC's zones-on kernel span, about 100.9 percent)"
    )
    ax.grid(axis="y", visible=False)
    heading(
        fig,
        "Anchor decomposition on the wall-setting core, every RISC: where each thread's kernel span goes",
        ANCHOR
        + ". Wall core (2,3), mean of iterations 1 and 2, zone sums corrected per occurrence. UNPACK spends 38.8 percent of the wall in "
        "cb_wait_front poll loops; the same wait shows on MATH as matmul zones 2.3x the model FPU time and on PACK as issue stalls attributed to MASK and EXP_INIT; "
        "the reader is inside noc.async_read_barrier for 80 percent of the wall and the writer waits for output the whole time.",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8.8)
    footer(
        fig,
        "MEASURED (zones on the wall core; classes as report_tables.md T3b). Data: data/bh_zones/decomp_t21_causal_q128k128.csv (cycles_corr, iterations 1 and 2), "
        "wall from T1. The reader's inner-zone tax (2,565 accumulate occurrences) sits inside the derived R_ISSUE_AND_PUSH and R_CONTROL parts per the report's accounting.",
    )
    save(fig, "a_anchor_threads.png")


# ---------------------------------------------------------------------------------------------
# 3. a_anchor_waterfall.png
# ---------------------------------------------------------------------------------------------
def fig_anchor_waterfall():
    T3 = rt_T3_anchor()
    wall = T3["zones-off wall (dev, = wall-core span)"]
    steps = [
        (
            "model floor\non wall core",
            T3["model floor on wall core (per-step floor x actual steps)"],
            AQUA,
            "MODEL",
            "",
        ),
        ("measured\nprologue", T3["init measured (KERNEL start to first QCHUNK, TRISC1)"], GRAY, "MEASURED", ""),
        ("waits on\nTRISC1", T3["waits on TRISC1 (K/V/Q/qkt_im/reserve/pack_done)"], BLUE, "MEASURED", ""),
        (
            "front end\n(reconfig, mask,\nexp_init, pushes,\npops)",
            T3["front end on TRISC1 (reconfig, mask, exp_init, pushes, pops)"],
            ORANGE,
            "MEASURED",
            "",
        ),
        (
            "in-phase excess\n(math zones\nminus floor)",
            T3["in-phase excess = math zones on TRISC1 - floor_wc"],
            RED,
            "MEASURED",
            "",
        ),
        (
            "un-zoned\ncontrol flow",
            T3["un-zoned control flow (in-step + between steps + tail)"],
            DARKGRAY,
            "MEASURED",
            "",
        ),
        (
            "slack displaced\nby the zone tax",
            T3[
                "slack displaced by the zone tax = tax_T1 - (wall_zon - wall_zoff) (INFERRED: waiting that the instrumentation overhead replaced)"
            ],
            BG,
            "INFERRED",
            "////",
        ),
    ]
    fig, ax = plt.subplots(figsize=(W_IN, 7.4))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.81, bottom=0.2)
    base = 0.0
    xs = []
    for i, (name, v, col, tag, hatch) in enumerate(steps):
        ax.bar(i, v, bottom=base, width=0.72, color=col, edgecolor=INK if hatch else BG, hatch=hatch, linewidth=1.1)
        pct = 100 * v / wall
        top = base + v
        if v > 250000:
            ax.text(i, base + v / 2, f"{v:,.0f}\n{pct:.1f} %", ha="center", va="center", fontsize=9.5, color=BG)
            ax.text(i, top + 25000, tag, ha="center", va="bottom", fontsize=8.8, color=MUTED)
        else:
            ax.text(i, top + 25000, f"{v:,.0f} ({pct:.1f} %)\n{tag}", ha="center", va="bottom", fontsize=8.8, color=INK)
        ax.plot([i + 0.36, i + 1 - 0.36], [top, top], color=MUTED, lw=0.9, ls=(0, (3, 3)))
        xs.append(name)
        base = top
    ax.bar(len(steps), wall, width=0.72, color=BG, edgecolor=INK, linewidth=1.4)
    ax.text(
        len(steps),
        wall / 2,
        f"zones-off wall\n{wall:,.0f}\n= 1898.3 us",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
    )
    ax.text(len(steps), wall + 25000, "MEASURED", ha="center", va="bottom", fontsize=8.8, color=MUTED)
    gap = wall - base
    ax.text(
        len(steps) + 0.4,
        2.8e6,
        f"sum of the seven steps {base:,.0f}; gap to the wall {gap:,.0f} cycles ({100 * gap / wall:.2f} %)",
        fontsize=8.8,
        color=INK,
        ha="right",
        va="bottom",
    )
    xs.append("device wall")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=9.2)
    ax.set_ylabel("cycles on the wall-setting core, TRISC1 view")
    ax.set_ylim(0, 3.05e6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v / 1e6:.1f} M"))
    heading(
        fig,
        "Anchor waterfall, TRISC1 view: model floor on the wall core to the measured device wall",
        ANCHOR
        + ". Floor = model per-step floor 852,327 / 153.6 x 165 steps of the wall core. Zone parts from the zones-on run on the wall core, "
        "mean of iterations 1 and 2; the in-phase excess is the MATH thread stalling inside matmul MOPs while the unpacker has no data (the DRAM stream seen from "
        "MATH). Displaced slack = TRISC1 zone tax 177,480 minus the gross wall growth 23,186: waiting that the instrumentation replaced on a non-critical thread.",
    )
    footer(
        fig,
        "MEASURED except the floor (MODEL, analysis/roofline.py predict) and the hatched displaced slack (INFERRED). Data: data/bh_zones/report_tables.md T3 "
        "(t21_causal_q128k128), built from decomp_t21_causal_q128k128.csv; zone_decomposition.md s4.3.",
    )
    save(fig, "a_anchor_waterfall.png")


# ---------------------------------------------------------------------------------------------
# 4. a_reader_breakdown.png
# ---------------------------------------------------------------------------------------------
def fig_reader_breakdown():
    parts, spans = decomp("t21_causal_q128k128")
    T1 = rt_T1()["t21_causal_q128k128"]
    wall = T1["wall_zoff_mean"]
    steps = T1["it1_steps_wc"]
    r = lambda p: parts[("NCRISC", p)]
    loop = r("R_KCHUNK")
    top = [
        ("noc.async_read_barrier", r("R_BARRIER"), BLUE),
        ("read issue + cb_push_back", r("R_ISSUE_AND_PUSH"), ORANGE),
        ("cb_reserve_back (back pressure)", r("R_RESERVE"), AQUA),
        ("un-zoned control in the k loop", r("R_CONTROL_IN_KCHUNK"), GRAY),
    ]
    per_step = [
        ("one K chunk read (16 tiles)", r("R_K_READ") / steps, BLUE),
        ("one V chunk read (16 tiles)", r("R_V_READ") / steps, ORANGE),
        ("Q read share (10 q chunks over 165 steps)", r("R_Q_READ") / steps, AQUA),
        ("control", r("R_CONTROL_IN_KCHUNK") / steps, GRAY),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_IN, 7.6), gridspec_kw=dict(height_ratios=[1, 1], hspace=0.55))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.79, bottom=0.17)
    # whole kernel
    left = 0.0
    for name, v, col in top:
        pct = 100 * v / wall
        ax1.barh(0, pct, left=left, color=col, edgecolor=BG, linewidth=1.2, height=0.6, label=name)
        if pct > 30:
            ax1.text(
                left + pct / 2,
                0,
                f"{name}\n{v:,.0f} cycles = {pct:.1f} % of wall",
                ha="center",
                va="center",
                fontsize=9.5,
                color=BG,
            )
        elif pct > 6:
            ax1.text(
                left + pct / 2,
                0,
                f"issue + push\n{v:,.0f} = {pct:.1f} %",
                ha="center",
                va="center",
                fontsize=9.3,
                color=BG,
            )
        elif pct > 2:
            ax1.text(left + pct / 2, 0, f"{pct:.1f} %", ha="center", va="center", fontsize=9, color=BG)
        left += pct
    ax1.text(
        0.5, 0.9, "Reader (NCRISC) k-loop time by activity, whole kernel", fontsize=11.5, fontweight="bold", va="center"
    )
    nbar = r("R_BARRIER")
    ax1.annotate(
        f"1,710 barriers x {nbar / 1710:,.0f} cycles each; 4 reads of 1,088 B in flight per barrier",
        xy=(40, 0.3),
        xytext=(40, 0.52),
        ha="center",
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7),
    )
    ax1.annotate(
        f"32 async reads per step, {r('R_ISSUE_AND_PUSH') / steps / 32:.0f} cycles per issue + push",
        xy=(87.9, 0.3),
        xytext=(80.5, 0.52),
        ha="center",
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7),
    )
    ax1.text(
        50,
        -0.5,
        f"R_KCHUNK loop total {loop:,.0f} cycles = {100 * loop / wall:.1f} % of the wall; barrier, issue + push, reserve ({100 * r('R_RESERVE') / wall:.1f} %) and control ({100 * r('R_CONTROL_IN_KCHUNK') / wall:.1f} %) tile it exactly",
        fontsize=8.8,
        color=MUTED,
        ha="center",
        va="top",
    )
    ax1.text(
        50,
        -0.74,
        f"the Q read zone ({r('R_Q_READ'):,.0f} cycles = {100 * r('R_Q_READ') / wall:.1f} % of wall) is nested inside the loop: its barrier share is in barrier, the rest in issue + push",
        fontsize=8.8,
        color=MUTED,
        ha="center",
        va="top",
    )
    ax1.set_xlim(0, 100.5)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_yticks([])
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("percent of the zones-off wall 2,562,642 (wall core, mean of iterations 1 and 2)")
    # per step
    left = 0.0
    for name, v, col in per_step:
        ax2.barh(0, v, left=left, color=col, edgecolor=BG, linewidth=1.2, height=0.6, label=name)
        if v > 1500:
            ax2.text(left + v / 2, 0, f"{name}\n{v:,.0f} cycles", ha="center", va="center", fontsize=9.5, color=BG)
        elif v > 300:
            ax2.text(left + v / 2, 0, f"{v:,.0f}", ha="center", va="center", fontsize=8.8, color=BG)
        left += v
    ax2.text(
        80,
        0.9,
        "The same time per k-chunk step: what one K chunk and one V chunk cost the reader",
        fontsize=11.5,
        fontweight="bold",
        va="center",
    )
    ax2.text(
        loop / steps / 2,
        -0.5,
        f"one k-chunk step on the reader = {loop / steps:,.0f} cycles (device wall / 165 steps = {wall / steps:,.0f}); K, V, Q share and control tile it exactly",
        fontsize=8.8,
        color=MUTED,
        ha="center",
        va="top",
    )
    ax2.text(
        loop / steps / 2,
        -0.74,
        f"one chunk = 16 tiles x 1,088 B = 17,408 B: {r('R_K_READ') / steps / 17408:.2f} cycles per byte per core, 34,816 B of K + V per core per step",
        fontsize=8.8,
        color=MUTED,
        ha="center",
        va="top",
    )
    ax2.set_xlim(0, loop / steps * 1.005)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_yticks([])
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("cycles per k-chunk step (R_KCHUNK / 165 steps of the wall core)")
    heading(
        fig,
        "Reader breakdown at the anchor: 80 percent of the wall inside the NoC read barrier, 7.4k cycles per 16-tile chunk",
        ANCHOR
        + ". NCRISC on the wall core, zones on, mean of iterations 1 and 2, sums corrected per occurrence. The reader is busy 99.6 percent of the wall and "
        "never waits for compute except the 3.4 percent of cb_reserve_back back pressure; every core reads its own K and V chunk from DRAM on every step (no forwarding "
        "chains on the causal path).",
    )
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=9)
    footer(
        fig,
        "MEASURED. Data: data/bh_zones/decomp_t21_causal_q128k128.csv (NCRISC rows, cycles_corr); percentages as report_tables.md T2b; zone_decomposition.md s4.2.",
    )
    save(fig, "a_reader_breakdown.png")


# ---------------------------------------------------------------------------------------------
# 5. a_ablation_walls.png
# ---------------------------------------------------------------------------------------------
def fig_ablation_walls():
    T6 = rt_T6()
    a4 = next(r for r in T6 if r["ablation"] == "a4" and r["config"] == "causal_q128k128")
    abls = [
        ("a4", f"A4 reader stub\n(no K/V NoC reads)\nk128 wall {a4['wall_base']:,.0f} to {a4['wall_abl']:,.0f}"),
        ("a4b", "A4b barrier threshold\n4 to 64 reads in flight"),
        ("a2", "A2 causal mask\nbracket off"),
        ("a6", "A6 SFPU exp stub\n(16 exps per step removed)"),
    ]
    ks = [
        ("causal_q128k128", "k_chunk 128", BLUE),
        ("causal_q128k256", "k_chunk 256", ORANGE),
        ("causal_q128k512", "k_chunk 512", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(W_IN, 6.8))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.81, bottom=0.2)
    gw = 0.72
    bw = gw / 3
    for gi, (ab, lab) in enumerate(abls):
        for si, (cfg, klab, col) in enumerate(ks):
            row = next(r for r in T6 if r["ablation"] == ab and r["config"] == cfg)
            d = 100 * (row["wall_abl"] - row["wall_base"]) / row["wall_base"]
            x = gi - gw / 2 + bw * (si + 0.5)
            ax.bar(x, d, width=bw * 0.9, color=col, edgecolor=BG, linewidth=0.8, label=klab if gi == 0 else None)
            fmt = f"{d:+.1f}" if abs(d) >= 1 else f"{d:+.2f}"
            ax.text(
                x,
                d + (1.2 if d >= 0 else -1.2),
                fmt,
                ha="center",
                va="bottom" if d >= 0 else "top",
                fontsize=9,
                color=INK,
            )
    ax.axhline(0, color=INK, lw=1.0)
    ax.axhspan(-0.4, 0.4, color=GRAY, alpha=0.35, lw=0)
    ax.text(
        0.55,
        -74,
        "shaded band: iteration spread of every ablated wall is under 0.4 percent (3 iterations, t22_*_zoff_runs.csv)",
        fontsize=8.8,
        color=MUTED,
        ha="left",
    )
    ax.set_xticks(range(len(abls)))
    ax.set_xticklabels([l for _, l in abls], fontsize=9.8)
    ax.set_ylabel("wall delta, percent of the base zones-off wall")
    ax.set_ylim(-78, 24)
    heading(
        fig,
        "Ablations: what the device wall does when one part is removed (causal q128, three k_chunks)",
        "causal S4096 q128 nh32 nkv8 bfp8 HiFi2, 110 cores, p100a, firmware 19.9.0 (today); k_chunk 128 / 256 / 512; zones off, mean of iterations 1 and 2, "
        "base = t21 zones-off wall of the same config. Removing the DRAM K/V stream (A4) removes 43 to 62 percent of the wall; 64 reads in flight (A4b) make it 13 to 14 percent "
        "slower; the causal mask bracket (A2) and the 16 SFPU exps per step (A6) move the wall by under 0.4 percent.",
    )
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    footer(
        fig,
        "MEASURED. Data: data/bh_zones/report_tables.md T6 (wall_base, wall_abl per ablation and config; walls from t22_abl_*_zoff_runs.csv); zone_decomposition.md s5.1.",
    )
    save(fig, "a_ablation_walls.png")


# ---------------------------------------------------------------------------------------------
# 6. a_ablation_parts_heatmap.png
# ---------------------------------------------------------------------------------------------
def fig_ablation_heatmap():
    rows = list(csv.DictReader(open(os.path.join(D, "ablation_deltas_q128k128.csv"))))
    val = {(r["thread"], r["part"]): r for r in rows}
    abls = ["a4", "a4b", "a2", "a6"]
    spec = [
        ("TRISC_0", "K_WAIT", "K_WAIT (UNPACK)"),
        ("TRISC_0", "V_WAIT", "V_WAIT (UNPACK)"),
        ("TRISC_0", "Q_WAIT", "Q_WAIT (UNPACK)"),
        ("TRISC_1", "QK_MM", "QK_MM (MATH)"),
        ("TRISC_1", "PV_MM", "PV_MM (MATH)"),
        ("TRISC_2", "MASK", "MASK (PACK)"),
        ("TRISC_2", "MASK_DIAG", "MASK_DIAG (PACK)"),
        ("TRISC_2", "EXP_INIT", "EXP_INIT (PACK)"),
        ("TRISC_2", "EXP", "EXP (PACK)"),
        ("TRISC_2", "SUBEXP", "SUBEXP (PACK)"),
        ("TRISC_2", "REDUCE", "REDUCE (PACK)"),
        ("TRISC_0", "RECONFIG", "RECONFIG (UNPACK)"),
        ("TRISC_1", "RECONFIG", "RECONFIG (MATH)"),
        ("TRISC_2", "RECONFIG", "RECONFIG (PACK)"),
    ]
    labels, M = [], []
    for t, p, lab in spec:
        r = val[(t, p)]
        labels.append(f"{lab}\nbase {float(r['base']):,.0f}")
        M.append([float(r[a]) for a in abls])
    grp = ["REDUCE", "SALAD_CORR", "SALAD_EXP", "NORM", "PUSHES"]
    pb, _ = decomp("t21_causal_q128k128")
    pa = {a: decomp(f"t22_abl_{a}_causal_q128k128")[0] for a in abls}
    for t, tl in [("TRISC_0", "UNPACK"), ("TRISC_1", "MATH")]:
        b = sum(pb[(t, g)] for g in grp)
        labels.append(f"REDUCE + SALAD_CORR + SALAD_EXP\n+ NORM + PUSHES ({tl}, summed)\nbase {b:,.0f}")
        M.append([sum(pa[a][(t, g)] for g in grp) - b for a in abls])
    for p, lab in [
        ("R_BARRIER", "R_BARRIER (reader)"),
        ("R_RESERVE", "R_RESERVE (reader)"),
        ("R_ISSUE_AND_PUSH", "R_ISSUE_AND_PUSH (reader)"),
    ]:
        r = val[("NCRISC", p)]
        labels.append(f"{lab}\nbase {float(r['base']):,.0f}")
        M.append([float(r[a]) for a in abls])
    r = val[("BRISC", "W_WAIT")]
    labels.append(f"W_WAIT (writer)\nbase {float(r['base']):,.0f}")
    M.append([float(r[a]) for a in abls])
    M = np.array(M)
    cmap = LinearSegmentedColormap.from_list("bo", [BLUE, "#9dc1ea", BG, "#f5b79c", ORANGE])
    norm = SymLogNorm(linthresh=3000, linscale=0.6, vmin=-2.2e6, vmax=2.2e6, base=10)
    fig, ax = plt.subplots(figsize=(W_IN, 11.2))
    fig.subplots_adjust(left=0.29, right=0.885, top=0.775, bottom=0.05)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = f"{v / 1000:+,.0f}k" if abs(v) >= 1000 else f"{v:+.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.8, color=BG if abs(v) > 250000 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        ["A4\nreader stub", "A4b\n64 reads in flight", "A2\nmask bracket off", "A6\nSFPU exp stub"], fontsize=10
    )
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, 3.5)
    for y in [2.5, 4.5, 10.5, 13.5, 15.5, 18.5]:
        ax.axhline(y, color=INK, lw=0.6)
    cax = fig.add_axes([0.925, 0.10, 0.017, 0.60])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-2e6, -1e6, -3e5, -1e5, -3e4, -1e4, 0, 1e4, 3e4, 1e5, 3e5, 1e6, 2e6])
    cb.set_ticklabels(
        ["-2 M", "-1 M", "-300k", "-100k", "-30k", "-10k", "0", "+10k", "+30k", "+100k", "+300k", "+1 M", "+2 M"]
    )
    cb.ax.tick_params(labelsize=8.5)
    fig.text(
        0.9335,
        0.715,
        "cycles moved\n(ablated minus\nbase, symmetric\nlog scale)",
        fontsize=8.5,
        ha="center",
        va="bottom",
        color=INK,
    )
    heading(
        fig,
        "Ablation heatmap at the anchor: which zone parts move when a part is removed (cycles, ablated minus base, wall core)",
        ANCHOR
        + ". Zones on, wall core, mean of iterations 1 and 2, signed cycles. A4 removes the three UNPACK waits and, by the same amounts, the MATH matmul "
        "excess and the PACK MASK and EXP_INIT stalls (they were the K/V wait seen from other threads); A2 moves the PACK MASK stall one zone downstream into REDUCE "
        "and changes the wall by 9,398 cycles; A6 removes the exp zones and the PACK stall reappears in EXP_INIT and MASK. Blue = fewer cycles, orange = more.",
        y=0.982,
        sub_w=160,
    )
    footer(
        fig,
        "MEASURED. Data: data/bh_zones/ablation_deltas_q128k128.csv (18 rows; base = t21 value); the two summed rows from decomp_t21_causal_q128k128.csv and "
        "decomp_t22_abl_{a4,a4b,a2,a6}_causal_q128k128.csv (cycles_corr, mean of iterations 1 and 2). Table: zone_decomposition.md s5.1 second table.",
        w=175,
    )
    save(fig, "a_ablation_parts_heatmap.png")


# ---------------------------------------------------------------------------------------------
# 7. a_residual_composition.png
# ---------------------------------------------------------------------------------------------
def fig_residual_composition():
    T1 = rt_T1()
    T3 = rt_T3_anchor()
    T6 = rt_T6()
    wall = T1["t21_causal_q128k128"]["wall_zoff_mean"]
    floor_wc = T3["model floor on wall core (per-step floor x actual steps)"]
    init = T3["init measured (KERNEL start to first QCHUNK, TRISC1)"]
    resid = wall - floor_wc - init
    a4 = next(r for r in T6 if r["ablation"] == "a4" and r["config"] == "causal_q128k128")
    a2 = next(r for r in T6 if r["ablation"] == "a2" and r["config"] == "causal_q128k128")
    stream = wall - a4["wall_abl"]  # 1,095,844
    fe = a4["wall_abl"] - floor_wc - init  # 551,041
    reconfig = 139214.0  # T2 TRISC_1 RECONFIG (ZD s5.3)
    control = T3["un-zoned control flow (in-step + between steps + tail)"]  # 81,963
    pops = 15000.0  # ZD s5.3: pops, pushes, push_hold, reserves, pack_done (MATH)
    mask = wall - a2["wall_abl"]  # 9,398
    exp = 0.0  # A6: 0 on the wall
    remainder = fe - reconfig - control - pops - mask - exp  # about 305,000
    # (legend label, short label, in-bar name, value, colour, tag, hatch)
    parts = [
        (
            "exposed DRAM K/V read stream (wall minus A4 wall)",
            "DRAM K/V stream",
            "exposed DRAM\nK/V read stream",
            stream,
            BLUE,
            "MEASURED (A4)",
            "",
        ),
        (
            "RECONFIG, 12 sites per step (MATH view)",
            "RECONFIG (12 sites per step, MATH view)",
            "",
            reconfig,
            ORANGE,
            "MEASURED zone",
            "",
        ),
        (
            "un-zoned control flow (MATH)",
            "un-zoned control flow (MATH)",
            "",
            control,
            DARKGRAY,
            "MEASURED (zone remainder)",
            "",
        ),
        (
            "pops, pushes, reserves, pack_done (MATH)",
            "pops, pushes, reserves, pack_done (MATH)",
            "",
            pops,
            GRAY,
            "MEASURED zones",
            "",
        ),
        (
            "causal mask bracket (A2 wall cost)",
            "causal mask bracket (A2 wall cost)",
            "",
            mask,
            RED,
            "MEASURED (A2)",
            "",
        ),
        ("SFPU exp on the wall (A6)", "SFPU exp on the wall (A6)", "", exp, AQUA, "MEASURED (A6)", ""),
        (
            "remainder: in-phase excess of the math zones with a free reader\n(matmul init, MOP fill, DEST sync, SFPU init on PACK, packer STALLWAITs)",
            "remainder: in-phase excess of the math zones with a free reader",
            "",
            remainder,
            BG,
            "INFERRED (A4 wall minus the zoned parts)",
            "////",
        ),
    ]
    fig, ax = plt.subplots(figsize=(W_IN, 8.8))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.845, bottom=0.265)
    x0, x1 = 0.0, 2.7
    bw = 1.45
    base = 0.0
    wall_parts = [
        ("model floor on\nthe wall core", floor_wc, AQUA, "MODEL"),
        ("measured prologue", init, GRAY, ""),
        ("exposed DRAM\nK/V read stream", stream, BLUE, "MEASURED (A4)"),
        ("compute front end\nwith a free reader", fe, ORANGE, "MEASURED (A4)"),
    ]
    for name, v, col, tag in wall_parts:
        pct = 100 * v / wall
        ax.bar(x0, pct, bottom=base, width=bw, color=col, edgecolor=BG, linewidth=1.2)
        if pct > 3:
            ax.text(
                x0,
                base + pct / 2,
                f"{name}\n{v:,.0f} = {pct:.1f} % of wall\n{tag}",
                ha="center",
                va="center",
                fontsize=9.3,
                color=BG,
            )
        base += pct
    ax.text(x0, 101.5, f"device wall {wall:,.0f} = 100 %", ha="center", va="bottom", fontsize=9.5, color=INK)
    scale = 100 * resid / wall  # 64.3
    base = 0.0
    handles = []
    side = []
    for legend_name, short, inbar, v, col, tag, hatch in parts:
        pr = 100 * v / resid
        h = pr * scale / 100
        ax.bar(x1, h, bottom=base, width=bw, color=col, edgecolor=INK if hatch else BG, hatch=hatch, linewidth=1.1)
        pw = 100 * v / wall
        if h > 20:
            ax.text(
                x1,
                base + h / 2,
                f"{inbar}\n{v:,.0f} cycles\n{pr:.1f} % of residual\n{pw:.1f} % of wall\n{tag}",
                ha="center",
                va="center",
                fontsize=9.0,
                color=BG,
            )
        else:
            side.append((base + h / 2, f"{short}: {v:,.0f} cycles\n{pr:.1f} % of residual, {pw:.1f} % of wall, {tag}"))
        handles.append(
            Patch(
                facecolor=col,
                edgecolor=INK if hatch else BG,
                hatch=hatch,
                label=f"{legend_name}: {pr:.1f} % of residual, {tag}",
            )
        )
        base += h
    # side notes for the thin segments, top to bottom in stacking order, leader lines that do not cross
    side_y = 62.0
    for y_seg, txt in reversed(side):
        ax.annotate(
            txt,
            xy=(x1 + bw / 2, y_seg),
            xytext=(x1 + bw / 2 + 0.14, side_y),
            fontsize=8.4,
            color=INK,
            va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, connectionstyle="arc3,rad=0"),
        )
        side_y -= 6.3
    ax.text(
        x1,
        scale + 1.5,
        f"residual = wall - floor - prologue = {resid:,.0f} = {scale:.1f} % of wall",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=INK,
    )
    y_lo = 100 * (floor_wc + init) / wall
    ax.plot([x0 + bw / 2, x1 - bw / 2], [y_lo, 0], color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.plot([x0 + bw / 2, x1 - bw / 2], [100, scale], color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(
        [
            "whole device wall\n(percent of wall)",
            "the 64.3 percent residual, expanded\n(bar height = its share of the wall)",
        ],
        fontsize=10,
    )
    ax.set_xlim(-0.9, 7.3)
    ax.set_ylim(0, 108)
    ax.set_ylabel("percent of the device wall")
    heading(
        fig,
        "What the residual above the wall-core floor is made of at the anchor: the 64.3 percent of the wall that is not compute floor",
        ANCHOR
        + ". Left: the wall as floor + exposed DRAM K/V stream (wall minus A4 wall) + compute front end with a free reader (A4 wall minus floor minus prologue). "
        "Right: the residual split into the named parts of zone_decomposition.md s5.3; the hatched remainder is the A4 wall minus the zoned parts and was not separately ablated.",
    )
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=8.6, labelspacing=0.5)
    footer(
        fig,
        "Data: zone_decomposition.md s5.3 table (basis column); walls from report_tables.md T1 and T6 (A4, A2 at q128 k128), floor and control from T3, RECONFIG from T2 (TRISC_1). "
        "MEASURED rows by ablation subtraction or zone sums; the remainder is INFERRED.",
    )
    save(fig, "a_residual_composition.png")


# ---------------------------------------------------------------------------------------------
# 8. a_counters_vs_zones.png
# ---------------------------------------------------------------------------------------------
def fig_counters_vs_zones():
    T7 = rt_T7()["t21_causal_q128k128"]
    parts, spans = decomp("t21_causal_q128k128")
    wc = wallcore_counters_anchor()
    wall = rt_T1()["t21_causal_q128k128"]["wall_zoff_mean"]
    left = [
        ("WAITING_FOR_\nNONZERO_SEM_0\n(unpacker)", "WAITING_FOR_NONZERO_SEM_0"),
        ("WAITING_FOR_\nSRCA_VALID\n(math)", "WAITING_FOR_SRCA_VALID"),
        ("WAITING_FOR_\nSRCB_VALID\n(math)", "WAITING_FOR_SRCB_VALID"),
        ("WAITING_FOR_\nNONZERO_SEM_2\n(packer)", "WAITING_FOR_NONZERO_SEM_2"),
    ]
    right = [
        ("K_WAIT\ncb_wait_front(K)", parts[("TRISC_0", "K_WAIT")]),
        ("V_WAIT\ncb_wait_front(V)", parts[("TRISC_0", "V_WAIT")]),
        ("Q_WAIT\ncb_wait_front(Q)", parts[("TRISC_0", "Q_WAIT")]),
    ]
    fig, ax = plt.subplots(figsize=(W_IN, 7.4))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.19)
    xs_l = [0, 1, 2, 3]
    xs_r = [4.6, 5.6, 6.6]
    bw = 0.62
    for x, (lab, key) in zip(xs_l, left):
        v = T7[key]
        ax.bar(x, v, width=bw, color=DARKGRAY, edgecolor=BG, linewidth=1.0)
        ax.text(
            x,
            v + 12000,
            f"{v:,.0f}\n({100 * v / wall:.1f} % of wall)",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=INK,
        )
        ax.plot([x], [wc[key]], marker="_", ms=16, color=INK, markeredgewidth=2.0, ls="", zorder=5)
    tot = 0
    for x, (lab, v) in zip(xs_r, right):
        ax.bar(x, v, width=bw, color=BLUE, edgecolor=BG, linewidth=1.0)
        ax.text(
            x,
            v + 12000,
            f"{v:,.0f}\n({100 * v / wall:.1f} % of wall)",
            ha="center",
            va="bottom",
            fontsize=9.2,
            color=INK,
        )
        tot += v
    ax.plot([xs_r[0] - bw / 2, xs_r[-1] + bw / 2], [tot, tot], color=BLUE, lw=1.2, ls=(0, (4, 3)))
    ax.text(
        xs_r[1],
        tot + 15000,
        f"UNPACK data waits, sum {tot:,.0f} = {100 * tot / wall:.1f} % of wall",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=BLUE,
    )
    ax.set_xticks(xs_l + xs_r)
    ax.set_xticklabels([l for l, _ in left] + [l for l, _ in right], fontsize=9)
    ax.set_ylabel("cycles (linear scale)")
    ax.set_ylim(0, 1.5e6)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, p: f"{v / 1e6:.1f} M" if v >= 1e6 else f"{v / 1e3:.0f}k")
    )
    ax.axvline(3.85, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    ax.text(
        1.5,
        1.44e6,
        "Tensix perf counters (mean of 110 cores, zoned multipass run)",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        5.6,
        1.44e6,
        "in-kernel zones, UNPACK thread, wall core",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        1.1,
        1.37e6,
        f"ticks = wall-core values: SEM_0 {wc['WAITING_FOR_NONZERO_SEM_0']:,.0f};\nSRCA / SRCB {wc['WAITING_FOR_SRCA_VALID']:.0f} / {wc['WAITING_FOR_SRCB_VALID']:.0f}; SEM_2 {wc['WAITING_FOR_NONZERO_SEM_2']:,.0f}",
        ha="center",
        va="top",
        fontsize=8.6,
        color=MUTED,
    )
    ax.text(
        5.6,
        0.78e6,
        "RISC-V poll loops are not Tensix stalls:\ncb_wait_front spins on tiles_received in the\nRISC-V, so the semaphore and SRC-valid counters\nnever see the 980k cycles of data waiting",
        ha="center",
        va="center",
        fontsize=9.6,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=BG, edgecolor=INK, lw=0.8),
    )
    ax.text(
        1.05,
        0.66e6,
        "the same counters in percent of the mean TRISC1 span\n"
        f"zoned run: SEM_0 {100 * T7['WAITING_FOR_NONZERO_SEM_0'] / T7['ref_cnt']:.1f} %, SRCA/SRCB "
        f"{100 * T7['WAITING_FOR_SRCA_VALID'] / T7['ref_cnt']:.1f}/{100 * T7['WAITING_FOR_SRCB_VALID'] / T7['ref_cnt']:.1f} %, "
        f"SEM_2 {100 * T7['WAITING_FOR_NONZERO_SEM_2'] / T7['ref_cnt']:.1f} %\n"
        "unmodified kernel: SEM_2 1,139,421 cycles (53.8 %)",
        ha="center",
        va="center",
        fontsize=8.2,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=BG, edgecolor=MUTED, lw=0.6),
    )
    heading(
        fig,
        "Why the counters said zero: Tensix stall counters against the in-kernel wait zones at the anchor",
        ANCHOR
        + ". Counter bars: mean over the 110 cores of the zones-on multipass run (report_tables.md T7), wall-core value as a tick. Zone bars: UNPACK (TRISC_0) "
        "cb_wait_front zones on the wall core, mean of iterations 1 and 2, corrected per occurrence. The packer semaphore counter (SEM_2) is the pack thread waiting for "
        "math, not for data. Zones do not change the Tensix instruction stream (THREAD_INSTRUCTIONS_0/1/2 differ by 11 / 0 / 59, FPU and SFPU counters identical).",
    )
    footer(
        fig,
        "MEASURED. Data: data/bh_zones/report_tables.md T7 (t21_causal_q128k128 zon_mp, mean of 110 cores and iterations 1 and 2; ref_cnt as the span); wall-core ticks from "
        "t21_causal_q128k128_zon_mp_counters.csv (cores (2,3) and (1,4)); zones from decomp_t21_causal_q128k128.csv; "
        "unmodified-kernel SEM_2 from zone_decomposition.md s4.6 (t01_causal_q128k128_mp).",
    )
    save(fig, "a_counters_vs_zones.png")


# ---------------------------------------------------------------------------------------------
# 9. a_per_ktile_composition.png
# ---------------------------------------------------------------------------------------------
def fig_per_ktile():
    T1 = rt_T1()
    T4 = rt_T4()
    cfgs = [
        ("t21_causal_q128k128", "causal q128 k128", "q128\nk128"),
        ("t21_causal_q128k256", "causal q128 k256", "q128\nk256"),
        ("t21_causal_q128k512", "causal q128 k512", "q128\nk512"),
        ("t21_causal_q64k128", "causal q64 k128", "q64\nk128"),
        ("t21_causal_q256k128", "causal q256 k128", "q256\nk128"),
        ("t21_causal_q512k128", "causal q512 k128", "q512\nk128"),
        ("t21_causal_q512k512", "causal q512 k512", "q512\nk512"),
    ]
    fig, ax = plt.subplots(figsize=(W_IN, 8.0))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.81, bottom=0.265)
    ticklabels = []
    for i, (tag, t4key, lab) in enumerate(cfgs):
        r1 = T1[tag]
        r4 = T4[t4key]
        kct = r1["k_chunk"] / 32
        kt = r1["it1_steps_wc"] * kct
        wall = r1["wall_zoff_mean"]
        init = r1["init_measured_pre_first_qchunk_T1"]
        floor_wc = wall - r1["resid_wc_init_meas"] - init
        floor_kt = floor_wc / kt
        wall_kt = wall / kt
        kv_kt = r4["T0_KV_wait_per_ktile"]
        resid_kt = r4["per_ktile_wc_init_meas"]
        rest_kt = wall_kt - floor_kt - kv_kt
        segs = [
            (floor_kt, AQUA, "model floor per k-tile on the wall core (MODEL)"),
            (kv_kt, BLUE, "UNPACK K + V wait per k-tile (MEASURED)"),
            (rest_kt, ORANGE, "rest of the residual per k-tile: Q wait, front end, in-phase excess, control (DERIVED)"),
        ]
        base = 0.0
        for v, col, name in segs:
            ax.bar(i, v, bottom=base, width=0.7, color=col, edgecolor=BG, linewidth=1.2, label=name if i == 0 else None)
            if v > 380:
                ax.text(i, base + v / 2, f"{v:,.0f}", ha="center", va="center", fontsize=9.3, color=BG)
            elif v > 200:
                ax.text(i, base + v / 2, f"{v:,.0f}", ha="center", va="center", fontsize=8.3, color=BG)
            else:
                ax.text(i + 0.37, base + v / 2, f"{v:,.0f}", ha="left", va="center", fontsize=8.3, color=INK)
            base += v
        ax.text(
            i,
            base + 90,
            f"{wall_kt:,.0f} per k-tile\n{kt:,.0f} k-tiles",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
        ax.text(i, -230, f"residual {resid_kt:,.0f}", ha="center", va="top", fontsize=8.6, color=MUTED)
        ticklabels.append(f"{lab}\nwall {wall / 1e6:.2f} M\n{r1['it1_steps_wc']:.0f} steps")
    ax.axvline(2.5, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    ax.axvline(5.5, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    ax.text(
        1,
        9750,
        "k_chunk sweep at q128: wall per k-tile\nflat within 2.4 percent",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
    )
    ax.text(
        4,
        9750,
        "q_chunk sweep at k128: the floor grows,\nthe K/V wait vanishes from q256 up",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
    )
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(ticklabels, fontsize=9.5)
    ax.set_ylabel("cycles per k-tile on the wall-setting core")
    ax.set_ylim(-600, 10400)
    ax.set_yticks(range(0, 10001, 1000))
    heading(
        fig,
        "Per-k-tile composition on the wall core, seven causal grid-1 configs: floor, UNPACK K/V wait, rest of the residual",
        "causal S4096 nh32 nkv8 bfp8 HiFi2, 110 cores, p100a, firmware 19.9.0 (today); q_chunk and k_chunk as labelled. Bar total = zones-off wall / (steps of the wall core x "
        "k tiles per chunk). Floor = wall minus wall-core residual minus measured prologue (model per-step floor x actual steps). At q128 the wall per k-tile is the K+V bytes per "
        "k-tile at the sustained DRAM rate; at q256 and q512 the K/V wait is 1 to 2 percent and the kernel is compute-thread bound (q512 k128 is off trend on every basis).",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=9)
    footer(
        fig,
        "Floor MODEL (analysis/roofline.py predict x measured step count); K/V wait and residual MEASURED (zones, wall core, mean of iterations 1 and 2); the orange rest is DERIVED "
        "(wall per k-tile minus floor minus K/V wait). "
        "Data: data/bh_zones/report_tables.md T1 (wall_zoff_mean, resid_wc_init_meas, it1_steps_wc, init) and T4 (T0_KV_wait_per_ktile, per_ktile_wc_init_meas); zone_decomposition.md s4.4.",
    )
    save(fig, "a_per_ktile_composition.png")


# ---------------------------------------------------------------------------------------------
# 10. a_util_ladder.png
# ---------------------------------------------------------------------------------------------
def fig_util_ladder():
    T7 = rt_T7()

    def util(tag):
        m = T7[tag]["MATH_COUNTER"]
        return 100 * m / runs_trisc1_mean(tag, "zon_mp"), 100 * m / runs_trisc1_mean(tag, "zoff")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_IN, 7.0), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.22))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.775, bottom=0.3)
    series = [("causal", "causal", BLUE, "o"), ("noncausal", "non-causal", ORANGE, "s")]
    for ax, sweep in [(ax1, "q"), (ax2, "k")]:
        if sweep == "q":
            pts = [(64, "q64k128"), (128, "q128k128"), (256, "q256k128"), (512, "q512k128")]
            ax.set_xlabel("q_chunk at k_chunk 128")
        else:
            pts = [(128, "q128k128"), (256, "q128k256"), (512, "q128k512")]
            ax.set_xlabel("k_chunk at q_chunk 128")
        xs = list(range(len(pts)))
        labels = defaultdict(list)  # (x, side) -> [(value, colour)]
        for key, name, col, mk in series:
            u_on, u_off = zip(*[util(f"t21_{key}_{cfg}") for _, cfg in pts])
            ax.plot(
                xs,
                u_on,
                color=col,
                marker=mk,
                ms=8,
                lw=2.0,
                label=f"{name}: MATH_COUNTER / mean TRISC1 span of the same zoned multipass run (MEASURED)"
                if sweep == "q"
                else None,
            )
            ax.plot(
                xs,
                u_off,
                color=col,
                marker=mk,
                ms=8,
                lw=1.2,
                ls=(0, (4, 3)),
                markerfacecolor=BG,
                markeredgewidth=1.6,
                label=f"{name}: same counter / mean TRISC1 span of the zones-off twin (upper estimate, DERIVED)"
                if sweep == "q"
                else None,
            )
            for x, a, b in zip(xs, u_on, u_off):
                labels[(x, "L")].append((a, col))
                labels[(x, "R")].append((b, col))
        # value labels: filled series to the left, hollow to the right; same-side labels pushed at least 2.8 points apart
        for (x, sd), items in labels.items():
            items.sort()
            ys = [v for v, _ in items]
            for i in range(1, len(ys)):
                if ys[i] - ys[i - 1] < 2.8:
                    ys[i] = ys[i - 1] + 2.8
            for (v, col), y in zip(items, ys):
                ax.text(
                    x - 0.13 if sd == "L" else x + 0.13,
                    y,
                    f"{v:.1f}",
                    ha="right" if sd == "L" else "left",
                    va="center",
                    fontsize=8.6,
                    color=col,
                    bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none", alpha=0.9),
                    zorder=7,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(p) for p, _ in pts])
        ax.set_xlim(-0.45, len(pts) - 0.55)
        ax.set_ylim(0, 85)
        ax.set_ylabel("MATH_COUNTER over the kernel window, percent")
    ax1.set_title("versus q_chunk (k_chunk 128)", fontsize=11.5, loc="left")
    ax2.set_title("versus k_chunk (q_chunk 128)", fontsize=11.5, loc="left")
    heading(
        fig,
        "Utilization ladder: measured MATH busy fraction over the kernel window, no model curves",
        GRID1
        + ". MATH_COUNTER is the mean over the 110 cores of the zones-on multipass run (mean of iterations 1 and 2); the window is the mean per-core TRISC1 kernel span "
        "of the same run (filled) or of the zones-off twin (hollow, since the zone tax lengthens compute-bound spans by 5 to 11 percent while FPU and SFPU counters are unchanged). "
        "Causal q128 sits at 43 to 44 percent because the K/V DRAM stream, not compute, sets the wall; growing q_chunk raises the floor against a fixed byte stream.",
    )
    ax1.legend(loc="upper center", bbox_to_anchor=(0.95, -0.15), ncol=1, fontsize=8.8)
    footer(
        fig,
        "MEASURED counters, DERIVED ratios. Data: data/bh_zones/report_tables.md T7 (MATH_COUNTER per config); t21_*_zon_mp_runs.csv and t21_*_zoff_runs.csv (trisc1_mean, run_idx 1 and 2). "
        "T7 ref_cnt equals the multipass TRISC1 mean span within 0.1 percent.",
    )
    save(fig, "a_util_ladder.png")


if __name__ == "__main__":
    fig_zone_tax()
    fig_anchor_threads()
    fig_anchor_waterfall()
    fig_reader_breakdown()
    fig_ablation_walls()
    fig_ablation_heatmap()
    fig_residual_composition()
    fig_counters_vs_zones()
    fig_per_ktile()
    fig_util_ladder()
