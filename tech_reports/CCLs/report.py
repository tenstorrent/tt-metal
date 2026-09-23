#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Merge every run under data/runs/ into reports and figures.

    python tech_reports/CCLs/report.py

A cell measured by several runs keeps its latest measurement. Per machine, that
is arch, dtype and packet, this writes:

    results/SUMMARY_<machine>.md     one curve per (op, n): ring over line, DRAM over L1
    results/FULL_<machine>.md        a section per (topology, memory)
    images/bw_<machine>_n<n>.png     DRAM, a panel per topology
    images/memcfg_<machine>_n<n>.png where an op was measured in both L1 and DRAM
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
IMAGE_DIR = HERE / "images"

BYTE_TARGETS = [1 << k for k in range(10, 35)]
OP_ORDER = ["all_reduce", "all_to_all", "all_gather", "reduce_scatter"]
TOPOLOGIES = ["ring", "line"]  # preference order for the summary
MEMORIES = ["dram", "l1"]

# Mesh shape and cluster axis stay in cells.csv but out of the reports: readers
# compare device counts.
MACHINE = ["arch", "dtype", "packet"]
CELL = MACHINE + ["topology", "memory", "op", "n", "target_bytes"]

# Okabe-Ito, chosen for hue separation under color-vision deficiency.
PALETTE = [("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^"), ("#CC79A7", "D")]


def load():
    paths = sorted(DATA_DIR.glob("runs/*/cells.csv"))
    if not paths:
        raise SystemExit(f"no runs/*/cells.csv in {DATA_DIR}. Run run_bench.sh first.")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # Run names are timestamps, so sorting by them puts the latest measurement last.
    df = df.sort_values("run").drop_duplicates(CELL, keep="last")
    return add_metrics(fill_links(df))


# ----------------------------------------------------------------- metrics


def fill_links(df):
    """Not every op reports its link count (all_to_all does not). The count
    depends on the machine, topology and device count, not the collective, so a
    blank takes the one count the other ops there agree on."""
    agreed = df.groupby(MACHINE + ["topology", "n"])["links"].transform(
        lambda s: s.dropna().iloc[0] if s.nunique() == 1 else float("nan")
    )
    df["links"] = df["links"].fillna(agreed)
    return df


def busbw_factor(op, n):
    """nccl-tests bus bandwidth correction, doc/PERFORMANCE.md."""
    return 2.0 * (n - 1) / n if op == "all_reduce" else (n - 1) / n


def link_factor(op, n):
    """Bottleneck bytes as a multiple of the total array.

    all_to_all differs because each chunk has one destination, so relay hops on
    a ring are extra traffic rather than a substitute for a direct send.
    """
    if op == "all_reduce":
        return 2.0 * (n - 1) / n
    if op == "all_to_all":
        return (n // 2) * ((n + 1) // 2) / n
    return (n - 1) / n


def add_metrics(df):
    """GB/s throughout. linkbw is per link per direction, blank without a link count."""
    secs = df["us"] * 1e-6
    df["algbw_gbps"] = df["bytes"] / secs / 1e9
    df["busbw_gbps"] = df["algbw_gbps"] * [busbw_factor(op, n) for op, n in zip(df["op"], df["n"])]
    directions = (df["resolved"] == "Ring").map({True: 2, False: 1})
    link_bytes = df["bytes"] * [link_factor(op, n) for op, n in zip(df["op"], df["n"])]
    df["linkbw_gbps"] = link_bytes / (directions * df["links"]) / secs / 1e9
    df["pct_of_line_rate"] = df["linkbw_gbps"] / df["line_rate_gbps"] * 100.0
    df["roofline_pct"] = df["ideal_us"] / df["us"] * 100.0
    return df


def stem(machine):
    return "_".join(str(x) for x in machine)


# ---------------------------------------------------------------- markdown

NOTES = [
    "Byte targets follow nccl-tests (`-b 1K -e 16G -f 2`), rounded to whole tiles.",
    "`size` is what was achieved. `algbw` is size/time. `busbw` is nccl's correction,",
    "`algbw * 2(n-1)/n` for all_reduce and `algbw * (n-1)/n` for the rest.",
    "",
    "`linkbw` is per link per direction. It divides the bottleneck traffic by the",
    "links carrying it, doubled on a ring. The topology is what the CCL ops resolve",
    "for the devices (`ttnn.get_usable_topology`). The link count is the count the",
    "ops discovered rather than used: a program may clamp below it to fit its worker",
    "cores. all_to_all uses a different factor from busbw, because on a ring its",
    "chunks relay through intermediate chips instead of arriving in one hop.",
    "`line rate` is linkbw against the per-link line rate recorded above.",
    "",
    "`roofline` is the op's own performance model over the measured time. The model",
    "takes the tightest of several ceilings and does not record which one bound, so",
    "rows compare within a size regime, not across the column. It is not link",
    "utilization, and it is blank for ops with no model.",
    "",
    "Empty rows had no tiled shape at that device count, or did not fit in memory.",
    "Each device must hold at least one tile of the split, so the total array is at",
    "least 1024*n elements.",
]


def header(title, g):
    r = g.iloc[0]
    return (
        [
            f"# {title}",
            "",
            "```",
            f"arch          {r['arch']}  line rate {r['line_rate_gbps']} GB/s per link per direction",
            f"dtype         {r['dtype']}  ({r['page_size']} B pages)",
            f"packet        {r['packet']} B",
            f"runs          {', '.join(sorted(g['run'].unique()))}",
            "```",
            "",
        ]
        + NOTES
        + [""]
    )


def _fmt(v, spec):
    return "" if pd.isna(v) else format(v, spec)


def op_tables(g, level, with_config=False):
    out = []
    config_head, config_rule = ("config | ", "--|") if with_config else ("", "")
    for op in OP_ORDER:
        per_op = g[g["op"] == op]
        if per_op.empty:
            continue
        out += [
            f"{level} {op}",
            "",
            f"| n | {config_head}target (B) | size (B) | count | pages | time (us) | "
            "algbw (GB/s) | busbw (GB/s) | linkbw (GB/s) | line rate (%) | roofline (%) |",
            f"|--:|{config_rule}--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
        for n, per_n in per_op.groupby("n"):
            config = ""
            if with_config:
                r = per_n.iloc[0]
                config = f"{r['topology']} {r['memory']} | "
            cells = per_n.set_index("target_bytes")
            for target in BYTE_TARGETS:
                if target not in cells.index:
                    out.append(f"| {n} | {config}{target} | | | | | | | | | |")
                    continue
                r = cells.loc[target]
                out.append(
                    f"| {n} | {config}{target} | {r['bytes']} | {r['count']} | {r['num_pages']} | "
                    f"{r['us']:.2f} | {r['algbw_gbps']:.2f} | {r['busbw_gbps']:.2f} | "
                    f"{_fmt(r['linkbw_gbps'], '.2f')} | {_fmt(r['pct_of_line_rate'], '.1f')} | "
                    f"{_fmt(r['roofline_pct'], '.1f')} |"
                )
        out.append("")
    return out


def links(g):
    seen = sorted({int(x) for x in g["links"].dropna()})
    return ", ".join(map(str, seen)) or "?"


def raw_report(g):
    out = header("CCL benchmark, all configurations", g)
    for topology in TOPOLOGIES:
        for memory in MEMORIES:
            sec = g[(g["topology"] == topology) & (g["memory"] == memory)]
            if sec.empty:
                continue
            out += [f"## {topology}, {memory.upper()}", "", f"links per direction: {links(sec)}", ""]
            out += op_tables(sec, "###")
    return out


def best(g):
    """One (topology, memory) per (op, n), so no curve mixes configurations."""
    preference = [(t, m) for t in TOPOLOGIES for m in MEMORIES]
    picked = []
    for _, per_curve in g.groupby(["op", "n"]):
        measured = set(zip(per_curve["topology"], per_curve["memory"]))
        topology, memory = next(c for c in preference if c in measured)
        picked.append(per_curve[(per_curve["topology"] == topology) & (per_curve["memory"] == memory)])
    return pd.concat(picked)


def best_report(g):
    out = header("CCL benchmark, best configuration", g)
    out += [
        "Each (op, n) shows one configuration, the first measured of ring over line",
        "and DRAM over L1. Ring needs a wraparound link, so it exists only at the",
        "device count that closes the axis.",
        "",
    ]
    return out + op_tables(best(g), "##", with_config=True)


# ----------------------------------------------------------------- figures


def human(n):
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:g}{unit}"
        n /= 1024


def label(name):
    return name.replace("_", "-").capitalize()


def panel(ax, groups, title, ymax, xlim):
    """groups: [(label, color, marker, linestyle, frame)]. busbw on the axis. The
    gutter holds each peak, with the link utilization measured at that size."""
    groups = sorted(groups, key=lambda g: -g[4]["busbw_gbps"].max())
    for name, color, marker, linestyle, g in groups:
        ax.plot(
            g["bytes"],
            g["busbw_gbps"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=4,
            linewidth=1.6,
            label=name,
        )

    # Nudged apart where peaks land close.
    y_transform = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    placed = math.inf
    for _, color, _, _, g in groups:
        at_peak = g.loc[g["busbw_gbps"].idxmax()]
        peak, pct = at_peak["busbw_gbps"], at_peak["pct_of_line_rate"]
        text = f"{peak:.1f} GB/s" + ("" if pd.isna(pct) else f" · {pct:.0f}%")
        y = min(peak, placed - ymax * 0.06)
        ax.text(1.02, y, text, transform=y_transform, va="center", fontsize=8, color=color)
        placed = y

    ax.text(1.02, 1.01, "peak · % line rate", transform=ax.transAxes, fontsize=7, color="0.45", va="bottom")

    lo, hi = xlim
    ax.set_xscale("log", base=2)
    # Counted down from the largest size, so the asymptote always has a tick.
    ax.set_xticks([2**k for k in range(int(math.log2(hi)), int(math.log2(lo)) - 1, -4)])
    ax.xaxis.set_major_formatter(lambda v, _: human(v))
    ax.set_xlim(lo / 1.5, hi * 1.5)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Tensor size (bytes)", fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)


def figure(panels, title, subtitle, dest):
    """panels: [(panel_title, groups)], one subplot each, sharing both axes."""
    frames = [g for _, groups in panels for *_, g in groups]
    ymax = 1.1 * max(g["busbw_gbps"].max() for g in frames)
    xlim = (min(g["bytes"].min() for g in frames), max(g["bytes"].max() for g in frames))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.6), squeeze=False)
    for ax, (panel_title, groups) in zip(axes[0], panels):
        panel(ax, groups, panel_title, ymax, xlim)
    axes[0][0].set_ylabel("Bus bandwidth (GB/s)", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.90, subtitle, fontsize=9, color="0.35", ha="center")
    # wspace leaves room for each panel's gutter labels.
    fig.subplots_adjust(top=0.78, right=0.82, wspace=0.55)
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dest}")


def subtitle(machine, n, g, memory=None):
    arch, dtype, packet = machine
    count = links(g)
    parts = [f"{n} devices", f"{count} link{'' if count == '1' else 's'}/direction", dtype]
    if memory:
        parts.append(memory)
    return ", ".join(parts + [f"{packet} B packet"])


def bw_figures(machine, g, style):
    dram = g[g["memory"] == "dram"]
    for n, per_n in dram.groupby("n"):
        panels = []
        for topology in ("line", "ring"):
            per_topo = per_n[per_n["topology"] == topology]
            if not per_topo.empty:
                groups = [(label(op), *style[op], "-", gg) for op, gg in per_topo.groupby("op")]
                panels.append((f"{topology} topology", groups))
        figure(
            panels,
            f"CCL bus bandwidth - {machine[0].capitalize()}",
            subtitle(machine, n, per_n, "DRAM"),
            IMAGE_DIR / f"bw_{stem(machine)}_n{n}.png",
        )


def memcfg_figures(machine, g):
    for (n, topology), per_n in g.groupby(["n", "topology"]):
        # Only ops measured in both memory configs say anything about the memory limit.
        ops = [op for op, gg in per_n.groupby("op") if set(MEMORIES) <= set(gg["memory"])]
        if not ops:
            continue
        # Colored per (op, memory) pair: the memory config is the comparison here,
        # and with a single op the op color would make both series identical.
        groups = []
        for i, (op, memory) in enumerate((op, m) for op in ops for m in MEMORIES):
            gg = per_n[(per_n["op"] == op) & (per_n["memory"] == memory)]
            color, marker = PALETTE[i % len(PALETTE)]
            groups.append((f"{label(op)}, {memory.upper()}", color, marker, "-" if memory == "dram" else "--", gg))
        figure(
            [(f"{topology} topology", groups)],
            f"CCL mem config - {machine[0].capitalize()}",
            subtitle(machine, n, per_n),
            IMAGE_DIR / f"memcfg_{stem(machine)}_n{n}.png",
        )


def main():
    df = load()
    IMAGE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    # Assigned once over everything, so an op keeps its color across figures.
    style = {op: PALETTE[i % len(PALETTE)] for i, op in enumerate(sorted(df["op"].unique()))}
    for machine, g in df.groupby(MACHINE):
        g = g.sort_values("bytes")
        for kind, lines in (("SUMMARY", best_report(g)), ("FULL", raw_report(g))):
            dest = RESULTS_DIR / f"{kind}_{stem(machine)}.md"
            dest.write_text("\n".join(lines).rstrip() + "\n")
            print(f"wrote {dest}")
        bw_figures(machine, g, style)
        memcfg_figures(machine, g)


if __name__ == "__main__":
    main()
