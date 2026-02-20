#!/usr/bin/env python3
"""Analyze D2H socket benchmark results (throughput or latency).

Usage:
  python3 analyze_d2h_throughput.py --throughput results.csv
  python3 analyze_d2h_throughput.py --latency   latency_results.csv
  python3 analyze_d2h_throughput.py --multichip multichip_bench.log
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CYCLES_PER_US = 1350.0


def human_bytes(n):
    for u in ("", "K", "M", "G"):
        if n < 1024:
            return f"{int(n)}{u}" if n == int(n) else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}T"


# ── data loading ─────────────────────────────────────────────────────────────


def _read_split_rows(path):
    header = ""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Header: first comma-containing line that doesn't start with a digit
            if not header and "," in line and not line[0].isdigit():
                header = line
                continue
            # Data row: first token must be numeric
            try:
                float(line.split(",")[0])
            except (ValueError, IndexError):
                continue
            rows.append(line.split(","))
    return header, rows


def _to_dataframe(rows, cols, numeric_cols):
    df = pd.DataFrame([r[: len(cols)] for r in rows], columns=cols)
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c])
    return df


def load_throughput_csv(path):
    """Load throughput benchmark CSV. Handles commas in MeshCoordinate."""
    header, rows = _read_split_rows(path)

    hcols = [h.strip() for h in header.split(",")]
    if "throughput_gbps" in hcols:
        nc, cols = 10, [
            "page_size",
            "socket_fifo_size",
            "total_data",
            "data_size",
            "pages_per_iter",
            "num_iterations",
            "total_pages",
            "avg_per_page_us",
            "avg_per_page_cycles",
            "throughput_gbps",
        ]
    elif "data_size" in hcols:
        nc, cols = 7, [
            "page_size",
            "socket_fifo_size",
            "total_data",
            "data_size",
            "num_iterations",
            "avg_per_page_us",
            "avg_per_page_cycles",
        ]
    else:
        nc, cols = 6, [
            "page_size",
            "socket_fifo_size",
            "total_data",
            "num_iterations",
            "avg_per_page_us",
            "avg_per_page_cycles",
        ]

    df = _to_dataframe(rows, cols, cols)

    if "data_size" not in df:
        df["data_size"] = (df.total_data / df.num_iterations).astype(int)
    if "throughput_gbps" not in df:
        df["throughput_gbps"] = df.page_size / (df.avg_per_page_us * 1e3)
    return df


def load_latency_csv(path):
    """Load latency benchmark CSV (pcie_socket_data_ping output)."""
    _, rows = _read_split_rows(path)

    cols = [
        "page_size",
        "socket_fifo_size",
        "num_iterations",
        "avg_us",
        "min_us",
        "max_us",
        "p50_us",
        "p99_us",
        "avg_cycles",
        "min_cycles",
        "max_cycles",
    ]
    return _to_dataframe(rows, cols, cols)


def load_h2d_latency_csv(path):
    """Load H2D latency benchmark CSV (h2d_socket_data_ping output with h2d_mode column)."""
    _, rows = _read_split_rows(path)

    cols = [
        "page_size",
        "socket_fifo_size",
        "h2d_mode",
        "num_iterations",
        "avg_us",
        "min_us",
        "max_us",
        "p50_us",
        "p99_us",
        "avg_cycles",
        "min_cycles",
        "max_cycles",
    ]
    numeric_cols = [c for c in cols if c != "h2d_mode"]
    return _to_dataframe(rows, cols, numeric_cols)


def load_h2d_throughput_csv(path):
    """Load H2D throughput benchmark CSV (with h2d_mode column)."""
    _, rows = _read_split_rows(path)

    cols = [
        "page_size",
        "socket_fifo_size",
        "h2d_mode",
        "total_data",
        "data_size",
        "pages_per_iter",
        "num_iterations",
        "total_pages",
        "avg_per_page_us",
        "avg_per_page_cycles",
        "throughput_gbps",
    ]
    numeric_cols = [c for c in cols if c != "h2d_mode"]
    return _to_dataframe(rows, cols, numeric_cols)


# ── helpers ──────────────────────────────────────────────────────────────────


def median_over_fifo(df):
    return (
        df.groupby(["page_size", "total_data"])
        .agg(throughput_gbps=("throughput_gbps", "median"), avg_per_page_us=("avg_per_page_us", "median"))
        .reset_index()
    )


def _annotate_heatmap(ax, data, vmax, fontsize=5):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white" if v > vmax * 0.7 else "black",
                )


# ══════════════════════════════════════════════════════════════════════════════
#  THROUGHPUT
# ══════════════════════════════════════════════════════════════════════════════


def tp_print_report(df):
    agg = median_over_fifo(df)
    nf = df.socket_fifo_size.nunique()

    print(f"\n{'='*70}\n  D2H Steady-State Throughput  ({len(df)} rows)\n{'='*70}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(
        f"  FIFO sizes : {nf} values, {human_bytes(df.socket_fifo_size.min())} .. "
        f"{human_bytes(df.socket_fifo_size.max())}"
    )
    print(f"  Total data : {[human_bytes(x) for x in sorted(df.total_data.unique())]}")

    pk = agg.sort_values("throughput_gbps", ascending=False).iloc[0]
    print(
        f"\n  Peak: {pk.throughput_gbps:.3f} GB/s "
        f"(page={human_bytes(pk.page_size)}, total={human_bytes(pk.total_data)})"
    )

    tds = sorted(agg.total_data.unique())
    hdr = f"  {'page':>6}" + "".join(f"{human_bytes(t):>8}" for t in tds)
    print(f"\n  Throughput (GB/s) -- median across {nf} FIFO sizes:\n{hdr}\n  {'-'*(len(hdr)-2)}")
    for ps in sorted(agg.page_size.unique()):
        row = f"  {human_bytes(ps):>6}"
        for td in tds:
            c = agg[(agg.page_size == ps) & (agg.total_data == td)]
            row += f"{c.throughput_gbps.values[0]:>8.3f}" if len(c) else f"{'--':>8}"
        print(row)

    td = df.total_data.max()
    sub = df[df.total_data == td]
    print(f"\n  FIFO impact at total_data={human_bytes(td)}:")
    print(f"  {'page':>6} {'min':>8} {'median':>8} {'max':>8} {'CV%':>6}")
    print(f"  {'-'*38}")
    for ps in sorted(sub.page_size.unique()):
        g = sub[sub.page_size == ps].throughput_gbps
        cv = g.std() / g.mean() * 100 if g.mean() > 0 else 0
        print(f"  {human_bytes(ps):>6} {g.min():>8.3f} {g.median():>8.3f} {g.max():>8.3f} {cv:>5.1f}%")

    agg2 = median_over_fifo(df[df.total_data == td])
    print(f"\n  Per-page timing (total_data={human_bytes(td)}):")
    print(f"  {'page':>6} {'us/page':>10} {'cycles':>10} {'GB/s':>8}\n  {'-'*38}")
    for _, r in agg2.sort_values("page_size").iterrows():
        print(
            f"  {human_bytes(r.page_size):>6} {r.avg_per_page_us:>10.3f} "
            f"{r.avg_per_page_us * CYCLES_PER_US:>10.0f} {r.throughput_gbps:>8.3f}"
        )


def tp_plot_throughput(df, out="d2h_throughput.png"):
    fs_max = df.socket_fifo_size.max()
    sub = df[df.socket_fifo_size == fs_max]
    agg = sub.groupby("page_size").agg(throughput_gbps=("throughput_gbps", "median")).reset_index()
    # keep per-total_data lines
    fig, ax = plt.subplots(figsize=(12, 6))
    for td in sorted(sub.total_data.unique()):
        g = sub[sub.total_data == td].sort_values("page_size")
        ax.plot(g.page_size, g.throughput_gbps, "o-", label=f"{human_bytes(td)}", ms=5, lw=2)
    ax.set(xscale="log", xlabel="Page Size", ylabel="Throughput (GB/s)",
           title=f"D2H Throughput vs Page Size — FIFO={human_bytes(fs_max)} (max), each line = one total transfer size")
    ax.set_xticks(t := sorted(sub.page_size.unique()))
    ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out}")


def tp_plot_vs_fifo(df, out="d2h_tp_vs_fifo.png"):
    td = df.total_data.max()
    sub = df[df.total_data == td]
    fig, ax = plt.subplots(figsize=(12, 6))
    for ps in sorted(sub.page_size.unique()):
        g = sub[sub.page_size == ps].sort_values("socket_fifo_size")
        ax.plot(g.socket_fifo_size, g.throughput_gbps, "o-", label=human_bytes(ps), ms=4, lw=1.5)
    ax.set(
        xscale="log",
        xlabel="Socket FIFO Size",
        ylabel="GB/s",
        title=f"D2H Throughput vs Socket FIFO Size — total transfer size = {human_bytes(td)}, each line = one page size",
    )
    ax.set_xticks(t := sorted(sub.socket_fifo_size.unique()))
    ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=7)
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="Page Size")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def tp_plot_heatmap_grid(df, out="d2h_tp_heatmap_grid.png"):
    td_vals = sorted(df.total_data.unique())
    ncols, n = 4, len(td_vals)
    nrows = -(-n // ncols)
    vmin, vmax = df.throughput_gbps.min(), df.throughput_gbps.max()
    all_ps, all_fs = sorted(df.page_size.unique()), sorted(df.socket_fifo_size.unique())

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.5 * nrows))
    for idx, ax in enumerate(axes.flatten()):
        if idx >= n:
            ax.axis("off")
            continue
        td = td_vals[idx]
        pv = (
            df[df.total_data == td]
            .pivot_table(index="page_size", columns="socket_fifo_size", values="throughput_gbps")
            .reindex(index=all_ps, columns=all_fs)
        )
        im = ax.imshow(pv.values, aspect="auto", cmap="YlGn", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"total_data = {human_bytes(td)}", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(all_fs)))
        ax.set_xticklabels([human_bytes(f) for f in all_fs], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(all_ps)))
        ax.set_yticklabels([human_bytes(p) for p in all_ps], fontsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Page Size", fontsize=9)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("FIFO Size", fontsize=9)
        _annotate_heatmap(ax, pv.values, vmax, fontsize=4)

    fig.suptitle("D2H Throughput (GB/s) — each subplot = one total transfer size, X = FIFO size, Y = page size", fontsize=13, fontweight="bold", y=0.98)
    cb = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cb, label="GB/s")
    fig.subplots_adjust(left=0.04, right=0.9, hspace=0.45, wspace=0.25)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def tp_export_csv(df, out="d2h_throughput_summary.csv"):
    repr_fifos = [1024, 4096, 32768, 512 * 1024 * 1024]
    repr_fifos = [f for f in repr_fifos if f in df.socket_fifo_size.values]
    nf = df.socket_fifo_size.nunique()

    def pivot(sub, label):
        pv = sub.pivot_table(index="page_size", columns="total_data", values="throughput_gbps")
        pv.columns = [f"D0 Read {human_bytes(c)}" for c in pv.columns]
        pv.index.name = label
        return pv

    tables = [pivot(median_over_fifo(df), f"page_size (median over {nf} FIFO sizes)")]
    for fs in repr_fifos:
        sub = df[df.socket_fifo_size == fs]
        if not sub.empty:
            tables.append(pivot(sub, f"page_size (FIFO={human_bytes(fs)})"))

    with open(out, "w") as f:
        for i, t in enumerate(tables):
            if i > 0:
                f.write("\n")
            t.to_csv(f, float_format="%.6f")
    print(f"  Saved {out}")


def run_throughput(path, prefix=""):
    df = load_throughput_csv(path)
    tp_print_report(df)
    tp_plot_throughput(df, out=f"{prefix}d2h_throughput.png")
    tp_plot_vs_fifo(df, out=f"{prefix}d2h_tp_vs_fifo.png")
    tp_plot_heatmap_grid(df, out=f"{prefix}d2h_tp_heatmap_grid.png")
    tp_export_csv(df, out=f"{prefix}d2h_throughput_summary.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  LATENCY
# ══════════════════════════════════════════════════════════════════════════════


def lat_print_report(df):
    nf = df.socket_fifo_size.nunique()
    print(f"\n{'='*70}\n  D2H Round-Trip Latency  ({len(df)} rows)\n{'='*70}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(f"  FIFO sizes : {[human_bytes(x) for x in sorted(df.socket_fifo_size.unique())]}")
    print(f"  Iterations : {int(df.num_iterations.iloc[0])}")

    # p50 table: rows=page_size, cols=fifo_size
    fifos = sorted(df.socket_fifo_size.unique())
    hdr = f"  {'page':>6}" + "".join(f"{human_bytes(f):>10}" for f in fifos)
    print(f"\n  p50 latency (us):\n{hdr}\n  {'-'*(len(hdr)-2)}")
    for ps in sorted(df.page_size.unique()):
        row = f"  {human_bytes(ps):>6}"
        for fs in fifos:
            c = df[(df.page_size == ps) & (df.socket_fifo_size == fs)]
            row += f"{c.p50_us.values[0]:>10.2f}" if len(c) else f"{'--':>10}"
        print(row)

    # spread: min vs p50 vs max at largest FIFO
    fs_max = df.socket_fifo_size.max()
    sub = df[df.socket_fifo_size == fs_max]
    print(f"\n  Latency spread at FIFO={human_bytes(fs_max)}:")
    print(f"  {'page':>6} {'min':>8} {'p50':>8} {'p99':>8} {'max':>8} {'avg':>8}")
    print(f"  {'-'*50}")
    for ps in sorted(sub.page_size.unique()):
        r = sub[sub.page_size == ps].iloc[0]
        print(
            f"  {human_bytes(ps):>6} {r.min_us:>8.2f} {r.p50_us:>8.2f} "
            f"{r.p99_us:>8.2f} {r.max_us:>8.2f} {r.avg_us:>8.2f}"
        )

    # FIFO impact
    print(f"\n  FIFO impact on p50 latency:")
    print(f"  {'page':>6} " + "".join(f"{human_bytes(f):>10}" for f in fifos))
    print(f"  {'-'*(6 + 10*len(fifos))}")
    for ps in sorted(df.page_size.unique()):
        row = f"  {human_bytes(ps):>6}"
        for fs in fifos:
            c = df[(df.page_size == ps) & (df.socket_fifo_size == fs)]
            row += f"{c.p50_us.values[0]:>10.2f}" if len(c) else f"{'--':>10}"
        print(row)


def lat_plot(df, out="d2h_latency.png"):
    """Latency vs page_size, one line per FIFO size. Dashed min/max lines instead of fill."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for fs in sorted(df.socket_fifo_size.unique()):
        g = df[df.socket_fifo_size == fs].sort_values("page_size")
        color = ax.plot(g.page_size, g.p50_us, "o-", label=f"FIFO={human_bytes(fs)}", ms=5, lw=2)[0].get_color()
        ax.plot(g.page_size, g.min_us, "--", color=color, lw=0.8, alpha=0.5)
        ax.plot(g.page_size, g.max_us, "--", color=color, lw=0.8, alpha=0.5)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Page Size (bytes)",
        ylabel="Latency (us)",
        title="D2H Round-Trip Latency (p50 solid, min/max dashed)",
    )
    ax.set_xticks(t := sorted(df.page_size.unique()))
    ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=9)
    ax.minorticks_off()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out}")


def lat_plot_breakdown(df, out="d2h_latency_breakdown.png"):
    """Bar chart: p50 latency stacked by page_size, grouped by FIFO size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: p50 latency
    fifos = sorted(df.socket_fifo_size.unique())
    pages = sorted(df.page_size.unique())
    x = np.arange(len(pages))
    w = 0.8 / len(fifos)
    for i, fs in enumerate(fifos):
        g = df[df.socket_fifo_size == fs].set_index("page_size").reindex(pages)
        axes[0].bar(x + i * w, g.p50_us, w, label=f"FIFO={human_bytes(fs)}")
    axes[0].set(xlabel="Page Size", ylabel="p50 Latency (us)", title="D2H p50 Latency")
    axes[0].set_xticks(x + w * (len(fifos) - 1) / 2)
    axes[0].set_xticklabels([human_bytes(p) for p in pages], rotation=45, fontsize=8)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2, axis="y")

    # Right: max/p99 outlier view
    for i, fs in enumerate(fifos):
        g = df[df.socket_fifo_size == fs].set_index("page_size").reindex(pages)
        axes[1].bar(x + i * w, g.max_us, w, label=f"FIFO={human_bytes(fs)}", alpha=0.6)
    axes[1].set(xlabel="Page Size", ylabel="Max Latency (us)", title="D2H Max Latency")
    axes[1].set_xticks(x + w * (len(fifos) - 1) / 2)
    axes[1].set_xticklabels([human_bytes(p) for p in pages], rotation=45, fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def lat_export_csv(df, out="d2h_latency_summary.csv"):
    """Pivot: rows=page_size, one set of columns per FIFO size."""
    fifos = sorted(df.socket_fifo_size.unique())
    result = pd.DataFrame(index=sorted(df.page_size.unique()))
    result.index.name = "page_size"
    for fs in fifos:
        sub = df[df.socket_fifo_size == fs].set_index("page_size")
        tag = human_bytes(fs)
        for col in ["p50_us", "min_us", "max_us", "p99_us", "avg_us"]:
            result[f"{col} (FIFO={tag})"] = sub[col]
    result.to_csv(out, float_format="%.4f")
    print(f"  Saved {out}")


def run_latency(path):
    df = load_latency_csv(path)
    lat_print_report(df)
    lat_plot(df)
    lat_plot_breakdown(df)
    lat_export_csv(df)


# -- H2D throughput ----


def h2d_median_over_fifo(df):
    return (
        df.groupby(["h2d_mode", "page_size", "total_data"])
        .agg(throughput_gbps=("throughput_gbps", "median"), avg_per_page_us=("avg_per_page_us", "median"))
        .reset_index()
    )


def h2d_tp_print_report(df):
    modes = sorted(df.h2d_mode.unique())
    nf = df.socket_fifo_size.nunique()
    agg = h2d_median_over_fifo(df)

    print(f"\n{'='*70}\n  H2D Steady-State Throughput  ({len(df)} rows)\n{'='*70}")
    print(f"  Modes      : {modes}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(
        f"  FIFO sizes : {nf} values, {human_bytes(df.socket_fifo_size.min())} .. "
        f"{human_bytes(df.socket_fifo_size.max())}"
    )
    print(f"  Total data : {[human_bytes(x) for x in sorted(df.total_data.unique())]}")

    for mode in modes:
        sub = df[df.h2d_mode == mode]
        pk = sub.sort_values("throughput_gbps", ascending=False).iloc[0]
        print(
            f"\n  Peak {mode}: {pk.throughput_gbps:.3f} GB/s "
            f"(page={human_bytes(pk.page_size)}, FIFO={human_bytes(pk.socket_fifo_size)}, "
            f"total={human_bytes(pk.total_data)})"
        )

    tds = sorted(agg.total_data.unique())
    for mode in modes:
        m_agg = agg[agg.h2d_mode == mode]
        hdr = f"  {'page':>6}" + "".join(f"{human_bytes(t):>8}" for t in tds)
        print(f"\n  {mode} throughput (GB/s) -- median across {nf} FIFO sizes:\n{hdr}\n  {'-'*(len(hdr)-2)}")
        for ps in sorted(m_agg.page_size.unique()):
            row = f"  {human_bytes(ps):>6}"
            for td in tds:
                c = m_agg[(m_agg.page_size == ps) & (m_agg.total_data == td)]
                row += f"{c.throughput_gbps.values[0]:>8.3f}" if len(c) else f"{'--':>8}"
            print(row)

    td = df.total_data.max()
    for mode in modes:
        sub = df[(df.total_data == td) & (df.h2d_mode == mode)]
        if sub.empty:
            continue
        print(f"\n  {mode} FIFO impact at total_data={human_bytes(td)}:")
        print(f"  {'page':>6} {'min':>8} {'median':>8} {'max':>8} {'CV%':>6}")
        print(f"  {'-'*38}")
        for ps in sorted(sub.page_size.unique()):
            g = sub[sub.page_size == ps].throughput_gbps
            cv = g.std() / g.mean() * 100 if g.mean() > 0 else 0
            print(f"  {human_bytes(ps):>6} {g.min():>8.3f} {g.median():>8.3f} {g.max():>8.3f} {cv:>5.1f}%")

    agg2 = h2d_median_over_fifo(df[df.total_data == td])
    print(f"\n  Per-page timing (total_data={human_bytes(td)}, median over FIFO sizes):")
    print(f"  {'page':>6} {'mode':>14} {'us/page':>10} {'cycles':>10} {'GB/s':>8}\n  {'-'*52}")
    for ps in sorted(agg2.page_size.unique()):
        for mode in modes:
            r = agg2[(agg2.page_size == ps) & (agg2.h2d_mode == mode)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(
                f"  {human_bytes(r.page_size):>6} {mode:>14} {r.avg_per_page_us:>10.3f} "
                f"{r.avg_per_page_us * CYCLES_PER_US:>10.0f} {r.throughput_gbps:>8.3f}"
            )


def h2d_tp_plot(df, out="h2d_throughput.png"):
    td_max = df.total_data.max()
    fs_max = df.socket_fifo_size.max()
    sub = df[(df.total_data == td_max) & (df.socket_fifo_size == fs_max)]
    modes = sorted(sub.h2d_mode.unique())
    agg = sub.groupby(["h2d_mode", "page_size"]).agg(throughput_gbps=("throughput_gbps", "median")).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for mode in modes:
        g = agg[agg.h2d_mode == mode].sort_values("page_size")
        ax.plot(g.page_size, g.throughput_gbps, "o-", label=mode, ms=5, lw=2)
    ax.set(
        xscale="log",
        xlabel="Page Size",
        ylabel="GB/s",
        title=f"H2D Throughput vs Page Size — FIFO={human_bytes(fs_max)} (max), total transfer size={human_bytes(td_max)} (max)",
    )
    ax.set_xticks(t := sorted(agg.page_size.unique()))
    ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out}")


def h2d_tp_plot_vs_fifo(df, out="h2d_tp_vs_fifo.png"):
    """Throughput vs FIFO size — all page sizes, one subplot per mode."""
    td = df.total_data.max()
    sub = df[df.total_data == td]
    modes = sorted(sub.h2d_mode.unique())
    # Include small, medium and new large page sizes
    all_page_sizes = sorted(sub.page_size.unique())
    rep_pages = [p for p in [64, 256, 1024, 4096, 16384, 32768, 65536, 131072, 262144]
                 if p in all_page_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        mode_df = sub[sub.h2d_mode == mode]
        for ps in rep_pages:
            g = mode_df[mode_df.page_size == ps].sort_values("socket_fifo_size")
            if not g.empty:
                ax.plot(g.socket_fifo_size, g.throughput_gbps, "o-", label=human_bytes(ps), ms=4, lw=1.5)
        ax.set(xscale="log", xlabel="Socket FIFO Size", ylabel="Throughput (GB/s)" if idx == 0 else "", title=f"H2D mode: {mode}")
        fifos = sorted(mode_df.socket_fifo_size.unique())
        tick_fifos = fifos[:: max(1, len(fifos) // 6)]
        ax.set_xticks(tick_fifos)
        ax.set_xticklabels([human_bytes(x) for x in tick_fifos], rotation=45, fontsize=7)
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9, title="Page Size")

    fig.suptitle(f"H2D Throughput vs Socket FIFO Size — total transfer size={human_bytes(td)} (max), each line = one page size", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def h2d_tp_plot_at_max_fifo(df, out="h2d_tp_at_max_fifo.png"):
    """GB/s vs page size at the maximum FIFO size, HOST_PUSH and DEVICE_PULL side by side."""
    fs_max = df.socket_fifo_size.max()
    sub = df[df.socket_fifo_size == fs_max]
    modes = sorted(sub.h2d_mode.unique())
    td_vals = sorted(sub.total_data.unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        mode_df = sub[sub.h2d_mode == mode]
        for td in td_vals:
            g = mode_df[mode_df.total_data == td].sort_values("page_size")
            if not g.empty:
                ax.plot(g.page_size, g.throughput_gbps, "o-", label=human_bytes(td), ms=5, lw=2)
        ax.set(
            xscale="log",
            xlabel="Page Size",
            ylabel="GB/s" if idx == 0 else "",
            title=f"H2D mode: {mode}  — FIFO={human_bytes(fs_max)} (max)",
        )
        page_ticks = sorted(mode_df.page_size.unique())
        ax.set_xticks(page_ticks)
        ax.set_xticklabels([human_bytes(p) for p in page_ticks], rotation=45, fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9, title="Total Data")

    fig.suptitle(
        f"H2D Throughput vs Page Size — FIFO={human_bytes(fs_max)} (max), each line = one total transfer size",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def h2d_tp_export_csv(df, out="h2d_throughput_summary.csv"):
    modes = sorted(df.h2d_mode.unique())
    nf = df.socket_fifo_size.nunique()
    repr_fifos = [f for f in [1024, 16384, 524288, 1048576] if f in df.socket_fifo_size.values]

    def pivot_combined(sub, label):
        parts = []
        for mode in modes:
            pv = sub[sub.h2d_mode == mode].pivot_table(
                index="page_size", columns="total_data", values="throughput_gbps"
            )
            pv.columns = [f"{mode} {human_bytes(c)}" for c in pv.columns]
            parts.append(pv)
        merged = parts[0].join(parts[1:], how="outer")
        merged.index.name = label
        return merged

    agg = h2d_median_over_fifo(df)
    tables = [pivot_combined(agg, f"page_size (median over {nf} FIFO sizes)")]
    for fs in repr_fifos:
        sub = df[df.socket_fifo_size == fs]
        if not sub.empty:
            tables.append(pivot_combined(sub, f"page_size (FIFO={human_bytes(fs)})"))

    with open(out, "w") as f:
        for i, t in enumerate(tables):
            if i > 0:
                f.write("\n")
            t.to_csv(f, float_format="%.6f")
    print(f"  Saved {out}")


def run_h2d_throughput(path, prefix=""):
    df = load_h2d_throughput_csv(path)
    h2d_tp_print_report(df)
    h2d_tp_plot(df, out=f"{prefix}h2d_throughput.png")
    h2d_tp_plot_vs_fifo(df, out=f"{prefix}h2d_tp_vs_fifo.png")
    h2d_tp_plot_at_max_fifo(df, out=f"{prefix}h2d_tp_at_max_fifo.png")
    h2d_tp_export_csv(df, out=f"{prefix}h2d_throughput_summary.csv")


# -- H2D latency ----


def h2d_print_report(df):
    modes = sorted(df.h2d_mode.unique())
    print(f"\n{'='*70}\n  H2D Round-Trip Latency  ({len(df)} rows)\n{'='*70}")
    print(f"  Modes      : {modes}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(f"  FIFO sizes : {[human_bytes(x) for x in sorted(df.socket_fifo_size.unique())]}")
    print(f"  Iterations : {int(df.num_iterations.iloc[0])}")

    # p50 latency comparison: HOST_PUSH vs DEVICE_PULL at largest FIFO
    fs_max = df.socket_fifo_size.max()
    print(f"\n  p50 latency (us) at FIFO={human_bytes(fs_max)}:")
    print(f"  {'page':>6}", end="")
    for m in modes:
        print(f" {m:>12}", end="")
    print(f" {'delta':>10}")
    print(f"  {'-'*50}")
    for ps in sorted(df.page_size.unique()):
        print(f"  {human_bytes(ps):>6}", end="")
        vals = {}
        for m in modes:
            c = df[(df.page_size == ps) & (df.socket_fifo_size == fs_max) & (df.h2d_mode == m)]
            val = c.p50_us.values[0] if len(c) else None
            vals[m] = val
            print(f" {val:>12.2f}" if val else f" {'--':>12}", end="")
        if len(vals) == 2 and all(v is not None for v in vals.values()):
            delta = list(vals.values())[1] - list(vals.values())[0]
            print(f" {delta:>+10.2f}")
        else:
            print()

    # Per-mode breakdown
    for mode in modes:
        sub = df[df.h2d_mode == mode]
        fifos = sorted(sub.socket_fifo_size.unique())
        print(f"\n  {mode} - p50 latency (us):")
        hdr = f"  {'page':>6}" + "".join(f"{human_bytes(f):>10}" for f in fifos)
        print(f"{hdr}\n  {'-'*(len(hdr)-2)}")
        for ps in sorted(sub.page_size.unique()):
            row = f"  {human_bytes(ps):>6}"
            for fs in fifos:
                c = sub[(sub.page_size == ps) & (sub.socket_fifo_size == fs)]
                row += f"{c.p50_us.values[0]:>10.2f}" if len(c) else f"{'--':>10}"
            print(row)


def h2d_plot(df, out="h2d_latency.png"):
    """Main H2D latency chart: HOST_PUSH vs DEVICE_PULL comparison at largest FIFO."""
    fs_max = df.socket_fifo_size.max()
    sub = df[df.socket_fifo_size == fs_max]
    modes = sorted(sub.h2d_mode.unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in modes:
        g = sub[sub.h2d_mode == mode].sort_values("page_size")
        color = ax.plot(g.page_size, g.p50_us, "o-", label=f"{mode} (p50)", ms=6, lw=2.5)[0].get_color()
        ax.plot(g.page_size, g.min_us, "--", color=color, lw=1, alpha=0.4, label=f"{mode} (min/max)")
        ax.plot(g.page_size, g.max_us, "--", color=color, lw=1, alpha=0.4)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Page Size (bytes)",
        ylabel="Latency (us)",
        title=f"H2D Round-Trip Latency: HOST_PUSH vs DEVICE_PULL (FIFO={human_bytes(fs_max)})",
    )
    ax.set_xticks(t := sorted(sub.page_size.unique()))
    ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=9)
    ax.minorticks_off()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out}")


def h2d_plot_breakdown(df, out="h2d_latency_breakdown.png"):
    """Detailed per-mode view with all FIFO sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    modes = sorted(df.h2d_mode.unique())

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        sub = df[df.h2d_mode == mode]
        for fs in sorted(sub.socket_fifo_size.unique()):
            g = sub[sub.socket_fifo_size == fs].sort_values("page_size")
            ax.plot(g.page_size, g.p50_us, "o-", label=f"FIFO={human_bytes(fs)}", ms=4, lw=1.5, alpha=0.7)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Page Size (bytes)",
            ylabel="Latency (us)" if idx == 0 else "",
            title=f"{mode}",
        )
        ax.set_xticks(t := sorted(sub.page_size.unique()))
        ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=9)
        ax.minorticks_off()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("H2D Latency by FIFO Size (p50)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def h2d_plot_comparison(df, out="h2d_latency_comparison.png"):
    """Compare HOST_PUSH vs DEVICE_PULL directly at largest FIFO."""
    fs_max = df.socket_fifo_size.max()
    sub = df[df.socket_fifo_size == fs_max]
    modes = sorted(sub.h2d_mode.unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in modes:
        g = sub[sub.h2d_mode == mode].sort_values("page_size")
        ax.plot(g.page_size, g.p50_us, "o-", label=mode, ms=6, lw=2.5)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Page Size (bytes)",
        ylabel="Latency (us)",
        title=f"H2D: HOST_PUSH vs DEVICE_PULL (FIFO={human_bytes(fs_max)}, p50)",
    )
    ax.set_xticks(t := sorted(sub.page_size.unique()))
    ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=9)
    ax.minorticks_off()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def h2d_plot_fifo_impact(df, out="h2d_latency_vs_fifo.png"):
    """Plot latency vs FIFO size for representative page sizes, separate subplots per mode."""
    modes = sorted(df.h2d_mode.unique())
    page_sizes = [64, 256, 1024, 4096, 16384]
    page_sizes = [p for p in page_sizes if p in df.page_size.values]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for idx, mode in enumerate(modes):
        ax = axes[idx]
        sub = df[df.h2d_mode == mode]
        for ps in page_sizes:
            g = sub[sub.page_size == ps].sort_values("socket_fifo_size")
            if not g.empty:
                ax.plot(g.socket_fifo_size, g.p50_us, "o-", label=human_bytes(ps), ms=5, lw=2)
        ax.set(
            xscale="log",
            xlabel="Socket FIFO Size (bytes)",
            ylabel="Latency (us)" if idx == 0 else "",
            title=f"{mode}",
        )
        ax.set_xticks(t := sorted(sub.socket_fifo_size.unique()))
        ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=8)
        ax.minorticks_off()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, title="Page Size")

    fig.suptitle("H2D Latency vs FIFO Size", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def h2d_export_csv(df, out="h2d_latency_summary.csv"):
    """Pivot: separate tables for each mode."""
    modes = sorted(df.h2d_mode.unique())
    tables = []
    for mode in modes:
        sub = df[df.h2d_mode == mode]
        fifos = sorted(sub.socket_fifo_size.unique())
        result = pd.DataFrame(index=sorted(sub.page_size.unique()))
        result.index.name = f"page_size ({mode})"
        for fs in fifos:
            g = sub[sub.socket_fifo_size == fs].set_index("page_size")
            tag = human_bytes(fs)
            for col in ["p50_us", "min_us", "max_us", "p99_us", "avg_us"]:
                result[f"{col} (FIFO={tag})"] = g[col]
        tables.append(result)

    with open(out, "w") as f:
        for i, t in enumerate(tables):
            if i > 0:
                f.write("\n")
            t.to_csv(f, float_format="%.4f")
    print(f"  Saved {out}")


def run_h2d_latency(path):
    df = load_h2d_latency_csv(path)
    h2d_print_report(df)
    h2d_plot(df)
    h2d_plot_breakdown(df)
    h2d_plot_fifo_impact(df)
    h2d_export_csv(df)


# -- D2H ping ----


def _resolve_input_path(path):
    return Path(path).expanduser().resolve()


def _resolve_d2h_ping_iteration_csv(path):
    p = _resolve_input_path(path)
    return p / "ping_iterations.csv" if p.is_dir() else p


def ping_plot_timeseries(path="ping_iterations.csv", out="d2h_ping_timeseries.png"):
    in_csv = _resolve_d2h_ping_iteration_csv(path)
    out_path = in_csv.parent / out
    try:
        df = pd.read_csv(in_csv)
    except FileNotFoundError:
        print(f"  Warning: {in_csv} not found")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.iteration, df.latency_us, "o-", ms=3, lw=1, alpha=0.7)
    ax.axhline(df.latency_us.median(), color="red", ls="--", lw=1.5, label=f"p50 = {df.latency_us.median():.2f} us")
    ax.axhline(df.latency_us.mean(), color="orange", ls=":", lw=1.5, label=f"avg = {df.latency_us.mean():.2f} us")
    ax.set(xlabel="Iteration", ylabel="Latency (us)", title="D2H Pure Ping: Per-Iteration Latency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out_path}")


def ping_export_csv(path="ping_iterations.csv", out="d2h_ping_summary.csv"):
    in_csv = _resolve_d2h_ping_iteration_csv(path)
    out_path = in_csv.parent / out
    try:
        df = pd.read_csv(in_csv)
        df.to_csv(out_path, index=False, float_format="%.4f")
        print(f"  Saved {out_path} ({len(df)} iterations)")
    except FileNotFoundError:
        print(f"  Warning: {in_csv} not found, skipping CSV export")


def run_ping(path):
    ping_plot_timeseries(path)
    ping_export_csv(path)


# -- H2D ping ----


def h2d_ping_plot(csv_dir, out="h2d_ping_timeseries.png"):
    csv_dir = _resolve_input_path(csv_dir)
    modes = {"HOST_PUSH": None, "DEVICE_PULL": None}
    for mode in modes:
        p = csv_dir / f"h2d_ping_iterations_{mode}.csv"
        try:
            modes[mode] = pd.read_csv(p)
        except FileNotFoundError:
            print(f"  Warning: {p} not found")

    present = {k: v for k, v in modes.items() if v is not None}
    if not present:
        return

    out_path = csv_dir / out
    fig, ax = plt.subplots(figsize=(12, 5))
    for mode, df in present.items():
        ax.plot(df.iteration, df.latency_us, "o-", ms=3, lw=1, alpha=0.7, label=mode)
        ax.axhline(
            df.latency_us.median(),
            ls="--",
            lw=1.5,
            alpha=0.6,
            label=f"{mode} p50 = {df.latency_us.median():.2f} us",
        )
    ax.set(xlabel="Iteration", ylabel="Latency (us)", title="H2D Pure Ping: Per-Iteration Latency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out_path}")


def h2d_ping_export_csv(csv_dir, out="h2d_ping_summary.csv"):
    csv_dir = _resolve_input_path(csv_dir)
    dfs = []
    for mode in ["HOST_PUSH", "DEVICE_PULL"]:
        p = csv_dir / f"h2d_ping_iterations_{mode}.csv"
        try:
            df = pd.read_csv(p)
            df["h2d_mode"] = mode
            dfs.append(df)
        except FileNotFoundError:
            pass
    if not dfs:
        print("  Warning: no H2D ping iteration files found")
        return
    out_path = csv_dir / out
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Saved {out_path} ({len(combined)} rows)")


def run_h2d_ping(path):
    p = _resolve_input_path(path)
    csv_dir = p if p.is_dir() else p.parent
    h2d_ping_plot(csv_dir)
    h2d_ping_export_csv(csv_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-CHIP MAX THROUGHPUT  (D2HSocketMultiChipMaxThroughputBenchmark)
# ══════════════════════════════════════════════════════════════════════════════


def load_multichip_csv(path):
    """Load the multi-chip benchmark output.

    The file may contain comment/blank lines (lines starting with '#' or empty)
    mixed in with the CSV.  We strip those out and parse the CSV rows, which have
    the columns:
        tray_id, asic_location, mesh_coord, socket_fifo_size, total_data,
        data_size, pages_per_iter, num_iterations, total_pages,
        avg_per_page_us, avg_per_page_cycles, throughput_gbps

    mesh_coord may contain commas (e.g. "(0,3)"), so we only take the first 12
    comma-separated tokens and label the rest as the coord string.
    """
    cols = [
        "tray_id",
        "asic_location",
        "mesh_coord",
        "socket_fifo_size",
        "total_data",
        "data_size",
        "pages_per_iter",
        "num_iterations",
        "total_pages",
        "avg_per_page_us",
        "avg_per_page_cycles",
        "throughput_gbps",
    ]
    numeric_cols = [c for c in cols if c not in ("mesh_coord",)]

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip header row and any non-CSV lines (UMD/logger output etc.)
            if line.startswith("tray_id"):
                continue
            parts = line.split(",")
            # First token must be a plain integer (tray_id); reject anything else
            try:
                int(parts[0])
            except (ValueError, IndexError):
                continue
            if len(parts) < len(cols):
                continue
            # mesh_coord may be "MeshCoordinate([row, col])" → two tokens after split;
            # collapse any extra middle tokens back into col index 2
            if len(parts) > len(cols):
                extra = len(parts) - len(cols)
                coord_str = ",".join(parts[2 : 3 + extra])
                parts = parts[:2] + [coord_str] + parts[3 + extra :]
            rows.append(parts[: len(cols)])

    df = pd.DataFrame(rows, columns=cols)
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c])

    # Build a human-readable chip label used throughout charts/reports
    df["chip_label"] = df.apply(
        lambda r: f"Tray{int(r.tray_id)}/ASIC{int(r.asic_location)} {r.mesh_coord}", axis=1
    )
    return df


# ── report ────────────────────────────────────────────────────────────────────


def mc_print_report(df):
    chips = df["chip_label"].unique()
    fifos = sorted(df["socket_fifo_size"].unique())

    print(f"\n{'='*70}")
    print(f"  D2H Multi-Chip Max Throughput  ({len(df)} rows, {len(chips)} chip(s))")
    print(f"{'='*70}")
    print(f"  Page size  : 64KB (fixed)")
    print(f"  Total data : {human_bytes(int(df.total_data.iloc[0]))}")
    print(f"  FIFO sizes : {[human_bytes(f) for f in fifos]}")
    print(f"  Chips      : {list(chips)}")

    # Peak per chip
    print(f"\n  Peak throughput per chip:")
    print(f"  {'Chip':<36} {'FIFO':>8} {'GB/s':>8}")
    print(f"  {'-'*54}")
    for chip in chips:
        sub = df[df.chip_label == chip]
        best = sub.sort_values("throughput_gbps", ascending=False).iloc[0]
        print(
            f"  {chip:<36} {human_bytes(int(best.socket_fifo_size)):>8} "
            f"{best.throughput_gbps:>8.3f}"
        )

    # Throughput table: rows=chip, cols=fifo_size
    print(f"\n  Throughput (GB/s) by chip × FIFO size:")
    hdr = f"  {'Chip':<36}" + "".join(f"{human_bytes(f):>9}" for f in fifos)
    print(f"{hdr}\n  {'-'*(len(hdr)-2)}")
    for chip in chips:
        row = f"  {chip:<36}"
        sub = df[df.chip_label == chip]
        for fs in fifos:
            c = sub[sub.socket_fifo_size == fs]
            row += f"{c.throughput_gbps.values[0]:>9.3f}" if len(c) else f"{'--':>9}"
        print(row)

    # Per-page timing at largest FIFO
    fs_max = max(fifos)
    sub_max = df[df.socket_fifo_size == fs_max]
    print(f"\n  Per-page timing at FIFO={human_bytes(fs_max)}:")
    print(f"  {'Chip':<36} {'us/page':>10} {'cycles':>10} {'GB/s':>8}")
    print(f"  {'-'*66}")
    for chip in chips:
        r = sub_max[sub_max.chip_label == chip]
        if r.empty:
            continue
        r = r.iloc[0]
        print(
            f"  {chip:<36} {r.avg_per_page_us:>10.3f} "
            f"{r.avg_per_page_cycles:>10.0f} {r.throughput_gbps:>8.3f}"
        )


# ── plots ─────────────────────────────────────────────────────────────────────


def mc_plot_throughput_vs_fifo(df, out="mc_d2h_throughput_vs_fifo.png"):
    """Line chart: throughput vs FIFO size, one line per chip."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for chip in sorted(df.chip_label.unique()):
        g = df[df.chip_label == chip].sort_values("socket_fifo_size")
        ax.plot(g.socket_fifo_size, g.throughput_gbps, "o-", label=chip, ms=6, lw=2)

    fifos = sorted(df.socket_fifo_size.unique())
    ax.set(
        xscale="log",
        xlabel="Socket FIFO Size",
        ylabel="GB/s",
        title="D2H Max Throughput vs FIFO Size — All Chips (page=64KB, total=1GB)",
    )
    ax.set_xticks(fifos)
    ax.set_xticklabels([human_bytes(f) for f in fifos], rotation=30, fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="Chip")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out}")


def mc_plot_bar_comparison(df, out="mc_d2h_throughput_bar.png"):
    """Grouped bar chart: one group per FIFO size, bars = chips."""
    chips = sorted(df.chip_label.unique())
    fifos = sorted(df.socket_fifo_size.unique())

    x = np.arange(len(fifos))
    width = 0.8 / max(len(chips), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, chip in enumerate(chips):
        sub = df[df.chip_label == chip].set_index("socket_fifo_size").reindex(fifos)
        ax.bar(x + i * width, sub.throughput_gbps, width, label=chip)

    ax.set_xticks(x + width * (len(chips) - 1) / 2)
    ax.set_xticklabels([human_bytes(f) for f in fifos], rotation=30, fontsize=9)
    ax.set(xlabel="Socket FIFO Size", ylabel="GB/s", title="D2H Max Throughput by Chip (page=64KB, total=1GB)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, title="Chip")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def mc_plot_heatmap(df, out="mc_d2h_throughput_heatmap.png"):
    """Heatmap: rows = chip, cols = FIFO size, values = GB/s."""
    chips = sorted(df.chip_label.unique())
    fifos = sorted(df.socket_fifo_size.unique())

    data = np.full((len(chips), len(fifos)), np.nan)
    for i, chip in enumerate(chips):
        for j, fs in enumerate(fifos):
            c = df[(df.chip_label == chip) & (df.socket_fifo_size == fs)]
            if not c.empty:
                data[i, j] = c.throughput_gbps.values[0]

    fig, ax = plt.subplots(figsize=(max(8, len(fifos) * 1.4), max(4, len(chips) * 0.8 + 2)))
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(fifos)))
    ax.set_xticklabels([human_bytes(f) for f in fifos], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(chips)))
    ax.set_yticklabels(chips, fontsize=9)
    ax.set_xlabel("Socket FIFO Size", fontsize=10)
    ax.set_ylabel("Chip (Tray/ASIC)", fontsize=10)
    ax.set_title("D2H Throughput (GB/s) — All Chips × FIFO Size (page=64KB, total=1GB)", fontsize=11)

    for i in range(len(chips)):
        for j in range(len(fifos)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > vmax * 0.7 else "black",
                )

    fig.colorbar(im, ax=ax, label="GB/s", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


# ── CSV export ────────────────────────────────────────────────────────────────


def mc_export_csv(df, out="mc_d2h_throughput_summary.csv"):
    """Pivot: rows = chip, cols = FIFO size, values = throughput GB/s."""
    pv = df.pivot_table(index="chip_label", columns="socket_fifo_size", values="throughput_gbps")
    pv.columns = [f"FIFO={human_bytes(c)} GB/s" for c in pv.columns]
    pv.index.name = "chip (tray/asic/coord)"

    # Also append avg_per_page_us columns for reference
    us_pv = df.pivot_table(index="chip_label", columns="socket_fifo_size", values="avg_per_page_us")
    us_pv.columns = [f"FIFO={human_bytes(c)} us/page" for c in us_pv.columns]
    us_pv.index.name = "chip (tray/asic/coord)"

    combined = pv.join(us_pv)
    combined.to_csv(out, float_format="%.6f")
    print(f"  Saved {out}")


def run_multichip(path, prefix=""):
    df = load_multichip_csv(path)
    mc_print_report(df)
    mc_plot_throughput_vs_fifo(df, out=f"{prefix}mc_d2h_throughput_vs_fifo.png")
    mc_plot_bar_comparison(df, out=f"{prefix}mc_d2h_throughput_bar.png")
    mc_plot_heatmap(df, out=f"{prefix}mc_d2h_throughput_heatmap.png")
    mc_export_csv(df, out=f"{prefix}mc_d2h_throughput_summary.csv")


# -- CLI ----

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze H2D/D2H benchmark results")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--throughput", action="store_true", help="D2H throughput")
    mode.add_argument("--latency", action="store_true", help="D2H latency")
    mode.add_argument("--h2d-throughput", action="store_true", help="H2D throughput")
    mode.add_argument("--h2d-latency", action="store_true", help="H2D latency")
    mode.add_argument("--ping", action="store_true", help="D2H pure ping")
    mode.add_argument("--h2d-ping", action="store_true", help="H2D pure ping")
    mode.add_argument("--multichip", action="store_true", help="D2H multi-chip max throughput")
    p.add_argument("csv", help="Path to benchmark CSV / log (or directory for ping modes)")
    p.add_argument(
        "--out-prefix", default="", metavar="PREFIX",
        help="Prefix prepended to every output file name (e.g. 'tray1_asic1_')"
    )
    args = p.parse_args()

    pfx = args.out_prefix
    if args.throughput:
        run_throughput(args.csv, prefix=pfx)
    elif args.latency:
        run_latency(args.csv)
    elif args.h2d_throughput:
        run_h2d_throughput(args.csv, prefix=pfx)
    elif args.h2d_latency:
        run_h2d_latency(args.csv)
    elif args.h2d_ping:
        run_h2d_ping(args.csv)
    elif args.multichip:
        run_multichip(args.csv, prefix=pfx)
    else:
        run_ping(args.csv)
