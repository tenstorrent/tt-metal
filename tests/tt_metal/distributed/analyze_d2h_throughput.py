#!/usr/bin/env python3
"""Analyze D2H socket benchmark results (throughput or latency).

Usage:
  python3 analyze_d2h_throughput.py --throughput results.csv
  python3 analyze_d2h_throughput.py --latency   latency_results.csv
"""

import argparse, sys, numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def human_bytes(n):
    for u in ("", "K", "M", "G"):
        if n < 1024:
            return f"{int(n)}{u}" if n == int(n) else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}T"


# ── data loading ─────────────────────────────────────────────────────────────


def load_throughput_csv(path):
    """Load throughput benchmark CSV. Handles commas in MeshCoordinate."""
    with open(path) as f:
        header = f.readline().strip()
        rows = [l.strip().split(",") for l in f if l.strip()]

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

    df = pd.DataFrame([r[:nc] for r in rows], columns=cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c])

    if "data_size" not in df:
        df["data_size"] = (df.total_data / df.num_iterations).astype(int)
    if "throughput_gbps" not in df:
        df["throughput_gbps"] = df.page_size / (df.avg_per_page_us * 1e3)
    return df


def load_latency_csv(path):
    """Load latency benchmark CSV (pcie_socket_data_ping output)."""
    with open(path) as f:
        header = f.readline().strip()
        rows = [l.strip().split(",") for l in f if l.strip()]

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
    nc = len(cols)
    df = pd.DataFrame([r[:nc] for r in rows], columns=cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c])
    return df


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
            f"{r.avg_per_page_us*1350:>10.0f} {r.throughput_gbps:>8.3f}"
        )


def tp_plot_throughput(df, out="d2h_throughput.png"):
    agg = median_over_fifo(df)
    fig, ax = plt.subplots(figsize=(12, 6))
    for td in sorted(agg.total_data.unique()):
        g = agg[agg.total_data == td].sort_values("page_size")
        ax.plot(g.page_size, g.throughput_gbps, "o-", label=f"D0 Read {human_bytes(td)}", ms=5, lw=2)
    ax.set(xscale="log", xlabel="Page Size", ylabel="GB/s", title="D2H BW")
    ax.set_xticks(t := sorted(agg.page_size.unique()))
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
        title=f"D2H BW vs FIFO Size (total_data={human_bytes(td)})",
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

    fig.suptitle("D2H Throughput (GB/s) -- Page Size x FIFO Size x Total Data", fontsize=13, fontweight="bold", y=0.98)
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


def run_throughput(path):
    df = load_throughput_csv(path)
    tp_print_report(df)
    tp_plot_throughput(df)
    tp_plot_vs_fifo(df)
    tp_plot_heatmap_grid(df)
    tp_export_csv(df)


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
    axes[1].set(xlabel="Page Size", ylabel="Max Latency (us)", title="D2H Max Latency (outliers)")
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


# ══════════════════════════════════════════════════════════════════════════════
#  PING  (pure signaling round-trip, no data DMA)
# ══════════════════════════════════════════════════════════════════════════════


def ping_print_report(df):
    print(f"\n{'='*70}\n  D2H Pure Ping (notify+barrier only)  ({len(df)} rows)\n{'='*70}")
    print(f"  Page sizes : {[human_bytes(x) for x in sorted(df.page_size.unique())]}")
    print(f"  FIFO sizes : {[human_bytes(x) for x in sorted(df.socket_fifo_size.unique())]}")
    print(f"  Iterations : {int(df.num_iterations.iloc[0])}")

    print(
        f"\n  Overall: p50 median={df.p50_us.median():.2f} us, "
        f"min={df.min_us.min():.2f} us, max-of-max={df.max_us.max():.2f} us"
    )

    fifos = sorted(df.socket_fifo_size.unique())
    hdr = f"  {'page':>6}" + "".join(f"{human_bytes(f):>10}" for f in fifos)
    print(f"\n  p50 (us):\n{hdr}\n  {'-'*(len(hdr)-2)}")
    for ps in sorted(df.page_size.unique()):
        row = f"  {human_bytes(ps):>6}"
        for fs in fifos:
            c = df[(df.page_size == ps) & (df.socket_fifo_size == fs)]
            row += f"{c.p50_us.values[0]:>10.2f}" if len(c) else f"{'--':>10}"
        print(row)


def ping_plot(df, out="d2h_ping.png"):
    """Ping latency vs page_size — should be flat. One line per FIFO."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for fs in sorted(df.socket_fifo_size.unique()):
        g = df[df.socket_fifo_size == fs].sort_values("page_size")
        color = ax.plot(g.page_size, g.p50_us, "o-", label=f"FIFO={human_bytes(fs)}", ms=5, lw=2)[0].get_color()
        ax.plot(g.page_size, g.min_us, "--", color=color, lw=0.8, alpha=0.5)
        ax.plot(g.page_size, g.max_us, "--", color=color, lw=0.8, alpha=0.5)
    ax.set(
        xscale="log",
        xlabel="Page Size (bytes)",
        ylabel="Latency (us)",
        title="D2H Pure Ping: notify→barrier round-trip (p50 solid, min/max dashed)",
    )
    ax.set_xticks(t := sorted(df.page_size.unique()))
    ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=9)
    ax.minorticks_off()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    med = df.p50_us.median()
    ax.axhline(med, color="gray", ls=":", lw=1, alpha=0.6)
    ax.annotate(f"median {med:.2f} us", xy=(df.page_size.min(), med), fontsize=8, color="gray", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out}")


def ping_plot_comparison(ping_df, latency_csv=None, out="d2h_ping_vs_latency.png"):
    """If latency CSV provided, overlay ping vs data_ping at largest common FIFO."""
    if latency_csv is None:
        return
    try:
        lat_df = load_latency_csv(latency_csv)
    except Exception:
        return

    common_fifos = set(ping_df.socket_fifo_size) & set(lat_df.socket_fifo_size)
    if not common_fifos:
        return
    fs = max(f for f in common_fifos if f <= 65536) if any(f <= 65536 for f in common_fifos) else max(common_fifos)

    p = ping_df[ping_df.socket_fifo_size == fs].sort_values("page_size")
    d = lat_df[lat_df.socket_fifo_size == fs].sort_values("page_size")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d.page_size, d.p50_us, "s-", label="Data ping (DMA + protocol)", ms=6, lw=2, color="C0")
    ax.plot(p.page_size, p.p50_us, "o-", label="Pure ping (protocol only)", ms=6, lw=2, color="C1")

    # Shade the DMA cost
    merged = (
        p.set_index("page_size")[["p50_us"]]
        .join(d.set_index("page_size")[["p50_us"]], lsuffix="_ping", rsuffix="_data")
        .dropna()
    )
    ax.fill_between(merged.index, merged.p50_us_ping, merged.p50_us_data, alpha=0.15, color="C0", label="DMA cost")

    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Page Size (bytes)",
        ylabel="Latency (us)",
        title=f"D2H Latency Breakdown: Protocol vs DMA (FIFO={human_bytes(fs)})",
    )
    ax.set_xticks(t := sorted(d.page_size.unique()))
    ax.set_xticklabels([human_bytes(x) for x in t], rotation=45, fontsize=9)
    ax.minorticks_off()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def ping_export_csv(df, out="d2h_ping_summary.csv"):
    """For ping, export the per-iteration timeseries instead of aggregated stats."""
    try:
        iters = pd.read_csv("ping_iterations.csv")
        iters.to_csv(out, index=False, float_format="%.4f")
        print(f"  Saved {out} ({len(iters)} iterations)")
    except FileNotFoundError:
        print(f"  Warning: ping_iterations.csv not found, skipping summary CSV")


def ping_plot_timeseries(path="ping_iterations.csv", out="d2h_ping_timeseries.png"):
    """Plot iteration-by-iteration latency."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.iteration, df.latency_us, "o-", ms=3, lw=1, alpha=0.7)
    ax.axhline(df.latency_us.median(), color="red", ls="--", lw=1.5, label=f"p50 = {df.latency_us.median():.2f} us")
    ax.axhline(df.latency_us.mean(), color="orange", ls=":", lw=1.5, label=f"avg = {df.latency_us.mean():.2f} us")
    ax.set(
        xlabel="Iteration",
        ylabel="Latency (us)",
        title=f"D2H Pure Ping: Per-Iteration Latency (page={df.iteration.count()} iters)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def run_ping(path, latency_csv=None):
    df = load_latency_csv(path)  # same CSV format
    ping_print_report(df)
    ping_plot(df)
    ping_plot_comparison(df, latency_csv)
    ping_plot_timeseries()
    ping_export_csv(df)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze D2H benchmark results")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--throughput", action="store_true", help="Analyze throughput benchmark")
    mode.add_argument("--latency", action="store_true", help="Analyze latency benchmark")
    mode.add_argument("--ping", action="store_true", help="Analyze pure ping benchmark")
    p.add_argument("csv", help="Path to benchmark CSV")
    p.add_argument("--latency-csv", help="(ping mode) Overlay data_ping latency for comparison")
    args = p.parse_args()

    if args.throughput:
        run_throughput(args.csv)
    elif args.latency:
        run_latency(args.csv)
    else:
        run_ping(args.csv, args.latency_csv)
