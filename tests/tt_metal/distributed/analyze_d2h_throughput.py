#!/usr/bin/env python3
"""Analyze D2H socket steady-state throughput benchmark results.

Usage: python3 analyze_d2h_throughput.py [results.csv]

Outputs:
  Console  — summary stats, per-page breakdown, FIFO impact
  PNGs     — d2h_throughput.png, d2h_tp_vs_fifo.png, d2h_tp_heatmap_grid.png
  CSV      — d2h_throughput_summary.csv (reference spreadsheet format)
"""

import sys, numpy as np, pandas as pd
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


def load_csv(path):
    """Load benchmark CSV. Auto-detects column format; handles commas in MeshCoordinate."""
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


def median_over_fifo(df):
    return (
        df.groupby(["page_size", "total_data"])
        .agg(
            throughput_gbps=("throughput_gbps", "median"),
            avg_per_page_us=("avg_per_page_us", "median"),
        )
        .reset_index()
    )


# ── console output ───────────────────────────────────────────────────────────


def print_report(df):
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

    # throughput table
    tds = sorted(agg.total_data.unique())
    hdr = f"  {'page':>6}" + "".join(f"{human_bytes(t):>8}" for t in tds)
    print(f"\n  Throughput (GB/s) -- median across {nf} FIFO sizes:\n{hdr}\n  {'-'*(len(hdr)-2)}")
    for ps in sorted(agg.page_size.unique()):
        row = f"  {human_bytes(ps):>6}"
        for td in tds:
            c = agg[(agg.page_size == ps) & (agg.total_data == td)]
            row += f"{c.throughput_gbps.values[0]:>8.3f}" if len(c) else f"{'--':>8}"
        print(row)

    # FIFO impact
    td = df.total_data.max()
    sub = df[df.total_data == td]
    print(f"\n  FIFO impact at total_data={human_bytes(td)}:")
    print(f"  {'page':>6} {'min':>8} {'median':>8} {'max':>8} {'CV%':>6}")
    print(f"  {'-'*38}")
    for ps in sorted(sub.page_size.unique()):
        g = sub[sub.page_size == ps].throughput_gbps
        cv = g.std() / g.mean() * 100 if g.mean() > 0 else 0
        print(f"  {human_bytes(ps):>6} {g.min():>8.3f} {g.median():>8.3f} {g.max():>8.3f} {cv:>5.1f}%")

    # per-page timing
    agg2 = median_over_fifo(df[df.total_data == td])
    print(f"\n  Per-page timing (total_data={human_bytes(td)}):")
    print(f"  {'page':>6} {'us/page':>10} {'cycles':>10} {'GB/s':>8}\n  {'-'*38}")
    for _, r in agg2.sort_values("page_size").iterrows():
        print(
            f"  {human_bytes(r.page_size):>6} {r.avg_per_page_us:>10.3f} "
            f"{r.avg_per_page_us*1350:>10.0f} {r.throughput_gbps:>8.3f}"
        )


# ── plots ────────────────────────────────────────────────────────────────────


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


def plot_throughput(df, out="d2h_throughput.png"):
    """BW vs page_size, one line per total_data."""
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


def plot_throughput_vs_fifo(df, out="d2h_tp_vs_fifo.png"):
    """BW vs FIFO size, one line per page_size, at largest total_data."""
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


def plot_throughput_heatmap_grid(df, out="d2h_tp_heatmap_grid.png"):
    """Faceted heatmaps: one per total_data, shared color scale."""
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


def export_csv(df, out="d2h_throughput_summary.csv"):
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


if __name__ == "__main__":
    df = load_csv(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
    print_report(df)
    plot_throughput(df)
    plot_throughput_vs_fifo(df)
    plot_throughput_heatmap_grid(df)
    export_csv(df)
