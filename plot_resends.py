#!/usr/bin/env python3
"""Generate per-host bar charts (TXQ0 resends and Uncorrected_CW, total + max)
from Ethernet Link Metrics log files.

Usage
-----
    python3 plot_resends.py \\
        --cluster-name "Rev C Blackhole Galaxy" \\
        --out-prefix revC \\
        [--out-dir OUT_DIR] \\
        [--log-scale auto|on|off] \\
        [--min-links N] \\
        LABEL=path/to/log1.txt  LABEL=path/to/log2.txt  ...

Each positional argument is one log file, optionally prefixed with
``LABEL=`` to give it a display name in the legend. If no ``LABEL=`` is
given, the label is derived from the filename stem (e.g. ``log_revc_2.txt``
becomes ``log_revc_2``).

Host ordering is auto-discovered from the union of all supplied logs.
Hosts with fewer than ``--min-links`` rows across all logs are discarded
as split-line artifacts (default 50).

Y-axis log-scale choice:
    auto  — use symlog if any value across runs exceeds 1000 (default)
    on    — force symlog
    off   — force linear

Four PNGs are written to ``OUT_DIR`` (default: current directory):
    {prefix}_resends_total.png   {prefix}_resends_max.png
    {prefix}_ucw_total.png       {prefix}_ucw_max.png
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROW_PAT = re.compile(
    r"(?:\]<stdout>:)?(bh-glx\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+?)\s*"
    r"(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s*B\s+\d+\s*B"
)

PORT_SHORT = {
    "QSFP_DD": "QSFP",
    "TRACE": "TRACE",
    "LINKING_BOARD_1": "LB1",
    "LINKING_BOARD_2": "LB2",
    "LINKING_BOARD_3": "LB3",
}


# ---- Parsing -----------------------------------------------------------------


def parse_rows(path):
    """Yield dicts per matched metric row in *path*."""
    with open(path) as f:
        for line in f:
            m = ROW_PAT.search(line)
            if not m:
                continue
            port_type = m.group(6).strip()
            yield {
                "host": m.group(1),
                "tray": int(m.group(2)),
                "asic": int(m.group(3)),
                "ch": int(m.group(4)),
                "port_type": port_type,
                "port_short": PORT_SHORT.get(port_type, port_type),
                # Hex columns: we only need Uncorrected_CW (group 10).
                "cw_uncorr": int(m.group(10), 16),
                # Integer columns: TXQ0 is group 11.
                "txq0": int(m.group(11)),
            }


def discover_hosts(log_paths, min_links):
    """Return a sorted list of hostnames that appear with at least *min_links*
    rows across the union of the given logs. Filters out split-line artifacts
    (hostnames that only appear once or twice because a log line was broken
    mid-hostname by an MPI prefix injection)."""
    counts = defaultdict(int)
    for p in log_paths:
        if not os.path.exists(p):
            print(f"WARN: missing {p}", file=sys.stderr)
            continue
        for row in parse_rows(p):
            counts[row["host"]] += 1
    hosts = sorted(h for h, c in counts.items() if c >= min_links)
    if not hosts:
        raise SystemExit(
            f"No hosts found with >= {min_links} rows across the supplied logs. "
            f"Observed host counts: {dict(counts)}"
        )
    return hosts


def aggregate_per_host(path, valid_hosts):
    """For one log, aggregate per-host totals and maxima for TXQ0 and UCW."""
    host_total_tx = defaultdict(int)
    host_max_tx = defaultdict(int)
    host_max_tx_link = {}
    host_total_ucw = defaultdict(int)
    host_max_ucw = defaultdict(int)
    host_max_ucw_link = {}
    valid = set(valid_hosts)

    for row in parse_rows(path):
        host = row["host"]
        if host not in valid:
            continue
        descriptor = f"t{row['tray']}/a{row['asic']}/ch{row['ch']} {row['port_short']}"
        tx = row["txq0"]
        ucw = row["cw_uncorr"]

        host_total_tx[host] += tx
        if tx > host_max_tx[host]:
            host_max_tx[host] = tx
            host_max_tx_link[host] = descriptor

        host_total_ucw[host] += ucw
        if ucw > host_max_ucw[host]:
            host_max_ucw[host] = ucw
            host_max_ucw_link[host] = descriptor

    return {
        h: {
            "total_txq0": host_total_tx.get(h, 0),
            "max_txq0": host_max_tx.get(h, 0),
            "max_txq0_link": host_max_tx_link.get(h, ""),
            "total_ucw": host_total_ucw.get(h, 0),
            "max_ucw": host_max_ucw.get(h, 0),
            "max_ucw_link": host_max_ucw_link.get(h, ""),
        }
        for h in valid_hosts
    }


# ---- Plotting ----------------------------------------------------------------


def plot_grouped_bars(
    cluster_name,
    host_order,
    run_labels,
    per_run_values,  # list (one per run) of np.array aligned to host_order
    title,
    ylabel,
    out_path,
    log_scale=False,
    annotations=None,  # list (per run) of list (per host) of str labels on bars
):
    n_hosts = len(host_order)
    n_runs = len(run_labels)
    x = np.arange(n_hosts, dtype=float)
    bar_width = 0.8 / max(n_runs, 1)
    cmap = plt.get_cmap("tab10")

    fig_h = 7.5 if annotations else 5.5
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n_hosts + 2), fig_h))

    for i, (label, vals) in enumerate(zip(run_labels, per_run_values)):
        offsets = x - 0.4 + bar_width / 2 + i * bar_width
        bars = ax.bar(
            offsets,
            vals,
            width=bar_width,
            color=cmap(i % 10),
            label=label,
            edgecolor="black",
            linewidth=0.3,
        )
        if annotations is not None:
            for hi, bar in enumerate(bars):
                text = annotations[i][hi]
                if not text:
                    continue
                val = vals[hi]
                if val <= 0:
                    continue
                ax.annotate(
                    text,
                    (bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=cmap(i % 10),
                )

    ax.set_xticks(x)
    short = [h.replace("bh-glx-120-", "").replace("bh-glx-", "") for h in host_order]
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_xlabel("Host")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{cluster_name} — {title}")
    if log_scale:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)

    if annotations is not None:
        _, top = ax.get_ylim()
        if log_scale:
            ax.set_ylim(top=top * 6)
        else:
            ax.set_ylim(top=top * 1.35)

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---- Main --------------------------------------------------------------------


def parse_log_spec(spec):
    """Parse a CLI arg of the form LABEL=PATH or PATH."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip() or os.path.splitext(os.path.basename(path))[0], path
    label = os.path.splitext(os.path.basename(spec))[0]
    return label, spec


def auto_log_scale(series_lists):
    """Return True if any value across any series exceeds 1,000 — the
    heuristic cutoff above which dynamic range is wide enough to warrant
    a log y-axis."""
    for arrays in series_lists:
        for arr in arrays:
            if len(arr) and np.max(arr) > 1_000:
                return True
    return False


def build_and_plot(cluster_name, logs, out_prefix, out_dir, log_scale_mode, min_links):
    log_paths = [path for _, path in logs]
    host_order = discover_hosts(log_paths, min_links=min_links)
    print(f"Discovered {len(host_order)} hosts (>= {min_links} rows each):")
    for h in host_order:
        print(f"  {h}")

    parsed = []
    for label, path in logs:
        if not os.path.exists(path):
            print(f"WARN: missing {path}", file=sys.stderr)
            parsed.append(
                (
                    label,
                    {
                        h: {
                            "total_txq0": 0,
                            "max_txq0": 0,
                            "max_txq0_link": "",
                            "total_ucw": 0,
                            "max_ucw": 0,
                            "max_ucw_link": "",
                        }
                        for h in host_order
                    },
                )
            )
            continue
        parsed.append((label, aggregate_per_host(path, host_order)))

    run_labels = [p[0] for p in parsed]

    def by_host(field):
        return [np.array([parsed[i][1][h][field] for h in host_order]) for i in range(len(parsed))]

    totals_tx = by_host("total_txq0")
    maxes_tx = by_host("max_txq0")
    totals_ucw = by_host("total_ucw")
    maxes_ucw = by_host("max_ucw")

    max_tx_links = [[parsed[i][1][h]["max_txq0_link"] for h in host_order] for i in range(len(parsed))]
    max_ucw_links = [[parsed[i][1][h]["max_ucw_link"] for h in host_order] for i in range(len(parsed))]

    def print_table(title, series):
        print(f"\n=== {cluster_name} — {title} ===")
        print(f"  {'Host':22s} " + " ".join(f"{r:>12s}" for r in run_labels))
        for hi, h in enumerate(host_order):
            print(f"  {h:22s} " + " ".join(f"{series[ri][hi]:>12,d}" for ri in range(len(parsed))))

    print_table("per-host TXQ0 totals", totals_tx)
    print_table("per-host TXQ0 max", maxes_tx)
    print_table("per-host Uncorrected_CW totals", totals_ucw)
    print_table("per-host Uncorrected_CW max", maxes_ucw)

    # Decide log scale per chart type (TXQ0 and UCW may have different dynamic ranges)
    def resolve_log(values):
        if log_scale_mode == "on":
            return True
        if log_scale_mode == "off":
            return False
        return auto_log_scale(values)

    os.makedirs(out_dir, exist_ok=True)

    plot_grouped_bars(
        cluster_name,
        host_order,
        run_labels,
        totals_tx,
        title="Per-host total TXQ0 resends across runs",
        ylabel="Total TXQ0 resends (sum across links on host)",
        out_path=os.path.join(out_dir, f"{out_prefix}_resends_total.png"),
        log_scale=resolve_log([totals_tx]),
    )
    plot_grouped_bars(
        cluster_name,
        host_order,
        run_labels,
        maxes_tx,
        title="Per-host max TXQ0 resends on any single link across runs",
        ylabel="Max TXQ0 resends on any single link",
        out_path=os.path.join(out_dir, f"{out_prefix}_resends_max.png"),
        log_scale=resolve_log([maxes_tx]),
        annotations=max_tx_links,
    )
    plot_grouped_bars(
        cluster_name,
        host_order,
        run_labels,
        totals_ucw,
        title="Per-host total Uncorrected_CW across runs",
        ylabel="Total Uncorrected_CW (sum across links on host)",
        out_path=os.path.join(out_dir, f"{out_prefix}_ucw_total.png"),
        log_scale=resolve_log([totals_ucw]),
    )
    plot_grouped_bars(
        cluster_name,
        host_order,
        run_labels,
        maxes_ucw,
        title="Per-host max Uncorrected_CW on any single link across runs",
        ylabel="Max Uncorrected_CW on any single link",
        out_path=os.path.join(out_dir, f"{out_prefix}_ucw_max.png"),
        log_scale=resolve_log([maxes_ucw]),
        annotations=max_ucw_links,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "logs",
        nargs="+",
        metavar="[LABEL=]PATH",
        help="One or more log files. Optional LABEL= prefix sets the legend name; "
        "without it, the filename stem is used.",
    )
    ap.add_argument("--cluster-name", default="Cluster", help="Title prefix for the charts (default: 'Cluster')")
    ap.add_argument(
        "--out-prefix", required=True, help="Filename prefix for output PNGs (e.g. 'revC' → revC_resends_total.png ...)"
    )
    ap.add_argument("--out-dir", default=".", help="Directory to write PNGs into (default: current directory)")
    ap.add_argument(
        "--log-scale",
        choices=["auto", "on", "off"],
        default="auto",
        help="Y-axis log-scale policy (default: auto — symlog if any value > 1000)",
    )
    ap.add_argument(
        "--min-links",
        type=int,
        default=50,
        help="Discard hostnames with fewer than this many metric rows across all logs, "
        "to filter out split-line artifacts (default: 50)",
    )

    args = ap.parse_args(argv)
    logs = [parse_log_spec(s) for s in args.logs]

    build_and_plot(
        cluster_name=args.cluster_name,
        logs=logs,
        out_prefix=args.out_prefix,
        out_dir=args.out_dir,
        log_scale_mode=args.log_scale,
        min_links=args.min_links,
    )


if __name__ == "__main__":
    main()
