#!/usr/bin/env python3
"""Parse tt-blaze multi-user benchmark perf output and generate charts + tables
across runs.

The benchmark output is the text block printed by
``dummy_pipeline_connector --sweep ...`` — one ``## Concurrency: N users``
block per concurrency level, with multiple metric sub-tables (Avg TTFT (ms),
P99 TTFT (ms), Avg TPS (per user), Aggregate Output tok/s, etc.), each
indexed by an ISL × OSL grid.

Usage
-----
    python3 plot_benchmark_perf.py \\
        --out-prefix myrun \\
        --out-dir ./out \\
        [LABEL=]path/to/run1.log  [LABEL=]path/to/run2.log ...

Each positional argument is one run's benchmark output (either the raw
stdout of ``dummy_pipeline_connector``, or any file that *contains* that
output — the parser scans for the well-known section headers). Prefix
with ``LABEL=`` to set the legend / table label; otherwise the filename
stem is used.

Outputs
-------
Under ``OUT_DIR`` (created if missing), using ``OUT_PREFIX`` as a filename
prefix:

    {prefix}_avg_tps_1u.png, {prefix}_avg_tps_4u.png, ...
    {prefix}_avg_ttft_1u.png, {prefix}_p99_ttft_1u.png, ...
    {prefix}_agg_tokps_1u.png, ...
    {prefix}_summary.csv     — one row per (run, concurrency) with summary columns
    {prefix}_detail.csv      — one row per (run, concurrency, ISL, OSL, metric) for pivoting
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Metric names we care about — canonicalised to the exact strings printed
# between the separator lines in the benchmark output.
PARSED_METRICS = [
    "Avg TTFT (ms)",
    "P50 TTFT (ms)",
    "P99 TTFT (ms)",
    "Avg TPS (per user)",
    "P50 TPS (per user)",
    "Min TPS",
    "Aggregate Output tok/s",
    "Elapsed (s)",
    "Errors",
]

# Metrics we generate a bar chart for (subset of PARSED_METRICS)
CHART_METRICS = [
    ("Avg TPS (per user)", "avg_tps", "Avg TPS (per user)"),
    ("P99 TTFT (ms)", "p99_ttft", "P99 TTFT (ms)"),
    ("Avg TTFT (ms)", "avg_ttft", "Avg TTFT (ms)"),
    ("Aggregate Output tok/s", "agg_tokps", "Aggregate Output tok/s"),
]

CONCURRENCY_HEADER_RE = re.compile(r"^##\s*Concurrency:\s*(\d+)\s*users", re.M)
SEPARATOR_RE = re.compile(r"^=+\s*$")
DASH_RE = re.compile(r"^-+\s*$")
OSL_HEADER_RE = re.compile(r"^\s*ISL\s*\\\s*OSL\s+(.+)$")


# ---- Parser ------------------------------------------------------------------


def parse_log(path):
    """Return {concurrency: {metric_name: {(isl, osl): value}}} for one log file."""
    with open(path) as f:
        lines = f.read().splitlines()

    out = defaultdict(lambda: defaultdict(dict))
    current_conc = None
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Concurrency block header
        m = CONCURRENCY_HEADER_RE.match(line)
        if m:
            current_conc = int(m.group(1))
            i += 1
            continue

        # Metric block header:  ===   /   METRIC_NAME   /   ===
        if SEPARATOR_RE.match(line) and i + 2 < n and SEPARATOR_RE.match(lines[i + 2]):
            metric_name = lines[i + 1].strip()
            i += 3  # past the bottom separator

            # Expect an "ISL \ OSL" header line next
            if i >= n:
                continue
            m2 = OSL_HEADER_RE.match(lines[i])
            if not m2:
                # Not a table we know how to parse; skip to the next separator
                continue
            osl_list = [int(x) for x in re.findall(r"\d+", m2.group(1))]
            i += 1

            # Skip the dash divider if present
            if i < n and DASH_RE.match(lines[i]):
                i += 1

            # Data rows until blank line / next separator / next concurrency header
            while i < n:
                row = lines[i]
                stripped = row.strip()
                if not stripped or SEPARATOR_RE.match(row) or CONCURRENCY_HEADER_RE.match(row):
                    break
                parts = stripped.split()
                if parts and parts[0].lstrip("-").isdigit():
                    try:
                        isl = int(parts[0])
                        vals = [float(v) for v in parts[1:]]
                    except ValueError:
                        i += 1
                        continue
                    for osl, v in zip(osl_list, vals):
                        if current_conc is not None and metric_name in PARSED_METRICS:
                            out[current_conc][metric_name][(isl, osl)] = v
                i += 1
            continue

        i += 1

    # Convert defaultdicts to regular dicts for cleaner downstream use
    return {c: dict(m) for c, m in out.items()}


# ---- Plotting ----------------------------------------------------------------


def plot_metric_across_runs(
    run_labels,
    per_run_values,  # list of dicts {osl: value} aligned to run_labels
    osls,
    metric_label,
    concurrency,
    out_path,
):
    """One grouped bar chart: OSL on X, one bar per run within each OSL group."""
    n_runs = len(run_labels)
    n_osls = len(osls)
    x = np.arange(n_osls, dtype=float)
    bar_width = 0.8 / max(n_runs, 1)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n_osls + 2), 5.5))
    for i, (label, vals) in enumerate(zip(run_labels, per_run_values)):
        arr = np.array([vals.get(o, np.nan) for o in osls], dtype=float)
        offsets = x - 0.4 + bar_width / 2 + i * bar_width
        ax.bar(
            offsets,
            arr,
            width=bar_width,
            color=cmap(i % 10),
            label=label,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in osls])
    ax.set_xlabel("OSL")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} — {concurrency} users — across runs")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---- Output: tables ----------------------------------------------------------


def write_summary_csv(path, run_labels, parsed_per_run):
    """One row per (run, concurrency) with common summary columns."""
    fields = [
        "run",
        "concurrency",
        "avg_tps_mean",
        "avg_tps_min",
        "avg_tps_max",
        "agg_tokps_peak",
        "agg_tokps_mean",
        "avg_ttft_mean_ms",
        "p50_ttft_ms",
        "p99_ttft_max_ms",
        "errors_total",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, data in zip(run_labels, parsed_per_run):
            for conc in sorted(data.keys()):
                metrics = data[conc]

                def vals(name):
                    return list(metrics.get(name, {}).values())

                avg_tps = vals("Avg TPS (per user)")
                agg = vals("Aggregate Output tok/s")
                avg_ttft = vals("Avg TTFT (ms)")
                p50_ttft = vals("P50 TTFT (ms)")
                p99_ttft = vals("P99 TTFT (ms)")
                errors = vals("Errors")

                row = {
                    "run": label,
                    "concurrency": conc,
                    "avg_tps_mean": round(sum(avg_tps) / len(avg_tps), 3) if avg_tps else "",
                    "avg_tps_min": round(min(avg_tps), 3) if avg_tps else "",
                    "avg_tps_max": round(max(avg_tps), 3) if avg_tps else "",
                    "agg_tokps_peak": round(max(agg), 2) if agg else "",
                    "agg_tokps_mean": round(sum(agg) / len(agg), 2) if agg else "",
                    "avg_ttft_mean_ms": round(sum(avg_ttft) / len(avg_ttft), 3) if avg_ttft else "",
                    "p50_ttft_ms": round(sum(p50_ttft) / len(p50_ttft), 3) if p50_ttft else "",
                    "p99_ttft_max_ms": round(max(p99_ttft), 3) if p99_ttft else "",
                    "errors_total": int(sum(errors)) if errors else "",
                }
                w.writerow(row)
    print(f"wrote {path}")


def write_detail_csv(path, run_labels, parsed_per_run):
    """One row per (run, concurrency, ISL, OSL, metric, value) — tall format,
    friendly for pivoting in Excel / pandas."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "concurrency", "isl", "osl", "metric", "value"])
        for label, data in zip(run_labels, parsed_per_run):
            for conc in sorted(data.keys()):
                for metric in PARSED_METRICS:
                    cells = data[conc].get(metric, {})
                    for (isl, osl), v in sorted(cells.items()):
                        w.writerow([label, conc, isl, osl, metric, v])
    print(f"wrote {path}")


def print_summary_table(run_labels, parsed_per_run):
    """Also print the summary inline to stdout as a Markdown-friendly table."""
    header = "| Run | Conc | Mean per-user TPS | Peak agg tok/s | Mean Avg TTFT (ms) " "| P99 TTFT max (ms) | Errors |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    print("\n" + header)
    print(sep)
    for label, data in zip(run_labels, parsed_per_run):
        for conc in sorted(data.keys()):
            metrics = data[conc]

            def cells(name):
                return list(metrics.get(name, {}).values())

            avg_tps = cells("Avg TPS (per user)")
            agg = cells("Aggregate Output tok/s")
            avg_ttft = cells("Avg TTFT (ms)")
            p99 = cells("P99 TTFT (ms)")
            err = cells("Errors")
            tps_mean = f"{sum(avg_tps)/len(avg_tps):.2f}" if avg_tps else "—"
            agg_peak = f"{max(agg):.1f}" if agg else "—"
            ttft_mean = f"{sum(avg_ttft)/len(avg_ttft):.2f}" if avg_ttft else "—"
            p99_max = f"{max(p99):.2f}" if p99 else "—"
            err_sum = int(sum(err)) if err else 0
            print(f"| {label} | {conc} | {tps_mean} | {agg_peak} | {ttft_mean} | {p99_max} | {err_sum} |")
    print()


# ---- Main --------------------------------------------------------------------


def parse_log_spec(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
        return (label.strip() or os.path.splitext(os.path.basename(path))[0], path)
    return (os.path.splitext(os.path.basename(spec))[0], spec)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("logs", nargs="+", metavar="[LABEL=]PATH", help="One or more log files containing benchmark output")
    ap.add_argument("--out-prefix", required=True, help="Filename prefix for output files")
    ap.add_argument("--out-dir", default=".", help="Output directory (default: .)")
    args = ap.parse_args(argv)

    logs = [parse_log_spec(s) for s in args.logs]
    os.makedirs(args.out_dir, exist_ok=True)

    run_labels = []
    parsed_per_run = []
    for label, path in logs:
        if not os.path.exists(path):
            print(f"WARN: missing {path}", file=sys.stderr)
            continue
        data = parse_log(path)
        if not data:
            print(f"WARN: no benchmark tables found in {path}", file=sys.stderr)
            continue
        run_labels.append(label)
        parsed_per_run.append(data)
        conc_list = sorted(data.keys())
        n_metrics = sum(len(m) for m in data.values())
        print(f"parsed {label:20s}  concurrencies={conc_list}  metric-entries={n_metrics}")

    if not run_labels:
        raise SystemExit("No parseable logs supplied.")

    # Union of concurrencies across all runs
    all_concs = sorted({c for d in parsed_per_run for c in d.keys()})

    # Union of OSLs per (concurrency, metric) — usually identical across runs
    for conc in all_concs:
        for metric_name, file_tag, y_label in CHART_METRICS:
            osls = sorted({o for data in parsed_per_run for (_, o) in data.get(conc, {}).get(metric_name, {}).keys()})
            if not osls:
                continue

            # Build per-run {osl: value} by averaging across ISLs if multiple ISLs exist
            per_run_values = []
            for data in parsed_per_run:
                cells = data.get(conc, {}).get(metric_name, {})
                vals_by_osl = defaultdict(list)
                for (isl, osl), v in cells.items():
                    vals_by_osl[osl].append(v)
                per_run_values.append({o: sum(vals_by_osl[o]) / len(vals_by_osl[o]) for o in vals_by_osl})

            out_path = os.path.join(args.out_dir, f"{args.out_prefix}_{file_tag}_{conc}u.png")
            plot_metric_across_runs(
                run_labels,
                per_run_values,
                osls,
                y_label,
                conc,
                out_path,
            )

    # Summary table (stdout + CSV) + detail CSV
    print_summary_table(run_labels, parsed_per_run)
    write_summary_csv(
        os.path.join(args.out_dir, f"{args.out_prefix}_summary.csv"),
        run_labels,
        parsed_per_run,
    )
    write_detail_csv(
        os.path.join(args.out_dir, f"{args.out_prefix}_detail.csv"),
        run_labels,
        parsed_per_run,
    )


if __name__ == "__main__":
    main()
