# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plot fixed 4x1K throughput against canonical 1x4K using recorded model runs."""

import argparse
import csv
import html
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def write_csv(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Directory containing canonical.json and chunked4.json"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--official-log", type=Path)
    parser.add_argument("--source-commit", default="unspecified")
    parser.add_argument("--layers", type=Path, help="Layer measurements from chunked_batch_layer_report.py")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {mode: json.loads((args.input / f"{mode}.json").read_text()) for mode in ("canonical", "chunked4")}
    for mode, run in runs.items():
        assert run["layers"] == 60 and run["context"] == 262144 and run["history"] == "real model calls"
        assert run["batch_size"] == (1 if mode == "canonical" else 4)
        assert run["chunk_size"] == (4096 if mode == "canonical" else 1024)
        (args.output / f"{mode}.json").write_text(json.dumps(run, indent=2) + "\n")
    full = {
        mode: {row["starts"][0]: row for row in run["records"] if row["label"] == "full"} for mode, run in runs.items()
    }
    rows = []
    for prefix, canonical in full["canonical"].items():
        batch = full["chunked4"][prefix]
        assert canonical["useful_tokens"] == 4096 and batch["useful_tokens"] == 4096
        rows.append(
            dict(
                prefix_tokens=prefix,
                canonical_tokens=4096,
                batch_tokens=4096,
                canonical_ms=canonical["median_ms"],
                batch_ms=batch["median_ms"],
                batch_latency_overhead_percent=100 * (batch["median_ms"] / canonical["median_ms"] - 1),
                canonical_tokens_per_second=canonical["useful_tokens_per_second"],
                batch_tokens_per_second=batch["useful_tokens_per_second"],
                batch_throughput_percent_of_canonical=100
                * batch["useful_tokens_per_second"]
                / canonical["useful_tokens_per_second"],
                canonical_repeats=len(canonical["samples_ms"]),
                batch_repeats=len(batch["samples_ms"]),
                canonical_staging_ms=canonical["staging_ms"],
                batch_staging_ms=batch["staging_ms"],
            )
        )
    write_csv(args.output / "comparison.csv", rows)
    official = []
    if args.official_log:
        for chunk, count, start, end, ms in re.findall(
            r"\[traced_perf\] chunk (\d+)/(\d+) \[(\d+), (\d+)\) device=([\d.]+)ms", args.official_log.read_text()
        ):
            official.append(
                dict(chunk=int(chunk), total_chunks=int(count), start=int(start), end=int(end), device_ms=float(ms))
            )
        assert len(official) == 64
        write_csv(args.output / "official_canonical.csv", official)
    validation = json.loads(args.validation.read_text()) if args.validation else {}
    layer_data = json.loads(args.layers.read_text()) if args.layers else None
    method = (
        "Gemma-4-31B-it, 60 layers, Blackhole CP8/TP4, original reduce-scatter, default activation placement. "
        "Each call processes 4,096 useful tokens: one 4K request in canonical, or four 1K requests in the fixed batch. "
        "Every request has the prefix shown on the chart. All histories were populated by real model calls. "
        "Five replays at prefixes 0, 8K, 32K, 64K, 128K, 192K and 248K; one replay at other positions. "
        "Timing covers trace replay and synchronization. Host staging is recorded separately; loading, compilation, "
        "output download and KV migration are excluded."
    )
    (args.output / "study.json").write_text(
        json.dumps(
            dict(
                source_commit=args.source_commit,
                method=method,
                validation=validation,
                memory_note="Eight 256K slots plus weights failed DRAM allocation. Four slots halve KV storage; the measured fixed batch uses four 256K slots.",
                kv_storage_bytes_per_device={"four_slots": 15151923200, "eight_slots": 30303846400},
                official_canonical=official,
                layer_profile_method=layer_data["method"] if layer_data else None,
                tt_perf_report_commit=layer_data["tt_perf_report_commit"] if layer_data else None,
            ),
            indent=2,
        )
        + "\n"
    )
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    pages = []

    def finish(fig, name, pdf):
        fig.savefig(args.output / f"{name}.png", dpi=140)
        path = args.output / f"{name}.svg"
        fig.savefig(path)
        svg = "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        path.write_text(svg)
        pages.append(
            f'<section id="{name}"><img src="{name}.svg" alt="{name.replace("_", " ")}" loading="lazy"></section>'
        )
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(args.output / "comparison.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 8.5))
        fig.subplots_adjust(left=0.09, right=0.96, top=0.77, bottom=0.38, wspace=0.30)
        fig.suptitle("Fixed 4×1K batching vs canonical 4K", x=0.06, y=0.96, ha="left", fontsize=23, fontweight="bold")
        fig.text(
            0.06, 0.905, "Same token count per call: 4,096 useful tokens. One request × 4K versus four requests × 1K."
        )
        for mode, label, color in (
            ("canonical", "Canonical: 1 request × 4K", "#3176b5"),
            ("chunked4", "Fixed batch: 4 requests × 1K", "#d7782c"),
        ):
            measurements = [full[mode][row["prefix_tokens"]] for row in rows]
            x = [row["prefix_tokens"] / 1024 for row in rows]
            axes[0].plot(
                x, [r["useful_tokens_per_second"] / 1000 for r in measurements], label=label, color=color, marker="."
            )
            axes[1].plot(x, [r["median_ms"] for r in measurements], label=label, color=color, marker=".")
        axes[0].set_title("Useful throughput — higher is better")
        axes[0].set_ylabel("Thousand useful tokens / second")
        axes[1].set_title("Time per 4K tokens — lower is better")
        axes[1].set_ylabel("Milliseconds per call")
        for axis in axes:
            axis.set_xlabel("Existing prefix per request (K tokens; K = 1024)")
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.2)
        fig.legend(
            *axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.055, 0.87), ncol=2, frameon=False
        )
        selected_rows = [row for row in rows if row["prefix_tokens"] in (0, 131072, 253952)]
        findings = []
        for row in selected_rows:
            findings.append(
                f"{row['prefix_tokens'] // 1024}K prefix: canonical {row['canonical_ms']:.1f} ms; "
                f"fixed batch {row['batch_ms']:.1f} ms. The batch takes "
                f"{row['batch_latency_overhead_percent']:.0f}% longer for the same token count."
            )
        y = 0.30
        for paragraph in (*findings, method):
            wrapped = textwrap.fill(paragraph, 140)
            fig.text(0.06, y, wrapped, va="top", fontsize=10, linespacing=1.3)
            y -= 0.023 * (wrapped.count("\n") + 1) + 0.014
        finish(fig, "throughput", pdf)

        padding = [full["chunked4"][0], *(r for r in runs["chunked4"]["records"] if r["label"] != "full")]
        fig, axes = plt.subplots(1, 2, figsize=(12, 8.5))
        fig.subplots_adjust(left=0.09, right=0.96, top=0.76, bottom=0.40, wspace=0.28)
        fig.suptitle("What does fixed-size padding cost?", x=0.06, y=0.96, ha="left", fontsize=23, fontweight="bold")
        fig.text(
            0.06, 0.905, "All cases execute the same 4×1K shape. Short final requests still occupy their full 1K lane."
        )
        labels = ["4 × 1024\n4096 useful", "4 × 512\n2048 useful", "1024 / 512 / 128 / 32\n1696 useful"]
        for axis, key, scale, ylabel in (
            (axes[0], "median_ms", 1, "Batch latency (ms)"),
            (axes[1], "useful_tokens_per_second", 0.001, "Thousand useful tokens / second"),
        ):
            values = [row[key] * scale for row in padding]
            axis.bar(range(3), values, color=("#3176b5", "#d7782c", "#829c9d"), width=0.65)
            axis.set_xticks(range(3), labels, fontsize=9)
            axis.set_ylabel(ylabel)
            axis.set_ylim(0, max(values) * 1.2)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            for i, value in enumerate(values):
                axis.annotate(f"{value:.1f}", (i, value), xytext=(0, 5), textcoords="offset points", ha="center")
        paragraphs = [
            "Why measure this? A fixed batch always "
            "computes 1K rows per request when fewer tokens are useful. All three cases start at zero and reuse the same trace.",
            "The model shares embedding, projections, norms, RoPE and MLP across four requests. Each attention layer "
            "runs four sequential 1K attention calls with local slices and one local concatenation. It adds no CP all-gathers.",
            "Memory: four 256K request caches fit. KV storage falls from 28.2 to 14.1 GiB/device when reducing eight slots to four. "
            "Weights and other buffers are additional. The canonical reference allocates one slot; both use full-history caches.",
            "Validation: " + str(validation.get("summary", "See study.json for validation results.")),
        ]
        y = 0.31
        for paragraph in paragraphs:
            wrapped = textwrap.fill(paragraph, 140)
            fig.text(0.06, y, wrapped, va="top", fontsize=10, linespacing=1.3)
            y -= 0.023 * (wrapped.count("\n") + 1) + 0.014
        finish(fig, "padding", pdf)
        if layer_data:
            from models.demos.gemma4_d_p.scripts.chunked_batch_layer_plots import append_layer_pages

            append_layer_pages(pdf, finish, layer_data, args.layers.parent)
    layer_links = ""
    if layer_data:
        reports = "".join(
            f'<li>{cell["mode"]}, {cell["layer"]}, chunk {cell["chunk_index"]}: '
            f'<a href="{cell["perf_report_csv"]}">tt-perf-report CSV</a> · '
            f'<a href="{cell["perf_report_csv"].replace(".csv", ".txt")}">text and advice</a></li>'
            for cell in layer_data["cells"]
        )
        layer_links = (
            '<p>Layer profiles: <a href="#layer_overview">Overview</a> · '
            '<a href="#global_op_comparison">Global operations</a> · '
            '<a href="#local_op_comparison">Sliding operations</a> · '
            '<a href="layer_comparison.csv">Comparison CSV</a> · '
            '<a href="layer_measurements.json">All measurements</a></p>'
            f"<details><summary>Per-layer tt-perf-report output</summary><ul>{reports}</ul></details>"
        )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma4 fixed 4×1K batching</title><style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;color:#223147}}img{{width:100%;height:auto}}</style>
<h1>Fixed 4×1K batching versus canonical 4K</h1>
<p><a href="comparison.pdf">PDF</a> · <a href="comparison.csv">Comparison CSV</a> · <a href="study.json">Method and validation</a> · <a href="reproduce.sh">Reproduction commands</a></p>
{layer_links}
{''.join(pages)}<p>{html.escape(method)}</p>
<p>Raw runs: <a href="canonical.json">canonical</a>, <a href="chunked4.json">fixed batch</a>.</p></html>
"""
    (args.output / "report.html").write_text("\n".join(line.rstrip() for line in body.splitlines()) + "\n")
    print(args.output / "comparison.pdf")


if __name__ == "__main__":
    main()
