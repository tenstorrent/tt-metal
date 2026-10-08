# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare the existing 8K path with full 8K ragged batches using recorded measurements.

python -m models.demos.gemma4_d_p.scripts.ragged_8k_report \
    --input models/demos/gemma4_d_p/docs/perf/ragged_load_2026_10_08/measurements.json \
    --output models/demos/gemma4_d_p/docs/perf/ragged_8k_2026_10_08
"""

import argparse
import html
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from models.demos.gemma4_d_p.scripts.ragged_load_report import csv_write, table

MODES = ("regular8192", "ragged")
NAMES = {"regular8192": "Chunked prefill (8K)", "ragged": "Ragged attention (full 8K batch)"}
COLORS = {"regular8192": "#2864a2", "ragged": "#cf5b26"}
LABELS = (
    "single_early",
    "tails2_early",
    "tails4_early",
    "single_late",
    "tails2_late",
    "tails4_late",
    "tails4_one_late",
    "tails4_two_late",
    "tails4_three_late",
    "tails2_mixed",
    "tails4_mixed",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.input.read_text())
    rows = {}
    for mode in MODES:
        data = source["loaded"][mode]
        assert data["chunk_size"] == 8192 and data["layers"] == 60
        assert data["activations_dram_only"] == "0"
        rows[mode] = {r["label"]: r for r in data["measurements"] if r["label"] in LABELS}
        assert set(rows[mode]) == set(LABELS)
        for row in rows[mode].values():
            assert sum(row["lengths"]) == row["useful_tokens"] == 8192
            assert row.get("activation_storage", "default L1") == "default L1"
            assert len(row["samples"]) == 5
    for label in LABELS:
        assert rows[MODES[0]][label]["lengths"] == rows[MODES[1]][label]["lengths"]
        assert rows[MODES[0]][label]["starts"] == rows[MODES[1]][label]["starts"]
        assert rows[MODES[0]][label]["ends"] == rows[MODES[1]][label]["ends"]

    def ms(mode, label):
        return 1000 * rows[mode][label]["median_seconds"]

    def extra(label):
        return 100 * (ms("ragged", label) / ms("regular8192", label) - 1)

    summary = [
        dict(
            case=label,
            request_tokens=str(rows["ragged"][label]["lengths"]),
            start_indices=str(rows["ragged"][label]["starts"]),
            end_indices_exclusive=str(rows["ragged"][label]["ends"]),
            total_useful_tokens=8192,
            chunked_ms=round(ms("regular8192", label), 2),
            ragged_ms=round(ms("ragged", label), 2),
            chunked_useful_tokens_per_second=round(rows["regular8192"][label]["useful_tokens_per_second"]),
            ragged_useful_tokens_per_second=round(rows["ragged"][label]["useful_tokens_per_second"]),
            ragged_extra_time_percent=round(extra(label), 2),
        )
        for label in LABELS
    ]
    csv_write(args.output / "comparison.csv", summary)
    (args.output / "measurements.json").write_text(
        json.dumps(
            dict(chunk_size=8192, useful_tokens_per_batch=8192, layers=60, mesh=[8, 4], repeats=5, measurements=rows),
            indent=2,
        )
        + "\n"
    )

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    pages = []
    pdf = PdfPages(args.output / "comparison.pdf")
    setup = (
        "Every ragged batch contains exactly 8,192 useful tokens. Both methods process the same requests. "
        "Chunked prefill handles each request separately with an 8K chunk; ragged attention handles them in one packed batch. "
        "A 2K or 4K label describes a request's contribution to the batch, not a different configured chunk size."
    )
    method = (
        "K = 1,024 tokens. Time means completion of ALL requests shown, not time per request. "
        "60 layers; CP8/TP4; median of five warmed runs, including staging and synchronization. "
        "Initial compilation/capture is excluded. Values reuse the completed 2026-10-08 measurements. "
        "In the current implementation, request pieces shorter than 8K must be final pieces."
    )

    def page(name, fig, title, why, conclusion, positions):
        fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=20, fontweight="bold")
        fig.text(0.06, 0.915, "Every ragged batch is FULL: 8,192 useful tokens. Lower time is better.", fontsize=12)
        notes = [why, positions, "Result: " + conclusion, method]
        y = 0.28
        for text in notes:
            wrapped = textwrap.fill(text, width=142)
            fig.text(0.06, y, wrapped, va="top", fontsize=10, linespacing=1.45)
            y -= 0.025 * (wrapped.count("\n") + 1) + 0.014
        fig.savefig(args.output / f"{name}.png", dpi=135)
        svg_path = args.output / f"{name}.svg"
        fig.savefig(svg_path)
        svg = "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
        svg_path.write_text(svg)
        pdf.savefig(fig)
        plt.close(fig)
        pages.append((title, why, conclusion, svg[svg.index("<svg") :]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.79, bottom=0.39, wspace=0.15)
    for axis, context in zip(axes, ("early", "late")):
        labels = [f"{work}_{context}" for work in ("single", "tails2", "tails4")]
        x = np.arange(3)
        for i, mode in enumerate(MODES):
            values = [ms(mode, label) for label in labels]
            bars = axis.bar(x + (i - 0.5) * 0.35, values, width=0.33, color=COLORS[mode], label=NAMES[mode])
            axis.bar_label(bars, labels=[f"{v:,.0f}" for v in values], padding=3, fontsize=11)
        axis.set_xticks(x, ("1 request\n8K", "2 requests\n4K + 4K", "4 requests\n2K + 2K + 2K + 2K"), fontsize=10)
        axis.set_title(
            "All requests start at token 0" if context == "early" else "All requests start at token 253,952",
            fontsize=12,
            pad=12,
        )
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_ylim(0, 1500)
    axes[0].set_ylabel("Time to process the full 8K of useful tokens (ms)")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.07, 0.89),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    page(
        "01_beginning_and_end",
        fig,
        "8K chunked prefill vs full 8K ragged batches",
        "Question: Does dividing a full 8K batch across more requests change which method is faster?",
        f"Chunked prefill is faster in all six cases. For four 2K requests, ragged takes {extra('tails4_early'):.0f}% more time "
        f"at the beginning and {extra('tails4_late'):.0f}% more time near the end.",
        "For 1 / 2 / 4 requests, chunked uses 1 / 2 / 4 separate 8K calls; ragged uses one full 8K call. "
        "Each pair of bars processes the same useful tokens for the same requests. "
        "Start index = tokens already processed for that request.",
    )

    fig, axis = plt.subplots(figsize=(12, 8.5))
    fig.subplots_adjust(left=0.085, right=0.97, top=0.80, bottom=0.40)
    labels = [f"tails4_{suffix}" for suffix in ("early", "one_late", "two_late", "three_late", "late")]
    for mode in MODES:
        values = [ms(mode, label) for label in labels]
        axis.plot(range(5), values, "o-", color=COLORS[mode], label=NAMES[mode], linewidth=2)
        for x, value in enumerate(values):
            axis.annotate(
                f"{value:,.0f} ms",
                (x, value),
                textcoords="offset points",
                xytext=(0, 10 if mode == "ragged" else -20),
                ha="center",
                fontsize=11,
            )
    axis.set_xticks(
        range(5), ("4 early\n0 late", "3 early\n1 late", "2 early\n2 late", "1 early\n3 late", "0 early\n4 late")
    )
    axis.set_ylabel("Time to process all four requests (ms)")
    axis.set_ylim(0, 1500)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", frameon=False, fontsize=11)
    page(
        "02_request_positions",
        fig,
        "Four requests: mostly beginning, mostly end, or mixed",
        "Question: With four requests contributing 2,048 tokens each, what happens as more have long histories?",
        "Both methods slow down as more requests have long histories. Ragged remains slower. Its extra time is "
        + ", ".join(f"{extra(label):.0f}%" for label in labels)
        + " as the number of late requests increases from zero to four.",
        "Every point is 4 × 2,048 = 8,192 useful tokens. Early requests start at 0; late requests start at 253,952. "
        "Every request's end index is its start index + 2,048.",
    )

    fig, axis = plt.subplots(figsize=(12, 8.5))
    fig.subplots_adjust(left=0.085, right=0.97, top=0.79, bottom=0.42)
    labels = ("tails2_mixed", "tails4_mixed")
    x = np.arange(2)
    for i, mode in enumerate(MODES):
        values = [ms(mode, label) for label in labels]
        bars = axis.bar(x + (i - 0.5) * 0.25, values, width=0.23, color=COLORS[mode], label=NAMES[mode])
        axis.bar_label(bars, labels=[f"{v:,.0f} ms" for v in values], padding=4)
    axis.set_xticks(x, ("4K + 4K\nStarts: 0 and 248K", "2K + 2K + 2K + 2K\nStarts: 0, 8K, 128K, 248K"))
    axis.set_ylim(0, 1100)
    axis.set_ylabel("Time to process the full 8K of useful tokens (ms)")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    fig.legend(
        *axis.get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.07, 0.89),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    page(
        "03_mixed_positions",
        fig,
        "One full 8K batch, requests at different positions",
        "Question: How do the two methods compare when requests in the batch have different amounts of history?",
        f"Ragged takes {extra('tails2_mixed'):.0f}% more time for the two-request batch and "
        f"{extra('tails4_mixed'):.0f}% more time for the four-request batch. Mixing positions does not produce a speedup here.",
        "Two-request ranges: [0, 4,096), [253,952, 258,048). Four-request ranges: [0, 2,048), "
        "[8,192, 10,240), [131,072, 133,120), [253,952, 256,000). End indices are exclusive.",
    )
    pdf.close()

    details = table(
        (
            "Request tokens",
            "Start indices",
            "End indices",
            "Chunked ms",
            "Ragged ms",
            "Extra time",
            "Chunked tokens/s",
            "Ragged tokens/s",
        ),
        [
            (
                r["request_tokens"],
                r["start_indices"],
                r["end_indices_exclusive"],
                f"{r['chunked_ms']:,.0f}",
                f"{r['ragged_ms']:,.0f}",
                f"{r['ragged_extra_time_percent']:.0f}%",
                f"{r['chunked_useful_tokens_per_second']:,}",
                f"{r['ragged_useful_tokens_per_second']:,}",
            )
            for r in summary
        ],
    )
    sections = "".join(f"<section>{svg}</section>" for title, why, conclusion, svg in pages)
    body = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>8K chunked prefill vs full 8K ragged batches</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#1c2939;background:#f3f5f8;margin:0}}main{{max-width:1150px;margin:auto;padding:25px}}section{{background:white;padding:20px;margin:20px 0}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}td,th{{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}}a{{color:#2864a2}}</style>
<main><h1>8K chunked prefill vs full 8K ragged batches</h1>
<p><b>The existing chunked path is faster in all 11 measured full-8K cases.</b></p>
<p>{html.escape(setup)}</p>
<p><a href="comparison.pdf">Three-page PDF</a> · <a href="comparison.csv">Times, useful-token throughput, and request ranges</a> · <a href="measurements.json">Raw measured samples</a></p>
{sections}<section><h2>All measured comparisons</h2>{details}<p>{html.escape(method)}</p>
<p>Source: the completed Gemma4 load study at implementation commit 8fc27f0c760. Both paths use their usual settings; all selected measurements use the default activation placement. Recreate with <code>python -m models.demos.gemma4_d_p.scripts.ragged_8k_report --input models/demos/gemma4_d_p/docs/perf/ragged_load_2026_10_08/measurements.json --output models/demos/gemma4_d_p/docs/perf/ragged_8k_2026_10_08</code>.</p>
</section></main></html>
"""
    (args.output / "report.html").write_text("\n".join(line.rstrip() for line in body.splitlines()) + "\n")
    print(f"Wrote {args.output / 'comparison.pdf'}: 11 full-8K cases, two methods, three pages")


if __name__ == "__main__":
    main()
