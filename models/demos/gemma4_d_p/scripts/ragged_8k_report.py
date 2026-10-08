# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare the existing 8K path with full 8K ragged batches using recorded measurements.

python -m models.demos.gemma4_d_p.scripts.ragged_8k_report \
    --input models/demos/gemma4_d_p/docs/perf/ragged_8k_reduce_scatter_2026_10_08/input.json \
    --before models/demos/gemma4_d_p/docs/perf/ragged_8k_2026_10_08/measurements.json \
    --output models/demos/gemma4_d_p/docs/perf/ragged_8k_reduce_scatter_2026_10_08
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
NAMES = {"regular8192": "Chunked: ONE 8K call, ONE request", "ragged": "Ragged: ONE full 8K batch"}
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
COMPARISONS = LABELS[:6]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", type=Path, help="Previous focused measurements.json for an FP32 before/after page")
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

    def reference_label(label):
        # Compare an equal 8K token budget, not independent calls for each
        # ragged request. Mixed prefixes have no single-call counterpart.
        assert label in COMPARISONS
        return "single_" + label.rsplit("_", 1)[1]

    def ms(mode, label):
        if mode == "regular8192":
            label = reference_label(label)
        return 1000 * rows[mode][label]["median_seconds"]

    def extra(label):
        return 100 * (ms("ragged", label) / ms("regular8192", label) - 1)

    def difference(label):
        value = extra(label)
        return f"{abs(value):.0f}% {'more' if value >= 0 else 'less'} time"

    before = json.loads(args.before.read_text())["measurements"] if args.before else None
    if before:
        assert all(source["loaded"][mode]["tp_reduction"] == "reduce_scatter" for mode in MODES)
        for label in LABELS:
            for key in ("lengths", "starts", "ends"):
                assert before["ragged"][label][key] == rows["ragged"][label][key]

    def old_ms(label):
        return 1000 * before["ragged"][label]["median_seconds"]

    def saved_ms(label):
        return old_ms(label) - ms("ragged", label)

    summary = [
        dict(
            case=label,
            request_tokens=str(rows["ragged"][label]["lengths"]),
            start_indices=str(rows["ragged"][label]["starts"]),
            end_indices_exclusive=str(rows["ragged"][label]["ends"]),
            chunked_reference_case=reference_label(label),
            chunked_request_tokens="[8192]",
            chunked_start_indices=str(rows["regular8192"][reference_label(label)]["starts"]),
            chunked_end_indices_exclusive=str(rows["regular8192"][reference_label(label)]["ends"]),
            chunked_calls=1,
            total_useful_tokens=8192,
            chunked_ms=round(ms("regular8192", label), 2),
            ragged_ms=round(ms("ragged", label), 2),
            chunked_useful_tokens_per_second=round(8192 * 1000 / ms("regular8192", label)),
            ragged_useful_tokens_per_second=round(rows["ragged"][label]["useful_tokens_per_second"]),
            ragged_extra_time_percent=round(extra(label), 2),
        )
        for label in COMPARISONS
    ]
    if before:
        for row in summary:
            label = row["case"]
            row.update(
                ragged_before_fp32_ms=round(old_ms(label), 2),
                ragged_saved_ms=round(saved_ms(label), 2),
                ragged_time_reduction_percent=round(100 * saved_ms(label) / old_ms(label), 2),
            )
    csv_write(args.output / "comparison.csv", summary)
    (args.output / "measurements.json").write_text(
        json.dumps(
            dict(
                chunk_size=8192,
                useful_tokens_per_batch=8192,
                layers=60,
                mesh=[8, 4],
                repeats=5,
                comparison_reference="One chunked 8K call for one request versus one ragged batch totaling 8K tokens",
                compared_cases=COMPARISONS,
                tp_reductions={mode: source["loaded"][mode].get("tp_reduction", "unrecorded") for mode in MODES},
                provenance=source.get("provenance", {}),
                validation=source.get("validation", {}),
                measurements=rows,
                before=before,
            ),
            indent=2,
        )
        + "\n"
    )

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    pages = []
    pdf = PdfPages(args.output / "comparison.pdf")
    setup = (
        "Every bar processes exactly 8,192 useful tokens in ONE call. The blue reference is always ONE request "
        "processing an 8K chunk. Orange packs the same total tokens across one, two, or four requests. "
        "The comparison fixes useful token count; request count differs. All late requests have a 248K prefix. "
        "Mixed-prefix samples remain in the raw data; they have no single chunk at one matching position."
    )
    method = (
        "K = 1,024 tokens. Time completes ONE call (8,192 useful tokens). 60 layers; CP8/TP4; median of five warmed runs "
        "including staging and sync. Compilation/capture excluded. Pieces shorter than 8K are final pieces."
    )
    if before:
        method += " Both current paths use the original reduce-scatter."

    def page(name, fig, title, why, conclusion, positions):
        fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=20, fontweight="bold")
        fig.text(0.06, 0.915, "Every bar = ONE call processing 8,192 useful tokens. Lower time is better.", fontsize=12)
        notes = [why, positions, "Result: " + conclusion, method]
        y = 0.30
        for text in notes:
            wrapped = textwrap.fill(text, width=142)
            fig.text(0.06, y, wrapped, va="top", fontsize=10, linespacing=1.45)
            y -= 0.025 * (wrapped.count("\n") + 1) + 0.014
        if source.get("validation", {}).get("note"):
            fig.text(0.06, 0.025, source["validation"]["note"], fontsize=9, color="#873c26")
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
        axis.set_xlabel("Requests in the RAGGED batch (orange only)", fontsize=10, labelpad=8)
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
        "One chunked 8K call vs one full 8K ragged batch",
        "Question: At the same 8K useful-token budget, how does splitting ragged work across requests affect time?",
        f"For ragged 4 × 2K versus chunked 1 × 8K: {ms('ragged', 'tails4_early'):.0f} vs "
        f"{ms('regular8192', 'tails4_early'):.0f} ms at the beginning "
        f"({ms('ragged', 'tails4_early') / ms('regular8192', 'tails4_early'):.2f}× the time); "
        f"{ms('ragged', 'tails4_late'):.0f} vs {ms('regular8192', 'tails4_late'):.0f} ms near the end "
        f"({ms('ragged', 'tails4_late') / ms('regular8192', 'tails4_late'):.2f}× the time).",
        "BLUE ALWAYS means one request processing one full 8K chunk. ORANGE means one packed batch totaling 8K. "
        "The request counts on the x-axis apply only to orange. Start index = existing prefix length for each request.",
    )

    if before:
        fig, axes = plt.subplots(1, 2, figsize=(12, 8.5), sharey=True)
        fig.subplots_adjust(left=0.085, right=0.975, top=0.79, bottom=0.39, wspace=0.15)
        for axis, context in zip(axes, ("early", "late")):
            labels = [f"{work}_{context}" for work in ("single", "tails2", "tails4")]
            x = np.arange(3)
            for i, (name, color, values) in enumerate(
                (
                    ("Before: ragged with FP32 sums", "#9299a5", [old_ms(label) for label in labels]),
                    ("Now: ragged with reduce-scatter", COLORS["ragged"], [ms("ragged", label) for label in labels]),
                )
            ):
                bars = axis.bar(x + (i - 0.5) * 0.35, values, width=0.33, color=color, label=name)
                axis.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=3, fontsize=11)
            axis.set_xticks(x, ("1 request\n8K", "2 requests\n4K + 4K", "4 requests\n2K + 2K + 2K + 2K"), fontsize=10)
            axis.set_title("All start at token 0" if context == "early" else "All start at token 253,952", pad=12)
            axis.set_ylim(0, 1500)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Ragged batch completion time (ms)")
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.07, 0.89),
            ncol=2,
            frameon=False,
            fontsize=11,
        )
        baseline_drift = max(
            abs(
                100
                * (rows["regular8192"][label]["median_seconds"] / before["regular8192"][label]["median_seconds"] - 1)
            )
            for label in LABELS
        )
        page(
            "02_removing_fp32",
            fig,
            "What did removing the FP32 sums save?",
            "Question: How much of ragged prefill's time came from gathering TP partials and adding them in FP32?",
            f"For the first 8K of one request: {old_ms('single_early'):.0f} -> {ms('ragged', 'single_early'):.0f} ms, "
            f"saving {saved_ms('single_early'):.0f} ms ({100 * saved_ms('single_early') / old_ms('single_early'):.0f}%). "
            f"Across all 11 cases, savings range from {min(map(saved_ms, LABELS)):.0f} to {max(map(saved_ms, LABELS)):.0f} ms. "
            "Packing and attention were unchanged.",
            "Before = previous five-replay medians; now = fresh five-replay medians with the same requests and positions. "
            f"The unchanged chunked baseline was rerun and differs by at most {baseline_drift:.1f}% across these cases.",
        )
    pdf.close()

    details = table(
        (
            "Ragged request tokens",
            "Start indices",
            "End indices",
            "Chunked ms",
            "Ragged ms",
            "Ragged vs chunked",
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
                difference(r["case"]),
                f"{r['chunked_useful_tokens_per_second']:,}",
                f"{r['ragged_useful_tokens_per_second']:,}",
            )
            for r in summary
        ],
    )
    before_details = ""
    if before:
        before_details = "<h2>Removing the FP32 sums</h2>" + table(
            ("Request tokens", "Start indices", "Before ms", "Now ms", "Saved ms", "Time reduction"),
            [
                (
                    r["request_tokens"],
                    r["start_indices"],
                    f"{r['ragged_before_fp32_ms']:,.0f}",
                    f"{r['ragged_ms']:,.0f}",
                    f"{r['ragged_saved_ms']:,.0f}",
                    f"{r['ragged_time_reduction_percent']:.0f}%",
                )
                for r in summary
            ],
        )
    headline = (
        f"With equal 8K useful-token budgets, ragged 4 × 2K takes "
        f"{ms('ragged', 'tails4_early') / ms('regular8192', 'tails4_early'):.2f}× the time of one chunked 8K call "
        f"at the beginning and {ms('ragged', 'tails4_late') / ms('regular8192', 'tails4_late'):.2f}× near the end."
    )
    source_note = html.escape(
        str(source.get("provenance", {}).get("description", "Recorded measurements supplied to this report."))
    )
    validation_note = html.escape(
        source.get("validation", {}).get("note", "Performance measurements do not qualify model accuracy.")
    )
    command = f"python -m models.demos.gemma4_d_p.scripts.ragged_8k_report --input {args.input} --output {args.output}"
    if args.before:
        command += f" --before {args.before}"
    sections = "".join(f"<section>{svg}</section>" for title, why, conclusion, svg in pages)
    body = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>8K chunked prefill vs full 8K ragged batches</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#1c2939;background:#f3f5f8;margin:0}}main{{max-width:1150px;margin:auto;padding:25px}}section{{background:white;padding:20px;margin:20px 0}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}td,th{{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}}a{{color:#2864a2}}</style>
<main><h1>8K chunked prefill vs full 8K ragged batches</h1>
<p><b>{headline}</b></p>
<p>{validation_note}</p>
<p>{html.escape(setup)}</p>
<p><a href="comparison.pdf">{len(pages)}-page PDF</a> · <a href="comparison.csv">Times, useful-token throughput, and request ranges</a> · <a href="measurements.json">Raw measured samples</a></p>
{sections}<section><h2>All measured comparisons</h2>{details}{before_details}<p>{html.escape(method)}</p>
<p>Source: {source_note} All selected measurements use the default activation placement. Recreate with <code>{html.escape(command)}</code>.</p>
</section></main></html>
"""
    (args.output / "report.html").write_text("\n".join(line.rstrip() for line in body.splitlines()) + "\n")
    print(
        f"Wrote {args.output / 'comparison.pdf'}: 6 equal-token comparisons, {len(pages)} pages; all 11 raw cases retained"
    )


if __name__ == "__main__":
    main()
