# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare layer_perf_report outputs for chunked 1x8K and ragged 4x2K calls."""

import argparse
import csv
import gzip
import html
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from models.demos.gemma4_d_p.scripts.ragged_load_report import csv_write, table

CATEGORIES = ("Attention", "CP gathering", "Layout copies", "Cache writes", "Matmuls", "Other ops")
COLORS = ("#cc5b28", "#9367bd", "#d7a524", "#8c564b", "#3975ab", "#89ada0")


def category(code, attributes):
    if "RingJointSDPA" in code:
        return "Attention"
    if "UpdatePaddedKvCache" in code:
        return "Cache writes"
    if "AllGather" in code and "'cluster_axis': '0'" in attributes:
        return "CP gathering"
    if any(name in code for name in ("SliceDevice", "PadDevice", "MeshPartitionDevice", "ConcatDevice")):
        if "NLPConcat" not in code:
            return "Layout copies"
    if "MatmulDevice" in code:
        return "Matmuls"
    return "Other ops"


def collect(input_root, output, layers):
    cells = []
    raw_dir = output / "raw"
    raw_dir.mkdir(exist_ok=True)
    for layer in layers:
        for mode in ("chunked", "ragged"):
            root = input_root / f"{layer}-{mode}" / "summaries" / "layer_perf"
            manifests = json.loads((root / "summary.json").read_text())
            for manifest in manifests:
                assert manifest["allocated_slots"] == 4 and manifest["tp_reduction"] == "reduce_scatter"
                for cell in manifest["cells"]:
                    chunk = cell["chunk_idx"]
                    assert chunk in (0, 31) and cell["useful_tokens"] == 8192
                    assert cell["request_lengths"] == ([8192] if mode == "chunked" else [2048] * 4)
                    path = next(root.glob(f"*/{layer}_chunk{chunk}.csv"))
                    raw_path = path.with_name(path.stem + "_ops.csv")
                    with raw_path.open() as source:
                        raw = {
                            int(float(row["GLOBAL CALL COUNT"])): row
                            for row in csv.DictReader(source)
                            if row["OP TYPE"] == "tt_dnn_device"
                        }
                    grouped = {name: dict(count=0, ms=0.0) for name in CATEGORIES}
                    ops = []
                    with path.open() as source:
                        for row in csv.DictReader(source):
                            if not row["Device Time"]:
                                continue
                            call = int(float(row["Global Call Count"]))
                            raw_row = raw[call]
                            code = raw_row["OP CODE"]
                            name = category(code, raw_row["ATTRIBUTES"])
                            duration = float(row["Device Time"]) / 1000
                            grouped[name]["count"] += 1
                            grouped[name]["ms"] += duration
                            ops.append(
                                dict(
                                    code=code,
                                    category=name,
                                    ms=duration,
                                    call_count=call,
                                    input0=[raw_row[f"INPUT_0_{dim}_PAD[LOGICAL]"] for dim in "WZYX"],
                                    input1=[raw_row[f"INPUT_1_{dim}_PAD[LOGICAL]"] for dim in "WZYX"],
                                    math_fidelity=raw_row["MATH FIDELITY"],
                                )
                            )
                    kernel_ms = cell["report"]["kernel_us"] / 1000
                    assert abs(sum(group["ms"] for group in grouped.values()) - kernel_ms) < 1e-6
                    cells.append(
                        dict(
                            layer=layer,
                            layer_index=cell["layer_idx"],
                            mode=mode,
                            chunk=chunk,
                            start=chunk * 8192,
                            request_lengths=cell["request_lengths"],
                            host_ms=cell["measured_ms"],
                            kernel_ms=kernel_ms,
                            span_ms=cell["report"]["span_us"] / 1000,
                            categories=grouped,
                            ops=ops,
                        )
                    )
                    stem = f"{layer}_{mode}_chunk{chunk}"
                    (raw_dir / f"{stem}_ops.csv.gz").write_bytes(gzip.compress(raw_path.read_bytes(), mtime=0))
                    # Preserve tt-perf-report's complete device-merged op table.
                    (raw_dir / f"{stem}_merged.csv").write_text(path.read_text())
    assert len(cells) == 4 * len(layers)
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-root", type=Path)
    source.add_argument("--measurements", type=Path, help="Regenerate from the archived measurements.json")
    parser.add_argument("--full-model", type=Path, help="Earlier ragged_8k_report measurements.json for context")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", nargs="+", choices=("global", "local"), default=("global", "local"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archived = json.loads(args.measurements.read_text()) if args.measurements else {}
    cells = (
        [cell for cell in archived["cells"] if cell["layer"] in args.layers]
        if archived
        else collect(args.input_root, args.output, args.layers)
    )
    full_model = archived.get("full_model", [])
    if args.full_model:
        full = json.loads(args.full_model.read_text())
        full_model = [
            dict(
                prefix_tokens=prefix,
                chunked_ms=1000 * full["measurements"]["regular8192"][f"single_{suffix}"]["median_seconds"],
                ragged_ms=1000 * full["measurements"]["ragged"][f"tails4_{suffix}"]["median_seconds"],
            )
            for prefix, suffix in ((0, "early"), (253952, "late"))
        ]
    lookup = {(cell["layer"], cell["mode"], cell["chunk"]): cell for cell in cells}
    method = (
        "Isolated layer, one traced replay per cell, CP8/TP4, four 256K cache slots initialized with random values. "
        "Embedding, RoPE preparation and host staging are outside the timed layer. Both paths use reduce-scatter. "
        "Bars sum device-kernel times: tt-perf-report 1.4.0 uses the maximum across devices for ordinary ops and "
        "the average for collectives. These are profiled per-layer measurements, not 60-layer model latency."
    )
    provenance = archived.get(
        "provenance",
        {
            "date": "2026-10-08",
            "model_commit": "4768be2777f",
            "test": "demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n with ragged benchmark extension",
            "trace_allocation_tracking": False,
            "note": "The existing two-trace isolated-layer harness ran with allocation tracking disabled; no model kernels changed.",
            "full_model_source": "../ragged_8k_reduce_scatter_2026_10_08/measurements.json",
        },
    )
    (args.output / "measurements.json").write_text(
        json.dumps(dict(method=method, provenance=provenance, full_model=full_model, cells=cells), indent=2) + "\n"
    )
    summary = []
    for layer in args.layers:
        for chunk in (0, 31):
            regular, ragged = (lookup[layer, mode, chunk] for mode in ("chunked", "ragged"))
            for name in CATEGORIES:
                a, b = regular["categories"][name], ragged["categories"][name]
                summary.append(
                    dict(
                        layer=layer,
                        prefix_tokens=chunk * 8192,
                        category=name,
                        chunked_count=a["count"],
                        ragged_count=b["count"],
                        chunked_ms=a["ms"],
                        ragged_ms=b["ms"],
                        extra_ms=b["ms"] - a["ms"],
                    )
                )
    csv_write(args.output / "comparison.csv", summary)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    pages = []
    with PdfPages(args.output / "layer_profile.pdf") as pdf:
        for layer in args.layers:
            title = "Global attention layers" if layer == "global" else "Sliding-window attention layers"
            fig, axes = plt.subplots(1, 2, figsize=(12, 8.5))
            fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.40, wspace=0.25)
            fig.suptitle(title, x=0.06, y=0.965, ha="left", fontsize=23, fontweight="bold")
            fig.text(
                0.06, 0.905, "One chunked 8K request vs one ragged 4×2K batch. Every bar processes 8,192 useful tokens."
            )
            for axis, chunk in zip(axes, (0, 31)):
                pair = [lookup[layer, mode, chunk] for mode in ("chunked", "ragged")]
                bottoms = [0.0, 0.0]
                for name, color in zip(CATEGORIES, COLORS):
                    heights = [cell["categories"][name]["ms"] for cell in pair]
                    axis.bar((0, 1), heights, bottom=bottoms, color=color, width=0.55, label=name)
                    bottoms = [a + b for a, b in zip(bottoms, heights)]
                for i, value in enumerate(bottoms):
                    axis.annotate(f"{value:.2f} ms", (i, value), xytext=(0, 6), textcoords="offset points", ha="center")
                axis.set_ylim(0, max(bottoms) * 1.2)
                axis.set_xticks((0, 1), ("Chunked\n1 × 8K", "Ragged\n4 × 2K"))
                axis.set_title("No existing prefix" if chunk == 0 else "248K prefix per request", pad=12)
                axis.set_ylabel("Device-kernel time per layer (ms)")
                axis.grid(axis="y", alpha=0.2)
                axis.set_axisbelow(True)
            fig.legend(
                *axes[0].get_legend_handles_labels(),
                loc="upper left",
                bbox_to_anchor=(0.055, 0.865),
                ncol=3,
                frameon=False,
            )
            early = lookup[layer, "ragged", 0]
            differences = [row for row in summary if row["layer"] == layer]
            explanations = []
            for prefix in (0, 253952):
                ordered = sorted(
                    (r for r in differences if r["prefix_tokens"] == prefix), key=lambda r: r["extra_ms"], reverse=True
                )
                explanations.append(
                    ("Beginning" if prefix == 0 else "248K prefix")
                    + ": largest added costs are "
                    + ", ".join(f"{r['category'].lower()} +{r['extra_ms']:.2f} ms" for r in ordered[:2])
                    + "."
                )
            count_text = (
                f"Per layer: attention calls 1 -> {early['categories']['Attention']['count']}; "
                f"extra CP gathers {early['categories']['CP gathering']['count']}; "
                f"matmuls {lookup[layer, 'chunked', 0]['categories']['Matmuls']['count']} -> {early['categories']['Matmuls']['count']}. "
                "Each 2K request is padded to 8K: four calls process 32K query rows for 8K useful rows. "
                "Panels use different y scales."
            )
            definitions = (
                "CP gathering collects token rows from the eight context-parallel devices. Layout copies slice, pad, "
                "repartition and reassemble those rows. Other ops include TP collectives, norms and RoPE."
            )
            y = 0.31
            for paragraph in (" ".join(explanations), count_text, definitions, method):
                wrapped = textwrap.fill(paragraph, width=142)
                fig.text(0.06, y, wrapped, va="top", fontsize=10, linespacing=1.4)
                y -= 0.025 * (wrapped.count("\n") + 1) + 0.014
            fig.savefig(args.output / f"{layer}.png", dpi=135)
            svg_path = args.output / f"{layer}.svg"
            fig.savefig(svg_path)
            svg = "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
            svg_path.write_text(svg)
            pages.append(svg[svg.index("<svg") :])
            pdf.savefig(fig)
            plt.close(fig)

        bridge = []
        if set(args.layers) == {"global", "local"} and full_model:
            for full in full_model:
                prefix = full["prefix_tokens"]
                weighted = {
                    name: sum(
                        row["extra_ms"] * (10 if row["layer"] == "global" else 50)
                        for row in summary
                        if row["prefix_tokens"] == prefix and row["category"] == name
                    )
                    for name in CATEGORIES
                }
                bridge.append(
                    dict(
                        **full,
                        actual_extra_ms=full["ragged_ms"] - full["chunked_ms"],
                        estimated_extra_ms=sum(weighted.values()),
                        categories=weighted,
                    )
                )
            csv_write(
                args.output / "model_context.csv",
                [{**{k: v for k, v in row.items() if k != "categories"}, **row["categories"]} for row in bridge],
            )
            fig, axis = plt.subplots(figsize=(12, 8.5))
            fig.subplots_adjust(left=0.18, right=0.91, top=0.72, bottom=0.44)
            fig.suptitle(
                "What explains the full-model gap?", x=0.06, y=0.965, ha="left", fontsize=23, fontweight="bold"
            )
            fig.text(0.06, 0.905, "One chunked 8K request vs one ragged 4×2K batch; both process 8,192 useful tokens.")
            bottoms = [0.0, 0.0]
            for name, color in zip(CATEGORIES, COLORS):
                widths = [row["categories"][name] for row in bridge]
                axis.barh((1, 0), widths, left=bottoms, color=color, height=0.5, label=name)
                bottoms = [a + b for a, b in zip(bottoms, widths)]
            for y_pos, row in zip((1, 0), bridge):
                axis.annotate(
                    f"{row['estimated_extra_ms']:.0f} ms",
                    (row["estimated_extra_ms"], y_pos),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                )
            axis.set_yticks((1, 0), ("No prefix", "248K prefix"))
            axis.set_xlim(0, max(bottoms) * 1.14)
            axis.set_xlabel("Estimated additional time: per-layer differences × 10 global + 50 sliding layers (ms)")
            axis.grid(axis="x", alpha=0.2)
            axis.set_axisbelow(True)
            fig.legend(
                *axis.get_legend_handles_labels(),
                loc="upper left",
                bbox_to_anchor=(0.055, 0.865),
                ncol=3,
                frameon=False,
            )
            wall_text = (
                "Earlier full-model measurements (chunked -> ragged, median of five replays): "
                + "; ".join(
                    ("no prefix" if row["prefix_tokens"] == 0 else "248K prefix")
                    + f" {row['chunked_ms']:.1f} -> {row['ragged_ms']:.1f} ms, an extra {row['actual_extra_ms']:.1f} ms"
                    for row in bridge
                )
                + "."
            )
            explanation = (
                "Why measure both layer types? There are 50 sliding layers and 10 global layers. "
                "Repeated data gathering and padding accumulate across all 60. Global attention performs four full-size "
                "passes over cached history, making its wasted padded-query work especially costly near 256K."
            )
            caveat = (
                "The bars are an extrapolation from isolated-layer kernel timings, not an exact breakdown of wall time. "
                "The earlier full-model measurements used populated histories; these profiles use random caches. "
                "The layer mix explains the size and direction of the gap, without implying every layer has identical timing. "
                "This is performance analysis, not numerical validation; the earlier report records an unresolved "
                "60-layer correctness failure for a separate [8192, 33] batch."
            )
            y = 0.34
            for paragraph in (wall_text, explanation, caveat):
                wrapped = textwrap.fill(paragraph, width=142)
                fig.text(0.06, y, wrapped, va="top", fontsize=10, linespacing=1.4)
                y -= 0.025 * (wrapped.count("\n") + 1) + 0.014
            fig.savefig(args.output / "model_context.png", dpi=135)
            svg_path = args.output / "model_context.svg"
            fig.savefig(svg_path)
            svg = "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
            svg_path.write_text(svg)
            pages.append(svg[svg.index("<svg") :])
            pdf.savefig(fig)
            plt.close(fig)

    details = table(
        ("Layer", "Prefix", "Category", "Calls: chunked / ragged", "Chunked ms", "Ragged ms", "Added ms"),
        [
            (
                "Global" if r["layer"] == "global" else "Sliding",
                r["prefix_tokens"],
                r["category"],
                f"{r['chunked_count']} / {r['ragged_count']}",
                f"{r['chunked_ms']:.3f}",
                f"{r['ragged_ms']:.3f}",
                f"{r['extra_ms']:+.3f}",
            )
            for r in summary
        ],
    )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>One chunked 8K call vs ragged 4×2K: layer profiles</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:30px auto;max-width:1200px;color:#213045}}svg{{width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>One chunked 8K call vs ragged 4×2K: layer profiles</h1>
<p><a href="layer_profile.pdf">PDF</a> · <a href="comparison.csv">Category timings and counts</a> · <a href="measurements.json">Per-operation measurements</a></p>
<p>Question: where does the extra time go when both paths process 8,192 useful tokens? We profile global layer 5 first, then sliding-window layer 0. At the late position, every request starts at token 253,952: the chunked call ends at 262,144 and each ragged request ends at 256,000 (exclusive).</p>
{''.join(pages)}<h2>Measured differences</h2>{details}<p>{html.escape(method)}</p>
<p>Layout copies group slice, pad, mesh partition and concat operations, including the shared operations present in both paths. Other ops include TP collectives, normalization, RoPE and elementwise work. Raw per-device CSV slices are compressed under <code>raw/</code>, alongside the unmodified device-merged tt-perf-report tables.</p></html>
"""
    (args.output / "report.html").write_text("\n".join(line.rstrip() for line in body.splitlines()) + "\n")
    print(args.output / "layer_profile.pdf")


if __name__ == "__main__":
    main()
