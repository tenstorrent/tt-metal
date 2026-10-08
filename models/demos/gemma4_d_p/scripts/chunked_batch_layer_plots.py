# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer comparison pages and tt-perf-report operation-table appendix."""

import csv
import textwrap

import matplotlib.pyplot as plt

BLUE, ORANGE = "#3176b5", "#d7782c"


def select(cells, layer, mode, position):
    subset = [c for c in cells if c["layer"] == layer and c["mode"] == mode]
    return (min if position == "first" else max)(subset, key=lambda c: c["start"])


def paragraphs(fig, texts, y, width=135, size=10):
    for paragraph in texts:
        wrapped = textwrap.fill(paragraph, width)
        fig.text(0.06, y, wrapped, va="top", fontsize=size, linespacing=1.3)
        y -= 0.023 * (wrapped.count("\n") + 1) + 0.012


def style_table(table, rows, total_row=None):
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce2e8")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#243c53")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row == total_row:
            cell.set_facecolor("#e2eaf1")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#f2f5f8" if row % 2 else "white")
        if col == 0:
            cell.get_text().set_ha("left")


def append_layer_pages(pdf, finish, data, root):
    cells = data["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.77, bottom=0.39, wspace=0.28)
    fig.suptitle("Where does batching add time?", x=0.06, y=0.96, ha="left", fontsize=23, fontweight="bold")
    fig.text(0.06, 0.905, "Isolated layer profiles. Both paths process 4,096 tokens; lower bars are better.")
    for axis, layer, title in zip(
        axes,
        ("global", "local"),
        ("Global layer 5 — 10 global layers/model", "Sliding layer 0 — 50 sliding layers/model"),
    ):
        highest = 0
        for mode, offset, color, label in (
            ("canonical", -0.18, BLUE, "Canonical 1×4K"),
            ("chunked4", 0.18, ORANGE, "Batch 4×1K"),
        ):
            values = [select(cells, layer, mode, pos)["kernel_ms"] for pos in ("first", "last")]
            highest = max(highest, *values)
            axis.bar([offset, 1 + offset], values, width=0.34, color=color, label=label)
            for i, value in enumerate(values):
                axis.text(i + offset, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
        axis.set_xticks([0, 1], ["First chunk", "Last chunk (ends at 256K)"])
        axis.set_ylim(0, highest * 1.2)
        axis.set_ylabel("Sum of operation kernel times (ms)")
        axis.set_title(title, fontsize=12)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.055, 0.865), ncol=2, frameon=False
    )
    notes = []
    for layer, label in (("global", "Global"), ("local", "Sliding")):
        a, b = [select(cells, layer, m, "last") for m in ("canonical", "chunked4")]
        extra = b["kernel_ms"] - a["kernel_ms"]
        attention_extra = (b["grouped"]["Attention (SDPA)"]["us"] - a["grouped"]["Attention (SDPA)"]["us"]) / 1000
        notes.append(
            f"{label}, last chunk: batching adds {extra:.3f} ms per layer; {attention_extra:.3f} ms of that is inside attention (SDPA)."
        )
    control = next(c for c in cells if c["mode"] == "chunked4" and c["layer"] == "global" and c["start"] == 258048)
    last = select(cells, "global", "chunked4", "last")
    notes += [
        f"Prefix control: batched global time is {control['kernel_ms']:.3f} ms at a 252K prefix and {last['kernel_ms']:.3f} ms at 255K. This measures the effect of the different final-chunk starts.",
        "Token ranges (K = 1024): first = canonical [0, 4K), batch 4 × [0, 1K). Last = canonical [252K, 256K), batch 4 × [255K, 256K).",
        "These are profiled kernel sums, not full-model latency. The layer test uses random KV histories and token embeddings; the full-model charts use actual model histories. Each panel has its own vertical scale.",
    ]
    paragraphs(fig, notes, 0.31)
    finish(fig, "layer_overview", pdf)

    for layer, title in (("global", "Global attention layer"), ("local", "Sliding-window attention layer")):
        selected = {(m, p): select(cells, layer, m, p) for m in ("canonical", "chunked4") for p in ("first", "last")}
        labels = list(dict.fromkeys(op["label"] for c in selected.values() for op in c["operations"]))
        rows = []
        for label in labels:
            a = selected["canonical", "first"]["grouped"].get(label, {"calls": 0})["calls"]
            b = selected["chunked4", "first"]["grouped"].get(label, {"calls": 0})["calls"]
            values = []
            for position in ("first", "last"):
                ca, cb = [
                    selected[m, position]["grouped"].get(label, {"us": 0})["us"] for m in ("canonical", "chunked4")
                ]
                values.extend((f"{ca:,.1f}", f"{cb:,.1f}", f"{cb-ca:+,.1f}"))
            rows.append([label, str(a), str(b), *values])
        totals = []
        for p in ("first", "last"):
            a, b = [selected[m, p]["kernel_ms"] * 1000 for m in ("canonical", "chunked4")]
            totals.extend((f"{a:,.1f}", f"{b:,.1f}", f"{b-a:+,.1f}"))
        rows.append(
            [
                "TOTAL",
                str(len(selected["canonical", "first"]["operations"])),
                str(len(selected["chunked4", "first"]["operations"])),
                *totals,
            ]
        )
        fig = plt.figure(figsize=(12, 8.5))
        fig.suptitle(title + ": operation comparison", x=0.05, y=0.96, ha="left", fontsize=22, fontweight="bold")
        fig.text(
            0.05,
            0.91,
            "All times are microseconds, summed over the calls in one layer. C = canonical 1×4K; B = batch 4×1K.",
        )
        axis = fig.add_axes([0.045, 0.24, 0.91, 0.60])
        axis.axis("off")
        table = axis.table(
            cellText=rows,
            colLabels=[
                "Operation",
                "Calls\nC",
                "Calls\nB",
                "First\nC µs",
                "First\nB µs",
                "Extra\nµs",
                "Last\nC µs",
                "Last\nB µs",
                "Extra\nµs",
            ],
            colWidths=[0.30, 0.045, 0.045, 0.105, 0.105, 0.085, 0.105, 0.105, 0.105],
            bbox=[0, 0, 1, 1],
            cellLoc="right",
        )
        style_table(table, rows, total_row=len(rows))
        for i, row in enumerate(rows, 1):
            if row[0] == "Attention (SDPA)":
                for col in range(9):
                    table[i, col].set_facecolor("#faeadb")
                    table[i, col].get_text().set_fontweight("bold")
        paragraphs(
            fig,
            [
                "The five projection/MLP matmuls have the same packed row count in both paths. Attention, cache writes and local slicing operate per request in the batch. Counts include all four requests.",
                "Local slices/concatenation copy tensor rows within each device. Norm redistribution changes each device's local memory layout. TP all-gather/reduce-scatter communicate between devices; these are listed separately.",
                "Timing source: tt-perf-report main (version 1.4.1), with the final warmed replay selected by signposts. Ordinary ops use the slowest device; collectives use the device average. Full per-call tables follow.",
            ],
            0.19,
            size=9,
        )
        finish(fig, f"{layer}_op_comparison", pdf)

    # Preserve the tool's per-call metrics instead of offering only grouped charts.
    for layer in ("global", "local"):
        for position in ("first", "last"):
            for mode in ("canonical", "chunked4"):
                cell = select(cells, layer, mode, position)
                with (root / cell["perf_report_csv"]).open(newline="") as handle:
                    source = [r for r in csv.DictReader(handle) if r.get("Device Time")]
                source.sort(key=lambda r: float(r["Global Call Count"]))
                for page_start in range(0, len(source), 64):
                    subset = source[page_start : page_start + 64]

                    def number(value, precision=1):
                        try:
                            return f"{float(value):,.{precision}f}"
                        except (ValueError, TypeError):
                            return "—"

                    rows = []
                    for i, row in enumerate(subset, page_start + 1):
                        name = row["OP Code"].lstrip("'").replace("DeviceOperation", "").replace("Operation", "")
                        name = name.replace("InterleavedToSharded", "Interleaved → sharded").replace(
                            "ShardedToInterleaved", "Sharded → interleaved"
                        )
                        rows.append(
                            [
                                str(i),
                                name,
                                number(row["Device Time"]),
                                number(row.get("Op-to-Op Gap")),
                                number(row.get("Cores"), 0),
                                number(row.get("DRAM")),
                                number(row.get("FLOPs")),
                                row.get("Bound", "") or "—",
                                row.get("Math Fidelity", "").split(" ")[0]
                                if row.get("Math Fidelity", "").startswith(("LoFi", "HiFi"))
                                else "—",
                                row.get("Input 0 Memory", "")
                                .replace("DEV_0_", "")
                                .replace("_INTERLEAVED", "")
                                .replace("_BLOCK_SHARDED", " block")
                                .replace("_HEIGHT_SHARDED", " height"),
                            ]
                        )
                    fig = plt.figure(figsize=(14, 10))
                    suffix = f" — {page_start//64+1}" if len(source) > 64 else ""
                    title = f"{layer.capitalize()}, {position}: {'canonical 1×4K' if mode=='canonical' else 'batch 4×1K'}{suffix}"
                    fig.suptitle(title, x=0.04, y=0.97, ha="left", fontsize=23, fontweight="bold")
                    fig.text(
                        0.04,
                        0.925,
                        f"tt-perf-report per-call table  |  request tokens [{cell['start']:,}, {cell['end']:,})  |  kernel sum {cell['kernel_ms']:.3f} ms",
                    )
                    axis = fig.add_axes([0.035, 0.13, 0.93, 0.75])
                    axis.axis("off")
                    table = axis.table(
                        cellText=rows,
                        colLabels=[
                            "#",
                            "Operation / matmul M × K × N",
                            "Device\nµs",
                            "Gap\nµs",
                            "Cores",
                            "DRAM\nGB/s",
                            "Compute\nTFLOP/s",
                            "Bound",
                            "Math",
                            "Input 0\nstorage",
                        ],
                        colWidths=[0.035, 0.325, 0.085, 0.075, 0.06, 0.085, 0.09, 0.065, 0.075, 0.105],
                        bbox=[0, 0, 1, 1],
                        cellLoc="right",
                    )
                    style_table(table, rows)
                    table.set_fontsize(8)
                    for i in range(1, len(rows) + 1):
                        table[i, 1].get_text().set_ha("left")
                        if "SDPA" in rows[i - 1][1]:
                            for j in range(10):
                                table[i, j].set_facecolor("#faeadb")
                        if rows[i - 1][7] == "SLOW":
                            table[i, 7].get_text().set_color("#a35523")
                    paragraphs(
                        fig,
                        [
                            "Source: tt-perf-report 1.4.1, main commit "
                            + data["tt_perf_report_commit"][:12]
                            + ". Rows follow execution order. The initial idle gap is excluded. Device columns summarize the 32 devices; they are not added together.",
                            "DRAM bandwidth, compute throughput and Bound are the tool's estimates. Blank entries mean the tool has no model for that operation; they do not mean zero cost. CSV/text reports retain the complete tool output and advice.",
                        ],
                        0.09,
                        width=165,
                        size=8,
                    )
                    finish(fig, f"ops_{layer}_{position}_{mode}_{page_start//64}", pdf)
