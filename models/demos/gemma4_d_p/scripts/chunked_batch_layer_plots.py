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


def style_table(table, rows, total_row=None, header_scale=2):
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce2e8")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_height(cell.get_height() * header_scale)
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
    batch_mode = next(c["mode"] for c in cells if c["mode"] != "canonical")
    batch = select(cells, "global", batch_mode, "first")
    lanes = batch["batch_size"]
    chunk = batch["end"] - batch["start"]
    total = batch["useful_tokens"]
    canonical_label = f"Canonical 1×{total//1024}K"
    batch_label = f"Batch {lanes}×{chunk//1024}K"
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.77, bottom=0.39, wspace=0.28)
    fig.suptitle("Where does batching change time?", x=0.06, y=0.96, ha="left", fontsize=23, fontweight="bold")
    fig.text(0.06, 0.905, f"Isolated layer profiles. Both paths process {total:,} tokens; lower bars are better.")
    for axis, layer, title in zip(
        axes,
        ("global", "local"),
        ("Global layer 5 — 10 global layers/model", "Sliding layer 0 — 50 sliding layers/model"),
    ):
        highest = 0
        for mode, offset, color, label in (
            ("canonical", -0.18, BLUE, canonical_label),
            (batch_mode, 0.18, ORANGE, batch_label),
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
    last_deltas = {}
    for layer, label in (("global", "Global"), ("local", "Sliding")):
        a, b = [select(cells, layer, m, "last") for m in ("canonical", batch_mode)]
        extra = b["kernel_ms"] - a["kernel_ms"]
        last_deltas[layer] = extra
        attention_extra = (b["grouped"]["Attention (SDPA)"]["us"] - a["grouped"]["Attention (SDPA)"]["us"]) / 1000
        notes.append(
            f"{label}, last: batching {'adds' if extra >= 0 else 'saves'} {abs(extra):.3f} ms per layer. The {lanes} attention calls total "
            f"{b['grouped']['Attention (SDPA)']['us']/1000:.3f} ms versus "
            f"{a['grouped']['Attention (SDPA)']['us']/1000:.3f} ms for canonical's one call, "
            f"Attention changes by {attention_extra:+.3f} ms."
        )
    canonical_last = select(cells, "global", "canonical", "last")
    control = next(
        c for c in cells if c["mode"] == batch_mode and c["layer"] == "global" and c["start"] == canonical_last["start"]
    )
    last = select(cells, "global", batch_mode, "last")
    notes += [
        f"Prefix control: batched global time is {control['kernel_ms']:.3f} ms at {control['start']//1024}K and {last['kernel_ms']:.3f} ms at {last['start']//1024}K. This isolates the effect of the different final-chunk starts.",
        f"Token ranges (K = 1024): first = canonical [0, {total//1024}K), batch {lanes} × [0, {chunk//1024}K). Last = canonical [{canonical_last['start']//1024}K, 256K), batch {lanes} × [{last['start']//1024}K, 256K).",
        "Kernel sums use isolated layers with random KV histories; full-model charts use actual histories. "
        + (
            "The 50 sliding layers offset much of the saving across 10 global layers. "
            if last_deltas["global"] < 0 < last_deltas["local"]
            else ""
        )
        + "Each panel has its own vertical scale.",
    ]
    paragraphs(fig, notes, 0.31)
    finish(fig, "layer_overview", pdf)

    for layer, title in (("global", "Global layer"), ("local", "Sliding-window layer")):
        selected = {(m, p): select(cells, layer, m, p) for m in ("canonical", batch_mode) for p in ("first", "last")}
        labels = list(dict.fromkeys(op["label"] for c in selected.values() for op in c["operations"]))
        rows = []
        for label in labels:
            a = selected["canonical", "first"]["grouped"].get(label, {"calls": 0})["calls"]
            b = selected[batch_mode, "first"]["grouped"].get(label, {"calls": 0})["calls"]
            values = []
            for position in ("first", "last"):
                ca, cb = [
                    selected[m, position]["grouped"].get(label, {"us": 0})["us"] for m in ("canonical", batch_mode)
                ]
                values.extend((f"{ca:,.1f}", f"{cb:,.1f}", f"{cb-ca:+,.1f}"))
            rows.append([label, str(a), str(b), *values])
        totals = []
        for p in ("first", "last"):
            a, b = [selected[m, p]["kernel_ms"] * 1000 for m in ("canonical", batch_mode)]
            totals.extend((f"{a:,.1f}", f"{b:,.1f}", f"{b-a:+,.1f}"))
        rows.append(
            [
                "TOTAL",
                str(len(selected["canonical", "first"]["operations"])),
                str(len(selected[batch_mode, "first"]["operations"])),
                *totals,
            ]
        )
        fig = plt.figure(figsize=(12, 8.5))
        fig.suptitle(title + ": operation comparison", x=0.05, y=0.96, ha="left", fontsize=22, fontweight="bold")
        fig.text(
            0.05,
            0.91,
            f"All times are microseconds, summed over the calls in one layer. C = {canonical_label}; B = {batch_label}.",
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
        notes = [
            f"The five projection/MLP matmuls have the same packed row count. Attention, cache writes and slicing operate per request. Counts include all {lanes} requests. Slices/concatenation copy local rows; norm redistribution changes local memory layout.",
            "The SDPA time includes its internal ring KV exchange. The separate TP all-gather/reduce-scatter operations communicate between tensor-parallel devices. Ordinary ops use the slowest device; collectives use the device average.",
        ]
        if total == 8192 and chunk == 4096:
            notes.append(
                "Existing global SDPA settings: canonical 8K uses Q blocks of 96, K blocks of 256, one K partition; each 4K call uses Q blocks of 128, K blocks of 256, three K partitions. Both use LoFi. These profiles do not isolate the effect of each setting."
                if layer == "global"
                else "Existing sliding SDPA settings: both paths use Q/K blocks of 128 tokens, one K partition and HiFi2. The batch executes the attention operation twice, once per request."
            )
        else:
            notes.append(
                "Timing source: tt-perf-report main (version 1.4.1), final warmed replay. Full per-call tables follow."
            )
        paragraphs(fig, notes, 0.19, size=9)
        finish(fig, f"{layer}_op_comparison", pdf)

    # Preserve the tool's per-call metrics instead of offering only grouped charts.
    for layer in ("global", "local"):
        for position in ("first", "last"):
            for mode in ("canonical", batch_mode):
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
                    title = f"{'Global' if layer == 'global' else 'Sliding'}, {position}: {canonical_label if mode=='canonical' else batch_label}{suffix}"
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
                    style_table(table, rows, header_scale=2.8)
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
