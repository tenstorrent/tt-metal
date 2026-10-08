# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build a standalone HTML/PDF performance report from canonical logs and load-study JSON.

python -m models.demos.gemma4_d_p.scripts.ragged_load_report --input /tmp/gemma4-ragged-load-study \
    --output models/demos/gemma4_d_p/docs/perf/ragged_load_2026_10_08
"""

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
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

MODES = ("regular2048", "regular4096", "regular8192", "stable8192", "ragged")
NAMES = {
    "regular2048": "Regular 2K",
    "regular4096": "Regular 4K",
    "regular8192": "Regular 8K",
    "stable8192": "Independent 8K, fixed FP32",
    "ragged": "Packed, 8K attention",
}
COLORS = dict(zip(MODES, ("#2678b2", "#289779", "#835ab3", "#d29627", "#d74848")))


def parse_canonical(path, chunk):
    log = path.read_text()
    if "1 passed" not in log:
        raise ValueError(f"{path} did not pass")
    pattern = re.compile(
        r"\[traced_perf\] chunk (\d+)/(\d+) \[(\d+), (\d+)\) device=([\d.]+)ms "
        r"\((\d+) tok/s\) \| total device=([\d.]+)ms wall=([\d.]+)ms"
    )
    rows = []
    previous_wall = 0.0
    for index, count, start, end, device_ms, tps, cumulative_device, wall in pattern.findall(log):
        rows.append(
            dict(
                chunk_size=chunk,
                index=int(index),
                start=int(start),
                end=int(end),
                device_ms=float(device_ms),
                device_tokens_per_second=int(tps),
                cumulative_device_ms=float(cumulative_device),
                cumulative_wall_ms=float(wall),
                wall_increment_ms=round(float(wall) - previous_wall, 1),
            )
        )
        previous_wall = float(wall)
    assert len(rows) == 262144 // chunk, (path, len(rows))
    total = re.search(r"\[traced_perf\] TOTAL 262144 tokens in ([\d.]+)ms \((\d+) tok/s\)", log)
    device = re.search(r"\[traced_perf\] DEVICE 262144 tokens in ([\d.]+)ms .*?staging ([\d.]+)ms", log)
    return dict(
        chunk_size=chunk,
        wall_ms=float(total[1]),
        tokens_per_second=int(total[2]),
        device_ms=float(device[1]),
        staging_ms=float(device[2]),
        chunks=rows,
    )


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    canonical = {
        str(chunk): parse_canonical(options.input / f"canonical-{chunk}.log", chunk) for chunk in (2048, 4096, 8192)
    }
    loaded = {mode: json.loads((options.input / f"loaded-{mode}.json").read_text()) for mode in MODES}
    streams = {
        mode: json.loads((options.input / filename).read_text())
        for mode, filename in (
            ("regular8192", "stream-regular8192.json"),
            ("regular8192_dram", "stream-regular8192-dram.json"),
            ("stable8192_dram", "stream-stable8192-dram.json"),
            ("ragged", "stream-ragged-dram.json"),
        )
    }
    measurements = {mode: {r["label"]: r for r in data["measurements"]} for mode, data in loaded.items()}
    labels = list(measurements["ragged"])
    assert len(labels) == 19, "The report requires the complete workload matrix"
    assert all(set(measurements[m]) == set(labels) and loaded[m]["layers"] == 60 for m in MODES)
    for label in labels:
        assert all(measurements[m][label]["lengths"] == measurements["ragged"][label]["lengths"] for m in MODES)
        assert all(measurements[m][label]["starts"] == measurements["ragged"][label]["starts"] for m in MODES)
        assert all(len(measurements[m][label]["samples"]) == 5 for m in MODES)
    for data in streams.values():
        stream = data["full_stream"]
        assert data["layers"] == 60 and stream["useful_tokens"] == 1048576
        assert [(r["start"], r["end"]) for r in stream["rounds"]] == [
            (start, start + 8192) for start in range(0, 262144, 8192)
        ]
    (options.output / "measurements.json").write_text(
        json.dumps(dict(canonical=canonical, loaded=loaded, streams=streams), indent=2) + "\n"
    )
    csv_write(options.output / "canonical_chunks.csv", [row for data in canonical.values() for row in data["chunks"]])
    csv_write(
        options.output / "loaded_summary.csv",
        [
            dict(
                mode=mode,
                label=label,
                lengths=str(row["lengths"]),
                starts=str(row["starts"]),
                ends=str(row["ends"]),
                useful_tokens=row["useful_tokens"],
                median_ms=1000 * row["median_seconds"],
                tokens_per_second=row["useful_tokens_per_second"],
                first_call_ms=1000 * row["first_call_seconds"],
                capture_needed=row["capture_needed"],
                activation_storage=row.get("activation_storage", "default L1"),
                request_completion_ms=str([1000 * value for value in row["median_request_completion_seconds"]]),
            )
            for mode in MODES
            for label, row in measurements[mode].items()
        ],
    )
    csv_write(
        options.output / "full_stream_chunks.csv",
        [
            dict(
                mode=mode,
                start=r["start"],
                end=r["end"],
                batch_seconds=r["seconds"],
                cumulative_wall_seconds=r["cumulative_wall_seconds"],
                useful_tokens_per_second=4 * (r["end"] - r["start"]) / r["seconds"],
            )
            for mode, data in streams.items()
            for r in data["full_stream"]["rounds"]
        ],
    )
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    sections = []
    pdf = PdfPages(options.output / "charts.pdf")

    def chart(name, fig, title, why, conclusion):
        fig.tight_layout()
        svg_path = options.output / f"{name}.svg"
        fig.savefig(svg_path, bbox_inches="tight")
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
        fig.savefig(options.output / f"{name}.png", dpi=130, bbox_inches="tight")
        caption = "\n".join(
            textwrap.fill(paragraph, width=145) for paragraph in (f"Why: {why}", f"Finding: {conclusion}")
        )
        fig.text(0.03, -0.035, caption, fontsize=10, va="top", linespacing=1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        svg = (options.output / f"{name}.svg").read_text()
        svg = svg[svg.index("<svg") :]
        sections.append(
            f"<section><h2>{title}</h2><p><b>Why measure this:</b> {why}</p>{svg}<p><b>What it shows:</b> {conclusion}</p></section>"
        )

    def seconds(mode, label):
        return measurements[mode][label]["median_seconds"]

    def best(label):
        return min(MODES[:3], key=lambda mode: seconds(mode, label))

    native_stream = streams["regular8192"]["full_stream"]
    packed_stream = streams["ragged"]["full_stream"]
    tails_ratios = [
        seconds("ragged", f"tails4_{ctx}") / seconds("regular8192", f"tails4_{ctx}")
        for ctx in ("early", "mixed", "late")
    ]
    headline = (
        f"The current packed implementation is slower under load. Four complete 256K prompts took "
        f"{packed_stream['wall_seconds']:.2f} s packed with DRAM activations versus "
        f"{native_stream['wall_seconds']:.2f} s with regular 8K chunks "
        f"({packed_stream['wall_seconds'] / native_stream['wall_seconds']:.2f}× the time). "
        "The default packed L1 configuration cannot run four full 8K chunks."
    )
    findings = [
        f"Regular 8K is the best full-context baseline: {canonical['8192']['tokens_per_second']:,} useful tokens/s, "
        f"versus {canonical['4096']['tokens_per_second']:,} for 4K and {canonical['2048']['tokens_per_second']:,} for 2K.",
        "Even when four 2K final tails fill an 8K pack, packed/native-8K time is "
        + ", ".join(f"{ratio:.2f}× {ctx}" for ratio, ctx in zip(tails_ratios, ("early", "mixed", "late")))
        + ". The relative gap narrows at long context because both paths still run 8K attention for each request.",
        f"For mixed-history 2K tails, the first request completes in "
        f"{measurements['regular2048']['tails4_mixed']['median_request_completion_seconds'][0] * 1000:.0f} ms "
        f"with native 2K, {measurements['regular8192']['tails4_mixed']['median_request_completion_seconds'][0] * 1000:.0f} ms "
        f"with native 8K, and {seconds('ragged', 'tails4_mixed') * 1000:.0f} ms packed. "
        "Packed outputs wait for the whole batch, including its long-history requests.",
        "Shape changes require another trace capture even for previously compiled shapes. Returning to four 2K tails costs "
        f"{measurements['ragged']['churn_tails4']['first_call_seconds']:.2f} s for the first call, "
        f"versus {seconds('ragged', 'churn_tails4'):.2f} s for a resident replay.",
        f"The underfilled 1K + 32-token batch is a small exception: at long history, packing takes "
        f"{seconds('ragged', 'small2_late') * 1000:.0f} ms versus "
        f"{seconds('regular8192', 'small2_late') * 1000:.0f} ms for native 8K. "
        f"Native 2K still takes only {seconds('regular2048', 'small2_late') * 1000:.0f} ms. "
        "These tiny tails fill only 13% of the 8K target; this is not a fully loaded batch win.",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for size, data in canonical.items():
        x = [r["end"] / 1024 for r in data["chunks"]]
        color = COLORS[f"regular{size}"]
        axes[0].plot(
            x,
            [r["device_tokens_per_second"] / 1000 for r in data["chunks"]],
            label=f"{int(size)//1024}K chunks",
            color=color,
        )
        axes[1].plot(
            x, [r["cumulative_wall_ms"] / 1000 for r in data["chunks"]], label=f"{int(size)//1024}K chunks", color=color
        )
    for axis in axes:
        axis.set_xlabel("Processed context (K tokens; K = 1024)")
        axis.set_xticks([0, 64, 128, 192, 256])
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Per-chunk device throughput (thousand tokens/s)")
    axes[0].set_title("Each chunk slows as its history grows")
    axes[1].set_ylabel("Cumulative prefill wall time (seconds)")
    axes[1].set_title("Larger chunks amortize repeated tokenwise work")
    chart(
        "01_canonical",
        fig,
        "1. Canonical full-context baseline",
        "Establish the best ordinary prefill path and retain its complete context curve.",
        "These are the requested unmodified 256K canonical tests. The left panel uses printed device timing; the right includes staging. "
        f"At the last chunk, 4K retains {canonical['4096']['chunks'][-1]['device_tokens_per_second'] / canonical['8192']['chunks'][-1]['device_tokens_per_second']:.1%} "
        f"of 8K's device throughput, with {canonical['4096']['chunks'][-1]['device_ms']:.1f} ms versus "
        f"{canonical['8192']['chunks'][-1]['device_ms']:.1f} ms per chunk. Smaller chunks provide shorter scheduling steps, "
        "but 8K remains fastest over the entire prompt. Every printed chunk is preserved in canonical_chunks.csv.",
    )

    workloads = ("tails4", "tails2", "full4", "small2")
    workload_names = (
        "4 × 2K tails\n8K useful",
        "2 × 4K tails\n8K useful",
        "4 × 8K ongoing\n32K useful; packed uses DRAM",
        "1K + 32 tails\n1,056 useful",
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, sharey=True)
    for axis, context in zip(axes, ("early", "mixed", "late")):
        x = np.arange(len(workloads))
        for j, mode in enumerate((*MODES[:3], "ragged")):
            rates = [measurements[mode][f"{work}_{context}"]["useful_tokens_per_second"] / 1000 for work in workloads]
            samples = [
                [
                    measurements[mode][f"{work}_{context}"]["useful_tokens"] / s["seconds"] / 1000
                    for s in measurements[mode][f"{work}_{context}"]["samples"]
                ]
                for work in workloads
            ]
            bars = axis.bar(
                x + (j - 1.5) * 0.19,
                rates,
                width=0.18,
                color=COLORS[mode],
                label=NAMES[mode],
                yerr=[[v - min(s) for v, s in zip(rates, samples)], [max(s) - v for v, s in zip(rates, samples)]],
                capsize=2,
            )
            if mode == "ragged":
                bars[2].set_hatch("///")
        description = {"early": "Early prefixes (0)", "mixed": "Mixed prefixes", "late": "Late prefixes (248K)"}[
            context
        ]
        axis.set_title(description, loc="left")
        axis.set_ylabel("Useful thousand tokens/s")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(ncol=2)
    axes[-1].set_xticks(np.arange(len(workloads)), workload_names)
    chart(
        "02_loaded_throughput",
        fig,
        "2. Equal useful work under load",
        "Compare real multi-slot batches against all three native chunk sizes, including final tails and full ongoing chunks.",
        "Five-replay medians; whiskers show the observed min–max range. Early = start 0; late = start 248K; mixed = [0, 8K, 128K, 248K] for four requests and [0, 248K] for two. Hatched full-batch packed bars use the explicit DRAM-activation override: the default L1 path failed at this size. Other bars use default activation placement. Packing retains 8K attention per request.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("Packed batch time ÷ native batch time (>1× means slower)", fontsize=14)
    matrices = []
    for reference in ("regular8192", "best"):
        matrices.append(
            np.array(
                [
                    [
                        seconds("ragged", f"{work}_{ctx}")
                        / seconds(best(f"{work}_{ctx}") if reference == "best" else reference, f"{work}_{ctx}")
                        for ctx in ("early", "mixed", "late")
                    ]
                    for work in workloads
                ]
            )
        )
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(2, *(float(v.max()) for v in matrices)))
    for axis, reference in zip(axes, ("regular8192", "best")):
        values = np.array(
            [
                [
                    seconds("ragged", f"{work}_{ctx}")
                    / seconds(best(f"{work}_{ctx}") if reference == "best" else reference, f"{work}_{ctx}")
                    for ctx in ("early", "mixed", "late")
                ]
                for work in workloads
            ]
        )
        axis.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        for row in range(len(workloads)):
            for col in range(3):
                axis.text(
                    col,
                    row,
                    f"{values[row, col]:.2f}×",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white" if norm(values[row, col]) > 0.8 else "#202020",
                )
        axis.set_xticks(range(3), ("Early", "Mixed", "Late"))
        axis.set_yticks(range(4), ("4 × 2K tails", "2 × 4K tails", "4 × 8K (packed DRAM)", "1K + 32 tails"))
        axis.set_title("Versus regular 8K" if reference != "best" else "Versus best regular chunk size")
    chart(
        "03_relative_cost",
        fig,
        "3. Does packing improve throughput?",
        "Separate comparison with the current 8K default from comparison with an appropriately chosen native chunk size.",
        "Cells are packed batch time ÷ native batch time. Below 1× favors packing (blue); above 1× means packing is slower (red). Four full packed chunks require the measured DRAM fallback. The best regular chunk is selected separately for each workload; changing chunk size within a live request would require compatible cache geometry, and is not a free scheduler decision.",
    )

    composition = ("early", "one_late", "two_late", "three_late", "late")
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for mode in (*MODES[:3], "ragged"):
        axis.plot(
            range(5),
            [seconds(mode, f"tails4_{c}") * 1000 for c in composition],
            "o-",
            color=COLORS[mode],
            label=NAMES[mode],
        )
    axis.set_xticks(range(5))
    axis.set_xlabel("Number of the four requests at 248K prefix (others start at 0)")
    axis.set_ylabel("Time to finish the 8K useful-token batch (ms)")
    axis.set_title("Same four 2K tails, changing only the prefix mix")
    axis.grid(alpha=0.2)
    axis.legend()
    packed_slope = (seconds("ragged", "tails4_late") - seconds("ragged", "tails4_early")) * 1000 / 4
    native_slope = (seconds("regular2048", "tails4_late") - seconds("regular2048", "tails4_early")) * 1000 / 4
    chart(
        "04_prefix_mix",
        fig,
        "4. What if most requests are late in their context?",
        "Hold useful tokens, number of requests, and packing shape fixed; vary only how many prefixes are long.",
        f"Moving one request from start 0 to 248K adds about {packed_slope:.1f} ms in the packed path versus {native_slope:.1f} ms with native 2K chunks (endpoint-average increments). Prefix changes reuse the resident trace, so this increase is execution work, not recapture. Mixed batches pay approximately the sum of their separate history costs, not the longest history for every request. Per-request 8K attention still executes on the packed path, including padded queries; shared projections do not eliminate this history-dependent work.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for axis, context in zip(axes, ("early", "late")):
        modes = ("regular8192", "stable8192", "ragged")
        values = [seconds(mode, f"single_{context}") * 1000 for mode in modes]
        bars = axis.bar(range(3), values, color=[COLORS[m] for m in modes])
        axis.bar_label(bars, fmt="%.0f ms", padding=3)
        axis.set_xticks(range(3), ("Native\nring reduction", "Independent\nfixed FP32", "Packed\nfixed FP32"))
        axis.set_title(f"One full 8K request, {context} prefix")
        axis.set_ylabel("Wall latency (ms)")
        axis.set_ylim(0, max(values) * 1.2)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    chart(
        "05_arithmetic_control",
        fig,
        "5. Where does the extra cost come from?",
        "Use a single full request so packing saves no rows, then isolate the stable-reduction choice from the remaining packed machinery.",
        f"The first-to-second difference measures replacing native TP ring reductions with rank-ordered FP32 additions: "
        f"{(seconds('stable8192', 'single_early') - seconds('regular8192', 'single_early')) * 1000:.0f} ms extra at the early prefix. "
        "The second-to-third difference includes CP redistribution, splitting/concatenating, and the packed cache/metadata path. "
        f"For four early 2K tails, packing takes {seconds('ragged', 'tails4_early') * 1000:.0f} ms "
        f"versus {seconds('stable8192', 'tails4_early') * 1000:.0f} ms with independent fixed-FP32 execution, "
        f"but the default native path takes only {seconds('regular8192', 'tails4_early') * 1000:.0f} ms. "
        "The saving against the FP32 control is real; it does not establish a win over the default. "
        "This is an end-to-end ablation, not a kernel-profiler attribution. Numerical equivalence to GPU is not established by this performance study.",
    )

    fig, axis = plt.subplots(figsize=(10, 4.8))
    for mode in ("regular2048", "regular8192", "ragged"):
        values = np.array(measurements[mode]["tails4_mixed"]["median_request_completion_seconds"]) * 1000
        axis.plot(range(4), values, "o-", color=COLORS[mode], label=NAMES[mode])
    axis.set_xticks(
        range(4), ("Request 0\nprefix 0", "Request 1\nprefix 8K", "Request 2\nprefix 128K", "Request 3\nprefix 248K")
    )
    axis.set_ylabel("Completion latency from shared batch boundary (ms)")
    axis.set_title("Four simultaneous 2K tails with mixed history lengths")
    axis.grid(alpha=0.2)
    axis.legend()
    chart(
        "06_request_latency",
        fig,
        "6. Does packing help individual request latency?",
        "Avoid mistaking aggregate throughput or batch time divided by request count for actual request completion latency.",
        "All requests are ready at time zero. Native paths visit active slots round-robin by their chunk size; packed outputs become available only after the entire graph completes. Earlier requests therefore inherit the long-history work of other requests in their packed batch. Arrival queueing and downstream transfer would add further latency.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    churn_labels = ("churn_tails4", "churn_tails2")
    for j, (axis, label) in enumerate(zip(axes, churn_labels)):
        row = measurements["ragged"][label]
        values = [row["first_call_seconds"], row["median_seconds"]]
        bars = axis.bar([0, 1], values, color=("#64748b", COLORS["ragged"]))
        axis.bar_label(bars, labels=[f"{v:.2f} s" for v in values], padding=3)
        axis.set_xticks([0, 1], ("Shape switch\nwarmup + capture + replay", "Resident shape\nsteady replay"))
        axis.set_title(("Four 2K tails", "Two 4K tails")[j])
        axis.set_ylabel("Wall time (seconds)")
        axis.set_ylim(0, max(measurements["ragged"][case]["first_call_seconds"] for case in churn_labels) * 1.2)
    chart(
        "07_shape_churn",
        fig,
        "7. What happens when shapes vary under load?",
        "Revisit shapes whose kernels have already run, after their traces have been released.",
        "These are warm-process shape switches, not weight loading. Only one graph is resident, so returning to an old shape still triggers warmup/capture. Fixed-shape replay throughput is an optimistic bound for a heterogeneous queue; frequent shape changes can dominate service time.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for mode, data in streams.items():
        stream_rows = data["full_stream"]["rounds"]
        name, color, style = {
            "regular8192": ("Regular 8K, default L1", COLORS["regular8192"], "-"),
            "regular8192_dram": ("Regular 8K, DRAM", COLORS["regular8192"], "--"),
            "stable8192_dram": ("Independent fixed FP32, DRAM", COLORS["stable8192"], ":"),
            "ragged": ("Packed 4 × 8K, DRAM fallback", COLORS["ragged"], "-"),
        }[mode]
        x = [r["end"] / 1024 for r in stream_rows]
        axes[0].plot(x, [4 * 8192 / r["seconds"] / 1000 for r in stream_rows], label=name, color=color, linestyle=style)
        axes[1].plot(x, [r["cumulative_wall_seconds"] for r in stream_rows], label=name, color=color, linestyle=style)
    for axis in axes:
        axis.set_xlabel("Context processed per request (K tokens)")
        axis.set_xticks([0, 64, 128, 192, 256])
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    axes[0].set_ylabel("Useful throughput (thousand tokens/s)")
    axes[0].set_title("Four ongoing requests keep the device busy")
    axes[1].set_ylabel("Elapsed prefill wall time (seconds)")
    axes[1].set_title("Four complete 256K prompts = 1,048,576 useful tokens")
    chart(
        "08_sustained_load",
        fig,
        "8. Does the result hold for four complete long prompts?",
        "Measure a real continuous prefill stream instead of extrapolating fixed-prefix tail measurements.",
        f"Each path runs all four prompts from 0 to 256K with real contiguous KV writes. Native 8K took {native_stream['wall_seconds']:.2f} s ({native_stream['useful_tokens_per_second']:,.0f} useful tokens/s); packed with DRAM activations took {packed_stream['wall_seconds']:.2f} s ({packed_stream['useful_tokens_per_second']:,.0f} tokens/s). Native 8K with DRAM took {streams['regular8192_dram']['full_stream']['wall_seconds']:.2f} s; independent fixed-FP32 execution with DRAM took {streams['stable8192_dram']['full_stream']['wall_seconds']:.2f} s. These controls separate storage placement and reductions from the remaining packing costs. This is one complete stream per mode, excluding initial warmup/capture. The packed default L1 configuration failed before this stream could run.",
    )
    stream_table = table(
        ("Execution", "Four 256K prompts", "Useful tokens/s", "Time / default native"),
        [
            (
                name,
                f"{streams[mode]['full_stream']['wall_seconds']:.2f} s",
                f"{streams[mode]['full_stream']['useful_tokens_per_second']:,.0f}",
                f"{streams[mode]['full_stream']['wall_seconds'] / native_stream['wall_seconds']:.2f}×",
            )
            for mode, name in (
                ("regular8192", "Native 8K, default L1"),
                ("regular8192_dram", "Native 8K, DRAM"),
                ("stable8192_dram", "Independent fixed FP32, DRAM"),
                ("ragged", "Packed, DRAM"),
            )
        ],
    )
    sections[-1] = sections[-1].replace("</section>", stream_table + "</section>")
    pdf.close()

    canonical_table = table(
        ("Chunk", "256K wall time", "Useful tokens/s", "First → last chunk, device"),
        [
            (
                f"{int(size)//1024}K",
                f"{data['wall_ms']/1000:.3f} s",
                f"{data['tokens_per_second']:,}",
                f"{data['chunks'][0]['device_ms']:.1f} → {data['chunks'][-1]['device_ms']:.1f} ms",
            )
            for size, data in canonical.items()
        ],
    )
    rows = []
    for label in labels:
        row = measurements["ragged"][label]
        native = best(label)
        rows.append(
            (
                label,
                str(row["lengths"]),
                str(row["starts"]),
                row.get("activation_storage", "default L1"),
                f"{seconds('ragged', label)*1000:.1f}",
                NAMES[native],
                f"{seconds(native, label)*1000:.1f}",
                f"{seconds('ragged', label)/seconds(native, label):.2f}×",
            )
        )
    detail = table(
        ("Case", "Lengths", "Starts", "Packed storage", "Packed ms", "Best native", "Native ms", "Packed / native"),
        rows,
    )
    title = "Gemma4 packed prefill under load — 2026-10-08"
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#1e293b;background:#f1f5f9;margin:0}}main{{max-width:1120px;margin:auto;padding:32px}}
h1{{font-size:30px;line-height:1.2}}h2{{font-size:23px}}section,.intro{{background:white;border-radius:12px;padding:24px;margin:24px 0}}
svg{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}}th,td{{padding:8px 12px;border-bottom:1px solid #e2e8f0;text-align:left;white-space:nowrap}}
th{{background:#eef2ff}}a{{color:#245c9d}}code{{background:#e2e8f0;padding:2px 4px}}.note{{color:#475569}}li{{margin:6px 0}}
</style><main><h1>{title}</h1><p>Blackhole Galaxy · CP8/TP4 · Gemma4-31B-it · 60 decoder layers</p>
<div class="intro"><h2>Measured conclusion</h2><p><b>{headline}</b></p><ul>{''.join(f'<li>{finding}</li>' for finding in findings)}</ul>
<p>The data supports keeping regular 8K as the current throughput baseline. Packing's tokenwise savings are outweighed by the current fixed-FP32 reductions, CP redistribution, and padded attention work. The controls below separate measured causes from possible improvements.</p></div>
<div class="intro"><h2>What was measured</h2><p>The three requested canonical runs process a complete 262,144-token text prompt with native ring reductions, one KV slot, and the original demo's direct trace replay. The loaded study uses the runtime API with four allocated 256K slots in every mode, comparing identical useful request lengths and prefixes against regular 2K, 4K and 8K paths plus an independent fixed-FP32 control. All histories are populated by model execution. No external server, fabricated valid-length-only history, or prefix sharing is used.</p>
{canonical_table}<p>Loaded numbers are five-replay median wall times, including host packing/staging, execution and synchronization. Prefix population, input token preparation, trace capture, output downloads and external migration are excluded from steady-state measurements. First-call/capture times are recorded separately. Regular paths use round-robin chunk scheduling. Repetitions rewrite fixed boundaries and restore host bookkeeping; they do not simulate request arrivals.</p>
<p class="note">Shorter-than-8K packed chunks are final: they model short prompts or final tails, not arbitrary 2K progress steps in an ongoing 8K-geometry request. Four ongoing requests therefore contribute 32K useful tokens. All packed runs use fixed-order FP32 TP reductions. The original GPU-accuracy qualification gap remains; this report measures performance, not model quality.</p>
<p class="note">Default L1 storage failed at the MLP gate projection for four full packed chunks, both after smaller shapes and in a fresh process. Full packed batches explicitly use <code>GEMMA4_ACTIVATIONS_DRAM_ONLY=1</code>; every such plot marks the fallback. All smaller matrix cases use default L1. This is a configuration limitation, not a measured default-path throughput number. See <a href="resource_limits.json">failure evidence</a>.</p>
<p><a href="charts.pdf">All charts with findings (PDF)</a> · <a href="canonical_chunks.csv">Every canonical chunk (CSV)</a> · <a href="loaded_summary.csv">Loaded results (CSV)</a> · <a href="full_stream_chunks.csv">Continuous-stream chunks (CSV)</a> · <a href="measurements.json">All raw timing samples (JSON)</a></p></div>
{''.join(sections)}<section><h2>Complete loaded comparison</h2>{detail}</section>
<section><h2>What to investigate next</h2><p>First reduce the fixed-FP32 collective cost without losing the required numerical behavior, and measure attention with true query lengths while preserving the existing cache mapping. Then test less expensive CP redistribution and bound the lifetime of MLP slab outputs so full batches fit. Finally evaluate a small set of reusable trace shapes. These are hypotheses suggested by the controls, not optimizations measured here.</p></section>
<section><h2>Reproduce and interpret</h2><p><a href="reproduce.sh">reproduce.sh</a> contains the exact model/cache environment, three canonical nodes, five loaded modes, four continuous streams, and report generation. Run it from the repository root with the Python environment activated and exclusive access to the device. <code>GEMMA4_LOAD_CASES</code> can filter scenarios for exploratory runs; this report requires all 19 scenarios and five repetitions. OMP_NUM_THREADS=16.</p>
<p>Implementation measured: commit <code>8fc27f0c760</code>, plus the accompanying benchmark harness; production model code was unchanged during the study. Device tests ran sequentially on one Blackhole Galaxy. There is one canonical run per chunk size, five fixed-boundary replays per loaded cell, and one full continuous stream per mode. Min–max whiskers describe those samples, not a statistical confidence interval. The native 2K/4K comparisons require that cache geometry from the start of the request; they do not imply free switching at a final tail. The study excludes arrival queueing, output download and external KV migration, so it does not establish server-level throughput or latency.</p></section></main></html>"""
    (options.output / "report.html").write_text("\n".join(line.rstrip() for line in body.splitlines()) + "\n")
    print(f"Wrote {options.output / 'report.html'}")


if __name__ == "__main__":
    main()
