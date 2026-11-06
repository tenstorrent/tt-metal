# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Charts for dest_write_path_report.md.

Usage, from the repository root: python3 docs_dest_write_path_results/plot_report.py
Reads docs_dest_write_path_results/logs and writes report_*.png next to this script.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
TPBS = [1, 2, 3, 4]

BLUE = "tab:blue"
LIGHT_BLUE = "#9ecae1"
ORANGE = "tab:orange"
LIGHT_ORANGE = "#fdae6b"
DARK_ORANGE = "#a63603"
GRAY = "#7f7f7f"
OPT_COLOR = {"base": "#fdd0a2", "addrmod": "#fdae6b", "bankunpack": "#f16913", "both": DARK_ORANGE}
OPT_LABEL = {
    "base": "Original",
    "addrmod": "Address-mode step",
    "bankunpack": "One UNPACR2 per bank",
    "both": "Both options",
}


def fit(xs, ys):
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
    return slope, ym - slope * xm


def finish(fig, out, source):
    fig.text(0.01, 0.005, f"Source: {source}. emu-quasar-1x3 emulator cycles.", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def label_bars(ax, rects, fmt="{:.0f}"):
    for rect in rects:
        ax.annotate(
            fmt.format(rect.get_height()),
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            xytext=(0, 2),
            textcoords="offset points",
        )


def load_stamps(path):
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(r["kernel"], r["path"], r["tiles_per_bank"], r.get("srcs_opt", "base"))] = r
    return out


def load_tile_loop(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["marker"] == "TILE_LOOP":
                rows.append(row)
    return rows


def srcs_opt_of(row):
    addr = row.get("dwp_srcs_addr_mod_step") == "True"
    bank = row.get("dwp_srcs_bank_unpack") == "True"
    return {(False, False): "base", (True, False): "addrmod", (False, True): "bankunpack", (True, True): "both"}[
        (addr, bank)
    ]


def plot_copy_fp32(summary, out):
    a = summary["copy_fp32_nosync_path_a"]
    b = summary["copy_fp32_srcs_opts_b_done"]["cycles_per_bank"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [0.8 + 0.05 * i for i in range(65)]
    ax.plot(
        xs,
        [a["fit"]["per_tile"] + a["fit"]["per_bank"] / x for x in xs],
        color=BLUE,
        linestyle=":",
        label="A: unpack to Dest (fit: 64 per tile + 35 per bank)",
    )
    ax.plot(
        [int(k) for k in a["cycles_per_tile"]],
        list(a["cycles_per_tile"].values()),
        color=BLUE,
        linestyle="none",
        marker="o",
        markersize=13,
        markerfacecolor="none",
        markeredgewidth=2,
        zorder=5,
        label="A: measured",
    )
    for opt, color in (("base", LIGHT_ORANGE), ("both", DARK_ORANGE)):
        ax.plot(
            TPBS,
            [c / t for c, t in zip(b[opt], TPBS)],
            color=color,
            marker="o",
            label=f"B: SrcS -> SFPU -> Dest, {OPT_LABEL[opt].lower()}",
        )
    ax.set_title("Copy, Float32 in and out")
    ax.set_xlabel("Tiles per Dest bank")
    ax.set_ylabel("Steady-state cycles per tile")
    ax.set_xticks(TPBS)
    ax.set_xlim(0.8, 4.2)
    ax.set_ylim(0, 125)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    finish(fig, out, "format sweep and SrcS option runs, stamp sync off, B uses the done-bit hand-off")


def plot_srcs_options(summary, rows, out):
    b = summary["copy_fp32_srcs_opts_b_done"]["cycles_per_bank"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

    opts = ["base", "addrmod", "bankunpack", "both"]
    per_tile, per_bank = [], []
    for opt in opts:
        slope, intercept = fit(TPBS, b[opt])
        per_tile.append(slope)
        per_bank.append(intercept)
    w = 0.38
    xs = range(len(opts))
    r1 = left.bar([x - w / 2 for x in xs], per_tile, w, color=ORANGE, label="Cycles per tile")
    r2 = left.bar([x + w / 2 for x in xs], per_bank, w, color=LIGHT_ORANGE, label="Extra cycles per bank")
    label_bars(left, r1)
    label_bars(left, r2)
    left.axhline(64, color=BLUE, linestyle="--", linewidth=1)
    left.text(3.45, 66, "A: 64 per tile", color=BLUE, fontsize=8, ha="right")
    left.set_xticks(list(xs))
    left.set_xticklabels([OPT_LABEL[o].replace(" ", "\n", 1) for o in opts], fontsize=9)
    left.set_ylabel("Cycles (straight-line fit over 1 to 4 tiles per bank)")
    left.set_title("Path B cost with each option (B-done)")
    left.set_ylim(0, 125)
    left.grid(axis="y", alpha=0.3)
    left.legend(fontsize=8)

    counters = [
        ("L1_TO_L1_thread3_ipc_pct", "TRISC3 issues\nan instruction"),
        ("L1_TO_L1_sfpu_utilization_pct", "SFPU\nbusy"),
        ("L1_TO_L1_srcs_stall_sfpu_pct", "SFPU waits\nfor SrcS"),
    ]
    picked = {srcs_opt_of(r): r for r in rows if r["dwp_path"].endswith("B_DONE") and r["dwp_tiles_per_bank"] == "4"}
    w = 0.8 / len(opts)
    for i, opt in enumerate(opts):
        vals = [float(picked[opt][col]) for col, _ in counters]
        rects = right.bar(
            [g + (i - (len(opts) - 1) / 2) * w for g in range(len(counters))],
            vals,
            w,
            color=OPT_COLOR[opt],
            label=OPT_LABEL[opt],
        )
        label_bars(right, rects)
    right.set_xticks(range(len(counters)))
    right.set_xticklabels([name for _, name in counters])
    right.set_ylabel("% of TILE_LOOP cycles")
    right.set_title("Why: perf counters at 4 tiles per bank (B-done)")
    right.set_ylim(0, 110)
    right.grid(axis="y", alpha=0.3)
    right.legend(fontsize=8, ncol=2, loc="upper right")
    finish(fig, out, "SrcS option runs, Float32 in and out, stamp sync off")


def plot_fp16(stamps, summary, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    lines = [
        ("A-drain", "base", BLUE, "-", "A: unpack to Dest, drain"),
        ("A-hw", "base", LIGHT_BLUE, "--", "A: unpack to Dest, SetDatValid"),
        ("B-done", "base", LIGHT_ORANGE, "-", "B: original (done bit)"),
        ("B-drain", "both", ORANGE, "--", "B: both options, drain"),
        ("B-done", "both", DARK_ORANGE, "-", "B: both options, done bit"),
    ]
    for path, opt, color, style, label in lines:
        left.plot(
            TPBS,
            [stamps[("Copy", path, t, opt)]["steady_cycles_per_tile"] for t in TPBS],
            color=color,
            linestyle=style,
            marker="o",
            label=label,
        )
    left.set_title("Copy, Float16_b in and out (32-bit Dest in between)")
    left.set_xlabel("Tiles per Dest bank")
    left.set_ylabel("Steady-state cycles per tile")
    left.set_xticks(TPBS)
    left.set_ylim(0, 125)
    left.grid(alpha=0.3)
    left.legend(fontsize=8, loc="lower left")

    def slope_of(path, opt):
        return fit(TPBS, [stamps[("Copy", path, t, opt)]["steady_cycles_per_bank"] for t in TPBS])[0]

    b = summary["copy_fp32_srcs_opts_b_done"]["cycles_per_bank"]
    groups = ["Float32 out", "Float16_b out"]
    series = [
        (
            "A: unpack to Dest",
            BLUE,
            [summary["copy_fp32_nosync_path_a"]["fit"]["per_tile"], slope_of("A-drain", "base")],
        ),
        ("B: original", LIGHT_ORANGE, [fit(TPBS, b["base"])[0], slope_of("B-done", "base")]),
        ("B: both options", DARK_ORANGE, [fit(TPBS, b["both"])[0], slope_of("B-done", "both")]),
    ]
    w = 0.8 / len(series)
    for i, (name, color, vals) in enumerate(series):
        rects = right.bar(
            [g + (i - (len(series) - 1) / 2) * w for g in range(len(groups))], vals, w, color=color, label=name
        )
        label_bars(right, rects)
    right.set_xticks(range(len(groups)))
    right.set_xticklabels(groups)
    right.set_ylabel("Cycles per tile (straight-line fit)")
    right.set_title("Cost per tile: Float32 out vs Float16_b out")
    right.set_ylim(0, 125)
    right.grid(axis="y", alpha=0.3)
    right.legend(fontsize=8)
    finish(fig, out, "format sweep runs, stamp sync off (B uses the done-bit hand-off on the right)")


def plot_matmul(summary, out):
    m = summary["matmul_cycles_per_bank"]
    series = [
        ("A-drain", BLUE),
        ("A-hw", LIGHT_BLUE),
        ("B-drain, original", LIGHT_ORANGE),
        ("B, both options", DARK_ORANGE),
        ("L1-acc", GRAY),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.8 / len(series)
    for i, (name, color) in enumerate(series):
        rects = ax.bar(
            [g + (i - (len(series) - 1) / 2) * w for g in range(len(m["dims"]))],
            m[name],
            w,
            color=color,
            label=name,
        )
        label_bars(ax, rects)
    ax.set_xticks(range(len(m["dims"])))
    ax.set_xticklabels(m["dims"])
    ax.set_ylabel("Steady-state cycles per Dest bank")
    ax.set_title("Matmul onto an fp32 partial in Dest (Out = Partial + A x B)")
    ax.set_ylim(0, 900)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    finish(fig, out, "matmul runs, stamp sync on")


def plot_sfpu(out):
    samples = {}
    for run in ("run1", "run2", "run3"):
        for line in (LOGS / run / "dest_write_path_bank_stamps.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["kernel"] in ("Square", "Exp"):
                samples.setdefault((r["kernel"], r["path"], r["tiles_per_bank"]), []).append(
                    r["steady_cycles_per_tile"]
                )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    paths = [("A-drain", BLUE), ("A-hw", LIGHT_BLUE), ("B-drain", ORANGE), ("B-done", LIGHT_ORANGE)]
    for ax, kernel, title in (
        (axes[0], "Square", "Square (light SFPU work)"),
        (axes[1], "Exp", "Exp, accurate fp32 (heavy SFPU work)"),
    ):
        w = 0.8 / len(paths)
        top = 0
        for i, (path, color) in enumerate(paths):
            vals = [sum(samples[(kernel, path, t)]) / len(samples[(kernel, path, t)]) for t in (1, 4)]
            top = max(top, *vals)
            rects = ax.bar([g + (i - 1.5) * w for g in range(2)], vals, w, color=color, label=path)
            label_bars(ax, rects)
        ax.set_xticks(range(2))
        ax.set_xticklabels(["1 tile per bank", "4 tiles per bank"])
        ax.set_ylabel("Steady-state cycles per tile")
        ax.set_title(title)
        ax.set_ylim(0, top * 1.25)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2, loc="upper center")
    finish(fig, out, "runs 1-3, original path B, stamp sync on, mean of 3 runs")


def main():
    summary = json.loads((LOGS / "earlier_runs_summary.json").read_text())
    fp16_stamps = load_stamps(LOGS / "fp16_out" / "dest_write_path_bank_stamps.jsonl")
    opt_rows = load_tile_loop(LOGS / "srcs_opts" / "perf_dest_write_path_quasar.csv")
    plot_copy_fp32(summary, HERE / "report_copy_fp32.png")
    plot_srcs_options(summary, opt_rows, HERE / "report_srcs_options.png")
    plot_fp16(fp16_stamps, summary, HERE / "report_fp16_out.png")
    plot_matmul(summary, HERE / "report_matmul.png")
    plot_sfpu(HERE / "report_sfpu_ops.png")


if __name__ == "__main__":
    main()
