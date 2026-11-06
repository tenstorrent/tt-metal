# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Plots the Dest write path perf runs (perf_dest_write_path_quasar.py, perf_dest_preload_matmul_quasar.py).

Usage: python3 plot_dest_write_path_quasar.py <output dir> <run dir> [<run dir> ...]
Each run dir holds the dest_write_path_bank_stamps.jsonl and the two perf CSVs of one run. When the same point
appears in several runs, the plot shows the mean, with whiskers from the lowest to the highest run.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATHS = ["A-drain", "A-hw", "B-drain", "B-done"]
CSV_PATH = {
    "DestWritePath.A_DRAIN": "A-drain",
    "DestWritePath.A_HW": "A-hw",
    "DestWritePath.B_DRAIN": "B-drain",
    "DestWritePath.B_DONE": "B-done",
}
COLOR = {
    "A-drain": "tab:blue",
    "A-hw": "#9ecae1",
    "B-drain": "tab:orange",
    "B-done": "#fdae6b",
}
LABEL = {
    "A-drain": "A: unpack to Dest, drain",
    "A-hw": "A: unpack to Dest, SetDatValid",
    "B-drain": "B: SrcS -> SFPU -> Dest, drain",
    "B-done": "B: SrcS -> SFPU -> Dest, done bit",
}
COUNTERS = [
    ("L1_TO_L1_unpack_thread_stall_pct", "Unpack thread\nstalled"),
    ("L1_TO_L1_pack_thread_stall_pct", "Pack thread\nstalled"),
    ("L1_TO_L1_dest_stall_unpack_pct", "Unpacker waits\nfor Dest"),
    ("L1_TO_L1_sfpu_utilization_pct", "SFPU\nbusy"),
    ("L1_TO_L1_srcs_stall_sfpu_pct", "SFPU waits\nfor SrcS"),
    ("L1_TO_L1_fpu_utilization_pct", "FPU\nbusy"),
]
SOURCE = "Source: perf_dest_write_path_quasar / perf_dest_preload_matmul_quasar, emu-quasar-1x3. {runs}"


class Stat:
    """Mean of the repeated samples of one point, with the lowest and highest sample."""

    def __init__(self, values):
        self.n = len(values)
        self.mean = sum(values) / self.n
        self.lo = min(values)
        self.hi = max(values)


def whiskers(stats):
    return [
        [max(0.0, s.mean - s.lo) for s in stats],
        [max(0.0, s.hi - s.mean) for s in stats],
    ]


def load_stamps(run_dirs):
    samples = defaultdict(list)
    for run in run_dirs:
        for line in (
            (Path(run) / "dest_write_path_bank_stamps.jsonl").read_text().splitlines()
        ):
            r = json.loads(line)
            key = (
                r["kernel"],
                r["path"],
                r["tiles_per_bank"],
                r["format"],
                r.get("kt_dim"),
                r.get("fidelity"),
            )
            samples[key].append(r)
    return samples


def stamp_stat(samples, field, **key):
    k = (
        key["kernel"],
        key["path"],
        key["tpb"],
        key.get("fmt", "Float32"),
        key.get("kt"),
        key.get("fidelity"),
    )
    return Stat([r[field] for r in samples[k]])


def load_counters(run_dirs):
    samples = defaultdict(lambda: defaultdict(list))
    for run in run_dirs:
        for name in (
            "perf_dest_write_path_quasar.csv",
            "perf_dest_preload_matmul_quasar.csv",
        ):
            path = Path(run) / name
            if not path.exists():
                continue
            with path.open() as f:
                for row in csv.DictReader(f):
                    if row["marker"] != "TILE_LOOP":
                        continue
                    key = (
                        row["dwp_kernel"].split(".")[-1],
                        CSV_PATH[row["dwp_path"]],
                        int(row["dwp_tiles_per_bank"]),
                        row["formats.input_A"].split(".")[-1],
                        int(row["dwp_kt_dim"]),
                        row.get("math_fidelity", "").split(".")[-1] or None,
                    )
                    for col, _ in COUNTERS:
                        if row.get(col) not in (None, ""):
                            samples[key][col].append(float(row[col]))
    return samples


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


def grouped(ax, groups, stats_by_path, ylabel, title, fmt="{:.0f}"):
    w = 0.8 / len(PATHS)
    for i, p in enumerate(PATHS):
        stats = stats_by_path[p]
        xs = [g + (i - (len(PATHS) - 1) / 2) * w for g in range(len(groups))]
        rects = ax.bar(
            xs,
            [s.mean for s in stats],
            w,
            yerr=whiskers(stats),
            capsize=2,
            color=COLOR[p],
            label=LABEL[p],
        )
        label_bars(ax, rects, fmt)
    top = max(s.hi for stats in stats_by_path.values() for s in stats)
    ax.set_ylim(0, top * 1.3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def finish(fig, out, runs):
    fig.text(0.01, 0.005, SOURCE.format(runs=runs), fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fit(xs, ys):
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum(
        (x - xm) ** 2 for x in xs
    )
    return slope, ym - slope * xm


def plot_copy(stamps, out, runs):
    tpbs = [1, 2, 3, 4]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for p in PATHS:
        tile = [
            stamp_stat(stamps, "steady_cycles_per_tile", kernel="Copy", path=p, tpb=t)
            for t in tpbs
        ]
        bank = [
            stamp_stat(stamps, "steady_cycles_per_bank", kernel="Copy", path=p, tpb=t)
            for t in tpbs
        ]
        style = "--" if p.endswith(("hw", "done")) else "-"
        left.errorbar(
            tpbs,
            [s.mean for s in tile],
            yerr=whiskers(tile),
            color=COLOR[p],
            linestyle=style,
            marker="o",
            capsize=3,
            label=LABEL[p],
        )
        slope, intercept = fit(tpbs, [s.mean for s in bank])
        right.errorbar(
            tpbs,
            [s.mean for s in bank],
            yerr=whiskers(bank),
            color=COLOR[p],
            linestyle="none",
            marker="o",
            capsize=3,
        )
        right.plot(
            [0, 4],
            [intercept, intercept + 4 * slope],
            color=COLOR[p],
            linestyle=style,
            label=f"{p}: {slope:.0f}/tile + {intercept:.0f}/bank",
        )
    left.set_title("Copy: cycles per tile")
    left.set_xlabel("Tiles per Dest bank")
    left.set_ylabel("Steady-state cycles per tile")
    left.set_xticks(tpbs)
    left.grid(alpha=0.3)
    left.legend(fontsize=8)
    right.set_title("Copy: cycles per bank, with a straight-line fit")
    right.set_xlabel("Tiles per Dest bank")
    right.set_ylabel("Steady-state cycles per bank")
    right.set_xticks([0] + tpbs)
    right.set_xlim(0, 4.2)
    right.set_ylim(0, None)
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    finish(fig, out, runs)


def plot_sfpu(stamps, out, runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, kernel, title in (
        (axes[0], "Square", "Square (light SFPU work)"),
        (axes[1], "Exp", "Exp, accurate fp32 (heavy SFPU work)"),
    ):
        stats = {
            p: [
                stamp_stat(
                    stamps, "steady_cycles_per_tile", kernel=kernel, path=p, tpb=t
                )
                for t in (1, 4)
            ]
            for p in PATHS
        }
        grouped(
            ax,
            ["1 tile per bank", "4 tiles per bank"],
            stats,
            "Steady-state cycles per tile",
            title,
        )
    axes[0].legend(fontsize=8, loc="upper center", ncol=2)
    finish(fig, out, runs)


def plot_matmul(stamps, out, runs):
    dims = [(1, 1), (1, 4), (4, 1), (4, 4)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    stats = {
        p: [
            stamp_stat(
                stamps,
                "steady_cycles_per_bank",
                kernel="Matmul",
                path=p,
                tpb=ct,
                kt=kt,
                fidelity="HiFi4",
            )
            for ct, kt in dims
        ]
        for p in PATHS
    }
    grouped(
        ax,
        [f"CT={ct}, KT={kt}" for ct, kt in dims],
        stats,
        "Steady-state cycles per Dest bank",
        "Matmul onto an fp32 partial preloaded in Dest (HiFi4; LoFi is identical)",
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    finish(fig, out, runs)


def plot_counters(counters, out, runs):
    cases = [
        (("Copy", 4, "Float32", 1, None), "Copy, 4 tiles per bank"),
        (("Exp", 4, "Float32", 1, None), "Exp, 4 tiles per bank"),
        (("Matmul", 4, "Float16_b", 4, "HiFi4"), "Matmul, CT=4, KT=4"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    for ax, ((kernel, tpb, fmt, kt, fid), title) in zip(axes, cases):
        stats = {}
        for p in PATHS:
            per = counters[(kernel, p, tpb, fmt, kt, fid)]
            stats[p] = [
                Stat(per[col]) if per[col] else Stat([0.0]) for col, _ in COUNTERS
            ]
        grouped(
            ax,
            [name for _, name in COUNTERS],
            stats,
            "% of TILE_LOOP cycles",
            title,
        )
        ax.set_ylim(0, 115)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    finish(fig, out, runs)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_dir = Path(sys.argv[1])
    run_dirs = sys.argv[2:]
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = f"{len(run_dirs)} run{'s' if len(run_dirs) > 1 else ''}, mean with min-max whiskers."
    stamps = load_stamps(run_dirs)
    counters = load_counters(run_dirs)
    plot_copy(stamps, out_dir / "copy.png", runs)
    plot_sfpu(stamps, out_dir / "sfpu_ops.png", runs)
    plot_matmul(stamps, out_dir / "matmul.png", runs)
    plot_counters(counters, out_dir / "where_time_goes.png", runs)


if __name__ == "__main__":
    main()
