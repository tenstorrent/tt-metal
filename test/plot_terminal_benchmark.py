#!/usr/bin/env python3
"""Parse pytest-benchmark table from a terminal log and save a comparison PNG."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch


def parse_benchmark_block(text: str, name_substr: str | None = None) -> tuple[list[tuple[str, str, float]], str | None]:
    """Return (rows, dtype_key) with rows as (variant, op, mean_us). dtype_key from first row."""
    rows: list[tuple[str, str, float]] = []
    in_block = False
    dtype_key: str | None = None
    for line in text.splitlines():
        if "benchmark:" in line and "tests" in line:
            in_block = True
            continue
        if in_block and line.strip().startswith("---") and "Name" not in line:
            if rows:  # closing separator after data
                break
            continue
        if not in_block or not line.strip() or line.startswith("Name (time"):
            continue
        if line.strip().startswith("---"):
            continue
        if name_substr is not None and name_substr not in line:
            continue
        m = re.search(r"\[(dram_optimized|ng_default)-([^\]]+)\]", line)
        if not m:
            continue
        variant, op = m.group(1), m.group(2)
        nums = re.findall(r"([\d,]+\.\d+)\s+\(", line)
        if len(nums) < 3:
            continue
        mean_us = float(nums[2].replace(",", ""))
        rows.append((variant, op, mean_us))
        if dtype_key is None:
            sm = re.search(r"test_benchmark_binary_ng_dram_interleaved_2048_([a-z0-9_]+)\[", line)
            if sm:
                dtype_key = sm.group(1)
    return rows, dtype_key


# variant-dtype-OP-512x512 (sweep) or variant-dtype-OP (2048 param only)
PAT_SELECT_OPS_SWEEP = re.compile(
    r"test_benchmark_binary_ng_select_ops_dtypes\[(dram_optimized|ng_default)" r"-([a-z0-9]+)-(.+)-(\d+x\d+)\]"
)
PAT_SELECT_OPS_2048 = re.compile(
    r"test_benchmark_binary_ng_select_ops_dtypes_2048\[(dram_optimized|ng_default)" r"-([a-z0-9]+)-([^\]]+)\]"
)


def parse_select_ops_duo_block(text: str, name_substr: str | None = None) -> list[tuple[str, str, str, str, float]]:
    """(variant, dtype, op, shape_key, mean_us).

    shape_key is e.g. \"512x512\" or \"2048\" for *_2048 test names without explicit shape in bracket.
    """
    rows: list[tuple[str, str, str, str, float]] = []
    in_block = False
    for line in text.splitlines():
        if "benchmark:" in line and "tests" in line:
            in_block = True
            continue
        if in_block and line.strip().startswith("---") and "Name" not in line:
            if rows:
                break
            continue
        if not in_block or not line.strip() or line.startswith("Name (time"):
            continue
        if line.strip().startswith("---"):
            continue
        if name_substr is not None and name_substr not in line:
            continue
        if "test_benchmark_binary_ng_select_ops_dtypes" not in line:
            continue
        m = PAT_SELECT_OPS_SWEEP.search(line)
        if m:
            variant, dtype, op, shape = m.group(1), m.group(2), m.group(3), m.group(4)
        else:
            m2 = PAT_SELECT_OPS_2048.search(line)
            if not m2:
                continue
            variant, dtype, op = m2.group(1), m2.group(2), m2.group(3)
            shape = "2048"
        nums = re.findall(r"([\d,]+\.\d+)\s+\(", line)
        if len(nums) < 3:
            continue
        mean_us = float(nums[2].replace(",", ""))
        rows.append((variant, dtype, op, shape, mean_us))
    return rows


def parse_select_ops_dtypes_block(text: str, name_substr: str | None = None) -> list[tuple[str, str, float]]:
    """(dtype tag, op name, mean_us) from test_benchmark_binary_ng_select_ops_dtypes_2048[bf4-SUB] lines."""
    rows: list[tuple[str, str, float]] = []
    in_block = False
    for line in text.splitlines():
        if "benchmark:" in line and "tests" in line:
            in_block = True
            continue
        if in_block and line.strip().startswith("---") and "Name" not in line:
            if rows:
                break
            continue
        if not in_block or not line.strip() or line.startswith("Name (time"):
            continue
        if line.strip().startswith("---"):
            continue
        if name_substr is not None and name_substr not in line:
            continue
        if "select_ops_dtypes_2048" not in line:
            continue
        m = re.search(
            r"test_benchmark_binary_ng_select_ops_dtypes_2048\[([a-z0-9]+)-([^\]]+)\]",
            line,
        )
        if not m:
            continue
        dtype, op = m.group(1), m.group(2)
        nums = re.findall(r"([\d,]+\.\d+)\s+\(", line)
        if len(nums) < 3:
            continue
        mean_us = float(nums[2].replace(",", ""))
        rows.append((dtype, op, mean_us))
    return rows


DTYPE_COLORS: dict[str, str] = {
    "bf4": "#5E35B1",
    "bf8": "#1565C0",
    "bf16": "#2E7D32",
    "f32": "#E65100",
    "i32": "#C62828",
    "u32": "#4E342E",
}


# Bytes per element. Block-float formats include the 1-byte shared exponent per
# 16-element block: bf4_b -> (16*4/8 + 1)/16 = 0.5625 B; bf8_b -> 1.0625 B.
DTYPE_BYTES_PER_ELEM: dict[str, float] = {
    "bf4": 9.0 / 16.0,
    "bf8": 17.0 / 16.0,
    "bf16": 2.0,
    "f16": 2.0,
    "f32": 4.0,
    "i32": 4.0,
    "u32": 4.0,
}


def shape_elements(sh: str) -> int | None:
    """Element count for a shape string like '512x512' or '2048' (treated as 2048x2048)."""
    if "x" in sh:
        a, b = sh.split("x", 1)
        if a.isdigit() and b.isdigit():
            return int(a) * int(b)
        return None
    if sh.isdigit():
        s = int(sh)
        return s * s
    return None


def dram_bound_us(dtype: str, shape: str, dram_bw_gbps: float, rw_factor: float) -> float | None:
    """Theoretical DRAM-bound time (µs) for a binary op at given bandwidth.

    rw_factor is total (read+write) bytes multiplier per output element. For an
    elementwise binary op (2 inputs read + 1 output written) it is 3.
    Returns None if dtype or shape is not recognized.
    """
    elems = shape_elements(shape)
    bpe = DTYPE_BYTES_PER_ELEM.get(dtype)
    if elems is None or bpe is None:
        return None
    bytes_total = rw_factor * elems * bpe
    return bytes_total / (dram_bw_gbps * 1e9) * 1e6


def title_dtype_label(dtype_key: str | None) -> str:
    if dtype_key == "bfloat8_b":
        return "2048×BFloat8 (bf8_b)"
    if dtype_key == "float32":
        return "2048×float32"
    if dtype_key:
        return f"2048×{dtype_key.replace('_', ' ')}"
    return "2048 (dtype from log)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "terminal_log",
        type=Path,
        nargs="?",
        default=Path("/home/aliaksei/.cursor/projects/home-aliaksei-wp-tt-tt-metal/terminals/8.txt"),
        help="Path to terminal log containing pytest-benchmark output",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "binary_ng_dram_interleaved_2048_benchmark.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--name-substr",
        default=None,
        metavar="SUBSTR",
        help="Only include benchmark lines containing this substring (e.g. bfloat8_b, float32)",
    )
    ap.add_argument(
        "--select-ops-dtypes",
        action="store_true",
        help="Parse select_ops_dtypes_2048 benchmarks: [bf4-ADD] (one bar) or [dram_optimized-bf4-ADD] (compare)",
    )
    ap.add_argument(
        "--compare-facet",
        choices=("all", "shape", "dtype", "op"),
        default="all",
        help=(
            "Compare mode only: which facet splits subplots. "
            "'all' writes three files: <stem>_by_shape.png, _by_dtype.png, _by_op.png"
        ),
    )
    ap.add_argument(
        "--dram-bw-gbps",
        type=float,
        default=270.0,
        help="DRAM bandwidth (GB/s) for the theoretical lower-bound tick (default: 270)",
    )
    ap.add_argument(
        "--rw-factor",
        type=float,
        default=3.0,
        help="Tensor-bytes multiplier for the lower bound (read+write). Default 3 = 2 reads + 1 write.",
    )
    args = ap.parse_args()
    text = args.terminal_log.read_text(encoding="utf-8", errors="replace")

    if args.select_ops_dtypes:
        duo = parse_select_ops_duo_block(text, name_substr=args.name_substr)
        if duo:
            return _main_select_ops_compare(args, duo)
        single = parse_select_ops_dtypes_block(text, name_substr=args.name_substr)
        return _main_select_ops_single(args, single)

    rows, dtype_key = parse_benchmark_block(text, name_substr=args.name_substr)
    if not rows:
        raise SystemExit(
            f"No benchmark rows found in {args.terminal_log}"
            + (f" (filter: {args.name_substr!r})" if args.name_substr else "")
        )

    by_op: dict[str, dict[str, float]] = defaultdict(dict)
    for variant, op, mean in rows:
        by_op[op][variant] = mean

    ops_sorted = sorted(
        by_op.keys(),
        key=lambda o: max(by_op[o].values()),
        reverse=True,
    )

    n = len(ops_sorted)
    height = max(10.0, 0.28 * n + 2.0)
    fig, ax = plt.subplots(figsize=(12, height), layout="constrained")

    y = [float(i) for i in range(n)]
    width = 0.36
    colors = {"dram_optimized": "#2E7D32", "ng_default": "#1565C0"}

    for shift, variant in ((-width / 2, "dram_optimized"), (width / 2, "ng_default")):
        heights: list[float] = []
        ys: list[float] = []
        for j, op in enumerate(ops_sorted):
            if variant in by_op[op]:
                heights.append(by_op[op][variant])
                ys.append(j + shift)
        if heights:
            ax.barh(ys, heights, width, label=variant, color=colors[variant], alpha=0.9)

    # Relative speedup: ng_default / dram_optimized (>1 → dram_optimized is faster).
    all_means = [t for o in ops_sorted for t in by_op[o].values()]
    x_max = max(all_means) if all_means else 1.0
    label_x = x_max * 1.04
    for j, op in enumerate(ops_sorted):
        d = by_op[op]
        if "dram_optimized" in d and "ng_default" in d:
            t_dram = d["dram_optimized"]
            t_ng = d["ng_default"]
            su = t_ng / t_dram
            if su > 1.0 + 1e-6:
                color, fw = "#1B5E20", "bold"
            elif su < 1.0 - 1e-6:
                color, fw = "#B71C1C", "bold"
            else:
                color, fw = "#555555", "normal"
            txt = f"{su:.2f}×"
        else:
            color, fw = "#888888", "normal"
            txt = "—"
        ax.text(
            label_x,
            j,
            txt,
            va="center",
            ha="left",
            fontsize=7,
            color=color,
            fontweight=fw,
        )

    ax.set_yticks(y, [op.replace("_", " ") for op in ops_sorted], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean time (µs)")
    ax.set_title(
        f"binary_ng DRAM interleaved {title_dtype_label(dtype_key)} — mean time (lower is better)\n"
        "Speedup = ng_default / dram_optimized (right column; >1 means dram_optimized is faster)"
    )
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    # Extra margin for speedup column (e.g. "10.91×")
    ax.set_xlim(0, x_max * 1.18)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output} ({len(rows)} rows, {n} unique ops)")


def _shape_area_sort_key(sh: str) -> int:
    """Order shapes by tensor element count (512x512 before 1024x1024, etc.)."""
    t = sh.replace(",", "")
    m = re.match(r"^(\d+)x(\d+)$", t)
    if m:
        return int(m.group(1)) * int(m.group(2))
    if t.isdigit():
        s = int(t)
        return s * s
    return 0


def _shape_label_pretty(sh: str) -> str:
    if sh == "2048":
        return "2048 (test param)"
    return sh.replace("x", "×")


DRAM_BOUND_COLOR = "#FFB300"


def _plot_select_ops_compare_panel(
    ax: Axes,
    keys: list[tuple[str, str, str]],
    by_key: dict[tuple[str, str, str], dict[str, float]],
    *,
    y_label_from_key: Callable[[tuple[str, str, str]], str],
    show_legend: bool = False,
    set_xlabel: bool = True,
    dram_bw_gbps: float = 270.0,
    rw_factor: float = 3.0,
) -> None:
    """Draw one horizontal bar group + speedup + DRAM-bound tick for one facet group."""
    n = len(keys)
    row_w = 0.36
    colors = {"dram_optimized": "#2E7D32", "ng_default": "#1565C0"}
    y = [float(i) for i in range(n)]
    for shift, variant in ((-row_w / 2, "dram_optimized"), (row_w / 2, "ng_default")):
        heights: list[float] = []
        ys: list[float] = []
        for j, k in enumerate(keys):
            if variant in by_key[k]:
                heights.append(by_key[k][variant])
                ys.append(j + shift)
        if heights:
            ax.barh(ys, heights, row_w, label=variant, color=colors[variant], alpha=0.9)
    all_m = [t for k in keys for t in by_key[k].values()]
    bound_xs: list[float] = []
    bound_ys: list[float] = []
    for j, k in enumerate(keys):
        dt, _op, sh = k
        b = dram_bound_us(dt, sh, dram_bw_gbps, rw_factor)
        if b is None:
            continue
        bound_xs.append(b)
        bound_ys.append(float(j))
    if bound_xs:
        # Per-row vertical tick spanning both stacked bars, at the DRAM-bound time.
        for bx, by in zip(bound_xs, bound_ys):
            ax.vlines(
                bx,
                by - 0.48,
                by + 0.48,
                colors=DRAM_BOUND_COLOR,
                linewidth=1.6,
                zorder=5,
            )
    x_max = max([*all_m, *bound_xs]) if (all_m or bound_xs) else 1.0
    label_x = x_max * 1.04
    fs = 5 if n > 50 else (6 if n > 30 else 7)
    for j, k in enumerate(keys):
        d = by_key[k]
        if "dram_optimized" in d and "ng_default" in d:
            su = d["ng_default"] / d["dram_optimized"]
            if su > 1.0 + 1e-6:
                tc, fw = "#1B5E20", "bold"
            elif su < 1.0 - 1e-6:
                tc, fw = "#B71C1C", "bold"
            else:
                tc, fw = "#555555", "normal"
            txt = f"{su:.2f}×"
        else:
            tc, fw = "#888888", "normal"
            txt = "—"
        ax.text(label_x, j, txt, va="center", ha="left", fontsize=fs, color=tc, fontweight=fw)
    y_labels = [y_label_from_key(k) for k in keys]
    tick_fs = 4 if n > 50 else (5 if n > 30 else 6)
    ax.set_yticks(y, y_labels, fontsize=tick_fs)
    ax.invert_yaxis()
    if set_xlabel:
        ax.set_xlabel("Mean time (µs)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_xlim(0, x_max * 1.18)
    if show_legend:
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Line2D(
                [0],
                [0],
                color=DRAM_BOUND_COLOR,
                linewidth=2.0,
                label=f"DRAM-bound @ {dram_bw_gbps:g} GB/s ({rw_factor:g}× tensor bytes)",
            )
        )
        labels.append(f"DRAM-bound @ {dram_bw_gbps:g} GB/s ({rw_factor:g}× tensor bytes)")
        ax.legend(handles=handles, labels=labels, loc="lower right", framealpha=0.95)


def _y_label_for_compare_facet(
    facet: Literal["shape", "dtype", "op"],
    k: tuple[str, str, str],
) -> str:
    dt, op, sh = k
    on = op.replace("_", " ")
    if facet == "shape":
        return f"{dt}  {on}"
    if facet == "dtype":
        return f"{on}  {_shape_label_pretty(sh)}"
    return f"{dt}  {_shape_label_pretty(sh)}"


def _group_keys_by_facet(
    by_key: dict[tuple[str, str, str], dict[str, float]],
    facet: Literal["shape", "dtype", "op"],
) -> tuple[list[str], dict[str, list[tuple[str, str, str]]]]:
    d: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for k in by_key:
        if facet == "shape":
            d[k[2]].append(k)
        elif facet == "dtype":
            d[k[0]].append(k)
        else:
            d[k[1]].append(k)
    if facet == "shape":
        order = sorted(d.keys(), key=_shape_area_sort_key)
    else:
        order = sorted(d.keys())
    return order, d


def _compare_panel_subtitle(facet: Literal["shape", "dtype", "op"], g: str, n: int) -> str:
    if facet == "shape":
        t = f"{g} (2048 test param)" if g == "2048" else g.replace("x", "×")
        return f"Tensor shape: {t}  ({n} (dtype, op) pairs)"
    if facet == "dtype":
        return f"dtype: {g}  ({n} (op, shape) pairs)"
    on = g.replace("_", " ")
    return f"op: {on}  ({n} (dtype, shape) pairs)"


def _compare_suptitle(facet: Literal["shape", "dtype", "op"], all_keys: list[tuple[str, str, str]]) -> str:
    if all(_k[2] == "2048" for _k in all_keys) and all_keys:
        sub = " (fixed 2048 problem size)"
    else:
        sub = ""
    by_what = {
        "shape": "by tensor shape",
        "dtype": "by data type (dtype)",
        "op": "by elementwise op",
    }[facet]
    return (
        f"binary_ng select_ops × dtypes{sub} — {by_what}; mean time (lower is better). "
        f"Speedup = ng_default / dram_optimized (>1 ⇒ dram faster)."
    )


def _write_select_ops_compare_figure(
    by_key: dict[tuple[str, str, str], dict[str, float]],
    output: Path,
    facet: Literal["shape", "dtype", "op"],
    *,
    dram_bw_gbps: float = 270.0,
    rw_factor: float = 3.0,
) -> int:
    """Build one faceted figure; return n keys plotted."""
    order, by_group = _group_keys_by_facet(by_key, facet)
    n_panels = len(order)

    def panel_height(nk: int) -> float:
        rh = 0.12 if nk > 40 else 0.18
        return max(2.2, rh * max(nk, 1) + 0.6)

    heights = [panel_height(len(by_group[g])) for g in order]
    total_h = sum(heights) + 1.2 + 0.5 * (n_panels - 1)
    max_rows = max(len(by_group[g]) for g in order) if order else 1
    fig_w = 14 if max_rows > 40 else 12
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(fig_w, min(total_h, 200)),
        height_ratios=heights,
        layout="constrained",
    )
    if n_panels == 1:
        axes = [axes]
    n_keys_total = 0

    def y_label(k: tuple[str, str, str]) -> str:
        return _y_label_for_compare_facet(facet, k)

    for ax, g in zip(axes, order):
        keys = sorted(
            by_group[g],
            key=lambda k2: max(by_key[k2].values()),
            reverse=True,
        )
        n_keys_total += len(keys)
        ax.set_title(
            _compare_panel_subtitle(facet, g, len(keys)),
            fontsize=9,
        )
        _plot_select_ops_compare_panel(
            ax,
            keys,
            by_key,
            y_label_from_key=y_label,
            show_legend=(ax is axes[-1]),
            set_xlabel=(ax is axes[-1]),
            dram_bw_gbps=dram_bw_gbps,
            rw_factor=rw_factor,
        )
    all_keys = list(by_key.keys())
    fig.suptitle(_compare_suptitle(facet, all_keys), fontsize=10, y=1.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return n_keys_total


def _main_select_ops_compare(args: argparse.Namespace, duo_rows: list[tuple[str, str, str, str, float]]) -> None:
    """dram_optimized vs ng_default; facet panels by shape, dtype, and/or op."""
    by_key: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for variant, dtype, op, _shape, mean in duo_rows:
        by_key[(dtype, op, _shape)][variant] = mean

    facets: tuple[Literal["shape", "dtype", "op"], ...]
    if args.compare_facet == "all":
        facets = ("shape", "dtype", "op")
    else:
        facets = (args.compare_facet,)

    for fct in facets:
        if len(facets) == 1:
            out = args.output
        else:
            stem = args.output.stem
            # Avoid awkward "by_by_<facet>" when user already passes a "by" stem.
            if stem in ("", "by"):
                new_stem = f"by_{fct}"
            elif stem.endswith("_by"):
                new_stem = f"{stem[:-3]}_by_{fct}"
            else:
                new_stem = f"{stem}_by_{fct}"
            out = args.output.with_name(f"{new_stem}{args.output.suffix}")
        printed = _write_select_ops_compare_figure(
            by_key,
            out,
            fct,
            dram_bw_gbps=args.dram_bw_gbps,
            rw_factor=args.rw_factor,
        )
        print(f"Wrote {out} ({len(duo_rows)} raw rows, {printed} (dtype, op, shape) pairs, {fct} facet)")


def _main_select_ops_single(args: argparse.Namespace, rows: list[tuple[str, str, float]]) -> None:
    if not rows:
        raise SystemExit(
            f"No select_ops_dtypes_2048 benchmark rows in {args.terminal_log}"
            + (f" (filter: {args.name_substr!r})" if args.name_substr else "")
        )
    # Slowest at top: sort by mean descending
    rows_sorted = sorted(rows, key=lambda r: r[2], reverse=True)
    n = len(rows_sorted)
    height = max(8.0, 0.22 * n + 1.5)
    fig, ax = plt.subplots(figsize=(12, height), layout="constrained")
    y = [float(i) for i in range(n)]
    heights = [r[2] for r in rows_sorted]
    bar_colors = [DTYPE_COLORS.get(r[0], "#757575") for r in rows_sorted]
    ax.barh(y, heights, color=bar_colors, alpha=0.92)
    x_max = max(heights) if heights else 1.0
    label_x = x_max * 1.01
    for j, (dt, op, mean) in enumerate(rows_sorted):
        ax.text(
            label_x,
            j,
            f"{mean:.1f}",
            va="center",
            ha="left",
            fontsize=6,
            color="#424242",
        )
    y_labels = [f"{dt}  {op.replace('_', ' ')}" for dt, op, _ in rows_sorted]
    ax.set_yticks(y, y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean time (µs)")
    ax.set_xlim(0, x_max * 1.12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    used_dtypes = list(dict.fromkeys(r[0] for r in rows_sorted))
    handles = [Patch(color=DTYPE_COLORS.get(d, "#757575"), label=d, alpha=0.92) for d in used_dtypes]
    ax.legend(handles=handles, loc="lower right", title="dtype", framealpha=0.95)
    ax.set_title(
        "binary_ng select_ops × dtypes 2048 — mean time per (dtype, op) (lower is better)\n"
        "Numbers at right: mean (µs)"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output} ({n} (dtype, op) rows)")


if __name__ == "__main__":
    main()
