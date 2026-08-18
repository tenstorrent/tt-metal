# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Training log comparison and visualization script.

This script parses log files from tt-train's main training binary (e.g., nano_gpt)
and generates comparison plots for:
  - Training loss curves
  - Loss differences between runs (relative to a baseline)
  - Step time performance

This is useful for evaluating kernel optimizations, fusion strategies, or
configuration changes by comparing multiple training runs side-by-side.

Usage:
    python plot_training_comparison.py --baseline run_baseline.txt --compare run_optimized.txt run_fused.txt \\
        --labels baseline optimized fused --output-dir ./plots

Expected log format:
    Step lines from nano_gpt (C++) or train.py (Python), e.g.:
        "Step: 1, Loss: 11.0234375, Time: 703.14 ms, ..."
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_log(filepath: str, warmup_steps: int = 15) -> Tuple[List[float], List[float]]:
    """
    Parse a training log file and extract step times and losses.

    Args:
        filepath: Path to the log file
        warmup_steps: Number of initial steps to skip for step time statistics
                      (warmup steps may have unreliable timing)

    Returns:
        Tuple of (step_times, losses) lists
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    step_times = []
    losses = []

    step_line_re = re.compile(r"Step: \d+, Loss: ([\d.]+), Time: ([\d.]+) ms")

    for line in lines:
        match = step_line_re.search(line)
        if match:
            losses.append(float(match.group(1)))
            step_times.append(float(match.group(2)))

    # Skip warmup steps for step time analysis
    step_times = step_times[warmup_steps:]

    return step_times, losses


def print_statistics(all_data: Dict[str, Dict[str, List[float]]]) -> None:
    """Print summary statistics for all runs."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print("\nMean Step Times:")
    for name, data in all_data.items():
        if data["step_times"]:
            mean_time = np.mean(data["step_times"])
            std_time = np.std(data["step_times"])
            print(f"  {name}: {mean_time:.2f} ms (std: {std_time:.2f} ms)")

    # Find baseline for speedup calculation
    names = list(all_data.keys())
    if len(names) > 1:
        baseline_name = names[0]
        if all_data[baseline_name]["step_times"]:
            baseline_time = np.mean(all_data[baseline_name]["step_times"])
            print(f"\nSpeedup relative to '{baseline_name}':")
            for name, data in all_data.items():
                if name != baseline_name and data["step_times"]:
                    mean_time = np.mean(data["step_times"])
                    speedup = baseline_time / mean_time
                    print(f"  {name}: {speedup:.3f}x")

    print("\nFinal Loss (last 100 steps average):")
    for name, data in all_data.items():
        if len(data["losses"]) >= 100:
            final_loss = np.mean(data["losses"][-100:])
            print(f"  {name}: {final_loss:.6f}")
        elif data["losses"]:
            final_loss = np.mean(data["losses"])
            print(f"  {name}: {final_loss:.6f} (all {len(data['losses'])} steps)")


def plot_loss_comparison(
    all_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    title_prefix: str = "",
    max_steps: Optional[int] = None,
    file_prefix: str = "",
) -> None:
    """Plot loss curves for all runs."""
    plt.figure(figsize=(20, 10))

    for name, data in all_data.items():
        losses = data["losses"]
        if max_steps:
            losses = losses[:max_steps]
        plt.plot(losses, label=name, linewidth=2)

    title = f"{title_prefix}Loss Comparison: All Runs" if title_prefix else "Loss Comparison: All Runs"
    plt.title(title, fontsize=20)
    plt.xlabel("Step", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=14)

    output_file = output_path / f"{file_prefix}losses.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_loss_difference(
    all_data: Dict[str, Dict[str, List[float]]],
    baseline_name: str,
    output_path: Path,
    title_prefix: str = "",
    max_steps: Optional[int] = None,
    file_prefix: str = "",
) -> None:
    """Plot loss differences relative to baseline."""
    if baseline_name not in all_data:
        print(f"Warning: Baseline '{baseline_name}' not found, skipping loss difference plot")
        return

    baseline_losses = all_data[baseline_name]["losses"]
    if max_steps:
        baseline_losses = baseline_losses[:max_steps]

    plt.figure(figsize=(20, 10))

    for name, data in all_data.items():
        if name != baseline_name:
            losses = data["losses"]
            if max_steps:
                losses = losses[:max_steps]

            # Ensure same length for comparison
            min_len = min(len(losses), len(baseline_losses))
            loss_diff = np.array(losses[:min_len]) - np.array(baseline_losses[:min_len])
            plt.plot(loss_diff, label=f"{name} vs {baseline_name}", linewidth=2)

    title = (
        f"{title_prefix}Loss Difference: Compared Runs vs Baseline"
        if title_prefix
        else "Loss Difference: Compared Runs vs Baseline"
    )
    plt.title(title, fontsize=20)
    plt.xlabel("Step", fontsize=16)
    plt.ylabel("Loss Difference", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=14)

    output_file = output_path / f"{file_prefix}losses_diff.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def plot_step_time(
    all_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    title_prefix: str = "",
    file_prefix: str = "",
) -> None:
    """Plot step time comparison."""
    plt.figure(figsize=(20, 10))

    for name, data in all_data.items():
        step_times = data["step_times"]
        if step_times:
            steps = range(len(step_times))
            plt.plot(steps, step_times, label=name, linewidth=2)

    title = f"{title_prefix}Step Time Comparison" if title_prefix else "Step Time Comparison"
    plt.title(title, fontsize=20)
    plt.xlabel("Step (after warmup)", fontsize=16)
    plt.ylabel("Time (ms)", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=14)

    output_file = output_path / f"{file_prefix}step_time.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


def _downsample(values: List[float], max_points: int) -> Tuple[List[int], List[float]]:
    """
    Downsample a series to at most evenly spaced points, `max_points`.

    Mermaid charts are rendered using SVG elements, so handling a large number of data points can lead to poor performance.

    Returns:
        Tuple of (x_indices, y_values) for the sampled points.
    """
    n = len(values)
    if n == 0:
        return [], []
    if n <= max_points:
        indices = list(range(n))
    else:
        # np.linspace includes both endpoints; unique() drops duplicates from rounding
        indices = sorted(set(int(round(i)) for i in np.linspace(0, n - 1, max_points)))
    return indices, [values[i] for i in indices]


# Legend palette: (emoji swatch, hex color). The hex values are pinned into
# Mermaid's plotColorPalette so line N always uses color N, and the emoji is
# chosen to visually match that hex.
_LEGEND_PALETTE: List[Tuple[str, str]] = [
    ("🟦", "#3b88c3"),
    ("🟧", "#f4900c"),
    ("🟩", "#78b159"),
    ("🟥", "#dd2e44"),
    ("🟪", "#aa8ed6"),
    ("🟨", "#fdcb58"),
    ("🟫", "#c1694f"),
    ("⬛", "#31373d"),
]


def _compute_x_range(series: List[Tuple[str, List[int], List[float]]]) -> Tuple[float, float]:
    """
    Compute the x-range to plot across all series.

    Matplotlib plots explicit (x, y) pairs, so runs of
    different lengths or step ranges align automatically. Mermaid's
    ``line [ys]`` has no x-positions. Instead, it spreads the y-values evenly
    across the whole axis. To align the runs, there should be a shared x-range and
    every series should be resampled onto a common grid.

    Preferred result is the *intersection* (the steps every series covers) so
    runs are compared only where they all have data. If that intersection is
    empty or a single point, fall back to the *union* (the full span) so the
    axis is still sensible.

    Examples (only each series' x-indices matter):
        - [0, 1, 2, 3] and [0, 1, 2, 3]  -> (0, 3)   # identical ranges
        - [0, 1, 2, 3, 4] and [2, 3, 4, 5, 6] -> (2, 4)   # overlap is steps 2-4
        - [0..99] and [0..49]            -> (0, 49)  # overlap ends with shorter run
        - [0, 1, 2] and [10, 11, 12]     -> (0, 12)  # disjoint -> union fallback
        - [5]                            -> (5, 5)   # single point -> union fallback
    """
    # Intersection: latest start and earliest end shared by every series.
    x_start = max(min(xs) for _, xs, _ in series)
    x_end = min(max(xs) for _, xs, _ in series)

    # No shared overlap (ranges are disjoint or a single point): use the union
    # span instead, from the earliest start to the latest end.
    if x_end <= x_start:
        x_start = min(min(xs) for _, xs, _ in series)
        x_end = max(max(xs) for _, xs, _ in series)
    return x_start, x_end


def _compute_padded_y_range(series: List[Tuple[str, List[float]]]) -> Tuple[float, float]:
    """
    Compute the y-axis range across all series, padded so lines aren't drawn
    flush against the top/bottom of the chart.

    Padding is 5% of the data's spread. When every value is identical the spread
    is zero, so we instead pad by 5% of the value's magnitude, and 1.0 as a last
    resort when that value is also zero (otherwise the range would be empty).

    Examples (values across all series):
        - min=0.0, max=10.0 -> pad=0.5   -> (-0.5, 10.5)   # normal spread
        - min=2.0, max=2.0  -> pad=0.1   -> (1.9, 2.1)     # flat line, pad on |value|
        - min=0.0, max=0.0  -> pad=1.0   -> (-1.0, 1.0)    # flat at zero, fallback pad
    """
    y_min = min(min(ys) for _, ys in series)
    y_max = max(max(ys) for _, ys in series)
    if y_min == y_max:
        # Flat line: no spread to base padding on, so pad by 5% of the value's
        # magnitude, falling back to 1.0 when the value itself is zero.
        pad = abs(y_min) * 0.05 or 1.0
    else:
        pad = (y_max - y_min) * 0.05
    return y_min - pad, y_max + pad


def _build_mermaid_xychart(
    title: str,
    x_label: str,
    y_label: str,
    series: List[Tuple[str, List[int], List[float]]],
) -> str:
    """
    Build a Mermaid ``xychart`` markdown code block for one or more line series.

    Mermaid xychart has no built-in legend and assigns line colors from the theme
    palette in series order. To make each line identifiable, the plot color
    palette is pinned via an init directive and a legend of matching colored-square
    emoji is emitted above the chart.

    Special functions are used to compute the x-range and y-range since Mermaid handles
    them different from Matplotlib.

    Args:
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        series: List of (label, x_indices, y_values) tuples.

    Returns:
        Markdown string containing the legend and a fenced mermaid block.
    """
    # Keep only series that have data to plot.
    series = [(label, xs, ys) for label, xs, ys in series if ys]
    if not series:
        return ""

    # Resample every series onto one shared grid so equal indices line up.
    x_start, x_end = _compute_x_range(series)
    grid_n = max(2, max(len(ys) for _, _, ys in series))
    grid = np.linspace(x_start, x_end, grid_n)
    # np.interp requires ascending sample x-values; downsampled indices are sorted.
    resampled = [(label, np.interp(grid, xs, ys).tolist()) for label, xs, ys in series]

    y_min, y_max = _compute_padded_y_range(resampled)

    # Pin each series' color so the legend emoji matches its Mermaid line color.
    swatches = [_LEGEND_PALETTE[i % len(_LEGEND_PALETTE)] for i in range(len(resampled))]
    palette = ", ".join(hex_color for _, hex_color in swatches)
    init_directive = f'%%{{init: {{"themeVariables": {{"xyChart": {{"plotColorPalette": "{palette}"}}}}}}}}%%'
    legend = "**Legend:**<br>" + "<br>".join(f"{emoji} {label}" for (emoji, _), (label, _) in zip(swatches, resampled))

    # Assemble the fenced Mermaid block.
    chart_lines = [
        "```mermaid",
        init_directive,
        "xychart",
        f'    title "{title}"',
        f'    x-axis "{x_label}" {int(round(x_start))} --> {int(round(x_end))}',
        f'    y-axis "{y_label}" {y_min:.4f} --> {y_max:.4f}',
    ]
    for _, ys in resampled:
        formatted = ", ".join(f"{v:.4f}" for v in ys)
        chart_lines.append(f"    line [{formatted}]")
    chart_lines.append("```")

    return "\n".join([legend, "", *chart_lines])


def _write_mermaid(output_dir: Path, filename: str, block: str) -> None:
    """
    Write a single Mermaid chart to a markdown file inside ``output_dir``.
    """
    # Strip any directory components so a crafted filename cannot escape output_dir.
    safe_name = os.path.basename(filename)
    base_dir = output_dir.resolve()
    output_file = (base_dir / safe_name).resolve()

    # Confine the write to base_dir (defense against traversal via symlinks/..).
    if base_dir != output_file.parent:
        raise ValueError(f"Refusing to write outside output directory: {output_file}")

    content = f"{block}\n"
    with open(output_file, "w") as f:
        f.write(content)
    print(f"Saved: {output_file}")


def export_loss_comparison_mermaid(
    all_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    title_prefix: str = "",
    max_steps: Optional[int] = None,
    max_points: int = 100,
    file_prefix: str = "",
) -> None:
    """Export loss curves for all runs as a Mermaid diagram (losses.md)."""
    series: List[Tuple[str, List[int], List[float]]] = []
    for name, data in all_data.items():
        losses = data["losses"]
        if max_steps:
            losses = losses[:max_steps]
        xs, ys = _downsample(losses, max_points)
        series.append((name, xs, ys))

    title = f"{title_prefix}Loss Comparison: All Runs"
    block = _build_mermaid_xychart(title, "Step", "Loss", series)
    if block:
        _write_mermaid(output_path, f"{file_prefix}losses.md", block)


def export_loss_difference_mermaid(
    all_data: Dict[str, Dict[str, List[float]]],
    baseline_name: str,
    output_path: Path,
    title_prefix: str = "",
    max_steps: Optional[int] = None,
    max_points: int = 100,
    file_prefix: str = "",
) -> None:
    """Export loss differences relative to baseline as a Mermaid diagram (losses_diff.md)."""
    if baseline_name not in all_data:
        print(f"Warning: Baseline '{baseline_name}' not found, skipping Mermaid loss difference export")
        return

    baseline_losses = all_data[baseline_name]["losses"]
    if max_steps:
        baseline_losses = baseline_losses[:max_steps]

    series: List[Tuple[str, List[int], List[float]]] = []
    for name, data in all_data.items():
        if name == baseline_name:
            continue
        losses = data["losses"]
        if max_steps:
            losses = losses[:max_steps]
        min_len = min(len(losses), len(baseline_losses))
        loss_diff = (np.array(losses[:min_len]) - np.array(baseline_losses[:min_len])).tolist()
        xs, ys = _downsample(loss_diff, max_points)
        series.append((f"{name} vs {baseline_name}", xs, ys))

    title = f"{title_prefix}Loss Difference: Compared Runs vs Baseline"
    block = _build_mermaid_xychart(title, "Step", "Loss Difference", series)
    if block:
        _write_mermaid(output_path, f"{file_prefix}losses_diff.md", block)


def export_step_time_mermaid(
    all_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    title_prefix: str = "",
    max_points: int = 100,
    file_prefix: str = "",
) -> None:
    """Export step time comparison as a Mermaid diagram (step_time.md)."""
    series: List[Tuple[str, List[int], List[float]]] = []
    for name, data in all_data.items():
        xs, ys = _downsample(data["step_times"], max_points)
        series.append((name, xs, ys))

    title = f"{title_prefix}Step Time Comparison"
    block = _build_mermaid_xychart(title, "Step (after warmup)", "Time (ms)", series)
    if block:
        _write_mermaid(output_path, f"{file_prefix}step_time.md", block)


def main(raw_args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Compare training logs and generate comparison plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare baseline against optimized version
    python plot_training_comparison.py --baseline run_baseline.txt --compare run_optimized.txt

    # Compare multiple runs with custom labels
    python plot_training_comparison.py --baseline baseline.txt \\
        --compare fusion_v1.txt fusion_v2.txt \\
        --labels baseline fusion-v1 fusion-v2 \\
        --title-prefix "SiLU Kernel "

    # Specify output directory and limit steps
    python plot_training_comparison.py --baseline run1.txt --compare run2.txt \\
        --output-dir ./my_plots --max-steps 5000
        """,
    )

    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline log file (used as reference for comparisons)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=[],
        help="Paths to log files to compare against baseline",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for the runs (baseline first, then compare runs). " "If not provided, filenames are used.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output plots (default: current directory)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=15,
        help="Number of warmup steps to skip for step time analysis (default: 15)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to include in loss plots (default: all)",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Prefix for plot titles (e.g., 'NanoLlama SiLU ')",
    )
    parser.add_argument(
        "--file-prefix",
        default="",
        help="Prefix prepended to every output filename (e.g. 'nano_gpt_n300_' produces "
        "'nano_gpt_n300_losses.png'). Non-alphanumeric characters (except '.', '_', '-')"
        "are replaced with '_'.",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Export the comparison charts as Mermaid diagrams in the output directory, "
        "mirroring the PNG outputs. Each file is GitHub-flavored markdown and "
        "can be appended to $GITHUB_STEP_SUMMARY.",
    )
    parser.add_argument(
        "--mermaid-max-points",
        type=int,
        default=100,
        help="Max data points to downsample per Mermaid line series. The higher the value, "
        "the closer the representation to the full PNG plot, but performance when rendering "
        "the chart may be slow. Must be at least 2. (default: 100)",
    )

    args = parser.parse_args(raw_args)

    # A chart needs at least two points; zero would silently emit no Markdown and
    # negative values would fail deep inside NumPy with an unclear error.
    if args.mermaid_max_points < 2:
        parser.error("--mermaid-max-points must be at least 2")

    # Sanitize the file prefix so it cannot introduce path separators/traversal.
    file_prefix = re.sub(r"[^A-Za-z0-9._-]", "_", args.file_prefix)

    # Collect all log files
    all_files = [args.baseline] + args.compare

    # Generate labels
    if args.labels:
        if len(args.labels) != len(all_files):
            print(f"Error: Number of labels ({len(args.labels)}) must match " f"number of files ({len(all_files)})")
            sys.exit(1)
        labels = args.labels
    else:
        labels = [Path(f).stem for f in all_files]

    # Fail fast if baseline file is missing (--baseline is required)
    if not Path(args.baseline).exists():
        print(f"Error: Baseline file not found: {args.baseline}")
        sys.exit(1)

    # Parse all log files
    print("Parsing log files...")
    all_data = {}
    for filepath, label in zip(all_files, labels):
        if not Path(filepath).exists():
            print(f"Warning: File not found: {filepath}")
            continue

        step_times, losses = parse_log(filepath, args.warmup_steps)
        all_data[label] = {"step_times": step_times, "losses": losses}
        print(f"  {label}: {len(losses)} loss values, {len(step_times)} step times")

    if not all_data:
        print("Error: No valid log files found")
        sys.exit(1)

    # Print statistics
    print_statistics(all_data)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    baseline_label = labels[0]

    plot_loss_comparison(all_data, output_path, args.title_prefix, args.max_steps, file_prefix)
    plot_step_time(all_data, output_path, args.title_prefix, file_prefix)

    if len(all_data) > 1:
        plot_loss_difference(all_data, baseline_label, output_path, args.title_prefix, args.max_steps, file_prefix)

    # Export Mermaid markdown if specified
    if args.mermaid:
        print("\nExporting Mermaid markdown...")
        export_loss_comparison_mermaid(
            all_data, output_path, args.title_prefix, args.max_steps, args.mermaid_max_points, file_prefix
        )
        export_step_time_mermaid(all_data, output_path, args.title_prefix, args.mermaid_max_points, file_prefix)

        if len(all_data) > 1:
            export_loss_difference_mermaid(
                all_data,
                baseline_label,
                output_path,
                args.title_prefix,
                args.max_steps,
                args.mermaid_max_points,
                file_prefix,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
