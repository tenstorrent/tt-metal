# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Memory usage analysis script for tt-train logs.
Parses memory usage summaries and compares with expected model/optimizer/gradient sizes.
"""

import argparse
import re
import sys
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_size_mb(text: str) -> float:
    """Extract size in MB from text like '+1863.25 MB' or '1863.25 MB'"""
    match = re.search(r"([+-]?\d+\.?\d*)\s*MB", text)
    if match:
        return float(match.group(1))
    return 0.0


def bytes_to_mb(bytes_val: float) -> float:
    """Convert bytes to MB"""
    return bytes_val / (1024 * 1024)


def extract_number_of_parameters(content: str, start_pos: int) -> Optional[int]:
    """Extract number of parameters from logs before the memory summary"""
    # Search forward from a reasonable position before start
    search_start = max(0, start_pos - 500)
    text_before = content[search_start:start_pos]
    match = re.search(r"Number of parameters:\s*(\d+)", text_before)
    if match:
        return int(match.group(1))
    return None


def parse_memory_section(section: str) -> Dict[str, Dict[str, float]]:
    """Parse a memory usage summary section and extract metrics"""
    metrics = {}

    # Extract metrics for each phase
    phases = ["MODEL_CREATION", "OPTIMIZER_CREATION", "FORWARD_PASS", "BACKWARD_PASS"]

    for phase in phases:
        phase_pattern = rf"--- {phase} ---\s*\n\s*DRAM: Segment Peak ([^,]+), Allocations ([^,]+), Deallocations ([^,]+), Segment Change ([^\n]+)\s*\n\s*DRAM: Cumulative Peak ([^,]+), Cumulative Current ([^\n]+)"
        match = re.search(phase_pattern, section)

        if match:
            metrics[phase] = {
                "segment_peak": parse_size_mb(match.group(1)),
                "allocations": parse_size_mb(match.group(2)),
                "deallocations": parse_size_mb(match.group(3)),
                "segment_change": parse_size_mb(match.group(4)),
                "cumulative_peak": parse_size_mb(match.group(5)),
                "cumulative_current": parse_size_mb(match.group(6)),
            }

    # Extract overall DRAM peak
    peak_match = re.search(r"Overall DRAM Peak:\s*([^\n,]+)", section)
    if peak_match:
        metrics["peak_dram"] = parse_size_mb(peak_match.group(1))

    return metrics


def find_memory_summaries(
    content: str,
) -> List[Tuple[str, Dict[str, Dict[str, float]], Optional[int]]]:
    """Find all memory usage summary sections in the log file"""
    summaries = []

    # Find all occurrences of memory usage summaries
    pattern = r"=== Memory Usage Summary ===.*?Overall DRAM Peak[^\n]*"
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        section = match.group(0)
        metrics = parse_memory_section(section)

        # Try to extract number of parameters
        num_params = extract_number_of_parameters(content, match.start())

        # Try to find a section name (e.g., "tinyllama (memory_efficient)")
        name_search_start = max(0, match.start() - 500)
        text_before = content[name_search_start : match.start()]

        # Look for the last non-empty line that's not all #'s
        lines = text_before.split("\n")
        section_name = "Unknown"
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("#") and "Number of parameters" not in line:
                section_name = line
                break

        summaries.append((section_name, metrics, num_params))

    return summaries


def calculate_percentage_diff(actual: float, expected: float) -> float:
    """Calculate percentage difference: (actual - expected) / expected * 100"""
    if expected == 0:
        return 0.0
    return ((actual - expected) / expected) * 100.0


def create_peak_memory_visualization(
    summaries_data: List[Tuple[str, Dict[str, float]]],
    title: Optional[str] = None,
    output_file: str = "memory_peak_visualization.png",
):
    """Create a visualization of peak memory usage breakdown for all summaries"""
    if not HAS_MATPLOTLIB:
        print(
            "Error: matplotlib is required for visualization. Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    if not summaries_data:
        print("No data to visualize", file=sys.stderr)
        return

    # Define colors
    colors = {
        "model": "#4da6b8",
        "optimizer": "#cec0fa",
        "activations": "#e38a42",
        "gradients_overhead": "#e789ab",
        "other": "#cccccc",
    }

    labels = {
        "model": "Model Parameters",
        "optimizer": "Optimizer State",
        "activations": "Activations",
        "gradients_overhead": "Gradients Overhead",
        "other": "Other",
    }

    num_summaries = len(summaries_data)
    fig, axes = plt.subplots(1, num_summaries, figsize=(6 * num_summaries, 8))

    # Handle single summary case
    if num_summaries == 1:
        axes = [axes]

    # Calculate common y-axis limit (max of all peaks and device memory)
    max_y = 0
    for name, breakdown in summaries_data:
        max_y = max(
            max_y, breakdown.get("total", 0.0), breakdown.get("device_memory", 0.0)
        )
    common_y_limit = max_y * 1.1

    for idx, (name, breakdown) in enumerate(summaries_data):
        ax = axes[idx]

        # Prepare data for stacked bar (bottom to top order)
        categories = [
            "other",
            "model",
            "gradients_overhead",
            "optimizer",
            "activations",
        ]
        values = [breakdown.get(cat, 0.0) for cat in categories]
        bar_colors = [colors[cat] for cat in categories]

        # Create stacked bar chart
        bottom = 0
        bars = []
        for i, (value, color) in enumerate(zip(values, bar_colors)):
            bar = ax.bar(
                0,
                value,
                bottom=bottom,
                color=color,
                width=0.5,
                edgecolor="white",
                linewidth=2,
            )
            bars.append(bar)
            bottom += value

        # Add device memory line
        device_memory_mb = breakdown.get("device_memory", 0.0)
        ax.axhline(
            y=device_memory_mb,
            color="#ff0000",
            linestyle="--",
            linewidth=2,
            label="Device Memory Limit",
        )

        # Formatting
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, common_y_limit)
        ax.set_ylabel("Memory (MB)", fontsize=12, fontweight="bold")
        ax.set_xticks([])

        # Add section name at the bottom
        ax.set_xlabel(name, fontsize=11, fontweight="bold", wrap=True)

        # Add value labels on each segment
        bottom = 0
        for value, category in zip(values, categories):
            if value > 0:
                text_y = bottom + value / 2
                ax.text(
                    0,
                    text_y,
                    f"{value:.0f} MB",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
                )
            bottom += value

        # Add total at the top
        total = breakdown.get("total", 0.0)
        ax.text(
            0,
            total,
            f"Total: {total:.0f} MB",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

        # Grid
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    # Create legend (shared for all subplots)
    legend_elements = [
        mpatches.Patch(facecolor=colors[cat], label=labels[cat])
        for cat in categories
        if any(s[1].get(cat, 0) > 0 for s in summaries_data)
    ]
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            color="#ff0000",
            linestyle="--",
            linewidth=2,
            label="Device Memory Limit",
        )
    )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=11,
        frameon=True,
    )

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
        plt.subplots_adjust(top=0.92)
    else:
        plt.subplots_adjust(top=0.95)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_file}")
    plt.close()


def analyze_memory_summary(
    name: str,
    metrics: Dict[str, Dict[str, float]],
    num_params: Optional[int],
    model_size_bytes: Optional[float],
    optimizer_size_bytes: float,
    gradients_size_bytes: Optional[float],
    device_memory_bytes: float,
    use_actual_sizes: bool = False,
) -> Dict[str, float]:
    """Analyze and print memory metrics for a single summary

    Returns a dictionary with memory breakdown for visualization

    Args:
        use_actual_sizes: If True, use measured values from logs for model/optimizer in visualization
    """
    print(f"\n{'='*80}")
    print(f"Section: {name}")
    print(f"{'='*80}")

    if num_params:
        print(f"Number of parameters: {num_params:,}")

    # Model size comparison
    if "MODEL_CREATION" in metrics:
        model_actual_mb = metrics["MODEL_CREATION"]["segment_change"]
        model_expected_mb = bytes_to_mb(model_size_bytes) if model_size_bytes else 0.0

        print(f"\n--- Model Size ---")
        print(f"  Actual (from logs):   {model_actual_mb:,.2f} MB")
        print(f"  Expected (input):     {model_expected_mb:,.2f} MB")
        if model_expected_mb > 0:
            diff_pct = calculate_percentage_diff(model_actual_mb, model_expected_mb)
            print(f"  Difference:           {diff_pct:+.2f}%")

    # Optimizer size comparison
    if "OPTIMIZER_CREATION" in metrics:
        optimizer_actual_mb = metrics["OPTIMIZER_CREATION"]["segment_change"]
        optimizer_expected_mb = bytes_to_mb(optimizer_size_bytes)

        print(f"\n--- Optimizer State ---")
        print(f"  Actual (from logs):   {optimizer_actual_mb:,.2f} MB")
        print(f"  Expected (input):     {optimizer_expected_mb:,.2f} MB")
        diff_pct = calculate_percentage_diff(optimizer_actual_mb, optimizer_expected_mb)
        print(f"  Difference:           {diff_pct:+.2f}%")

    # Activations size (FORWARD_PASS Segment Change)
    if "FORWARD_PASS" in metrics:
        activations_mb = metrics["FORWARD_PASS"]["segment_change"]
        print(f"\n--- Activations Size ---")
        print(f"  Activations:          {activations_mb:,.2f} MB")

    # Gradients size (calculated from logs)
    if "BACKWARD_PASS" in metrics and "OPTIMIZER_CREATION" in metrics:
        backward_cumulative = metrics["BACKWARD_PASS"]["cumulative_current"]
        optimizer_cumulative = metrics["OPTIMIZER_CREATION"]["cumulative_current"]
        gradients_actual_mb = backward_cumulative - optimizer_cumulative

        print(f"\n--- Gradients Size ---")
        print(f"  Actual (from logs):   {gradients_actual_mb:,.2f} MB")

        if gradients_size_bytes is not None:
            gradients_expected_mb = bytes_to_mb(gradients_size_bytes)
            print(f"  Expected (input):     {gradients_expected_mb:,.2f} MB")
            diff_pct = calculate_percentage_diff(
                gradients_actual_mb, gradients_expected_mb
            )
            print(f"  Difference:           {diff_pct:+.2f}%")

    # Peak DRAM usage
    peak_breakdown = {}
    if "peak_dram" in metrics:
        peak_dram_mb = metrics["peak_dram"]
        device_memory_mb = bytes_to_mb(device_memory_bytes)
        usage_percentage = (peak_dram_mb / device_memory_mb) * 100.0

        print(f"\n--- Peak DRAM Usage ---")
        print(f"  Peak DRAM:            {peak_dram_mb:,.2f} MB")
        print(f"  Device Memory:        {device_memory_mb:,.2f} MB")
        print(f"  Usage:                {usage_percentage:.2f}%")

        # Calculate breakdown for visualization
        if use_actual_sizes:
            # Use actual measured values from logs
            model_mb = metrics.get("MODEL_CREATION", {}).get("segment_change", 0.0)
            optimizer_mb = metrics.get("OPTIMIZER_CREATION", {}).get(
                "segment_change", 0.0
            )
        else:
            # Use theoretical values
            model_mb = bytes_to_mb(model_size_bytes) if model_size_bytes else 0.0
            optimizer_mb = bytes_to_mb(optimizer_size_bytes)

        activations_mb = metrics.get("FORWARD_PASS", {}).get("segment_change", 0.0)
        gradients_overhead_mb = metrics.get("BACKWARD_PASS", {}).get(
            "segment_peak", 0.0
        )

        # Calculate "other" as the difference
        accounted_mb = model_mb + optimizer_mb + activations_mb + gradients_overhead_mb
        other_mb = max(0.0, peak_dram_mb - accounted_mb)

        peak_breakdown = {
            "model": model_mb,
            "optimizer": optimizer_mb,
            "activations": activations_mb,
            "gradients_overhead": gradients_overhead_mb,
            "other": other_mb,
            "total": peak_dram_mb,
            "device_memory": device_memory_mb,
        }

    return peak_breakdown


def main():
    parser = argparse.ArgumentParser(
        description="Analyze memory usage from tt-train logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--logs",
        required=True,
        help="Path to log file containing memory usage summaries",
    )

    parser.add_argument(
        "--device_memory",
        type=float,
        default=12 * 1024 * 1024 * 1024,  # 12 GB default
        help="Available device memory in bytes (default: 12GB)",
    )

    parser.add_argument(
        "--model_size",
        type=float,
        help="Size of the model in bytes (can be extracted from logs if not provided, assumes bf16)",
    )

    parser.add_argument(
        "--optimizer_size",
        type=float,
        help="Size of optimizer in bytes (default: 2 * model_size, assumes bf16)",
    )

    parser.add_argument(
        "--gradients_size",
        type=float,
        help="Size of gradients in bytes (default: model_size, assumes bf16)",
    )

    parser.add_argument(
        "--visualize_peak",
        action="store_true",
        help="Create a histogram visualization of peak memory usage breakdown",
    )

    parser.add_argument(
        "--use_actual_sizes",
        action="store_true",
        help="Use actual measured values from logs for model/optimizer sizes in visualization instead of theoretical values",
    )

    parser.add_argument(
        "--title", type=str, help="Optional title for the visualization"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="memory_peak_visualization.png",
        help="Output filename for visualization (default: memory_peak_visualization.png)",
    )

    args = parser.parse_args()

    # Read log file
    with open(args.logs, "r") as f:
        content = f.read()

    # Find all memory summaries
    summaries = find_memory_summaries(content)

    if not summaries:
        raise ValueError("No memory usage summaries found in the log file")

    print(f"Found {len(summaries)} memory usage summary/summaries")

    # Determine model size
    model_size_bytes = args.model_size

    # If model size not provided, try to extract from logs
    if model_size_bytes is None:
        # Try to get from first summary with parameters
        for name, metrics, num_params in summaries:
            if num_params:
                # Assume 2 bytes per parameter (bf16)
                model_size_bytes = num_params * 2
                print(
                    f"Model size not provided, calculated from logs: {bytes_to_mb(model_size_bytes):.2f} MB ({num_params:,} parameters, assuming bf16)"
                )
                break

    if model_size_bytes is None:
        raise ValueError(
            "Error: Model size not provided and could not be extracted from logs"
        )

    # Determine optimizer size (default: 2 * model_size, bf16)
    optimizer_size_bytes = (
        args.optimizer_size
        if args.optimizer_size is not None
        else (2 * model_size_bytes)
    )

    # Gradients size (default: same as model_size for bf16, will be compared with logs)
    gradients_size_bytes = (
        args.gradients_size if args.gradients_size is not None else model_size_bytes
    )

    # Analyze each summary
    visualization_data = []
    for name, metrics, num_params in summaries:
        breakdown = analyze_memory_summary(
            name,
            metrics,
            num_params,
            model_size_bytes,
            optimizer_size_bytes,
            gradients_size_bytes,
            args.device_memory,
            args.use_actual_sizes,
        )
        if breakdown:
            visualization_data.append((name, breakdown))

    print(f"\n{'='*80}")
    print("Analysis complete")
    print(f"{'='*80}\n")

    # Create visualization if requested
    if args.visualize_peak:
        if not HAS_MATPLOTLIB:
            print(
                "Warning: Cannot create visualization. Please install matplotlib: pip install matplotlib",
                file=sys.stderr,
            )
        elif visualization_data:
            create_peak_memory_visualization(
                visualization_data, title=args.title, output_file=args.output
            )


if __name__ == "__main__":
    main()
