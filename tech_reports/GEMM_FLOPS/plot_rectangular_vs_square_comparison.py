# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Compare different matrix aspect ratios (1:1:1, 1:2:1, 1:2:4, etc.) with similar time complexity.
Matrix multiplication has O(m*k*n) time complexity, so we'll generate matrices with
different aspect ratios but similar total complexity and compare their performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config_utils import load_sweep_data, get_best_config_with_storage_precedence


def get_aspect_ratio_from_saved_columns(row):
    """
    Get aspect ratio pattern from saved CSV columns.
    Now we have direct access to aspect_ratio_m, aspect_ratio_k, aspect_ratio_n!
    """
    try:
        # Use the saved aspect ratio columns directly
        m_ratio = int(row["aspect_ratio_m"])
        k_ratio = int(row["aspect_ratio_k"])
        n_ratio = int(row["aspect_ratio_n"])
        return f"{m_ratio}:{k_ratio}:{n_ratio}"
    except (KeyError, ValueError):
        # Fallback to "1:1:1" if columns don't exist (for older data)
        return "1:1:1"


def generate_test_matrices(base_complexity):
    """
    Generate matrices with different aspect ratios but similar complexity.

    Args:
        base_complexity: Target complexity (m*k*n)

    Returns:
        List of (m, k, n, ratio_name) tuples
    """
    # Target cube root for base dimension
    base_dim = int(round(base_complexity ** (1 / 3)))

    matrices = []

    # Define aspect ratio patterns
    aspect_patterns = [
        (1, 1, 1, "1:1:1 (Square)"),
        (1, 2, 1, "1:2:1 (Wide K)"),
        (1, 1, 2, "1:1:2 (Wide N)"),
        (2, 1, 1, "2:1:1 (Wide M)"),
        (1, 2, 4, "1:2:4 (Very Rect)"),
        (1, 4, 2, "1:4:2 (Very Wide K)"),
        (2, 2, 1, "2:2:1 (Square MK)"),
        (1, 2, 2, "1:2:2 (Square KN)"),
    ]

    for m_ratio, k_ratio, n_ratio, name in aspect_patterns:
        # Calculate dimensions to achieve target complexity
        # m*k*n = base_complexity
        # m = base_dim * m_ratio * scale
        # k = base_dim * k_ratio * scale
        # n = base_dim * n_ratio * scale
        # So: (base_dim * scale)^3 * (m_ratio * k_ratio * n_ratio) = base_complexity
        # scale^3 = base_complexity / (base_dim^3 * m_ratio * k_ratio * n_ratio)

        ratio_product = m_ratio * k_ratio * n_ratio
        scale_cubed = base_complexity / (base_dim**3 * ratio_product)
        scale = scale_cubed ** (1 / 3)

        m = int(round(base_dim * m_ratio * scale))
        k = int(round(base_dim * k_ratio * scale))
        n = int(round(base_dim * n_ratio * scale))

        # Adjust to powers of 2 and reasonable sizes
        m = max(32, 2 ** round(np.log2(m)))
        k = max(32, 2 ** round(np.log2(k)))
        n = max(32, 2 ** round(np.log2(n)))

        matrices.append((m, k, n, name))

    return matrices


def find_similar_aspect_ratios_in_data(df, source=None):
    """
    Find matrices in the actual data that match different aspect ratio patterns.
    Now using saved aspect ratio columns - much simpler!
    """
    df = df.copy()
    df["complexity"] = df["m"] * df["k"] * df["n"]

    # Use the aspect_ratio_pattern column if it exists (from config_utils)
    # Otherwise create it from the saved columns
    if "aspect_ratio_pattern" not in df.columns:
        df["aspect_ratio_pattern"] = df.apply(get_aspect_ratio_from_saved_columns, axis=1)

    # Group by aspect ratio patterns
    aspect_groups = {}
    for pattern in df["aspect_ratio_pattern"].unique():
        pattern_data = df[df["aspect_ratio_pattern"] == pattern]
        if len(pattern_data) > 0:
            aspect_groups[pattern] = pattern_data

    return aspect_groups


def get_focused_performance_for_pattern(pattern, source_df, source):
    """
    Get performance for a specific pattern using the same focused approach as table analysis.
    Returns performance from the smallest bucket containing all three patterns.
    """
    # Find aspect ratio groups
    aspect_groups = find_similar_aspect_ratios_in_data(source_df, source)

    # Focus on smaller matrices (bottom 60% of complexity range)
    all_complexities = []
    for pattern_data in aspect_groups.values():
        all_complexities.extend(pattern_data["matrix_elements"].tolist())

    sorted_complexities = sorted(set(all_complexities))

    if len(sorted_complexities) > 5:
        max_complexity_for_focus = sorted_complexities[int(len(sorted_complexities) * 0.6)]
        small_complexities = [c for c in sorted_complexities if c <= max_complexity_for_focus]
    else:
        small_complexities = sorted_complexities

    # Create complexity buckets
    complexity_buckets = []
    if small_complexities:
        current_bucket = [small_complexities[0]]
        bucket_min = small_complexities[0]

        for complexity in small_complexities[1:]:
            if complexity <= bucket_min * 3:
                current_bucket.append(complexity)
            else:
                if len(current_bucket) >= 2:
                    complexity_buckets.append(current_bucket)
                current_bucket = [complexity]
                bucket_min = complexity

        if len(current_bucket) >= 2:
            complexity_buckets.append(current_bucket)

    # Find smallest bucket with all three patterns
    target_patterns = {"1:1:1", "1:2:4", "1:2:8"}
    smallest_complete_bucket = None

    for complexity_bucket in complexity_buckets:
        bucket_patterns = set()
        for p, pattern_data in aspect_groups.items():
            bucket_group = pattern_data[pattern_data["matrix_elements"].isin(complexity_bucket)]
            if not bucket_group.empty:
                bucket_patterns.add(p)

        if target_patterns.issubset(bucket_patterns):
            smallest_complete_bucket = complexity_bucket
            break

    # Get performance for the requested pattern from the focused bucket
    if smallest_complete_bucket is not None and pattern in aspect_groups:
        pattern_data = aspect_groups[pattern]
        bucket_group = pattern_data[pattern_data["matrix_elements"].isin(smallest_complete_bucket)]
        if not bucket_group.empty:
            best_config = get_best_config_with_storage_precedence(bucket_group)
            if not best_config.empty:
                return best_config["tflops"]

    return 0  # Return 0 if no focused data available


def plot_aspect_ratio_comparison(source):
    """Create single comparison plot for different aspect ratios with different colored bars for each dtype"""

    df = load_sweep_data()
    source_df = df[df["source"] == source]

    # Find aspect ratio groups in actual data
    aspect_groups = find_similar_aspect_ratios_in_data(source_df, source)

    print(f"\n=== Aspect ratio patterns found in {source.upper()} data ===")
    for pattern, data in aspect_groups.items():
        print(f"{pattern}: {len(data)} configurations")
        # Show sample matrix dimensions for this pattern
        sample = data.iloc[0]
        print(f"  Sample: {sample['m']}x{sample['k']}x{sample['n']}")

    # Define dtype configurations with colors
    dtype_configs = [
        ("BFLOAT4_B_LoFi", "#1f77b4", "BFLOAT4_B LoFi"),
        ("BFLOAT8_B_HiFi2", "#ff7f0e", "BFLOAT8_B HiFi2"),
        ("BFLOAT16_HiFi4", "#2ca02c", "BFLOAT16 HiFi4"),
    ]

    # Get all unique aspect ratio patterns and sort by squareness
    def calculate_squareness(pattern_str):
        """Calculate how 'rectangular' a pattern is. Lower = more square."""
        try:
            parts = pattern_str.split(":")
            ratios = [float(p) for p in parts]
            max_ratio = max(ratios)
            min_ratio = min(ratios)
            return max_ratio / min_ratio
        except:
            return 999

    all_patterns = list(aspect_groups.keys())
    pattern_squareness = [(p, calculate_squareness(p)) for p in all_patterns]
    pattern_squareness.sort(key=lambda x: x[1])  # Sort by squareness
    sorted_patterns = [p[0] for p in pattern_squareness]

    # Create single larger plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.suptitle(
        f"Matrix Aspect Ratio Performance Comparison ({source.upper()})\nM:K:N Ratios: Square (1:1:1) → Moderately Rectangular (1:2:4) → Very Rectangular (1:2:8)",
        fontsize=16,
    )

    # Calculate bar positions
    x_positions = []
    bar_width = 0.25
    group_width = len(dtype_configs) * bar_width

    for i, pattern in enumerate(sorted_patterns):
        base_x = i * (group_width + 0.1)  # Add spacing between groups
        for j in range(len(dtype_configs)):
            x_positions.append(base_x + j * bar_width)

    # Plot bars for each dtype
    all_bars = []
    all_performances = []
    all_labels = []
    all_squareness = []

    for dtype_idx, (dtype_fidelity, color, label) in enumerate(dtype_configs):
        dtype_df = source_df[source_df["dtype_fidelity"] == dtype_fidelity]

        for pattern_idx, pattern in enumerate(sorted_patterns):
            # Find data for this pattern and dtype
            if "aspect_ratio_pattern" in dtype_df.columns:
                pattern_data = dtype_df[dtype_df["aspect_ratio_pattern"] == pattern]
            else:
                pattern_data = dtype_df[dtype_df.apply(get_aspect_ratio_from_saved_columns, axis=1) == pattern]

            if pattern_data.empty:
                performance = 0
            else:
                # Use the same focused approach as table analysis - smallest bucket with all patterns
                performance = get_focused_performance_for_pattern(pattern, dtype_df, source)

            # Calculate position
            x_pos = pattern_idx * (group_width + 0.1) + dtype_idx * bar_width

            if performance > 0:
                bar = ax.bar(
                    x_pos, performance, bar_width, color=color, alpha=0.8, label=label if pattern_idx == 0 else ""
                )
                all_bars.append(bar[0])
                all_performances.append(performance)

                # Add value label on bar
                ax.text(
                    x_pos,
                    performance + max(all_performances if all_performances else [performance]) * 0.01,
                    f"{performance:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    # Add specific aspect ratio labels
    for pattern_idx, pattern in enumerate(sorted_patterns):
        if pattern == "1:1:1":
            sq_label = "Square Matrices"
        elif pattern == "1:2:4":
            sq_label = "Moderately Rectangular"
        elif pattern == "1:2:8":
            sq_label = "Very Rectangular"
        else:
            # Fallback for other patterns
            squareness = calculate_squareness(pattern)
            if squareness <= 2.0:
                sq_label = "Square-ish"
            elif squareness <= 5.0:
                sq_label = "Moderately Rectangular"
            else:
                sq_label = "Very Rectangular"

        group_center = pattern_idx * (group_width + 0.1) + group_width / 2 - bar_width / 2
        max_perf = max(all_performances) if all_performances else 100
        ax.text(
            group_center,
            -max_perf * 0.08,
            sq_label,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color="darkgray",
        )

    # Set x-tick labels
    x_tick_positions = [i * (group_width + 0.1) + group_width / 2 - bar_width / 2 for i in range(len(sorted_patterns))]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(sorted_patterns, rotation=45, ha="right", fontsize=11)

    # Formatting
    ax.set_xlabel("Aspect Ratio Pattern (M:K:N)", fontsize=12)
    ax.set_ylabel("Best TFLOPs", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust y-axis to make room for sub-labels
    if all_performances:
        max_perf = max(all_performances)
        ax.set_ylim(-max_perf * 0.15, max_perf * 1.15)

    plt.tight_layout()
    plt.savefig(f"tech_reports/GEMM_FLOPS/images/aspect_ratio_comparison_{source}.png", bbox_inches="tight", dpi=150)
    plt.close()


def create_detailed_aspect_ratio_table(source):
    """Create a detailed table showing aspect ratio performance with fair complexity comparison"""

    df = load_sweep_data()
    source_df = df[df["source"] == source]

    print(f"\n=== {source.upper()} Aspect Ratio Performance Analysis (Prioritizing Smaller Matrices) ===")

    dtype_configs = ["BFLOAT4_B_LoFi", "BFLOAT8_B_HiFi2", "BFLOAT16_HiFi4"]

    for dtype_fidelity in dtype_configs:
        print(f"\n--- {dtype_fidelity} ---")
        dtype_df = source_df[source_df["dtype_fidelity"] == dtype_fidelity]

        if dtype_df.empty:
            print("No data available")
            continue

        # Find aspect ratio groups for this dtype
        aspect_groups = find_similar_aspect_ratios_in_data(dtype_df, source)

        if not aspect_groups:
            print("No aspect ratio patterns found")
            continue

        print(f"{'Complexity Range':<25} {'Aspect Ratios Compared':<50} {'Performance Impact':<20}")
        print("-" * 95)

        # Focus on smaller matrices first - they show clearer aspect ratio effects
        all_complexities = []
        for pattern_data in aspect_groups.values():
            all_complexities.extend(pattern_data["matrix_elements"].tolist())

        sorted_complexities = sorted(set(all_complexities))

        # Filter to focus on smaller matrices (bottom 60% of complexity range)
        if len(sorted_complexities) > 5:
            max_complexity_for_focus = sorted_complexities[int(len(sorted_complexities) * 0.6)]
            small_complexities = [c for c in sorted_complexities if c <= max_complexity_for_focus]
            print(
                f"Focusing on smaller matrices: {len(small_complexities)} complexity levels (up to {max_complexity_for_focus:.1e})"
            )
        else:
            small_complexities = sorted_complexities
            print(
                f"Using all available complexity levels: {len(small_complexities)} (up to {max(small_complexities):.1e})"
            )

        # Create logarithmic buckets - group complexities within 3x of each other for smaller matrices
        import math

        complexity_buckets = []

        if small_complexities:
            current_bucket = [small_complexities[0]]
            bucket_min = small_complexities[0]

            for complexity in small_complexities[1:]:
                # If complexity is within 3x of bucket minimum, add to current bucket
                if complexity <= bucket_min * 3:  # Tighter range for small matrices
                    current_bucket.append(complexity)
                else:
                    # Start new bucket
                    if len(current_bucket) >= 2:  # Only keep buckets with multiple complexities
                        complexity_buckets.append(current_bucket)
                    current_bucket = [complexity]
                    bucket_min = complexity

            # Add the last bucket
            if len(current_bucket) >= 2:
                complexity_buckets.append(current_bucket)

        # Sort by bucket size (data richness) and prioritize smaller matrices
        def bucket_priority(bucket):
            avg_complexity = sum(bucket) / len(bucket)
            data_richness = len(bucket)
            # Lower complexity = higher priority, more data = higher priority
            return (-avg_complexity, data_richness)

        complexity_buckets.sort(key=bucket_priority, reverse=True)
        target_buckets = complexity_buckets[:4]  # Top 4 buckets prioritizing smaller matrices

        def calculate_squareness(pattern_str):
            """Calculate how 'rectangular' a pattern is. Lower = more square."""
            try:
                parts = pattern_str.split(":")
                ratios = [float(p) for p in parts]
                max_ratio = max(ratios)
                min_ratio = min(ratios)
                return max_ratio / min_ratio
            except:
                return 999

        # Find the smallest bucket that has all three aspect ratio patterns for fair comparison
        target_patterns = {"1:1:1", "1:2:4", "1:2:8"}
        smallest_complete_bucket = None

        for complexity_bucket in complexity_buckets:
            # Check if this bucket has all three patterns
            bucket_patterns = set()
            for pattern, pattern_data in aspect_groups.items():
                bucket_group = pattern_data[pattern_data["matrix_elements"].isin(complexity_bucket)]
                if not bucket_group.empty:
                    bucket_patterns.add(pattern)

            # If this bucket has all three patterns, it's our target
            if target_patterns.issubset(bucket_patterns):
                smallest_complete_bucket = complexity_bucket
                break

        if smallest_complete_bucket is not None:
            min_complexity = min(smallest_complete_bucket)
            max_complexity = max(smallest_complete_bucket)
            range_ratio = max_complexity / min_complexity
            print(f"\nSmallest complete bucket: {min_complexity:.1e} - {max_complexity:.1e} ({range_ratio:.1f}x range)")
            print("(Contains all three aspect ratios: 1:1:1, 1:2:4, 1:2:8)")

            # Get best performance for each aspect ratio within this complexity range
            bucket_results = []
            for pattern in ["1:1:1", "1:2:4", "1:2:8"]:  # Force specific order
                if pattern in aspect_groups:
                    pattern_data = aspect_groups[pattern]
                    bucket_group = pattern_data[pattern_data["matrix_elements"].isin(smallest_complete_bucket)]
                    if not bucket_group.empty:
                        best_config = get_best_config_with_storage_precedence(bucket_group)
                        if not best_config.empty:
                            squareness = calculate_squareness(pattern)
                            shape = f"{best_config['m']}x{best_config['k']}x{best_config['n']}"
                            complexity = best_config["matrix_elements"]
                            bucket_results.append((pattern, squareness, best_config["tflops"], shape, complexity))

            if len(bucket_results) == 3:  # All three patterns
                # Square is always first (1:1:1)
                baseline_perf = bucket_results[0][2]  # Square performance

                print(f"  {'Pattern':<12} {'Shape':<25} {'TFLOPs':<10} {'Complexity':<12} {'vs Square':<12}")
                for pattern, squareness, perf, shape, complexity in bucket_results:
                    relative_perf = (perf / baseline_perf - 1) * 100  # Percentage change
                    print(f"  {pattern:<12} {shape:<25} {perf:<10.2f} {complexity:<12.1e} {relative_perf:+6.1f}%")
            else:
                print(f"  Could not find complete data for all three patterns")
        else:
            print(f"  No bucket found with all three aspect ratio patterns")

        print(f"\nSummary: Focused on smallest matrices with complete aspect ratio comparison")


if __name__ == "__main__":
    # Create plots for both devices
    for source in ["n150", "p150"]:
        print(f"Analyzing aspect ratios for {source.upper()}...")
        plot_aspect_ratio_comparison(source)
        create_detailed_aspect_ratio_table(source)

    print(f"\nAspect ratio comparison plots saved!")
    print("Files created:")
    print("- tech_reports/GEMM_FLOPS/images/aspect_ratio_comparison_n150.png")
    print("- tech_reports/GEMM_FLOPS/images/aspect_ratio_comparison_p150.png")
