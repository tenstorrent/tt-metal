#!/usr/bin/env python3
"""
Compare different matrix aspect ratios (1:1:1, 1:2:1, 1:2:4, etc.) with similar time complexity.
Matrix multiplication has O(m*k*n) time complexity, so we'll generate matrices with
different aspect ratios but similar total complexity and compare their performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config_utils import load_sweep_data, get_best_config_with_storage_precedence


def get_core_grid_for_source(source):
    """Get the core grid size (X, Y) for a given source device, matching test_benchmark.py logic"""
    if source == "n150":
        return (8, 8)  # X=8, Y=8 (8x8 grid)
    elif source == "p150":
        return (13, 10)  # X=13, Y=10 (13x10 grid)
    else:
        return (8, 8)  # default


def get_base_dimensions_from_scaled(m, k, n, source):
    """
    Extract the true effective dimensions from scaled dimensions using exact test_benchmark.py logic.

    From test_benchmark.py scaling:
    m = m_base * shape_ratios[0] * grid_size[1]  # M scaled by Y (num cols)
    k = k_base * shape_ratios[1] * grid_size[0]  # K scaled by X (num rows)
    n = n_base * shape_ratios[2] * grid_size[0]  # N scaled by X (num rows)

    We need to find the effective (m_base * shape_ratios[0]) : (k_base * shape_ratios[1]) : (n_base * shape_ratios[2]) ratio.
    """
    # Get the exact grid size using test_benchmark.py logic
    grid_x, grid_y = get_core_grid_for_source(source)

    # Unscale by grid to get (base * shape_ratios) values
    # This matches the exact inverse of test_benchmark.py scaling
    effective_m = m / grid_y  # m / grid_size[1]
    effective_k = k / grid_x  # k / grid_size[0]
    effective_n = n / grid_x  # n / grid_size[0]

    return (effective_m, effective_k, effective_n)


def classify_matrix_aspect_ratio(m, k, n, source=None):
    """
    Classify matrix by its aspect ratio pattern based on test_benchmark.py shape_ratios.

    The sweep in test_benchmark.py uses these shape_ratios:
    - (1, 1, 1): Base configuration (1:1:1, 1:2:2, 1:2:4, etc. from base configs)
    - (1, 2, 4): Makes K 2x and N 4x (1:2:4, 1:4:8, 1:4:16, etc.)
    - (1, 2, 8): Makes K 2x and N 8x (1:2:8, 1:4:16, 1:4:32, etc.)
    """
    # If source is provided, get the effective dimensions (base * shape_ratios)
    if source:
        effective_m, effective_k, effective_n = get_base_dimensions_from_scaled(m, k, n, source)
    else:
        effective_m, effective_k, effective_n = m, k, n

    # Normalize by the smallest dimension to get the aspect ratio
    dims = [effective_m, effective_k, effective_n]
    min_dim = min(dims)
    normalized = [d / min_dim for d in dims]

    # Round to common ratios to handle floating point precision
    norm_m = round(normalized[0] * 2) / 2  # Round to nearest 0.5
    norm_k = round(normalized[1] * 2) / 2
    norm_n = round(normalized[2] * 2) / 2

    # Convert to integer ratios by finding a good scale
    scale = 2
    int_m = int(norm_m * scale)
    int_k = int(norm_k * scale)
    int_n = int(norm_n * scale)

    # Simplify by finding GCD
    from math import gcd

    def gcd_three(a, b, c):
        return gcd(gcd(a, b), c)

    common_gcd = gcd_three(int_m, int_k, int_n)
    if common_gcd > 0:
        int_m //= common_gcd
        int_k //= common_gcd
        int_n //= common_gcd

    # Classify based on known patterns from test_benchmark.py
    ratio_str = f"{int_m}:{int_k}:{int_n}"

    # Map to common expected patterns
    if int_m == int_k == int_n:
        return "1:1:1"
    elif (int_m, int_k, int_n) == (1, 2, 2):
        return "1:2:2"
    elif (int_m, int_k, int_n) == (1, 2, 4):
        return "1:2:4"
    elif (int_m, int_k, int_n) == (1, 2, 8):
        return "1:2:8"
    elif (int_m, int_k, int_n) == (1, 4, 8):
        return "1:4:8"
    elif (int_m, int_k, int_n) == (1, 4, 16):
        return "1:4:16"
    elif (int_m, int_k, int_n) == (1, 4, 32):
        return "1:4:32"
    elif (int_m, int_k, int_n) == (1, 1, 2):
        return "1:1:2"
    elif (int_m, int_k, int_n) == (2, 2, 1):
        return "2:2:1"
    elif (int_m, int_k, int_n) == (3, 3, 4):
        return "3:3:4"
    elif (int_m, int_k, int_n) == (3, 4, 4):
        return "3:4:4"
    else:
        # For any other ratios, use the computed ratio
        return ratio_str


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
    """
    df = df.copy()
    df["complexity"] = df["m"] * df["k"] * df["n"]
    df["aspect_ratio_pattern"] = df.apply(
        lambda row: classify_matrix_aspect_ratio(row["m"], row["k"], row["n"], source), axis=1
    )

    # Group by aspect ratio patterns
    aspect_groups = {}
    for pattern in df["aspect_ratio_pattern"].unique():
        pattern_data = df[df["aspect_ratio_pattern"] == pattern]
        if len(pattern_data) > 0:
            aspect_groups[pattern] = pattern_data

    return aspect_groups


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
        f"Matrix Aspect Ratio Performance Comparison ({source.upper()})\nM:K:N Ratios Ordered from Most Square → Most Rectangular",
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
            pattern_data = dtype_df[
                dtype_df.apply(
                    lambda row: classify_matrix_aspect_ratio(row["m"], row["k"], row["n"], source) == pattern, axis=1
                )
            ]

            if pattern_data.empty:
                performance = 0
            else:
                # Get best performance for this pattern
                best_perf = 0
                for matrix_elements in pattern_data["matrix_elements"].unique():
                    group = pattern_data[pattern_data["matrix_elements"] == matrix_elements]
                    best_config = get_best_config_with_storage_precedence(group)
                    if not best_config.empty and best_config["tflops"] > best_perf:
                        best_perf = best_config["tflops"]
                performance = best_perf

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

    # Add progressive squareness labels (Most Square → Most Rectangular)
    for pattern_idx, pattern in enumerate(sorted_patterns):
        squareness = calculate_squareness(pattern)
        if squareness <= 8.5:
            sq_label = "Most Square"
        elif squareness <= 11.0:
            sq_label = "→ More Rectangular"
        elif squareness <= 13.0:
            sq_label = "→ More Rectangular"
        elif squareness <= 17.0:
            sq_label = "→ Very Rectangular"
        else:
            sq_label = "Most Rectangular"

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
    """Create a detailed table showing aspect ratio performance"""

    df = load_sweep_data()
    source_df = df[df["source"] == source]

    print(f"\n=== {source.upper()} Aspect Ratio Performance Analysis ===")

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

        print(
            f"{'Aspect Ratio':<15} {'Squareness':<12} {'Best Matrix Shape':<25} {'Best TFLOPs':<15} {'Complexity':<15} {'Relative Perf':<15}"
        )
        print("-" * 105)

        # Get performance for each pattern
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

        pattern_results = []
        for pattern, pattern_data in aspect_groups.items():
            best_perf = 0
            best_shape = None
            best_complexity = 0

            for matrix_elements in pattern_data["matrix_elements"].unique():
                group = pattern_data[pattern_data["matrix_elements"] == matrix_elements]
                best_config = get_best_config_with_storage_precedence(group)
                if not best_config.empty and best_config["tflops"] > best_perf:
                    best_perf = best_config["tflops"]
                    best_shape = f"{best_config['m']}x{best_config['k']}x{best_config['n']}"
                    best_complexity = best_config["m"] * best_config["k"] * best_config["n"]

            if best_perf > 0:
                squareness = calculate_squareness(pattern)
                pattern_results.append((pattern, squareness, best_shape, best_perf, best_complexity))

        # Sort by squareness (most square first), then by performance
        pattern_results.sort(key=lambda x: (x[1], -x[3]))

        # Print results with relative performance
        if pattern_results:
            best_overall_perf = max(r[3] for r in pattern_results)  # Get max performance
            for pattern, squareness, shape, perf, complexity in pattern_results:
                relative_perf = perf / best_overall_perf
                squareness_label = f"{squareness:.1f}x" if squareness < 999 else "N/A"
                print(
                    f"{pattern:<15} {squareness_label:<12} {shape:<25} {perf:<15.2f} {complexity:<15.2e} {relative_perf:<15.3f}"
                )


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
