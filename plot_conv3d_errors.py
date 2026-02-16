#!/usr/bin/env python3
"""
Plot conv3d error analysis from CSV data.
Compares absolute errors across different input shapes and math fidelity levels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def plot_conv3d_errors(csv_path="conv3d_errors.csv", output_path="conv3d_errors_plot.png"):
    """
    Plot absolute errors from conv3d tests.

    Args:
        csv_path: Path to CSV file with error data
        output_path: Path to save the output PNG
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Run the conv3d tests first to generate the data.")
        sys.exit(1)

    # Read the CSV
    df = pd.read_csv(csv_path)

    if df.empty:
        print("Error: CSV file is empty")
        sys.exit(1)

    print(f"Loaded {len(df)} error records from {csv_path}")
    print(f"Shapes: {df['shape'].nunique()}")
    print(f"Math fidelities: {df['math_fidelity'].unique()}")

    # Aggregate: take max absolute error per shape and math_fidelity
    # (since we're interested in the worst case)
    agg_df = df.groupby(["shape", "math_fidelity"])["abs_error"].max().reset_index()

    # Sort shapes for better visualization
    shape_order = sorted(agg_df["shape"].unique())

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create bar plot
    sns.barplot(data=agg_df, x="shape", y="abs_error", hue="math_fidelity", order=shape_order, ax=ax)

    # Customize plot
    ax.set_xlabel("Input Shape", fontweight="bold")
    ax.set_ylabel("Maximum Absolute Error", fontweight="bold")
    ax.set_title("Conv3D Absolute Error by Input Shape and Math Fidelity", fontweight="bold")

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")

    # Adjust legend
    ax.legend(title="Math Fidelity", loc="upper left")

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    # Also print summary statistics
    print("\n=== Summary Statistics ===")
    summary = agg_df.groupby("math_fidelity")["abs_error"].describe()
    print(summary)

    plt.close()


if __name__ == "__main__":
    csv_file = "conv3d_errors.csv"
    output_file = "conv3d_errors_plot.png"

    # Allow command-line arguments
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    plot_conv3d_errors(csv_file, output_file)
