import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
import glob

from matplotlib.colors import LogNorm, BoundaryNorm


def load_csv(filename):
    """Load CSV file with binary operation accuracy results"""
    return pd.read_csv(filename, sep=",", index_col=False, skipinitialspace=True)


def create_heatmap(data, operation_name, output_path):
    """
    Create a heatmap from binary operation accuracy data

    Args:
        data: DataFrame with columns 'a', 'b', 'max_ulp_error'
        operation_name: Name of the operation for the title
        output_path: Path where to save the PNG file
    """
    # Create pivot table with a on x-axis, b on y-axis, max_ulp_error as values

    print(f"DATA = \n{data}")

    pivot_data = data.pivot(columns="a", index="b", values="max_ulp_error")

    print(f"COLUMNS = {pivot_data.columns.tolist()}")
    print(f"INDEX = {pivot_data.index.tolist()}")
    # print(f"VALUES = {pivot_data.values.tolist()}")

    # Ensure the pivot table has the right dimensions (512x512)
    print(f"Pivot table shape: {pivot_data.shape}")

    # Create figure with appropriate size
    plt.figure(figsize=(12, 10))

    # Create custom colormap
    # Colors: good (green) for <2 ULP, bad (red) for >5 ULP
    # Black for NaN values
    colors = [
        "#2E8B57",  # Sea Green (good, <2 ULP)
        "#FFD700",  # Gold (moderate, 2-5 ULP)
        "#FF6347",  # Tomato (bad, >5 ULP)
        "#8B0000",
    ]  # Dark Red (very bad, >10 ULP)

    # Define color boundaries
    boundaries = [0, 2, 5, 10, 1e9]

    # Create custom colormap
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # Handle NaN values by setting them to a specific color (black)
    # First, create a mask for NaN values
    # pivot_data = pivot_data.fillna(1e9)
    # pivot_data = pivot_data.replace([np.inf, -np.inf], 1e9)

    nan_mask = np.isnan(pivot_data.values)

    print(f"NAN MASK = {nan_mask}")
    # assert np.all(nan_mask == False)

    print(f"Pivot data shape: {pivot_data.shape}")
    print(f"Pivot data: {pivot_data}")

    # Create the color mesh
    # Get the x and y coordinates for pcolormesh
    x_coords = pivot_data.columns.values
    y_coords = pivot_data.index.values

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_coords, y_coords)

    levels = [1e-6, 1, 2, 3, 4, 5, 10, 100, 100]
    cmap = plt.colormaps["PiYG"]
    cmap.set_bad(color="purple")

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # norm=LogNorm(vmin=1e-1, vmax=100)
    # Create the color mesh
    mesh = plt.pcolormesh(X, Y, pivot_data.values, norm=norm, cmap="PuBu_r", shading="auto")

    # Create colorbar
    cbar = plt.colorbar(mesh, norm=norm, label="Max ULP Error", ticks=levels)

    # Set background color for NaN values to black
    plt.gca().set_facecolor("black")

    plt.gca().set_xscale("log")
    plt.gca().set_yscale("symlog")

    plt.gca().set_xlim(1e-12, 1e12)
    plt.gca().set_ylim(-1e3, 1e3)

    # Add a horizontal red line at B == 0 (y == 0)
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="y = a**0", alpha=0.5)
    plt.text(x=plt.gca().get_xlim()[0], y=0, s="$y = a^{0}$", color="red", va="bottom", ha="left")

    # Add a vertical red line at A == 1 (x == 1)
    plt.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="y = 1**b", alpha=0.5)
    plt.text(x=2, y=plt.gca().get_ylim()[1] * 0.9, s="$y = 1^{b}$", color="red", va="top", ha="left")

    # Set aspect ratio to be square
    # plt.gca().set_aspect('equal')

    # Set title and labels
    plt.title(f"{operation_name}(A, B) - ULP Error Heatmap\n(Black = NaN or Inf)", pad=20)
    plt.xlabel("A (base)")
    plt.ylabel("B (exponent)")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def preprocess_data(data):
    """
    Remove rows where 'a' or 'b' are either infinite or NaN.
    Assumes columns are named 'a' and 'b'.
    """
    # Remove rows where 'a' or 'b' are NaN or infinite
    mask = (np.isfinite(data["a"])) & (np.isfinite(data["b"]))
    print(f"Preprocessed data shape: {data[mask].shape}")

    data = data[mask]

    # mask_useful = (
    #    (data['a'] > -1e3) & (data['a'] < 1e3) &
    #    (data['b'] > -1e3) & (data['b'] < 1e3)
    # )

    # data = data[mask_useful]

    return data


def main():
    """Main function to process all binary operation CSV files"""
    # Default input and output directories

    sns.set(
        style="ticks",
        rc={
            "axes.grid": True,
            # "axes.edgecolor": None,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "font.size": 20,
            "legend.title_fontsize": 20,
            "legend.fontsize": 20,
            "lines.linewidth": 4,
            "axes.linewidth": 1,
            "font.serif": ["Latin Modern Math"],
            "lines.markersize": 8,
            "lines.markeredgecolor": "none",
        },
    )

    input_dir = "accuracy_results/results/binary"
    output_dir = "accuracy_results/plots/binary_heatmaps"

    # Allow command line arguments to override directories
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Create output directory: {output_dir}")
        os.makedirs(output_dir)

    # Create input directory if it doesn't exist
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        os.makedirs(input_dir)

    csv_file = "accuracy_results/results/binary/pow[bfloat16].csv"
    data = load_csv(csv_file)
    data = preprocess_data(data)
    create_heatmap(data, "pow", f"{output_dir}/pow.png")

    csv_file = "accuracy_results/results/binary/pow21f[bfloat16].csv"
    data = load_csv(csv_file)
    data = preprocess_data(data)
    create_heatmap(data, "pow21f", f"{output_dir}/pow21f.png")

    print("All heatmaps generated successfully!")


if __name__ == "__main__":
    main()
