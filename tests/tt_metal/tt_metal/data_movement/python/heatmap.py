import os
from tests.tt_metal.tt_metal.data_movement.python.stats_collector import StatsCollector
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tests.tt_metal.tt_metal.data_movement.python.plotter import Plotter
import numpy as np
from tests.tt_metal.tt_metal.data_movement.python.constants import *
from tt_metal.tools.profiler.process_device_log import extract_device_info
from tests.tt_metal.tt_metal.data_movement.python.config import DataMovementConfig
from tests.tt_metal.tt_metal.data_movement.python.test_metadata import TestMetadataLoader
from loguru import logger

if __name__ == "__main__":
    import sys  # is sys already imported by argparse?
    import argparse

    parser = argparse.ArgumentParser(description="Produce heatmap data from profiler log CSV")
    parser.add_argument("log_csv", type=str, help="Path to the log CSV file")
    # add architecture here or extract from csv?
    parser.add_argument("-r", "--risc", choices=["riscv_0", "riscv_1"], help="RISC-V processor to analyze")
    parser.add_argument("-n", "--num_transactions", type=int, help="Number of transactions")
    parser.add_argument("-s", "--transaction_size", type=int, help="Transaction size in bytes")
    parser.add_argument("-i", "--test_id", type=int, help="Test ID to analyze")
    args = parser.parse_args()

    # if len(sys.argv) < 2:
    #     print("Usage: python stats_collector.py <path_to_log_csv>")
    #     sys.exit(1)

    file_path = args.log_csv
    collector = StatsCollector(file_path, test_id_to_name={}, test_type_attributes={}, verbose=True)
    result, attrs = collector.gather_bw_per_core(args.num_transactions, args.transaction_size, args.risc, args.test_id)

    arch, _ = extract_device_info(file_path)
    print(f"Detected architecture: {arch}")
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "heatmap", arch)
    output_file = os.path.join(
        output_dir,
        f"heatmap_{attrs['Test id']}_{attrs['Risc']}_{attrs['Number of transactions']}_{attrs['Transaction size in bytes']}.png",
    )
    print(output_dir)
    print(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # extract from architecture later
    height_cores = 12
    width_cores = 17

    matrix = np.zeros((height_cores, width_cores))
    for (x, y), bw in result.items():
        # testing gradient display
        # if bw < 0.6:
        #     bw *= 20
        # elif bw < 1:
        #     bw *= 10
        # elif bw < 5:
        #     bw *= 3
        # elif bw < 11:
        #     bw *= 2
        matrix[y][x] = bw
        if bw > NOC_WIDTHS.get(arch, 64):
            logger.warning(f"Warning: Bandwidth {bw}b/c on core ({x}, {y}) exceeds maximum for {arch}")

    # matrix = np.random.rand(height_cores, width_cores) * 6 + 50
    # matrix[0][0] = 0
    # matrix[3][0] = 30

    bw_colors = [(0, "black"), (0.00001, "red"), (1, "green")]
    bw_cmap = LinearSegmentedColormap.from_list("bw_cmap", bw_colors)
    textcolor_threshold = matrix.max() / 2

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.figsize = (DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT)
    fig.set_size_inches(DEFAULT_PLOT_WIDTH * 2.5, DEFAULT_PLOT_HEIGHT * 1.5)

    im1 = ax1.imshow(matrix, cmap=bw_cmap, vmin=0, vmax=64)
    im2 = ax2.imshow(matrix, cmap="hot")
    fig.colorbar(im1)
    fig.colorbar(im2)

    for i in range(height_cores):
        for j in range(width_cores):
            label = "X" if matrix[i, j] == 0 else f"{matrix[i, j]:.2f}"
            ax1.text(j, i, label, ha="center", va="center", color="white")
            ax2.text(
                j, i, label, ha="center", va="center", color="white" if matrix[i, j] < textcolor_threshold else "black"
            )  # later try change color from black to white based on value

    config = DataMovementConfig()
    metadata_loader = TestMetadataLoader(config)
    test_id_to_name, test_id_to_comment, test_bounds, test_type_attributes = metadata_loader.get_test_mappings()
    test_name = test_id_to_name.get(attrs["Test id"], f"Test ID {attrs['Test id']}")
    # Add figure title
    fig.suptitle(f"{test_name} Heatmap ({arch.upper()})", fontsize=24, fontweight="bold")

    # Add a comment with the attributes
    fig.supxlabel(
        f"RISCV_{attrs['Risc']}, {attrs['Number of transactions']} transactions, {attrs['Transaction size in bytes']} bytes each",
        fontsize=16,
    )
    plt.savefig(output_file)
    plt.close()

    print("Printing result:")
    print(result)
