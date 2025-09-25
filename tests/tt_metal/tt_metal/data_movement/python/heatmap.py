# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tests.tt_metal.tt_metal.data_movement.python.stats_collector import StatsCollector
from tests.tt_metal.tt_metal.data_movement.python.constants import *
from tracy.process_device_log import extract_device_info
from tests.tt_metal.tt_metal.data_movement.python.config import DataMovementConfig
from tests.tt_metal.tt_metal.data_movement.python.test_metadata import TestMetadataLoader
from loguru import logger


def matrix_arch_fill(matrix, coords, val):
    for coord in coords:
        if isinstance(coord, str):
            x, y = map(int, coord.split("-"))
            matrix[y][x] = val
        else:
            raise ValueError(f"Unexpected coordinate type: {type(coord)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce heatmap data from profiler log CSV")
    parser.add_argument("-l", "--log_csv", type=str, help="Path to the log CSV file")
    parser.add_argument("-r", "--risc", choices=["riscv_0", "riscv_1"], help="RISC-V processor to analyze")
    parser.add_argument("-n", "--num_transactions", type=int, help="Number of transactions")
    parser.add_argument("-s", "--transaction_size", type=int, help="Transaction size in bytes")
    parser.add_argument("-i", "--test_id", type=int, help="Test ID to analyze")
    args = parser.parse_args()

    file_path = args.log_csv if args.log_csv else "generated/profiler/.logs/profile_log_device.csv"
    collector = StatsCollector(file_path, test_id_to_name={}, test_type_attributes={}, verbose=True)
    result, attrs = collector.gather_bw_per_core(args.num_transactions, args.transaction_size, args.risc, args.test_id)

    arch, _ = extract_device_info(file_path)
    print(f"Detected architecture: {arch}")
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "heatmap", arch)
    output_file = os.path.join(
        output_dir,
        f"heatmap_{attrs['Test id']}_{attrs['Risc']}_{attrs['Number of transactions']}_{attrs['Transaction size in bytes']}.png",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load architecture YAML
    arch_yaml_file_name = "blackhole_140_arch.yaml" if arch == "blackhole" else "wormhole_b0_80_arch.yaml"
    arch_yaml_path = os.path.join("tt_metal", "soc_descriptors", arch_yaml_file_name)
    with open(arch_yaml_path, "r") as arch_yaml_file:
        arch_data = yaml.safe_load(arch_yaml_file)

    height_cores = arch_data["grid"]["y_size"]
    width_cores = arch_data["grid"]["x_size"]
    HM_ARC = -0.000001
    HM_PCIE = -0.000002
    HM_DRAM = -0.000003
    HM_ETH = -0.000004
    HM_ROUTER = -0.000005

    # Fill in bandwidth per core
    matrix = np.zeros((height_cores, width_cores))
    for core, bw in result.items():
        if isinstance(core, tuple) and len(core) == 2:
            x, y = core
        else:
            raise ValueError(f"Unexpected core type {type(core)} for core {core}")
        matrix[y][x] = bw
        if bw > NOC_WIDTHS.get(arch, 64):
            logger.warning(f"Warning: Bandwidth {bw}b/c on core ({x}, {y}) exceeds maximum for {arch}")

    # Fill in architecture components
    matrix_arch_fill(matrix, arch_data["arc"], HM_ARC)
    matrix_arch_fill(matrix, arch_data["pcie"], HM_PCIE)
    for row in arch_data.get("dram", []):
        matrix_arch_fill(matrix, row, HM_DRAM)
    matrix_arch_fill(matrix, arch_data["eth"], HM_ETH)
    matrix_arch_fill(matrix, arch_data["router_only"], HM_ROUTER)

    bw_colors = [(0, "black"), (0.00001, "red"), (1, "green")]
    bw_cmap = LinearSegmentedColormap.from_list("bw_cmap", bw_colors)
    textcolor_threshold = matrix.max() / 2

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(DEFAULT_PLOT_WIDTH * 2.5, DEFAULT_PLOT_HEIGHT * 1.5)

    im1 = ax1.imshow(matrix, cmap=bw_cmap, vmin=0, vmax=NOC_WIDTHS.get(arch, 64))
    im2 = ax2.imshow(matrix, cmap="hot")
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar2 = fig.colorbar(im2, ax=ax2)

    for i in range(height_cores):
        for j in range(width_cores):
            if matrix[i, j] == HM_ARC:
                label = "ARC"
            elif matrix[i, j] == HM_PCIE:
                label = "PCIe"
            elif matrix[i, j] == HM_DRAM:
                label = "DRAM"
            elif matrix[i, j] == HM_ETH:
                label = "ETH"
            elif matrix[i, j] == HM_ROUTER:
                label = "RTR"
            elif matrix[i, j] == 0:
                label = "X"
            else:
                label = f"{matrix[i, j]:.2f}"
            ax1.text(j, i, label, ha="center", va="center", color="white")
            ax2.text(
                j, i, label, ha="center", va="center", color="white" if matrix[i, j] < textcolor_threshold else "black"
            )

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

    print("Bandwidth per core:")
    print(result)
    print(f"Saved heatmap to {output_file}")
