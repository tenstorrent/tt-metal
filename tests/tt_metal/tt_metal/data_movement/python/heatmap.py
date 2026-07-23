# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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

# Sentinels used to paint non-worker tiles (must be distinct negative values).
HM_ARC = -0.000001
HM_PCIE = -0.000002
HM_DRAM = -0.000003
HM_ETH = -0.000004
HM_ROUTER = -0.000005

_ARCH_LABELS = {
    HM_ARC: "ARC",
    HM_PCIE: "PCIe",
    HM_DRAM: "DRAM",
    HM_ETH: "ETH",
    HM_ROUTER: "RTR",
}

_METRIC_TITLES = {
    "bandwidth": "Bandwidth",
    "latency": "Latency",
    "both": "Bandwidth & Latency",
}


def matrix_arch_fill(matrix, coords, val):
    for coord in coords:
        if isinstance(coord, str):
            x, y = map(int, coord.split("-"))
            matrix[y][x] = val
        else:
            raise ValueError(f"Unexpected coordinate type: {type(coord)}")


def build_matrix(values_by_core, arch_data, height, width):
    """Place per-core values on the grid and paint non-worker tiles with sentinels."""
    matrix = np.zeros((height, width))
    for core, val in values_by_core.items():
        if isinstance(core, tuple) and len(core) == 2:
            x, y = core
        else:
            raise ValueError(f"Unexpected core type {type(core)} for core {core}")
        matrix[y][x] = val

    matrix_arch_fill(matrix, arch_data["arc"], HM_ARC)
    matrix_arch_fill(matrix, arch_data["pcie"], HM_PCIE)
    for row in arch_data.get("dram", []):
        matrix_arch_fill(matrix, row, HM_DRAM)
    matrix_arch_fill(matrix, arch_data["eth"], HM_ETH)
    matrix_arch_fill(matrix, arch_data["router_only"], HM_ROUTER)
    return matrix


def annotate(ax, matrix, fmt, text_threshold):
    height, width = matrix.shape
    for i in range(height):
        for j in range(width):
            v = matrix[i, j]
            if v in _ARCH_LABELS:
                label = _ARCH_LABELS[v]
            elif v == 0:
                label = "X"
            else:
                label = fmt(v)
            color = "white" if v < text_threshold else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)


def draw_bandwidth(ax, fig, matrix, arch, cmap_kind):
    """Per-core bandwidth panel. cmap_kind: 'capped' (NoC-width scaled) or 'hot'."""
    noc_width = NOC_WIDTHS.get(arch, 64)
    if cmap_kind == "capped":
        cmap = LinearSegmentedColormap.from_list("bw_cmap", [(0, "black"), (0.00001, "red"), (1, "green")])
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=noc_width)
        threshold = noc_width / 2
    else:
        im = ax.imshow(matrix, cmap="hot")
        threshold = matrix.max() / 2 if matrix.max() > 0 else 1
    fig.colorbar(im, ax=ax)
    ax.set_title("Per-core bandwidth (B/cyc)")
    annotate(ax, matrix, lambda v: f"{v:.2f}", threshold)


def draw_latency(ax, fig, matrix, cmap_kind):
    """Per-core latency panel. Non-worker tiles are hidden from the colour scale.

    cmap_kind 'inverse' maps low latency -> green, high -> red, i.e. the visual
    inverse of the bandwidth panel (bandwidth = data / latency), so the two maps
    should look like negatives of each other.
    """
    worker = matrix[matrix > 0]
    vmax = float(worker.max()) if worker.size else 1.0
    vmin = float(worker.min()) if worker.size else 0.0
    median = float(np.median(worker)) if worker.size else 0.0
    delta = vmax - vmin
    display = np.where(matrix < 0, np.nan, matrix)  # keep arch tiles off the scale
    if cmap_kind == "inverse":
        cmap = LinearSegmentedColormap.from_list("lat_cmap", [(0, "green"), (1, "red")])
    else:
        cmap = "hot"
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title("Per-core latency (cyc)")
    annotate(ax, matrix, lambda v: f"{v:.0f}", vmax / 2)
    stats = (
        f"Max    = {vmax:,.0f} cyc\n"
        f"Min    = {vmin:,.0f} cyc\n"
        f"Delta  = {delta:,.0f} cyc\n"
        f"Median = {median:,.0f} cyc"
    )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce heatmap data from profiler log CSV")
    parser.add_argument("-l", "--log_csv", type=str, help="Path to the log CSV file")
    parser.add_argument("-r", "--risc", choices=["riscv_0", "riscv_1"], help="RISC-V processor to analyze")
    parser.add_argument("-n", "--num_transactions", type=int, help="Number of transactions")
    parser.add_argument("-s", "--transaction_size", type=int, help="Transaction size in bytes")
    parser.add_argument("-i", "--test_id", type=int, help="Test ID to analyze")
    parser.add_argument(
        "-m",
        "--metric",
        choices=["bandwidth", "latency", "both"],
        default="bandwidth",
        help="Per-core metric(s) to plot. 'bandwidth' (default) preserves the original two-panel "
        "bandwidth map; 'latency' shows per-core latency; 'both' shows bandwidth vs latency side by side.",
    )
    args = parser.parse_args()

    file_path = args.log_csv if args.log_csv else "generated/profiler/.logs/profile_log_device.csv"
    collector = StatsCollector(file_path, test_id_to_name={}, test_type_attributes={}, verbose=True)
    bw_by_core, lat_by_core, attrs = collector.gather_bw_and_latency_per_core(
        args.num_transactions, args.transaction_size, args.risc, args.test_id
    )

    arch, _, _ = extract_device_info(file_path)
    print(f"Detected architecture: {arch}")

    # Nest outputs into data/<arch>/<group>/<subgroup>/ when the test is tagged in
    # test_information.yaml; otherwise keep the legacy data/heatmap/<arch>/ subtree.
    config = DataMovementConfig()
    metadata_loader = TestMetadataLoader(config)
    subdir = metadata_loader.get_output_subdir(attrs["Test id"])
    if subdir:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, arch, subdir)
    else:
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "heatmap", arch)
    # Preserve the original filename for the default bandwidth metric; suffix the others.
    prefix = "heatmap" if args.metric == "bandwidth" else f"heatmap_{args.metric}"
    output_file = os.path.join(
        output_dir,
        f"{prefix}_{attrs['Test id']}_{attrs['Risc']}_{attrs['Number of transactions']}_{attrs['Transaction size in bytes']}.png",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load architecture YAML
    arch_yaml_file_name = "blackhole_140_arch.yaml" if arch == "blackhole" else "wormhole_b0_80_arch.yaml"
    arch_yaml_path = os.path.join("tt_metal", "soc_descriptors", arch_yaml_file_name)
    with open(arch_yaml_path, "r") as arch_yaml_file:
        arch_data = yaml.safe_load(arch_yaml_file)

    height_cores = arch_data["grid"]["y_size"]
    width_cores = arch_data["grid"]["x_size"]

    bw_matrix = build_matrix(bw_by_core, arch_data, height_cores, width_cores)
    lat_matrix = build_matrix(lat_by_core, arch_data, height_cores, width_cores)

    # Warn about any core exceeding the NoC width (a sign of a measurement bug).
    noc_width = NOC_WIDTHS.get(arch, 64)
    for core, bw in bw_by_core.items():
        if bw > noc_width:
            logger.warning(f"Warning: Bandwidth {bw}b/c on core {core} exceeds maximum for {arch}")

    if args.metric == "latency":
        # Single panel (yellow->red). Stats box enlarged and placed on the right.
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(DEFAULT_PLOT_WIDTH * 2.2, DEFAULT_PLOT_HEIGHT * 1.6)
        stats = draw_latency(ax1, fig, lat_matrix, "hot")
        fig.subplots_adjust(right=0.72)
        fig.text(
            0.76,
            0.5,
            stats,
            fontsize=20,
            family="monospace",
            va="center",
            ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="gray"),
        )
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(DEFAULT_PLOT_WIDTH * 2.5, DEFAULT_PLOT_HEIGHT * 1.5)
        if args.metric == "bandwidth":
            draw_bandwidth(ax1, fig, bw_matrix, arch, "capped")
            draw_bandwidth(ax2, fig, bw_matrix, arch, "hot")
        else:  # both
            draw_bandwidth(ax1, fig, bw_matrix, arch, "capped")
            stats = draw_latency(ax2, fig, lat_matrix, "hot")
            ax2.set_xlabel(stats, fontsize=10, family="monospace", loc="left")

    test_id_to_name, test_id_to_comment, test_bounds, test_type_attributes = metadata_loader.get_test_mappings()
    test_name = test_id_to_name.get(attrs["Test id"], f"Test ID {attrs['Test id']}")
    fig.suptitle(f"{test_name} {_METRIC_TITLES[args.metric]} Heatmap ({arch.upper()})", fontsize=24, fontweight="bold")

    fig.supxlabel(
        f"RISCV_{attrs['Risc']}, {attrs['Number of transactions']} transactions, {attrs['Transaction size in bytes']} bytes each",
        fontsize=16,
    )
    plt.savefig(output_file)
    plt.close()

    print("Bandwidth per core:")
    print(dict(bw_by_core))
    if args.metric in ("latency", "both"):
        print("Latency per core (cyc):")
        print(dict(lat_by_core))
    print(f"Saved heatmap to {output_file}")
