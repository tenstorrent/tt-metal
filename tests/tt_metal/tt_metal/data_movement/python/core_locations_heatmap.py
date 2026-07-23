# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Full-grid per-core heatmap for the DRAM Interleaved Page *Core Locations* sweep
(test id 62) and any other sweep that issues one run per core set.

Why this exists
---------------
``heatmap.py`` groups runs by ``(Test id, Number of transactions, Transaction
size)`` and then renders only the *first* matching ``run_host_id``. That is the
right behaviour for a single run that spans many cores (matmul, multi_interleaved,
all_to_all). The Core Locations sweep is the opposite case: it issues ~140
separate runs, each activating a single core, all sharing the identical
parameter tuple. ``heatmap.py`` therefore discards 139 of them and lights up a
single core.

This script instead walks *every* ``run_host_id`` for the selected test id and
places each run's active core(s) onto the grid, so you get a fully-populated map
of the congestion-free baseline ("Phase 1"). It emits:

  * a two-panel PNG: per-core read bandwidth (B/cyc) and per-core latency (cyc)
  * a tidy long-format CSV: one row per (run_host_id, core)

It also works for multi-core runs (row / column / cross sweeps): if a core
appears in more than one run, its values are aggregated with the median.

Run from the tt-metal repo root, e.g.:
  python tests/tt_metal/tt_metal/data_movement/python/core_locations_heatmap.py \
      -i 62 -r riscv_1
"""

import os
import argparse
import csv as csv_module
from collections import defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from loguru import logger

from tests.tt_metal.tt_metal.data_movement.python.stats_collector import StatsCollector
from tests.tt_metal.tt_metal.data_movement.python.constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOT_WIDTH,
    DEFAULT_PLOT_HEIGHT,
    NOC_WIDTHS,
)
from tests.tt_metal.tt_metal.data_movement.python.config import DataMovementConfig
from tests.tt_metal.tt_metal.data_movement.python.test_metadata import TestMetadataLoader
from tracy.process_device_log import extract_device_info

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


def normalize_core(core):
    """Return (x, y) ints for a core key that may be a tuple or an 'x-y' string."""
    if isinstance(core, (tuple, list)) and len(core) == 2:
        return int(core[0]), int(core[1])
    if isinstance(core, str) and "-" in core:
        x, y = core.split("-")
        return int(x), int(y)
    raise ValueError(f"Unexpected core key: {core!r}")


def matrix_arch_fill(matrix, coords, val):
    for coord in coords:
        x, y = normalize_core(coord)
        matrix[y][x] = val


def collect_per_core(file_path, risc, test_id, num_transactions, transaction_size, verbose):
    """Walk every run_host_id and aggregate (median) bw + latency per core."""
    collector = StatsCollector(file_path, test_id_to_name={}, test_type_attributes={}, verbose=verbose)
    _, aggregate_stats = collector.gather_analysis_stats()

    if risc not in aggregate_stats or not aggregate_stats[risc]:
        raise ValueError(f"No data for {risc}. Available: {[r for r, v in aggregate_stats.items() if v]}")

    bw_by_core = defaultdict(list)
    lat_by_core = defaultdict(list)
    rows = []  # long-format records for the CSV
    matched_runs = 0
    attrs_seen = None

    for run_host_id, data in aggregate_stats[risc].items():
        attrs = data["attributes"]
        if test_id is not None and attrs.get("Test id") != test_id:
            continue
        if num_transactions is not None and attrs.get("Number of transactions") != num_transactions:
            continue
        if transaction_size is not None and attrs.get("Transaction size in bytes") != transaction_size:
            continue

        matched_runs += 1
        attrs_seen = attrs
        cores = data["all_cores"]
        bws = data["all_bandwidths"]
        durs = data["all_durations"]
        for core, bw, dur in zip(cores, bws, durs):
            x, y = normalize_core(core)
            bw_by_core[(x, y)].append(bw)
            lat_by_core[(x, y)].append(dur)
            rows.append(
                {
                    "run_host_id": run_host_id,
                    "core_x": x,
                    "core_y": y,
                    "test_id": attrs.get("Test id"),
                    "num_transactions": attrs.get("Number of transactions"),
                    "transaction_size_bytes": attrs.get("Transaction size in bytes"),
                    "bandwidth_bytes_per_cyc": bw,
                    "latency_cyc": dur,
                }
            )

    if matched_runs == 0:
        raise ValueError("No runs matched the given test_id / num_transactions / transaction_size filters.")

    bw_med = {core: float(np.median(v)) for core, v in bw_by_core.items()}
    lat_med = {core: float(np.median(v)) for core, v in lat_by_core.items()}
    return bw_med, lat_med, rows, attrs_seen, matched_runs


def load_arch_grid(arch):
    arch_yaml_file_name = "blackhole_140_arch.yaml" if arch == "blackhole" else "wormhole_b0_80_arch.yaml"
    arch_yaml_path = os.path.join("tt_metal", "soc_descriptors", arch_yaml_file_name)
    with open(arch_yaml_path, "r") as f:
        arch_data = yaml.safe_load(f)
    return arch_data


def build_matrix(values_by_core, arch_data, height, width, paint_arch=True):
    matrix = np.zeros((height, width))
    for (x, y), val in values_by_core.items():
        matrix[y][x] = val
    if paint_arch:
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
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=6)


def main():
    parser = argparse.ArgumentParser(description="Full-grid per-core heatmap for the Core Locations sweep")
    parser.add_argument("-l", "--log_csv", type=str, default="generated/profiler/.logs/profile_log_device.csv")
    parser.add_argument("-r", "--risc", choices=["riscv_0", "riscv_1"], default="riscv_1")
    parser.add_argument("-i", "--test_id", type=int, default=62)
    parser.add_argument("-n", "--num_transactions", type=int, default=None)
    parser.add_argument("-s", "--transaction_size", type=int, default=None)
    parser.add_argument("--csv_out", type=str, default=None, help="Path for the per-core long-format CSV.")
    parser.add_argument("--png_out", type=str, default=None, help="Path for the heatmap PNG.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    bw_med, lat_med, rows, attrs, matched_runs = collect_per_core(
        args.log_csv, args.risc, args.test_id, args.num_transactions, args.transaction_size, args.verbose
    )

    arch, _, _ = extract_device_info(args.log_csv)
    print(f"Detected architecture: {arch}; matched {matched_runs} run(s), {len(bw_med)} unique core(s).")

    arch_data = load_arch_grid(arch)
    height = arch_data["grid"]["y_size"]
    width = arch_data["grid"]["x_size"]

    risc_idx = 0 if args.risc == "riscv_0" else 1
    n_txn = attrs.get("Number of transactions")
    txn_sz = attrs.get("Transaction size in bytes")

    # Warn about any core exceeding the NoC width (a sign of a measurement bug).
    noc_width = NOC_WIDTHS.get(arch, 64)
    for core, bw in bw_med.items():
        if bw > noc_width:
            logger.warning(f"Bandwidth {bw:.2f} B/cyc on core {core} exceeds NoC width {noc_width} for {arch}")

    bw_matrix = build_matrix(bw_med, arch_data, height, width)
    lat_matrix = build_matrix(lat_med, arch_data, height, width)

    fig, (ax_bw, ax_lat) = plt.subplots(1, 2)
    fig.set_size_inches(DEFAULT_PLOT_WIDTH * 2.5, DEFAULT_PLOT_HEIGHT * 1.5)

    bw_cmap = LinearSegmentedColormap.from_list("bw_cmap", [(0, "black"), (0.00001, "red"), (1, "green")])
    im_bw = ax_bw.imshow(bw_matrix, cmap=bw_cmap, vmin=0, vmax=noc_width)
    fig.colorbar(im_bw, ax=ax_bw)
    ax_bw.set_title("Per-core read bandwidth (B/cyc)")
    annotate(ax_bw, bw_matrix, lambda v: f"{v:.1f}", noc_width / 2)

    # Latency: mask non-worker tiles so the colour scale tracks worker latency only.
    lat_workers = np.array([v for v in lat_med.values()]) if lat_med else np.array([0.0])
    lat_vmax = float(lat_workers.max()) if lat_workers.size else 1.0
    lat_display = np.where(lat_matrix < 0, np.nan, lat_matrix)
    im_lat = ax_lat.imshow(lat_display, cmap="viridis", vmin=0, vmax=lat_vmax)
    fig.colorbar(im_lat, ax=ax_lat)
    ax_lat.set_title("Per-core latency (cyc)")
    annotate(ax_lat, lat_matrix, lambda v: f"{v:.0f}", lat_vmax / 2)

    fig.suptitle(
        f"DRAM Interleaved Page Core Locations (test {args.test_id}, {arch.upper()})", fontsize=20, fontweight="bold"
    )
    fig.supxlabel(f"RISCV_{risc_idx}, {n_txn} transactions, {txn_sz} bytes each — {matched_runs} runs", fontsize=14)

    # Nest outputs into data/<arch>/<group>/<subgroup>/ when the test is tagged in
    # test_information.yaml; otherwise keep the legacy data/heatmap/<arch>/ subtree.
    metadata_loader = TestMetadataLoader(DataMovementConfig())
    subdir = metadata_loader.get_output_subdir(args.test_id)
    base_dir = (
        os.path.join(DEFAULT_OUTPUT_DIR, arch, subdir) if subdir else os.path.join(DEFAULT_OUTPUT_DIR, "heatmap", arch)
    )

    png_out = args.png_out or os.path.join(base_dir, f"core_locations_{args.test_id}_{risc_idx}_{n_txn}_{txn_sz}.png")
    os.makedirs(os.path.dirname(png_out), exist_ok=True)
    plt.savefig(png_out)
    plt.close()
    print(f"Saved heatmap to {png_out}")

    csv_out = args.csv_out or os.path.join(base_dir, f"core_locations_{args.test_id}_{risc_idx}_{n_txn}_{txn_sz}.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    rows.sort(key=lambda r: (r["core_y"], r["core_x"]))
    with open(csv_out, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-core CSV to {csv_out}")


if __name__ == "__main__":
    main()
