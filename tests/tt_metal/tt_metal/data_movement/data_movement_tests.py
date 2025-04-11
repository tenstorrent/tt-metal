#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
from argparse import ArgumentParser
from loguru import logger  # type: ignore
import matplotlib.pyplot as plt

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

# Corresponding test ids for each test
test_id_to_name = {0: "DRAM Interleaved Packet Sizes", 1: "DRAM Interleaved Core Locations", 2: "DRAM Sharded"}


def run_dm_tests(profile, gtest_filter):
    log_file_path = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"
    if profile or not os.path.exists(log_file_path) or gtest_filter:
        logger.info(f"Profiling Kernels...")
        cmd = f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/unit_tests_data_movement"

        if gtest_filter:
            cmd += f' --gtest_filter="*{gtest_filter}*"'

        os.system(cmd)

    # Configure post proc script
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = log_file_path
    setup.timerAnalysis = {
        "reader_kernel_analysis": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
        "writer_kernel_analysis": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
            "end": {"risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
        },
        "reader_events": {
            "across": "device",
            "type": "event",
            "marker": {"risc": "BRISC"},
        },
        "writer_events": {
            "across": "device",
            "type": "event",
            "marker": {"risc": "NCRISC"},
        },
    }

    # Gather stats from csv
    stats = import_log_run_stats(setup)
    cores = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"]
    dm_stats = {
        "reader": {
            "analysis": {"stats": dict(), "series": []},
            "attributes": dict(),
        },
        "writer": {
            "analysis": {"stats": dict(), "series": []},
            "attributes": dict(),
        },
    }

    # Gather analysis stats
    for core in cores:
        dm_stats["reader"]["analysis"]["stats"][core] = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"][
            "analysis"
        ]["reader_kernel_analysis"]["stats"]
        dm_stats["reader"]["analysis"]["series"].extend(
            stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]["reader_kernel_analysis"]["series"]
        )
        dm_stats["writer"]["analysis"]["stats"][core] = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"][
            "analysis"
        ]["writer_kernel_analysis"]["stats"]
        dm_stats["writer"]["analysis"]["series"].extend(
            stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]["writer_kernel_analysis"]["series"]
        )

    # Gather test attributes
    for kernel in dm_stats.keys():
        attributes = dm_stats[kernel]["attributes"]
        for event in stats["devices"][0]["cores"]["DEVICE"]["riscs"]["TENSIX"]["events"][kernel + "_events"]:
            run_host_id = event[0]["run_host_id"]
            if run_host_id in attributes.keys():
                attributes[run_host_id][event[0]["zone_name"]] = event[2]
            else:
                attributes[run_host_id] = {event[0]["zone_name"]: event[2]}

        dm_stats[kernel]["attributes"] = attributes

    # Stats per runtime host id
    for i in range(len(dm_stats["reader"]["analysis"]["series"])):
        run_host_id = dm_stats["reader"]["analysis"]["series"][i]["duration_type"][0]["run_host_id"]
        logger.info(f"Run host id: {run_host_id}")

        # Latency
        logger.info(f'Reader duration: {dm_stats["reader"]["analysis"]["series"][i]["duration_cycles"]}')
        logger.info(f'Writer duration: {dm_stats["writer"]["analysis"]["series"][i]["duration_cycles"]}')

        # Attributes
        logger.info(f"Attributes:")
        for attr, val in dm_stats["reader"]["attributes"][run_host_id].items():
            logger.info(f"  {attr}: {val}")
        logger.info(f"\n")

    # # # # # # Performance check method # # # # # #
    reader_cycles = dm_stats["reader"]["analysis"]["series"][0]["duration_cycles"]
    reader_cycles_lower_bound = 700
    reader_cycles_upper_bound = 900
    reader_cycles_within_bounds = reader_cycles_lower_bound <= reader_cycles <= reader_cycles_upper_bound
    reader_attributes = dm_stats["reader"]["attributes"][0]
    reader_bw = (
        reader_attributes["Number of transactions"] * reader_attributes["Transaction size in bytes"] / reader_cycles
    )
    reader_bw_lower_bound = 0.07
    reader_bw_within_bounds = reader_bw_lower_bound <= reader_bw

    if not reader_cycles_within_bounds:
        logger.warning(
            f"Reader cycles not within bounds. Received {reader_cycles}, was expecting between {reader_cycles_lower_bound} and {reader_cycles_upper_bound}"
        )
    else:
        logger.info(f"Reader cycles within bounds. Received {reader_cycles}")

    if not reader_bw_within_bounds:
        logger.warning(
            f"Reader bandwidth not within bounds. Received {reader_bw}, was expecting above {reader_bw_lower_bound}"
        )
    else:
        logger.info(f"Reader bandwidth within bounds. Received {reader_bw}")

    # assert reader_cycles_within_bounds
    # assert reader_bw_within_bounds

    plot_dm_stats(dm_stats)


def plot_dm_stats(dm_stats):
    # Extract data for plotting
    reader_series = dm_stats["reader"]["analysis"]["series"]
    writer_series = dm_stats["writer"]["analysis"]["series"]

    # Group data by Test id
    test_ids = set()
    for attributes in dm_stats["reader"]["attributes"].values():
        test_ids.add(attributes["Test id"])
    for attributes in dm_stats["writer"]["attributes"].values():
        test_ids.add(attributes["Test id"])

    test_ids = sorted(test_ids)  # Sort for consistent ordering

    # Create the main figure
    fig = plt.figure(layout="constrained", figsize=(18, 6 * len(test_ids)))

    # Create subfigures for each Test id
    subfigs = fig.subfigures(len(test_ids), 1)
    if len(test_ids) == 1:
        subfigs = [subfigs]

    for idx, (subfig, test_id) in enumerate(zip(subfigs, test_ids)):
        # Add a title for the current Test id
        test_name = test_id_to_name.get(test_id, f"Test ID {test_id}")
        subfig.suptitle(test_name, fontsize=16, weight="bold")

        # Create subplots within the subfigure
        axes = subfig.subplots(1, 3)

        # Filter data for the current Test id
        reader_filtered = [
            entry
            for entry in reader_series
            if dm_stats["reader"]["attributes"][entry["duration_type"][0]["run_host_id"]]["Test id"] == test_id
        ]
        writer_filtered = [
            entry
            for entry in writer_series
            if dm_stats["writer"]["attributes"][entry["duration_type"][0]["run_host_id"]]["Test id"] == test_id
        ]

        # Aggregate data across all runtime_host_ids for the current Test id
        reader_durations = [entry["duration_cycles"] for entry in reader_filtered]
        writer_durations = [entry["duration_cycles"] for entry in writer_filtered]

        reader_bandwidths = []
        writer_bandwidths = []
        reader_data_sizes = []
        writer_data_sizes = []

        for entry in reader_filtered:
            runtime_host_id = entry["duration_type"][0]["run_host_id"]
            attributes = dm_stats["reader"]["attributes"][runtime_host_id]
            transaction_size = attributes["Transaction size in bytes"]
            bandwidth = attributes["Number of transactions"] * transaction_size / entry["duration_cycles"]
            reader_bandwidths.append(bandwidth)
            reader_data_sizes.append(transaction_size)

        for entry in writer_filtered:
            runtime_host_id = entry["duration_type"][0]["run_host_id"]
            attributes = dm_stats["writer"]["attributes"][runtime_host_id]
            transaction_size = attributes["Transaction size in bytes"]
            bandwidth = attributes["Number of transactions"] * transaction_size / entry["duration_cycles"]
            writer_bandwidths.append(bandwidth)
            writer_data_sizes.append(transaction_size)

        # Plot durations
        ax = axes[0]
        ax.plot(reader_durations, label="Reader Duration (cycles)", marker="o")
        ax.plot(writer_durations, label="Writer Duration (cycles)", marker="o")
        ax.set_xlabel("Index")
        ax.set_ylabel("Duration (cycles)")
        ax.set_title("Kernel Durations")
        ax.legend()
        ax.grid()

        # Plot bandwidth
        ax = axes[1]
        ax.plot(reader_bandwidths, label="Reader Bandwidth (bytes/cycle)", marker="o")
        ax.plot(writer_bandwidths, label="Writer Bandwidth (bytes/cycle)", marker="o")
        ax.set_xlabel("Index")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title("Bandwidth Comparison")
        ax.legend()
        ax.grid()

        # Plot size of data transferred vs bandwidth
        ax = axes[2]
        ax.scatter(reader_data_sizes, reader_bandwidths, label="Reader", marker="o")
        ax.scatter(writer_data_sizes, writer_bandwidths, label="Writer", marker="o")
        ax.set_xlabel("Transaction Size (bytes)")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title("Data Size vs Bandwidth")
        ax.legend()
        ax.grid()

    # Save the combined plot
    plt.savefig("dm_stats_plot.png")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate reference outputs for LLaMA accuracy testing.")
    parser.add_argument("-p", "--profile", action="store_true")
    parser.add_argument("-g", "--gtest-filter", dest="gtest_filter")
    args = parser.parse_args()

    run_dm_tests(*vars(args).values())
