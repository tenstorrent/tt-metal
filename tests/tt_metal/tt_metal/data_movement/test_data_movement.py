#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import csv
import sys
from loguru import logger  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import itertools

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

# Corresponding test ids for each test
test_id_to_name = {
    0: "DRAM Interleaved Packet Sizes",
    1: "DRAM Interleaved Core Locations",
    2: "DRAM Sharded",
    3: "DRAM Directed Ideal",
    4: "One to One Packet Sizes",
    5: "One from One Packet Sizes",
    6: "One to All 2x2 Packet Sizes",
    7: "One to All 4x4 Packet Sizes",
    8: "One to All 10x10 Packet Sizes",
    9: "One to All Multicast 2x2 Packet Sizes",
    10: "One to All Multicast 5x5 Packet Sizes",
    11: "One to All Multicast 11x10 Packet Sizes",
    12: "One to All Multicast Linked 2x2 Packet Sizes",
    13: "One to All Multicast Linked 5x5 Packet Sizes",
    14: "One to All Multicast Linked 11x10 Packet Sizes",
    15: "One from All Packet Sizes",
    16: "Loopback Packet Sizes",
}

# Correspondng test bounds for each arch, test id, riscv core
# NOTE: These bounds are aggregated averages of large test sweeps and
# are subject to change with new directed tests.
test_bounds = {
    "wormhole_b0": {
        0: {
            "riscv_1": {"latency": {"lower": 500, "upper": 42000}, "bandwidth": 0.12},
            "riscv_0": {"latency": {"lower": 400, "upper": 28000}, "bandwidth": 0.15},
        },
        1: {
            "riscv_1": {"latency": {"lower": 400, "upper": 700}, "bandwidth": 0.19},
            "riscv_0": {"latency": {"lower": 300, "upper": 500}, "bandwidth": 0.29},
        },
        2: {
            "riscv_1": {"latency": {"lower": 500, "upper": 600}, "bandwidth": 0.24},
            "riscv_0": {"latency": {"lower": 400, "upper": 500}, "bandwidth": 0.30},
        },
        3: {
            "riscv_1": {"latency": {"lower": 33000, "upper": 35000}, "bandwidth": 22},
            "riscv_0": {"latency": {"lower": 33000, "upper": 35000}, "bandwidth": 21},
        },
        4: {
            "riscv_1": {"latency": {"lower": 4000, "upper": 12000}, "bandwidth": 0.007},
            "riscv_0": {"latency": {"lower": 300, "upper": 4700}, "bandwidth": 0.17},
        },
        5: {
            "riscv_1": {"latency": {"lower": 200, "upper": 5000}, "bandwidth": 0.1},
        },
        6: {
            "riscv_0": {"latency": {"lower": 200, "upper": 70000}, "bandwidth": 0.4},
        },
        7: {
            "riscv_0": {"latency": {"lower": 200, "upper": 30000}, "bandwidth": 1.0},
        },
        8: {
            "riscv_0": {"latency": {"lower": 2000, "upper": 2000000}, "bandwidth": 1.0},
        },
        9: {
            "riscv_0": {"latency": {"lower": 200, "upper": 300000}, "bandwidth": 0.1},
        },
        10: {
            "riscv_0": {"latency": {"lower": 200, "upper": 70000}, "bandwidth": 0.1},
        },
        11: {
            "riscv_0": {"latency": {"lower": 200, "upper": 200000}, "bandwidth": 0.04},
        },
        12: {
            "riscv_0": {"latency": {"lower": 200, "upper": 30000}, "bandwidth": 0.1},
        },
        13: {
            "riscv_0": {"latency": {"lower": 200, "upper": 30000}, "bandwidth": 0.1},
        },
        14: {
            "riscv_0": {"latency": {"lower": 200, "upper": 100000}, "bandwidth": 0.04},
        },
        15: {
            "riscv_1": {"latency": {"lower": 700, "upper": 120000}, "bandwidth": 0.7},
        },
        16: {
            "riscv_0": {"latency": {"lower": 50, "upper": 30000}, "bandwidth": 0.4},
        },
    },
    "blackhole": {
        0: {
            "riscv_1": {"latency": {"lower": 500, "upper": 42000}, "bandwidth": 0.12},
            "riscv_0": {"latency": {"lower": 400, "upper": 28000}, "bandwidth": 0.15},
        },
        1: {
            "riscv_1": {"latency": {"lower": 300, "upper": 700}, "bandwidth": 0.17},
            "riscv_0": {"latency": {"lower": 200, "upper": 500}, "bandwidth": 0.23},
        },
        2: {
            "riscv_1": {"latency": {"lower": 400, "upper": 600}, "bandwidth": 0.13},
            "riscv_0": {"latency": {"lower": 300, "upper": 500}, "bandwidth": 0.16},
        },
        3: {
            "riscv_1": {"latency": {"lower": 42000, "upper": 44000}, "bandwidth": 33},
            "riscv_0": {"latency": {"lower": 42000, "upper": 44000}, "bandwidth": 34},
        },
        4: {
            "riscv_1": {"latency": {"lower": 4000, "upper": 12000}, "bandwidth": 0.007},
            "riscv_0": {"latency": {"lower": 300, "upper": 4700}, "bandwidth": 0.17},
        },
        5: {
            "riscv_1": {"latency": {"lower": 300, "upper": 4700}, "bandwidth": 0.17},
        },
        6: {
            "riscv_0": {"latency": {"lower": 200, "upper": 70000}, "bandwidth": 0.4},
        },
        7: {
            "riscv_0": {"latency": {"lower": 200, "upper": 300000}, "bandwidth": 1.0},
        },
        8: {
            "riscv_0": {"latency": {"lower": 2000, "upper": 2000000}, "bandwidth": 1.0},
        },
        9: {
            "riscv_0": {"latency": {"lower": 200, "upper": 30000}, "bandwidth": 0.1},
        },
        10: {
            "riscv_0": {"latency": {"lower": 200, "upper": 70000}, "bandwidth": 0.1},
        },
        11: {
            "riscv_0": {"latency": {"lower": 200, "upper": 200000}, "bandwidth": 0.04},
        },
        12: {
            "riscv_0": {"latency": {"lower": 200, "upper": 30000}, "bandwidth": 0.1},
        },
        13: {
            "riscv_0": {"latency": {"lower": 200, "upper": 30000}, "bandwidth": 0.1},
        },
        14: {
            "riscv_0": {"latency": {"lower": 200, "upper": 100000}, "bandwidth": 0.04},
        },
        15: {
            "riscv_1": {"latency": {"lower": 800, "upper": 135000}, "bandwidth": 1.19},
        },
        16: {
            "riscv_0": {"latency": {"lower": 50, "upper": 30000}, "bandwidth": 0.4},
        },
    },
}


def run_dm_tests(profile, verbose, gtest_filter, plot, report, arch_name):
    logger.info("Starting data movement tests...")
    log_file_path = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"

    # Get architecture
    arch = get_arch(arch_name)
    if verbose:
        logger.info(f"Running data movement tests on architecture: {arch}")

    # Profile tests
    if profile or not os.path.exists(log_file_path) or gtest_filter:
        profile_dm_tests(verbose=verbose, gtest_filter=gtest_filter)

    # Gather analysis stats
    dm_stats = gather_analysis_stats(log_file_path, verbose=verbose)

    # Print stats if explicitly requested
    if verbose:
        print_stats(dm_stats)

    # Plot results
    if plot:
        plot_dm_stats(dm_stats)

    # Export results to csv
    if report:
        export_dm_stats_to_csv(dm_stats)

    # Check performance (TODO: enable assertions)
    performance_check(dm_stats, arch=arch, verbose=verbose)

    logger.info("Data movement tests completed.")


def get_arch(arch_name):
    # Get architecture from environment variable or command line argument
    if arch_name:
        return arch_name

    arch = os.environ.get("ARCH_NAME", None)
    if arch is None:
        logger.warning("ARCH_NAME environment variable is not set, defaulting to 'blackhole'.")
        return "blackhole"
    elif arch not in test_bounds.keys():
        logger.error(f"ARCH_NAME '{arch}' is not recognized.")
        sys.exit(1)
    return arch


def profile_dm_tests(verbose=False, gtest_filter=None):
    if verbose:
        logger.info(f"Profiling Kernels...")
    cmd = f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/unit_tests_data_movement"

    if gtest_filter:
        cmd += f' --gtest_filter="*{gtest_filter}*"'

    os.system(cmd)


def gather_analysis_stats(file_path, verbose=False):
    # Gather stats from csv and set up analysis
    stats = gather_stats_from_csv(file_path, verbose=verbose)
    cores = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"]
    dm_stats = {
        "riscv_1": {
            "analysis": {"stats": dict(), "series": []},
            "attributes": dict(),
        },
        "riscv_0": {
            "analysis": {"stats": dict(), "series": []},
            "attributes": dict(),
        },
    }

    # Gather analysis stats
    # Statistics are recorded per core, but timeseries data is aggregated for all cores
    for core in cores:
        core_analysis = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]
        if "riscv_1_analysis" in core_analysis.keys():
            dm_stats["riscv_1"]["analysis"]["stats"][core] = core_analysis["riscv_1_analysis"]["stats"]
            dm_stats["riscv_1"]["analysis"]["series"].extend(core_analysis["riscv_1_analysis"]["series"])

        if "riscv_0_analysis" in core_analysis.keys():
            dm_stats["riscv_0"]["analysis"]["stats"][core] = core_analysis["riscv_0_analysis"]["stats"]
            dm_stats["riscv_0"]["analysis"]["series"].extend(core_analysis["riscv_0_analysis"]["series"])

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

    return dm_stats


def gather_stats_from_csv(file_path, verbose=False):
    # Configure post proc script
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = file_path
    setup.timerAnalysis = {
        "riscv_1_analysis": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "zone_name": "RISCV1"},
            "end": {"risc": "ANY", "zone_name": "RISCV1"},
        },
        "riscv_0_analysis": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "zone_name": "RISCV0"},
            "end": {"risc": "ANY", "zone_name": "RISCV0"},
        },
        "riscv_0_events": {
            "across": "device",
            "type": "event",
            "marker": {"risc": "BRISC"},
        },
        "riscv_1_events": {
            "across": "device",
            "type": "event",
            "marker": {"risc": "NCRISC"},
        },
    }

    # Gather stats from csv
    if not verbose:
        logger.disable("tt_metal.tools.profiler.process_device_log")

    return import_log_run_stats(setup)


def performance_check(dm_stats, arch="blackhole", verbose=False):
    # Tidy results' ranges
    results_bounds = {}
    for riscv in dm_stats.keys():
        for run in dm_stats[riscv]["analysis"]["series"]:
            run_host_id = run["duration_type"][0]["run_host_id"]
            test_id = dm_stats[riscv]["attributes"][run_host_id]["Test id"]
            if test_id not in results_bounds.keys():
                results_bounds[test_id] = {
                    riscv: {"latency": {"lower": float("inf"), "upper": 0}, "bandwidth": float("inf")}
                }
            elif riscv not in results_bounds[test_id].keys():
                results_bounds[test_id][riscv] = {
                    "latency": {"lower": float("inf"), "upper": 0},
                    "bandwidth": float("inf"),
                }

            cycles = run["duration_cycles"]
            results_bounds[test_id][riscv]["latency"]["lower"] = min(
                results_bounds[test_id][riscv]["latency"]["lower"], cycles
            )
            results_bounds[test_id][riscv]["latency"]["upper"] = max(
                results_bounds[test_id][riscv]["latency"]["upper"], cycles
            )

            attributes = dm_stats[riscv]["attributes"][run_host_id]
            bandwidth = attributes["Number of transactions"] * attributes["Transaction size in bytes"] / cycles
            results_bounds[test_id][riscv]["bandwidth"] = min(results_bounds[test_id][riscv]["bandwidth"], bandwidth)

    # Performance checks per test
    for test_id, bounds in results_bounds.items():
        # Print latency and bandwidth perf results
        if verbose:
            logger.info("")
            logger.info(f"Perf results for test id: {test_id}")
            logger.info(f"Latency")
            for riscv in bounds.keys():
                if bounds[riscv]["latency"]["lower"] != float("inf"):
                    logger.info(
                        f"  {riscv}: {bounds[riscv]['latency']['lower']}-{bounds[riscv]['latency']['upper']} cycles"
                    )

            logger.info(f"Bandwidth")
            for riscv in bounds.keys():
                if bounds[riscv]["bandwidth"] != float("inf"):
                    logger.info(f"  {riscv}: {bounds[riscv]['bandwidth']} Bytes/cycle")

        if test_id not in test_bounds[arch].keys():
            logger.warning(f"Test id {test_id} not found in {arch} test bounds.")
            continue

        for riscv in bounds.keys():
            if riscv not in test_bounds[arch][test_id].keys():
                continue
            cycles_within_bounds = (
                test_bounds[arch][test_id][riscv]["latency"]["lower"] <= bounds[riscv]["latency"]["lower"]
                and bounds[riscv]["latency"]["upper"] <= test_bounds[arch][test_id][riscv]["latency"]["upper"]
            )
            bw_within_bounds = test_bounds[arch][test_id][riscv]["bandwidth"] <= bounds[riscv]["bandwidth"]

            # Print bounds check results
            if verbose:
                if not cycles_within_bounds:
                    logger.warning(f"{riscv} cycles not within perf bounds.")
                else:
                    logger.info(f"{riscv} cycles within perf bounds.")
                if not bw_within_bounds:
                    logger.warning(f"{riscv} bandwidth not within perf bounds.")
                else:
                    logger.info(f"{riscv} bandwidth within perf bounds.")

            assert cycles_within_bounds
            assert bw_within_bounds


def print_stats(dm_stats):
    # Print stats per runtime host id
    for riscv1_run, riscv0_run in itertools.zip_longest(
        dm_stats["riscv_1"]["analysis"]["series"], dm_stats["riscv_0"]["analysis"]["series"], fillvalue=None
    ):
        run_host_id = (riscv1_run if riscv1_run else riscv0_run)["duration_type"][0]["run_host_id"]

        logger.info(f"Run host id: {run_host_id}")

        if riscv1_run:
            logger.info(f'RISCV 1 duration: {riscv1_run["duration_cycles"]}')

        if riscv0_run:
            logger.info(f'RISCV 0 duration: {riscv0_run["duration_cycles"]}')

        logger.info(f"Attributes:")
        for attr, val in dm_stats["riscv_1" if riscv1_run else "riscv_0"]["attributes"][run_host_id].items():
            logger.info(f"  {attr}: {val}")
        logger.info("")


def plot_dm_stats(dm_stats, output_file="dm_stats_plot.png"):
    # Extract data for plotting
    riscv_1_series = dm_stats["riscv_1"]["analysis"]["series"]
    riscv_0_series = dm_stats["riscv_0"]["analysis"]["series"]

    # Group data by Test id
    test_ids = set()
    for attributes in dm_stats["riscv_1"]["attributes"].values():
        test_ids.add(attributes["Test id"])
    for attributes in dm_stats["riscv_0"]["attributes"].values():
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
        riscv_1_filtered = [
            entry
            for entry in riscv_1_series
            if dm_stats["riscv_1"]["attributes"][entry["duration_type"][0]["run_host_id"]]["Test id"] == test_id
        ]
        riscv_0_filtered = [
            entry
            for entry in riscv_0_series
            if dm_stats["riscv_0"]["attributes"][entry["duration_type"][0]["run_host_id"]]["Test id"] == test_id
        ]

        # Aggregate data across all runtime_host_ids for the current Test id
        riscv_1_durations = [entry["duration_cycles"] for entry in riscv_1_filtered]
        riscv_0_durations = [entry["duration_cycles"] for entry in riscv_0_filtered]

        riscv_1_bandwidths = []
        riscv_0_bandwidths = []
        riscv_1_data_sizes = []
        riscv_0_data_sizes = []

        for entry in riscv_1_filtered:
            runtime_host_id = entry["duration_type"][0]["run_host_id"]
            attributes = dm_stats["riscv_1"]["attributes"][runtime_host_id]
            transaction_size = attributes["Transaction size in bytes"]
            bandwidth = attributes["Number of transactions"] * transaction_size / entry["duration_cycles"]
            riscv_1_bandwidths.append(bandwidth)
            riscv_1_data_sizes.append(transaction_size)

        for entry in riscv_0_filtered:
            runtime_host_id = entry["duration_type"][0]["run_host_id"]
            attributes = dm_stats["riscv_0"]["attributes"][runtime_host_id]
            transaction_size = attributes["Transaction size in bytes"]
            bandwidth = attributes["Number of transactions"] * transaction_size / entry["duration_cycles"]
            riscv_0_bandwidths.append(bandwidth)
            riscv_0_data_sizes.append(transaction_size)

        # Plot durations
        ax = axes[0]
        ax.plot(riscv_1_durations, label="RISCV 1 Duration (cycles)", marker="o")
        ax.plot(riscv_0_durations, label="RISCV 0 Duration (cycles)", marker="o")
        ax.set_xlabel("Index")
        ax.set_ylabel("Duration (cycles)")
        ax.set_title("Kernel Durations")
        ax.legend()
        ax.grid()

        # Plot bandwidth
        ax = axes[1]
        ax.plot(riscv_1_bandwidths, label="RISCV 1 Bandwidth (bytes/cycle)", marker="o")
        ax.plot(riscv_0_bandwidths, label="RISCV 0 Bandwidth (bytes/cycle)", marker="o")
        ax.set_xlabel("Index")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title("Bandwidth Comparison")
        ax.legend()
        ax.grid()

        # Plot size of data transferred vs bandwidth
        ax = axes[2]
        ax.scatter(riscv_1_data_sizes, riscv_1_bandwidths, label="RISCV 1", marker="o")
        ax.scatter(riscv_0_data_sizes, riscv_0_bandwidths, label="RISCV 0", marker="o")
        ax.set_xlabel("Transaction Size (bytes)")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title("Data Size vs Bandwidth")
        ax.legend()
        ax.grid()

    # Save the combined plot
    plt.savefig(output_file)
    plt.close()
    logger.info(f"dm_stats plots saved at {output_file}")


def export_dm_stats_to_csv(dm_stats, output_file="dm_stats.csv"):
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(["Kernel", "Run Host ID", "Test ID", "Latency (cycles)", "Bandwidth (bytes/cycle)"])

        # Iterate over the dm_stats object
        for kernel, kernel_data in dm_stats.items():
            for run_host_id, attributes in kernel_data["attributes"].items():
                test_id = attributes.get("Test id", "N/A")
                duration_cycles = next(
                    (
                        entry["duration_cycles"]
                        for entry in kernel_data["analysis"]["series"]
                        if entry["duration_type"][0]["run_host_id"] == run_host_id
                    ),
                    None,
                )
                if duration_cycles:
                    transaction_size = attributes.get("Transaction size in bytes", 0)
                    num_transactions = attributes.get("Number of transactions", 0)
                    bandwidth = (num_transactions * transaction_size) / duration_cycles if duration_cycles else 0
                    writer.writerow([kernel, run_host_id, test_id, duration_cycles, bandwidth])

    logger.info(f"dm_stats exported to {output_file}")


def test_data_movement(
    no_profile: bool,
    verbose_log: bool,
    gtest_filter: str,
    plot: bool,
    report: bool,
    arch: str,
):
    run_dm_tests(
        profile=not no_profile, verbose=verbose_log, gtest_filter=gtest_filter, plot=plot, report=report, arch_name=arch
    )
