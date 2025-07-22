#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import csv
import sys
import yaml
from loguru import logger  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.gridspec import GridSpec
import itertools
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

from tests.tt_metal.tt_metal.data_movement._python_script.config import DataMovementConfig
from tests.tt_metal.tt_metal.data_movement._python_script.test_metadata import TestMetadataLoader
from tests.tt_metal.tt_metal.data_movement._python_script.constants import (
    RISCV_PROCESSORS,
    NOC_WIDTHS,
    MULTICAST_SCHEMES_TEST_IDS,
    DEFAULT_PLOT_WIDTH,
    DEFAULT_PLOT_HEIGHT,
    DEFAULT_COMMENT_HEIGHT_RATIO,
    RISC_TO_KERNEL_MAP,
    DEFAULT_OUTPUT_DIR,
)


def run_dm_tests(profile, verbose, gtest_filter, plot, report, arch_name):
    logger.info("Starting data movement tests...")
    log_file_path = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"

    # Initialize configuration and load test metadata
    config = DataMovementConfig()
    metadata_loader = TestMetadataLoader(config)
    test_id_to_name, test_id_to_comment, test_bounds = metadata_loader.get_test_mappings()

    # Get architecture
    arch = config.get_arch(arch_name, test_bounds)
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
        plot_dm_stats(dm_stats, arch=arch, test_id_to_name=test_id_to_name, test_id_to_comment=test_id_to_comment)

    # Export results to csv
    if report:
        export_dm_stats_to_csv(dm_stats, test_id_to_name=test_id_to_name)

    # Check performance
    performance_check(dm_stats, arch=arch, verbose=verbose, test_bounds=test_bounds, test_id_to_name=test_id_to_name)

    logger.info("Data movement tests completed.")


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


def performance_check(dm_stats, arch="blackhole", verbose=False, test_bounds=None, test_id_to_name=None):
    # Tidy results' ranges
    results_bounds = {}
    for riscv in dm_stats.keys():
        for run in dm_stats[riscv]["analysis"]["series"]:
            run_host_id = run["duration_type"][0]["run_host_id"]
            test_id = dm_stats[riscv]["attributes"][run_host_id]["Test id"]
            if test_id not in results_bounds.keys():
                results_bounds[test_id] = {riscv: {"latency": {"upper": 0}, "bandwidth": float("inf")}}
            elif riscv not in results_bounds[test_id].keys():
                results_bounds[test_id][riscv] = {
                    "latency": {"upper": 0},
                    "bandwidth": float("inf"),
                }

            cycles = run["duration_cycles"]
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
                if bounds[riscv]["latency"]["upper"] != float("inf"):
                    logger.info(f"  {riscv}: {bounds[riscv]['latency']['upper']} cycles")

            logger.info(f"Bandwidth")
            for riscv in bounds.keys():
                if bounds[riscv]["bandwidth"] != float("inf"):
                    logger.info(f"  {riscv}: {bounds[riscv]['bandwidth']} Bytes/cycle")

        if test_bounds is None or test_id not in test_bounds[arch].keys():
            logger.warning(f"Test id {test_id} not found in {arch} test bounds.")
            continue

        for riscv in bounds.keys():
            if riscv not in test_bounds[arch][test_id].keys():
                continue
            cycles_within_bounds = bounds[riscv]["latency"]["upper"] <= test_bounds[arch][test_id][riscv]["latency"]
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
        logger.info(f"")
        logger.info(f"")

        run_host_id = (riscv1_run if riscv1_run else riscv0_run)["duration_type"][0]["run_host_id"]

        logger.info(f"Run host id: {run_host_id}")

        for riscv, run in [("RISCV 1", riscv1_run), ("RISCV 0", riscv0_run)]:
            if run:
                logger.info(f"")
                logger.info(f'{riscv} duration: {run["duration_cycles"]}')
                logger.info("Attributes:")
                for attr, val in dm_stats[riscv.lower().replace(" ", "_")]["attributes"][run_host_id].items():
                    logger.info(f"  {attr}: {val}")

    logger.info(f"")
    logger.info(f"")


def aggregate_performance(dm_stats, method="median"):
    """
    Aggregates duration and bandwidth per run_host_id for each kernel,
    and includes the attributes for each run_host_id.

    Args:
        dm_stats: nested dict as produced by gather_analysis_stats
        method: 'median' or 'average'

    Returns:
        Dict: {kernel: {run_host_id: {
            "duration_cycles": aggregated_value,
            "bandwidth": aggregated_value,
            "attributes": attributes_dict,
            "all_durations": [...],
            "all_bandwidths": [...],
        }}}
    """
    result = {}
    for kernel in dm_stats.keys():
        grouped_durations = defaultdict(list)
        grouped_bandwidths = defaultdict(list)
        for entry in dm_stats[kernel]["analysis"]["series"]:
            run_host_id = entry["duration_type"][0]["run_host_id"]
            attributes = dm_stats[kernel]["attributes"][run_host_id]
            num_transactions = attributes["Number of transactions"]
            transaction_size = attributes["Transaction size in bytes"]
            duration = entry["duration_cycles"]
            bandwidth = num_transactions * transaction_size / duration if duration else 0
            grouped_durations[run_host_id].append(duration)
            grouped_bandwidths[run_host_id].append(bandwidth)

        agg = {}
        for run_host_id in grouped_durations:
            durations = grouped_durations[run_host_id]
            bandwidths = grouped_bandwidths[run_host_id]
            if method == "median":
                agg_duration = float(np.median(durations))
                agg_bandwidth = float(np.median(bandwidths))
            elif method == "average":
                agg_duration = float(np.mean(durations))
                agg_bandwidth = float(np.mean(bandwidths))
            else:
                raise ValueError(f"Unknown method: {method}")
            agg[run_host_id] = {
                "duration_cycles": agg_duration,
                "bandwidth": agg_bandwidth,
                "attributes": dm_stats[kernel]["attributes"][run_host_id],
                "all_durations": durations,
                "all_bandwidths": bandwidths,
            }
        result[kernel] = agg
    return result


def plot_dm_stats(
    dm_stats, output_dir=DEFAULT_OUTPUT_DIR, arch="blackhole", test_id_to_name=None, test_id_to_comment=None
):
    # Set noc_width based on architecture
    noc_width = NOC_WIDTHS.get(arch, 64)  # Default to 64 if architecture not found
    multicast_schemes_test_ids = MULTICAST_SCHEMES_TEST_IDS

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Helper: Extract data for a specific test_id
    def extract_data(series, attributes, test_id):
        filtered = [
            entry for entry in series if attributes[entry["duration_type"][0]["run_host_id"]]["Test id"] == test_id
        ]

        data = {
            "durations": [],
            "data_sizes": [],
            "bandwidths": [],
            "transactions": [],
        }
        if test_id in multicast_schemes_test_ids:
            data["noc_index"] = []
            data["multicast_scheme_number"] = []
            data["grid_dimensions"] = []

        for entry in filtered:
            runtime_host_id = entry["duration_type"][0]["run_host_id"]
            attr = attributes[runtime_host_id]

            duration = entry["duration_cycles"]
            transaction_size = attr["Transaction size in bytes"]
            num_transactions = attr["Number of transactions"]
            bandwidth = num_transactions * transaction_size / duration

            data["durations"].append(duration)
            data["data_sizes"].append(transaction_size)
            data["bandwidths"].append(bandwidth)
            data["transactions"].append(num_transactions)

            if test_id in multicast_schemes_test_ids:
                noc_index = attr["NoC Index"]
                multicast_scheme_type = attr["Multicast Scheme Type"]
                grid_dimensions = f"{attr['Subordinate Grid Size X']} x {attr['Subordinate Grid Size Y']}"

                data["noc_index"].append(noc_index)
                data["multicast_scheme_number"].append(multicast_scheme_type)
                data["grid_dimensions"].append(grid_dimensions)

        return data

    # Helper: Plot Type 1 - Test Index vs Duration
    def plot_durations(ax, data):
        risc_to_kernel_map = RISC_TO_KERNEL_MAP

        unique_transactions = sorted(set(data["riscv_1"]["transactions"] + data["riscv_0"]["transactions"]))
        for num_transactions in unique_transactions:
            for riscv in RISCV_PROCESSORS:
                # Group and plot RISCV data
                grouped = [
                    (size, duration)
                    for size, duration, transactions in zip(
                        data[riscv]["data_sizes"], data[riscv]["durations"], data[riscv]["transactions"]
                    )
                    if transactions == num_transactions
                ]
                if grouped:
                    sizes, durations = zip(*grouped)
                    ax.plot(
                        sizes,
                        durations,
                        label=f"{risc_to_kernel_map[riscv]} (Number of Transactions={num_transactions})",
                        marker="o",
                    )

        # Adjust the plot area to leave space for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Shrink plot width to 80% of allocated space

        # Place the legend outside the plot but within the allocated subfigure space
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),  # Position the legend outside the plot area
            borderaxespad=0,
            fontsize=8,
        )

        ax.set_xlabel("Transaction Size (bytes)")
        ax.set_ylabel("Duration (cycles)")
        ax.set_title("Transaction Size vs Duration")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_yscale("log", base=10)
        ax.grid()

    # Helper: Plot Type 2 - Transaction Size vs Bandwidth
    def plot_data_size_vs_bandwidth(ax, data, noc_width):
        risc_to_kernel_map = RISC_TO_KERNEL_MAP

        unique_transactions = sorted(set(itertools.chain.from_iterable(data[riscv]["transactions"] for riscv in data)))

        for num_transactions in unique_transactions:
            grouped = {}
            for riscv in RISCV_PROCESSORS:
                grouped[riscv] = [
                    (size, bw)
                    for size, bw, transactions in zip(
                        data[riscv]["data_sizes"], data[riscv]["bandwidths"], data[riscv]["transactions"]
                    )
                    if transactions == num_transactions
                ]
            if grouped[riscv]:
                # Sort by data sizes (x-axis) before plotting
                grouped[riscv].sort(key=lambda x: x[0])
                sizes, bws = zip(*grouped[riscv])
                ax.plot(sizes, bws, label=f"{risc_to_kernel_map[riscv]} (Transactions={num_transactions})", marker="o")

        # Add theoretical max bandwidth curve
        transaction_sizes = sorted(set(data["riscv_1"]["data_sizes"] + data["riscv_0"]["data_sizes"]))
        max_bandwidths = [noc_width * ((size / noc_width) / ((size / noc_width) + 1)) for size in transaction_sizes]
        ax.plot(transaction_sizes, max_bandwidths, label="Theoretical Max BW", linestyle="--", color="black")

        # Adjust the plot area to leave space for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Shrink plot width to 80% of allocated space

        # Place the legend outside the plot but within the allocated subfigure space
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),  # Position the legend outside the plot area
            borderaxespad=0,
            fontsize=8,
        )

        # Set labels and title
        ax.set_xlabel("Transaction Size (bytes)")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title("Transaction Size vs Bandwidth")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.grid()

    # Helper: Plot Bandwidth for Multicast Schemes
    def plot_bandwidth_multicast(ax, data, x_axis, lines, riscv, noc_index):
        # Filter data where "noc_index" matches the input noc_index
        filtered_data = [
            (data[riscv][x_axis][i], data[riscv]["bandwidths"][i], data[riscv][lines][i])
            for i in range(len(data[riscv][x_axis]))
            if data[riscv].get("noc_index", [None])[i] == noc_index
        ]

        if not filtered_data:
            return  # No data to plot for the given noc_index

        # Sort the filtered data by the x_axis values
        filtered_data.sort(key=lambda x: x[0])

        # Extract sorted x_axis, bandwidths, and lines
        sorted_x_axis, sorted_bandwidths, sorted_lines = zip(*filtered_data)

        # Get unique line categories
        lines_list = sorted(set(sorted_lines))

        for line in lines_list:
            # Filter data for the current line category
            line_data = [(x, bw) for x, bw, l in zip(sorted_x_axis, sorted_bandwidths, sorted_lines) if l == line]

            if line_data:
                sizes, bws = zip(*line_data)
                ax.plot(
                    sizes,
                    bws,
                    label=f"{riscv.upper()}, NoC {noc_index}, {lines.replace('_', ' ').title()}: {line}",
                    marker="o",
                )

        # Adjust the plot area to leave space for the legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Shrink plot width to 80% of allocated space

        # Place the legend outside the plot but within the allocated subfigure space
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),  # Position the legend outside the plot area
            borderaxespad=0,
            fontsize=8,
        )

        ax.set_xlabel(f"{x_axis.replace('_', ' ').title()}")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title(f"{x_axis.replace('_', ' ').title()} vs Bandwidth")
        ax.grid()

    # Parameters for individual plot and figure layout
    plot_width = DEFAULT_PLOT_WIDTH  # Width of an individual plot
    plot_height = DEFAULT_PLOT_HEIGHT  # Height of an individual plot
    nrows_per_figure = 1  # Number of rows of plots per figure
    ncols_per_figure = 2  # Number of columns of plots per figure
    comment_section_height_ratio = DEFAULT_COMMENT_HEIGHT_RATIO  # Height ratio for the comment section

    agg_stats = aggregate_performance(dm_stats)

    # Extract data for plotting
    riscv_1_series = agg_stats["riscv_1"]
    riscv_0_series = agg_stats["riscv_0"]

    # Group data by Test id
    test_ids = set()
    for kernel in agg_stats.keys():
        for stats in agg_stats[kernel].values():
            test_ids.add(stats["attributes"]["Test id"])

    test_ids = sorted(test_ids)  # Sort for consistent ordering

    # Extract test IDs
    # test_ids = sorted(
    #    {attributes["Test id"] for riscv in RISCV_PROCESSORS for attributes in dm_stats[riscv]["attributes"].values()}
    # )

    # Extract data for plotting
    series = {riscv: dm_stats[riscv]["analysis"]["series"] for riscv in RISCV_PROCESSORS}

    # Iterate over test IDs and create figures
    for test_id in test_ids:
        test_name = test_id_to_name.get(test_id, f"Test ID {test_id}") if test_id_to_name else f"Test ID {test_id}"

        # Prepare figure for the current test ID

        figure_height = plot_height * nrows_per_figure + comment_section_height_ratio * plot_height

        fig = plt.figure(figsize=(plot_width * ncols_per_figure, figure_height))

        # Create a GridSpec layout
        gridspec = GridSpec(
            nrows_per_figure + 1,
            ncols_per_figure,
            height_ratios=[plot_height] * nrows_per_figure + [comment_section_height_ratio * plot_height],
        )

        # Create subplots within the figure
        axes = [fig.add_subplot(gridspec[0, col]) for col in range(ncols_per_figure)]

        # Extract data for riscv_1 and riscv_0
        data = {
            riscv: extract_data(series[riscv], dm_stats[riscv]["attributes"], test_id) for riscv in RISCV_PROCESSORS
        }

        # Generate plots based on test type
        if test_id in multicast_schemes_test_ids:
            plot_bandwidth_multicast(
                axes[0], data, x_axis="grid_dimensions", lines="multicast_scheme_number", riscv="riscv_0", noc_index=0
            )
            plot_bandwidth_multicast(
                axes[1], data, x_axis="grid_dimensions", lines="multicast_scheme_number", riscv="riscv_0", noc_index=1
            )
        else:  # Packet Sizes
            plot_durations(axes[0], data)
            plot_data_size_vs_bandwidth(axes[1], data, noc_width)

        # Add comments section to the figure below the plots
        comment_ax = fig.add_subplot(gridspec[-1, :])
        comment_ax.axis("off")  # Hide axes for the comments section
        comment_text = (
            test_id_to_comment.get(test_id, "No comment available, test has not been analyzed")
            if test_id_to_comment
            else "No comment available"
        )
        comment_ax.text(
            0.5,
            0.5,
            f"Comments: {comment_text}",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
        )

        # Save the plot for this test id
        test_name = test_id_to_name.get(test_id, f"Test ID {test_id}") if test_id_to_name else f"Test ID {test_id}"
        output_file = os.path.join(output_dir, f"{test_name}.png")
        plt.savefig(output_file)
        plt.close(fig)
        logger.info(f"dm_stats plot for test id {test_id} saved at {output_file}")


def export_dm_stats_to_csv(dm_stats, output_dir=DEFAULT_OUTPUT_DIR, test_id_to_name=None):
    os.makedirs(output_dir, exist_ok=True)
    agg_stats = aggregate_performance(dm_stats)

    # Group by test id
    test_ids = set()
    for riscv in agg_stats.keys():
        for test_run in agg_stats[riscv].values():
            test_ids.add(test_run["attributes"]["Test id"])
    test_ids = sorted(test_ids)

    for test_id in test_ids:
        test_name = test_id_to_name.get(test_id, f"Test ID {test_id}") if test_id_to_name else f"Test ID {test_id}"
        csv_file = os.path.join(output_dir, f"{test_name}.csv")
        with open(csv_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Kernel",
                    "Run Host ID",
                    "Transaction Size (bytes)",
                    "Number of Transactions",
                    "Latency (cycles)",
                    "Bandwidth (bytes/cycle)",
                ]
            )
            for kernel in agg_stats.keys():
                for run_host_id, run_stats in agg_stats[kernel].items():
                    # run_host_id = entry["duration_type"][0]["run_host_id"]
                    attributes = run_stats["attributes"]
                    if attributes.get("Test id") != test_id:
                        continue
                    transaction_size = attributes.get("Transaction size in bytes", 0)
                    num_transactions = attributes.get("Number of transactions", 0)
                    duration_cycles = run_stats["duration_cycles"]
                    bandwidth = run_stats["bandwidth"]
                    writer.writerow(
                        [
                            "Receiver" if kernel == "riscv_1" else "Sender",
                            run_host_id,
                            transaction_size,
                            num_transactions,
                            duration_cycles,
                            bandwidth,
                        ]
                    )
        logger.info(f"CSV report for test id {test_id} saved at {csv_file}")


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
