#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import csv
import sys
from loguru import logger  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import itertools
import matplotlib.ticker as mticker
import numpy as np

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
    6: "One to All Unicast 2x2 Packet Sizes",
    7: "One to All Unicast 4x4 Packet Sizes",
    8: "One to All Unicast 10x10 Packet Sizes",
    9: "One to All Multicast 2x2 Packet Sizes",
    10: "One to All Multicast 5x5 Packet Sizes",
    11: "One to All Multicast 11x10 Packet Sizes",
    12: "One to All Multicast Linked 2x2 Packet Sizes",
    13: "One to All Multicast Linked 5x5 Packet Sizes",
    14: "One to All Multicast Linked 11x10 Packet Sizes",
    15: "One from All Packet Sizes",
    16: "Loopback Packet Sizes",
    30: "One from All Directed Ideal",
    50: "One to One Directed Ideal",
    51: "One from One Directed Ideal",
    52: "One to All Directed Ideal",
    17: "Reshard Hardcoded Small",
    18: "Reshard Hardcoded Medium",
    19: "Reshard Hardcoded Many Cores",
    20: "Reshard Hardcoded 2 Cores to Many Cores",
    200: "Deinterleave Single Core",
    201: "Deinterleave Multi Core",
    21: "Conv Act with halo 3x3",
    22: "Conv Act with halo 3x3 Small",
    23: "Conv Halo Gather",
    60: "All to All Packet Sizes",
    70: "All from All Packet Sizes",
}

# Comments for each test explaining why we get the perf that we do
test_id_to_comment = {
    1: "This test appears to be broken. The graph is showing numbers that dont make sense.",
    2: "This test appears to be broken. The graph is showing numbers that dont make sense.",
    3: "This test shows the ideal read and write bandwidth when transfering multiple 8KB packets. \n\
        The read bandwidth is what is expected, however write bandwidth is expected to be 64 \n\
        B/cycle rather than 35 B/cycle. There may be some configuration problem with the dram \n\
        controller/phy or this may be the physical limit of the dram.",
    17: "This is a 2 reader reshard. It seems to be getting expected perf based on number of transactions \n\
        and transactions size. Reshard perf is dictated based on the number of transactions and the \n\
        transaction size. A small number of transactions will result in small perf due to large \n\
        round trip latency. It is suggested to use a large number of transactions, with large transaction \n\
        size to get the best performance.",
    18: "This is a 2 reader reshard. It seems to be getting expected perf based on number of transactions \n\
        and transactions size. Reshard perf is dictated based on the number of transactions and the \n\
        transaction size. A small number of transactions will result in small perf due to large \n\
        round trip latency. It is suggested to use a large number of transactions, with large transaction \n\
        size to get the best performance.",
    19: "This is a 8 reader reshard. It seems to be getting expected perf based on number of transactions \n\
        and transactions size. Reshard perf is dictated based on the number of transactions and the \n\
        transaction size. A small number of transactions will result in small perf due to large \n\
        round trip latency. It is suggested to use a large number of transactions, with large transaction \n\
        size to get the best performance.",
    20: "This is a 2 core to 8 reader reshard. It seems to be getting expected perf based on number of \n\
        transactions and transactions size. Reshard perf is dictated based on the number of transactions \n\
        and the transaction size. A small number of transactions will result in small perf due to large \n\
        round trip latency. It is suggested to use a large number of transactions, with large transaction \n\
        size to get the best performance.",
    21: "Convolution has a large number of transactions and a small transaction size. The performance is \n\
        similar to what it would be for a similarly configured one from one. Convolution may benefit from \n\
        having multiple cores doing different parts of the convolution at the same time. This would \n\
        result in a larger effective bandwidth.",
    22: "Convolution has a large number of transactions and a small transaction size. The performance is \n\
        similar to what it would be for a similarly configured one from one. Convolution may benefit from \n\
        having multiple cores doing different parts of the convolution at the same time. This would \n\
        result in a larger effective bandwidth.",
    23: "The performance of this test is similar to how other tests perform based on the number of \n\
        transactions and the transaction size, but with extra degradation due to needing to read \n\
        parameters from L1.",
    200: "With a single core the graphs shows performance increases as the theshold increases. \n\
         This is because frequent flushes dont hide the round trip latency.",
    201: "With multiple cores the graph shows that a small theshold always provides bad performance. \n\
          This is because frequent flushes dont hide the round trip latency. At larger thesholds, \n\
          the performance starts to fluctuate due to head-of-line blocking and unfairness in the NOC. \n\
          Performance fluctuates because the flush disturbes the steady state and will randomly create \n\
          traffic that sometimes has head of line blocking, and sometimes not.",
}

# Correspondng test bounds for each arch, test id, riscv core
# NOTE: These bounds are aggregated averages of large test sweeps and
# are subject to change with new directed tests.
test_bounds = {
    "wormhole_b0": {
        # 0: {
        #     "riscv_1": {"latency": {"lower": 300, "upper": 24000}, "bandwidth": 0.08},
        #     "riscv_0": {"latency": {"lower": 300, "upper": 25000}, "bandwidth": 0.07},
        # },
        1: {
            "riscv_1": {"latency": {"lower": 23000, "upper": 24000}, "bandwidth": 21},
            "riscv_0": {"latency": {"lower": 24000, "upper": 25000}, "bandwidth": 21},
        },
        2: {
            "riscv_1": {"latency": {"lower": 300, "upper": 600}, "bandwidth": 0.08},
            "riscv_0": {"latency": {"lower": 300, "upper": 500}, "bandwidth": 0.08},
        },
        3: {  # DRAM Unary Directed Ideal
            "riscv_1": {"latency": {"lower": 33000, "upper": 35000}, "bandwidth": 22},
            "riscv_0": {"latency": {"lower": 33000, "upper": 35000}, "bandwidth": 21},
        },
        # 4: {
        #     "riscv_0": {"latency": {"lower": 200, "upper": 18000}, "bandwidth": 0.1},
        # },
        # 5: {
        #     "riscv_1": {"latency": {"lower": 200, "upper": 19000}, "bandwidth": 0.1},
        # },
        # 6: {
        #     "riscv_0": {"latency": {"lower": 400, "upper": 70000}, "bandwidth": 0.3},
        # },
        # 7: {
        #     "riscv_0": {"latency": {"lower": 800, "upper": 300000}, "bandwidth": 0.6},
        # },
        # 8: {
        #     "riscv_0": {"latency": {"lower": 1900, "upper": 900000}, "bandwidth": 0.8},
        # },
        # 9: {
        #     "riscv_0": {"latency": {"lower": 300, "upper": 30000}, "bandwidth": 0.09},
        # },
        # 10: {
        #     "riscv_0": {"latency": {"lower": 400, "upper": 60000}, "bandwidth": 0.07},
        # },
        # 11: {
        #     "riscv_0": {"latency": {"lower": 500, "upper": 90000}, "bandwidth": 0.04},
        # },
        # 12: {
        #     "riscv_0": {"latency": {"lower": 200, "upper": 20000}, "bandwidth": 0.09},
        # },
        # 13: {
        #     "riscv_0": {"latency": {"lower": 400, "upper": 30000}, "bandwidth": 0.07},
        # },
        # 14: {
        #     "riscv_0": {"latency": {"lower": 500, "upper": 40000}, "bandwidth": 0.04},
        # },
        # 15: {
        #     "riscv_1": {"latency": {"lower": 700, "upper": 85000}, "bandwidth": 0.71},
        # },
        # 16: {
        #     "riscv_0": {"latency": {"lower": 50, "upper": 30000}, "bandwidth": 0.4},
        # },
        30: {  # One from All Directed Ideal
            "riscv_1": {"latency": {"lower": 20000, "upper": 37000}, "bandwidth": 30},
        },
        50: {  # One to One Directed Ideal
            "riscv_0": {"latency": {"lower": 28000, "upper": 36000}, "bandwidth": 29},  # 33832
        },
        51: {  # One from One Directed Ideal
            "riscv_1": {"latency": {"lower": 32700, "upper": 37500}, "bandwidth": 28},  # 18596, 28.2
        },
        52: {  # One to All Unicast Directed Ideal
            "riscv_0": {"latency": {"lower": 0, "upper": 4350000}, "bandwidth": 31},  #  4294196, 31.25
        },
        53: {  # One to All Multicast Directed Ideal
            "riscv_0": {"latency": {"lower": 0, "upper": 150000}, "bandwidth": 14},  # 137035, 15.3
        },
        54: {  # One to All Multicast Linked Directed Ideal
            "riscv_0": {"latency": {"lower": 0, "upper": 90000}, "bandwidth": 22},  # 88542, 23.69
        },
        17: {
            "riscv_1": {"latency": {"lower": 50, "upper": 700}, "bandwidth": 3},
        },
        18: {
            "riscv_1": {"latency": {"lower": 500, "upper": 3000}, "bandwidth": 15},
        },
        19: {
            "riscv_1": {"latency": {"lower": 500, "upper": 3000}, "bandwidth": 10},
        },
        200: {
            "riscv_1": {"latency": {"lower": 15000, "upper": 30000}, "bandwidth": 1.7},
        },
        201: {
            "riscv_1": {"latency": {"lower": 10000, "upper": 60000}, "bandwidth": 2},
        },
        20: {
            "riscv_1": {"latency": {"lower": 100, "upper": 1000}, "bandwidth": 3},
        },
        21: {
            "riscv_1": {"latency": {"lower": 150000, "upper": 300000}, "bandwidth": 3},
        },
        22: {
            "riscv_1": {"latency": {"lower": 1000000, "upper": 1100000}, "bandwidth": 0.3},
        },
        23: {
            "riscv_1": {"latency": {"lower": 500, "upper": 1000}, "bandwidth": 10},
        },
    },
    "blackhole": {
        # 0: {
        #     "riscv_1": {"latency": {"lower": 400, "upper": 17000}, "bandwidth": 0.1},
        #     "riscv_0": {"latency": {"lower": 300, "upper": 16000}, "bandwidth": 0.15},
        # },
        1: {
            "riscv_1": {"latency": {"lower": 20000, "upper": 33000}, "bandwidth": 32},
            "riscv_0": {"latency": {"lower": 20000, "upper": 33000}, "bandwidth": 33},
        },
        2: {
            "riscv_1": {"latency": {"lower": 400, "upper": 600}, "bandwidth": 0.13},
            "riscv_0": {"latency": {"lower": 300, "upper": 500}, "bandwidth": 0.16},
        },
        3: {  # DRAM Unary Directed Ideal
            "riscv_1": {"latency": {"lower": 42000, "upper": 44000}, "bandwidth": 33},
            "riscv_0": {"latency": {"lower": 42000, "upper": 44000}, "bandwidth": 34},
        },
        # 4: {
        #     "riscv_0": {"latency": {"lower": 200, "upper": 19000}, "bandwidth": 0.17},
        # },
        # 5: {
        #     "riscv_1": {"latency": {"lower": 300, "upper": 18000}, "bandwidth": 0.17},
        # },
        # 6: {
        #     "riscv_0": {"latency": {"lower": 400, "upper": 70000}, "bandwidth": 0.5},
        # },
        # 7: {
        #     "riscv_0": {"latency": {"lower": 900, "upper": 275000}, "bandwidth": 1.00},
        # },
        # 8: {
        #     "riscv_0": {"latency": {"lower": 3800, "upper": 1700000}, "bandwidth": 1.65},
        # },
        # 9: {
        #     "riscv_0": {"latency": {"lower": 300, "upper": 30000}, "bandwidth": 0.16},
        # },
        # 10: {
        #     "riscv_0": {"latency": {"lower": 450, "upper": 70000}, "bandwidth": 0.12},
        # },
        # 11: {
        #     "riscv_0": {"latency": {"lower": 700, "upper": 115000}, "bandwidth": 0.08},
        # },
        # 12: {
        #     "riscv_0": {"latency": {"lower": 300, "upper": 20000}, "bandwidth": 0.16},
        # },
        # 13: {
        #     "riscv_0": {"latency": {"lower": 500, "upper": 24000}, "bandwidth": 0.12},
        # },
        # 14: {
        #     "riscv_0": {"latency": {"lower": 700, "upper": 46000}, "bandwidth": 0.08},
        # },
        # 15: {
        #     "riscv_1": {"latency": {"lower": 800, "upper": 87000}, "bandwidth": 1.19},
        # },
        # 16: {
        #     "riscv_0": {"latency": {"lower": 50, "upper": 30000}, "bandwidth": 0.4},
        # },
        30: {"riscv_1": {"latency": {"lower": 10000, "upper": 18000}, "bandwidth": 60}},  # One from All Directed Ideal
        50: {  # One to One Directed Ideal
            "riscv_0": {"latency": {"lower": 12000, "upper": 19000}, "bandwidth": 59},  # 17000
        },
        51: {  # One from One Directed Ideal
            "riscv_1": {"latency": {"lower": 16000, "upper": 17800}, "bandwidth": 59},  # 8730, 60.1
        },
        52: {  # One to All Unicast Directed Ideal
            "riscv_0": {"latency": {"lower": 0, "upper": 7500000}, "bandwidth": 62},  # 7359113, 62.69
        },
        53: {  # One to All Multicast Directed Ideal
            "riscv_0": {"latency": {"lower": 0, "upper": 180000}, "bandwidth": 24},  # 170221, 24.6
        },
        54: {  # One to All Multicast Linked Directed Ideal
            "riscv_0": {"latency": {"lower": 0, "upper": 110000}, "bandwidth": 41},  # 101088, 41.4
        },
        17: {
            "riscv_1": {"latency": {"lower": 50, "upper": 700}, "bandwidth": 7},
        },
        18: {
            "riscv_1": {"latency": {"lower": 500, "upper": 3000}, "bandwidth": 30},
        },
        19: {
            "riscv_1": {"latency": {"lower": 500, "upper": 3000}, "bandwidth": 25},
        },
        200: {
            "riscv_1": {"latency": {"lower": 15000, "upper": 30000}, "bandwidth": 1.7},
        },
        201: {
            "riscv_1": {"latency": {"lower": 10000, "upper": 60000}, "bandwidth": 2},
        },
        20: {
            "riscv_1": {"latency": {"lower": 100, "upper": 1000}, "bandwidth": 7},
        },
        21: {
            "riscv_1": {"latency": {"lower": 150000, "upper": 300000}, "bandwidth": 6},
        },
        22: {
            "riscv_1": {"latency": {"lower": 1000000, "upper": 1100000}, "bandwidth": 0.6},
        },
        23: {
            "riscv_1": {"latency": {"lower": 500, "upper": 1000}, "bandwidth": 20},
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
        plot_dm_stats(dm_stats, arch=arch)

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


def plot_dm_stats(dm_stats, output_dir="tests/tt_metal/tt_metal/data_movement", arch="blackhole"):
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

    # Set noc_width based on architecture
    noc_width = 32 if arch == "wormhole_b0" else 64

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for test_id in test_ids:
        # Create the figure for this test id
        fig = plt.figure(layout="constrained", figsize=(18, 6))

        # Add a title for the current Test id
        test_name = test_id_to_name.get(test_id, f"Test ID {test_id}")
        subsubfig = fig.subfigures(2, 1, height_ratios=[100, 1])
        subsubfig[0].suptitle(test_name, fontsize=16, weight="bold")

        # Create subplots within the subfigure
        axes = subsubfig[0].subplots(1, 2)

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
        riscv_1_durations = []
        riscv_0_durations = []
        riscv_1_data_sizes = []
        riscv_0_data_sizes = []
        riscv_1_bandwidths = []
        riscv_0_bandwidths = []
        riscv_1_transactions = []
        riscv_0_transactions = []

        for entry in riscv_1_filtered:
            duration = entry["duration_cycles"]
            runtime_host_id = entry["duration_type"][0]["run_host_id"]
            attributes = dm_stats["riscv_1"]["attributes"][runtime_host_id]
            transaction_size = attributes["Transaction size in bytes"]
            num_transactions = attributes["Number of transactions"]
            bandwidth = num_transactions * transaction_size / duration
            riscv_1_durations.append(duration)
            riscv_1_data_sizes.append(transaction_size)
            riscv_1_bandwidths.append(bandwidth)
            riscv_1_transactions.append(num_transactions)

        for entry in riscv_0_filtered:
            duration = entry["duration_cycles"]
            runtime_host_id = entry["duration_type"][0]["run_host_id"]
            attributes = dm_stats["riscv_0"]["attributes"][runtime_host_id]
            transaction_size = attributes["Transaction size in bytes"]
            num_transactions = attributes["Number of transactions"]
            bandwidth = num_transactions * transaction_size / entry["duration_cycles"]
            riscv_0_durations.append(duration)
            riscv_0_data_sizes.append(transaction_size)
            riscv_0_bandwidths.append(bandwidth)
            riscv_0_transactions.append(num_transactions)

        # Plot durations
        ax = axes[0]
        unique_transactions = sorted(set(riscv_1_transactions + riscv_0_transactions))  # Ensure ascending order
        for num_transactions in unique_transactions:
            # Group and plot RISCV 1 data
            riscv_1_grouped = [
                (sizes, durations)
                for sizes, durations, trans in zip(riscv_1_data_sizes, riscv_1_durations, riscv_1_transactions)
                if trans == num_transactions
            ]
            if riscv_1_grouped:
                sizes, durations = zip(*riscv_1_grouped)
                ax.plot(sizes, durations, label=f"Receiver (Number of Transactions={num_transactions})", marker="o")

            # Group and plot RISCV 0 data
            riscv_0_grouped = [
                (sizes, durations)
                for sizes, durations, trans in zip(riscv_0_data_sizes, riscv_0_durations, riscv_0_transactions)
                if trans == num_transactions
            ]
            if riscv_0_grouped:
                sizes, durations = zip(*riscv_0_grouped)
                ax.plot(sizes, durations, label=f"Sender (Number of Transactions={num_transactions})", marker="o")

        ax.set_xlabel("Transaction Size (bytes)")
        ax.set_ylabel("Duration (cycles)")
        ax.set_title("Transaction Size vs Duration")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.set_yscale("log", base=10)
        # ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.legend()
        ax.grid()

        # Plot size of data transferred vs bandwidth
        ax = axes[1]
        for num_transactions in unique_transactions:
            # Group and plot RISCV 1 data
            riscv_1_grouped = [
                (size, bw)
                for size, bw, trans in zip(riscv_1_data_sizes, riscv_1_bandwidths, riscv_1_transactions)
                if trans == num_transactions
            ]
            if riscv_1_grouped:
                sizes, bws = zip(*riscv_1_grouped)
                ax.plot(sizes, bws, label=f"Receiver (Number of Transactions={num_transactions})", marker="o")

            # Group and plot RISCV 0 data
            riscv_0_grouped = [
                (size, bw)
                for size, bw, trans in zip(riscv_0_data_sizes, riscv_0_bandwidths, riscv_0_transactions)
                if trans == num_transactions
            ]
            if riscv_0_grouped:
                sizes, bws = zip(*riscv_0_grouped)
                ax.plot(sizes, bws, label=f"Sender (Number of Transactions={num_transactions})", marker="o")

        # Add theoretical max bandwidth curve
        transaction_sizes = sorted(set(riscv_1_data_sizes + riscv_0_data_sizes))
        max_bandwidths = [noc_width * ((size / noc_width) / ((size / noc_width) + 1)) for size in transaction_sizes]
        ax.plot(transaction_sizes, max_bandwidths, label="Theoretical Max BW", linestyle="--", color="black")

        ax.set_xlabel("Transaction Size (bytes)")
        ax.set_ylabel("Bandwidth (bytes/cycle)")
        ax.set_title("Transaction Size vs Bandwidth")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.legend()
        ax.grid()

        # Add a comment section below the plots
        if test_id in test_id_to_comment.keys():
            txtObj = subsubfig[1].text(
                0.5,
                0,
                f"Comments: {test_id_to_comment.get(test_id, 'No comment available, test has not been analyzed')}",
                ha="center",
                fontsize=10,
                style="italic",
                wrap=True,
            )
            txtObj._get_wrap_line_width = lambda: 0.9 * subsubfig[1].bbox.width

        # Save the plot for this test id
        output_file = os.path.join(output_dir, f"{test_id_to_name.get(test_id, f'Test ID {test_id}')}.png")
        plt.savefig(output_file)
        plt.close(fig)
        logger.info(f"dm_stats plot for test id {test_id} saved at {output_file}")


def export_dm_stats_to_csv(dm_stats, output_dir="tests/tt_metal/tt_metal/data_movement"):
    os.makedirs(output_dir, exist_ok=True)
    # Group by test id
    test_ids = set()
    for riscv in dm_stats.keys():
        for attributes in dm_stats[riscv]["attributes"].values():
            test_ids.add(attributes["Test id"])
    test_ids = sorted(test_ids)

    for test_id in test_ids:
        test_name = test_id_to_name.get(test_id, f"Test ID {test_id}")
        csv_file = os.path.join(output_dir, f"{test_name}.csv")
        with open(csv_file, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Kernel",
                    "Run Host ID",
                    "Log2 of Transaction Size (bytes)",
                    "Number of Transactions",
                    "Latency (cycles)",
                    "Bandwidth (bytes/cycle)",
                ]
            )
            for kernel in ["riscv_1", "riscv_0"]:
                kernel_series = dm_stats[kernel]["analysis"]["series"]
                for entry in kernel_series:
                    run_host_id = entry["duration_type"][0]["run_host_id"]
                    attributes = dm_stats[kernel]["attributes"].get(run_host_id, {})
                    if attributes.get("Test id") != test_id:
                        continue
                    transaction_size = attributes.get("Transaction size in bytes", 0)
                    num_transactions = attributes.get("Number of transactions", 0)
                    duration_cycles = entry["duration_cycles"]
                    bandwidth = (num_transactions * transaction_size) / duration_cycles if duration_cycles else 0
                    log2_transaction_size = int(np.log2(transaction_size)) if transaction_size > 0 else 0
                    writer.writerow(
                        [
                            "Receiver" if kernel == "riscv_1" else "Sender",
                            run_host_id,
                            log2_transaction_size,
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
