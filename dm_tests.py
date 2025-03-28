#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser
from loguru import logger  # type: ignore

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

# import pytest


def run_dm_tests(profile, gtest_filter):
    log_file_path = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"
    if profile or not os.path.exists(log_file_path) or gtest_filter:
        logger.info(f"Profiling Kernels...")
        cmd = f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/unit_tests_dm"

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
    # TODO: Print/log for each core
    core = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"][0]
    dm_stats = {
        "reader": {
            "analysis": stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]["reader_kernel_analysis"],
            "attributes": dict(),
        },
        "writer": {
            "analysis": stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]["writer_kernel_analysis"],
            "attributes": dict(),
        },
    }

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

    # Analysis average stats (Not very meaningful)
    logger.info(f"Averages")
    logger.info(f'Reader duration: {dm_stats["reader"]["analysis"]["stats"]["Average"]}')
    logger.info(f'Writer duration: {dm_stats["writer"]["analysis"]["stats"]["Average"]}\n')

    # # # # # # Performance check method # # # # # #
    reader_cycles = dm_stats["reader"]["analysis"]["series"][0]["duration_cycles"]
    reader_cycles_lower_bound = 700
    reader_cycles_upper_bound = 800
    reader_cycles_within_bounds = reader_cycles_lower_bound <= reader_cycles <= reader_cycles_upper_bound

    if not reader_cycles_within_bounds:
        logger.warning(
            f"Reader cycles not within bounds. Received {reader_cycles}, was expecting between {reader_cycles_lower_bound} and {reader_cycles_upper_bound}"
        )
    else:
        logger.info(f"Reader cycles within bounds. Received {reader_cycles}")

    # assert reader_cycles_within_bounds


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate reference outputs for LLaMA accuracy testing.")
    parser.add_argument("-p", "--profile", action="store_true")
    parser.add_argument("-g", "--gtest-filter", dest="gtest_filter")
    args = parser.parse_args()

    run_dm_tests(*vars(args).values())
