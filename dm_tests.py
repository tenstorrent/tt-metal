#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser
from loguru import logger

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

# import pytest


def run_dm_tests(profile):
    if profile:
        os.system(
            f"TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1 {os.environ['TT_METAL_HOME']}/build/test/tt_metal/unit_tests_dm"
        )

    setup = device_post_proc_config.default_setup()

    setup.timerAnalysis = {
        "reader_kernel_latency": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
        "writer_kernel_latency": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
            "end": {"risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
        },
    }

    setup.deviceInputLog = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"
    stats = import_log_run_stats(setup)
    core = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"][0]
    reader_cycles = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]["reader_kernel_latency"]["stats"][
        "First"
    ]
    writer_cycles = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"]["writer_kernel_latency"]["stats"][
        "First"
    ]

    logger.info(f"Data Movement Performance Numbers")
    logger.info(f"Reader cycles: {reader_cycles}")
    logger.info(f"Writer cycles: {writer_cycles}")

    # unicast_cycles_lower_bound = 300
    # unicast_cycles_upper_bound = 400
    # unicast_cycles_within_bounds = unicast_cycles_lower_bound <= unicast_cycles <= unicast_cycles_upper_bound

    # if not unicast_cycles_within_bounds:
    #     logger.warning(
    #         f"Unicast cycles not within bounds. Received {unicast_cycles}, was expecting between {unicast_cycles_lower_bound} and {unicast_cycles_upper_bound}"
    #     )
    # else:
    #     logger.info(f"Unicast cycles within bounds. Received {unicast_cycles}")

    # assert unicast_cycles_within_bounds


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate reference outputs for LLaMA accuracy testing.")
    parser.add_argument("-p", "--profile", action="store_true")
    args = parser.parse_args()
    run_dm_tests(args.profile)
