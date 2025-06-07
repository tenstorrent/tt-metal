# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import os
import re
import subprocess
import pytest
from loguru import logger
from pathlib import Path

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


# Helper function to sort strings naturally
def natural_key_from_path(s):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s.name)]


"""
Runs tests/ttnn/unit_tests/gtests/accessor/test_accessor_benchmarks.cpp test and parses output.
See the C++ test for more details on what is benchmarked and generated.
"""


def test_accessors():
    ENV = os.environ.copy()
    ENV["TT_METAL_DEVICE_PROFILER"] = "1"
    BASE = Path(ENV["TT_METAL_HOME"])

    binary_path = Path(BASE / "build" / "test" / "ttnn" / "unit_tests_ttnn_accessor")
    subprocess.run([binary_path, "--gtest_filter=AccessorTests/AccessorBenchmarks.*"], env=ENV)

    setup = device_post_proc_config.default_setup()
    setup.timerAnalysis = {
        "SHARDED_ACCESSOR_CTA_GET_NOC_ADDR": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "SHARDED_ACCESSOR_CTA"},
            "end": {"risc": "BRISC", "zone_name": "SHARDED_ACCESSOR_CTA"},
        },
        "SHARDED_ACCESSOR_CRTA_DDS_GET_NOC_ADDR": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "SHARDED_ACCESSOR_CRTA_DDS"},
            "end": {"risc": "BRISC", "zone_name": "SHARDED_ACCESSOR_CRTA_DDS"},
        },
        "SHARDED_ACCESSOR_CRTA_SSD_GET_NOC_ADDR": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "SHARDED_ACCESSOR_CRTA_SSD"},
            "end": {"risc": "BRISC", "zone_name": "SHARDED_ACCESSOR_CRTA_SSD"},
        },
        "SHARDED_ACCESSOR_CRTA_DDD_GET_NOC_ADDR": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "SHARDED_ACCESSOR_CRTA_DDD"},
            "end": {"risc": "BRISC", "zone_name": "SHARDED_ACCESSOR_CRTA_DDD"},
        },
        "INTERLEAVED_ACCESSOR_GET_NOC_ADDR": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "INTERLEAVED_ACCESSOR"},
            "end": {"risc": "BRISC", "zone_name": "INTERLEAVED_ACCESSOR"},
        },
    }

    results_dir = Path(BASE / "accessor_benchmarks")
    profile_log_file = Path(".logs/profile_log_device.csv")
    for test_dir in sorted(results_dir.iterdir(), key=natural_key_from_path):
        if not test_dir.is_dir():
            raise IsADirectoryError(f"Expected {test_dir} to be a directory containing {profile_log_file}")
        profile_log_path = test_dir / "profile_log_device.csv"
        setup.deviceInputLog = profile_log_path

        stats = import_log_run_stats(setup)
        core = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"][0]
        sharded_stats_cta = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"][
            "SHARDED_ACCESSOR_CTA_GET_NOC_ADDR"
        ]["stats"]
        sharded_stats_crta_dds = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"][
            "SHARDED_ACCESSOR_CRTA_DDS_GET_NOC_ADDR"
        ]["stats"]
        sharded_stats_crta_ssd = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"][
            "SHARDED_ACCESSOR_CRTA_SSD_GET_NOC_ADDR"
        ]["stats"]
        sharded_stats_crta_ddd = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"][
            "SHARDED_ACCESSOR_CRTA_DDD_GET_NOC_ADDR"
        ]["stats"]
        interleaved_stats = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"][
            "INTERLEAVED_ACCESSOR_GET_NOC_ADDR"
        ]["stats"]

        logger.info(f"Results for {test_dir}:")
        logger.info(f"Sharded Stats: {sharded_stats_cta}")
        logger.info(f"Sharded CRTA DDS Stats: {sharded_stats_crta_dds}")
        logger.info(f"Sharded CRTA SSD Stats: {sharded_stats_crta_ssd}")
        logger.info(f"Sharded CRTA DDD Stats: {sharded_stats_crta_ddd}")
        logger.info(f"Interleaved Stats: {interleaved_stats}")
        logger.info(f"Sharded Average: {sharded_stats_cta['Average']} (cycles)")
        logger.info(f"Sharded CRTA DDS Average: {sharded_stats_crta_dds['Average']} (cycles)")
        logger.info(f"Sharded CRTA SSD Average: {sharded_stats_crta_ssd['Average']} (cycles)")
        logger.info(f"Sharded CRTA DDD Average: {sharded_stats_crta_ddd['Average']} (cycles)")
        logger.info(f"Interleaved Average: {interleaved_stats['Average']} (cycles)")
