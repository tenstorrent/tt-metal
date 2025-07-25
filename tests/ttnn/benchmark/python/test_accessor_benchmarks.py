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

Make sure that tt-metal is built with profiler enabled, e.g.:
./build_metal.sh --release --enable-ccache --build-tests -e -p
"""

# Bits Encoding: [Bank coordinates][Tensor shape][Shard shape][Number of banks][Rank]01
# 1 means dynamic, 0 means static
# Last 01 means that data is in l1 (not dram) and sharded
ARGS_CONFIGS = [
    "0000001",  # Everything static
    "0010001",  # dynamic tensor shape
    "0100001",  # dynamic shard shape
    "0110001",  # dynamic tensor shape and shard shape
    "0110101",  # static number of banks and banks coordinates
    "1000001",  # dynamic bank coordinates
    "1001001",  # dynamic bank coordinates and number of banks
    "1010001",  # dynamic bank coordinates and tensor shape
    "1011001",  # static rank and shard shape
    "1100001",  # dynamic bank coordinates and shard shape
    "1101001",  # static rank and tensor shape
    "1110001",  # static rank and number of banks
    "1110101",  # static number of banks
    "1111001",  # static rank
    "1111101",  # Everything dynamic
]


def impl_test(gtest_filter, res_dir):
    ENV = os.environ.copy()
    ENV["TT_METAL_DEVICE_PROFILER"] = "1"
    BASE = Path(ENV["TT_METAL_HOME"])

    binary_path = Path(BASE / "build" / "test" / "ttnn" / "unit_tests_ttnn_accessor")
    subprocess.run([binary_path, f"--gtest_filter={gtest_filter}"], env=ENV)

    setup = device_post_proc_config.default_setup()
    zone_names = []
    timerAnalysis = {}
    for args_config in ARGS_CONFIGS:
        zone_name = f"SHARDED_ACCESSOR_{args_config}"
        timerAnalysis[zone_name] = {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": zone_name},
            "end": {"risc": "BRISC", "zone_name": zone_name},
        }
        zone_names.append(zone_name)
    setup.timerAnalysis = timerAnalysis

    results_dir = Path(BASE / res_dir)
    profile_log_file = Path(".logs/profile_log_device.csv")
    for test_dir in sorted(results_dir.iterdir(), key=natural_key_from_path):
        if not test_dir.is_dir():
            raise IsADirectoryError(f"Expected {test_dir} to be a directory containing {profile_log_file}")
        profile_log_path = test_dir / "profile_log_device.csv"
        setup.deviceInputLog = profile_log_path

        stats = import_log_run_stats(setup)
        logger.info(f"Results for {test_dir}:")
        for zone_name in zone_names:
            core = [key for key in stats["devices"][0]["cores"].keys() if key != "DEVICE"][0]
            st = stats["devices"][0]["cores"][core]["riscs"]["TENSIX"]["analysis"][zone_name]["stats"]
            logger.info(f"Zone: {zone_name}: Average: {st['Average']} (cycles)")


def test_get_noc_addr_page_id():
    impl_test("AccessorTests/AccessorBenchmarks.GetNocAddr/*", res_dir="accessor_get_noc_addr_benchmarks")


def test_get_noc_addr_page_coord():
    impl_test(
        "AccessorTests/AccessorBenchmarks.GetNocAddrPageCoord/*", res_dir="accessor_get_noc_addr_page_coord_benchmarks"
    )


def test_constructor():
    impl_test("AccessorTests/AccessorBenchmarks.Constructor/*", res_dir="accessor_constructor_benchmarks")
