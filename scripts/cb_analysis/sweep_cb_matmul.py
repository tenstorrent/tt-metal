# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import os
from pathlib import Path
import csv
import ttnn
import numpy as np
import shutil
import datetime

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG, rm

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

test_to_run = {
    "dm": "TT_METAL_DEVICE_PROFILER=1 pytest tests/didt/test_lm_head_matmul.py::test_lm_head_matmul -k '1chips' --didt-workload-iterations 1",
    "compute": "TT_METAL_DEVICE_PROFILER=1 pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k 'with_gelu and 1chips' --didt-workload-iterations 1",
}

test_variants = [
    "zone",
    "counter",
]

stas_header = [
    "core_compute_cb_wait_front",
    "core_compute_cb_reserve_back",
    "core_writer_cb_wait_front",
    "core_reader_cb_reserve_back",
]


def get_test_case_name(test_case):
    if test_case == "dm":
        return "1D matmul - DM bound"
    elif test_case == "compute":
        return "2D matmul with gelu - Compute bound"


def update_hpp(variant):
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
    hpp_file = TT_METAL_HOME / "tt_metal" / "tools" / "profiler" / "test_cb.hpp"

    # Ensure the directory exists
    hpp_file.parent.mkdir(parents=True, exist_ok=True)

    if variant == "zone":
        # Create empty file
        with open(hpp_file, "w") as f:
            pass  # Creates empty file
        logger.info(f"Created empty {hpp_file} for zone variant")
    elif variant == "counter":
        # Write the define line
        with open(hpp_file, "w") as f:
            f.write("#define USE_ZONE_COUNTER 1\n")
        logger.info(f"Created {hpp_file} with USE_ZONE_COUNTER define for counter variant")
    else:
        logger.warning(f"Unknown variant: {variant}. No changes made to {hpp_file}")


def get_profiler_stats(log_path, header):
    setup = device_post_proc_config.perf_analysis()
    setup.deviceInputLog = log_path
    device_data = import_log_run_stats(setup)
    analysis_data = device_data["devices"][0]["cores"]["DEVICE"]["analysis"]
    analysis_res = []

    for entry in header:
        if entry not in analysis_data:
            analysis_res.append(0.0)
        else:
            analysis_res.append(round(analysis_data[entry]["stats"]["Average"], 0))

    return analysis_res


def get_bottleneck_type(max_index):
    """Convert stats index to bottleneck description"""
    bottleneck_map = {
        0: "DM reader",  # core_compute_cb_wait_front
        1: "DM writer",  # core_compute_cb_reserve_back
        2: "Compute pack",  # core_writer_cb_wait_front
        3: "Compute unpack",  # core_reader_cb_reserve_back
    }
    return bottleneck_map.get(max_index, "Unknown")


def test_matmul_cb_analysis():
    """Run cb analysis for matmul and save results to csv"""
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = TT_METAL_HOME / "cb_matmul_length_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create one combined CSV file for all stats
    combined_stats_path = log_dir / "cb_matmul_length_all_stats.csv"

    with open(combined_stats_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        header = ["test case", "variant"] + stas_header + ["DM/Compute bound"]
        csv_writer.writerow(header)

        for variant in test_variants:
            for test_case, command in test_to_run.items():
                # Clear profiler log file if it exists
                if os.path.exists(profiler_log_path):
                    rm(profiler_log_path)

                # Update the test_cb.hpp file based on the variant
                update_hpp(variant)

                # Run the test and get the profiler stats
                logger.info(f"Running {get_test_case_name(test_case)} with variant {variant}")
                os.system(command)
                assert os.path.exists(profiler_log_path), "Profiler log file does not exist."

                # Copy raw profiler log file
                log_filename = f"{test_case}_{variant}.csv"
                log_path = log_dir / log_filename
                shutil.copy2(profiler_log_path, log_path)

                # Get the profiler stats
                stats = get_profiler_stats(profiler_log_path, stas_header)

                # Determine the bottleneck type
                max_index = np.argmax(stats)
                bottleneck_type = get_bottleneck_type(max_index)

                # Write the stats to the CSV
                csv_writer.writerow([get_test_case_name(test_case), variant] + stats + [bottleneck_type])
