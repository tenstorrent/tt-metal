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

# CPP test parameters
# (x, y, z) --> reader, writer, compute kernel funcionality
# argument = 0 --> no functionality in kernel, only CB sync
# argument = 9999 --> use default behavoir of kernel, NOC transaction for DM or data processing for compute
# argument > 0 && argument != 9999 --> execute riscv wait of argument cycles in the kernel, no noc or compute
test_config = [(0, 0, 0), (1000, 0, 0), (0, 1000, 0), (0, 0, 1000), (9999, 9999, 9999)]


def config_to_str(config):
    if config == 0:
        return "cb"
    elif config == 9999:
        return "noc/compute"
    else:
        return f"wait {config} cycles"


test_variants = [
    "baseline",
    "zone",
    "counter",
]

stas_header = [
    "trisc0_compute_block_duration",
    "trisc1_compute_block_duration",
    "trisc2_compute_block_duration",
    "reader_block_duration",
    "writer_block_duration",
    "core_compute_cb_wait_front",
    "core_compute_cb_reserve_back",
    "core_writer_cb_wait_front",
    "core_reader_cb_reserve_back",
]


def get_profiler_stats(log_path):
    setup = device_post_proc_config.perf_analysis()
    setup.deviceInputLog = log_path
    device_data = import_log_run_stats(setup)
    analysis_data = device_data["devices"][0]["cores"]["DEVICE"]["analysis"]
    analysis_res = []

    for entry in stas_header:
        if entry not in analysis_data:
            analysis_res.append(0.0)
        else:
            analysis_res.append(analysis_data[entry]["stats"]["Average"])

    return analysis_res


def get_bottleneck_type(max_index):
    """Convert stats index to bottleneck description"""
    bottleneck_map = {
        5: "DM reader",  # core_compute_cb_wait_front
        6: "DM writer",  # core_compute_cb_reserve_back
        7: "Compute pack",  # core_writer_cb_wait_front
        8: "Compute unpack",  # core_reader_cb_reserve_back
    }
    return bottleneck_map.get(max_index, "Unknown")


def test_cb_length():
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = TT_METAL_HOME / "cb_length_logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create one combined CSV file for all stats
    combined_stats_path = log_dir / "cb_length_all_stats.csv"

    with open(combined_stats_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header with variant, config, and stats columns
        header = ["variant", "reader", "writer", "compute"] + stas_header + ["DM/Compute bound"]
        csv_writer.writerow(header)

        for variant in test_variants:
            for reader, writer, compute in test_config:
                logger.info(
                    f"Testing variant: {variant} with reader: {config_to_str(reader)}, writer: {config_to_str(writer)}, compute: {config_to_str(compute)}"
                )

                # Clear profiler log file if it exists
                if os.path.exists(profiler_log_path):
                    rm(profiler_log_path)

                cmd = f"TT_METAL_DEVICE_PROFILER=1 ./build_Release_tracy/test/tt_metal/test_eltwise_binary --reader {reader} --writer {writer} --compute {compute}"
                if variant == "zone":
                    cmd += " --measure-cb-timings"
                if variant == "counter":
                    cmd += " --measure-cb-timings --use-zone-counter"
                os.system(cmd)
                assert os.path.exists(profiler_log_path), "Profiler log file does not exist."

                # Copy raw profiler log file
                log_filename = f"{variant}_r{reader}_w{writer}_c{compute}.csv"
                log_path = log_dir / log_filename
                shutil.copy2(profiler_log_path, log_path)

                # Get stats and write to combined CSV file
                stats = get_profiler_stats(profiler_log_path)

                # Determine bottleneck from last 4 entries only
                if variant != "baseline":
                    last_4_stats = stats[-4:]  # Get last 4 entries
                    max_value = max(last_4_stats)
                    local_max_index = last_4_stats.index(max_value)
                    global_max_index = local_max_index + 5  # Add offset to get global index (5 = len(stats) - 4)

                    bottleneck = get_bottleneck_type(global_max_index)
                else:
                    bottleneck = "N/A"

                # Combine variant, config, and stats data
                row_data = (
                    [variant, config_to_str(reader), config_to_str(writer), config_to_str(compute)]
                    + stats
                    + [bottleneck]
                )
                csv_writer.writerow(row_data)
                csvfile.flush()  # Ensure data is written immediately
