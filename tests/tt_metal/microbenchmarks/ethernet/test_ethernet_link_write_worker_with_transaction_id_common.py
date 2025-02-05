# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
import csv
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config

from models.utility_functions import is_grayskull

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def append_to_csv(file_path, header, data, write_header=True):
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or write_header:
            writer.writerow(header)
        writer.writerows([data])


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def profile_results(sample_size, sample_count, channel_count, num_directions, test_latency, file_name):
    freq = get_device_freq() / 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    main_test_body_string = "MAIN-TEST-BODY"
    setup.timerAnalysis = {
        main_test_body_string: {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ERISC", "zone_name": main_test_body_string},
            "end": {"core": "ANY", "risc": "ERISC", "zone_name": main_test_body_string},
        },
    }
    devices_data = import_log_run_stats(setup)
    device_0 = list(devices_data["devices"].keys())[0]
    device_1 = list(devices_data["devices"].keys())[1]

    # MAIN-TEST-BODY
    main_loop_cycle = devices_data["devices"][device_0]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"][
        "Average"
    ]

    if test_latency == 1:
        main_loop_latency = main_loop_cycle / freq
        header = [
            "NUM_DIRECTIONS",
            "SAMPLE_SIZE",
            "LATENCY (ns)",
        ]
        res = main_loop_latency
    else:
        main_loop_latency = main_loop_cycle / freq / sample_count / channel_count
        bw = sample_size / main_loop_latency
        header = [
            "NUM_DIRECTIONS",
            "SAMPLE_SIZE",
            "BW (B/c)",
        ]
        res = bw
    write_header = not os.path.exists(file_name)
    append_to_csv(
        file_name,
        header,
        [num_directions, sample_size, res],
        write_header,
    )
    return main_loop_latency


def run_erisc_write_worker(
    sample_count, sample_size_expected_latency, channel_count, num_directions, test_latency, enable_worker, file_name
):
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    sample_size = sample_size_expected_latency[0]
    expected_latency = sample_size_expected_latency[1]
    expected_latency_lower_bound = expected_latency - 0.5
    expected_latency_upper_bound = expected_latency + 0.5

    ARCH_NAME = os.getenv("ARCH_NAME")
    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_write_worker_latency_no_edm_{ARCH_NAME} \
                {sample_count} \
                {sample_size} \
                {channel_count} \
                {num_directions} \
                {test_latency} \
                {enable_worker}"
    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    main_loop_latency = profile_results(
        sample_size, sample_count, channel_count, num_directions, test_latency, file_name
    )
    logger.info(f"sender_loop_latency {main_loop_latency}")
    if test_latency != 1:
        logger.info(f"sender_loop_bw {sample_size / main_loop_latency}")

    assert expected_latency_lower_bound <= main_loop_latency <= expected_latency_upper_bound
