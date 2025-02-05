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

FILE_NAME = PROFILER_LOGS_DIR / "test_ethernet_link_write_worker_latency.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


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


def profile_results(sample_size, sample_count, channel_count, num_directions):
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
    main_loop_latency = main_loop_cycle / freq / sample_count / channel_count
    bw = sample_size / main_loop_latency

    header = [
        "NUM_DIRECTIONS",
        "SAMPLE_SIZE",
        "BW (B/c)",
    ]
    write_header = not os.path.exists(FILE_NAME)
    append_to_csv(
        FILE_NAME,
        header,
        [num_directions, sample_size, bw],
        write_header,
    )
    return main_loop_latency


def run_erisc_write_worker(sample_count, sample_size_expected_latency, channel_count, num_directions):
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
                {num_directions} "
    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    main_loop_latency = profile_results(sample_size, sample_count, channel_count, num_directions)
    logger.info(f"sender_loop_latency {main_loop_latency}")
    logger.info(f"result BW (B/c): {sample_size / main_loop_latency}")

    assert expected_latency_lower_bound <= main_loop_latency <= expected_latency_upper_bound


# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [1])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [(16, 97.2), (128, 97.2), (256, 98.0), (512, 98.0), (1024, 99.0), (2048, 173.0), (4096, 340.0), (8192, 678.5)],
)
def test_erisc_write_worker_bw_uni_dir(sample_count, sample_size_expected_latency, channel_count, num_directions):
    run_erisc_write_worker(sample_count, sample_size_expected_latency, channel_count, num_directions)


# bi-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_directions", [2])
@pytest.mark.parametrize(
    "sample_size_expected_latency",
    [(16, 148.0), (128, 148.0), (256, 148.7), (512, 148.8), (1024, 149.2), (2048, 178.2), (4096, 344.2)],
)
def test_erisc_write_worker_bw_bi_dir(sample_count, sample_size_expected_latency, channel_count, num_directions):
    run_erisc_write_worker(sample_count, sample_size_expected_latency, channel_count, num_directions)
