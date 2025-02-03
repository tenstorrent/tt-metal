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


def profile_results(sample_size, sample_count, channel_count, num_writes_skip_barrier):
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
        "SAMPLE_SIZE",
        "NUM_WRITES_BEFORE_BARRIER",
        "BW (B/c)",
    ]
    write_header = not os.path.exists(FILE_NAME)
    append_to_csv(
        FILE_NAME,
        header,
        [sample_size, num_writes_skip_barrier, bw],
        write_header,
    )
    return main_loop_latency


@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("sample_count", [1])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_writes_skip_barrier", [1])
@pytest.mark.parametrize("sample_size", [16])
def test_erisc_write_worker_latency(sample_count, sample_size, channel_count, num_writes_skip_barrier):
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    ARCH_NAME = os.getenv("ARCH_NAME")
    cmd = f"\
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_ethernet_write_worker_latency_no_edm_{ARCH_NAME} \
                {sample_count} \
                {sample_size} \
                {channel_count} \
                {num_writes_skip_barrier} \
    #         "
    print(cmd)
    rc = os.system(cmd)
    if rc != 0:
        print("Error in running the test")
        assert False

    main_loop_latency = profile_results(sample_size, sample_count, channel_count, num_writes_skip_barrier)
    print(f"sender_loop_latency {main_loop_latency}")
    print(f"result BW (B/c): {sample_size / main_loop_latency}")

    return True
