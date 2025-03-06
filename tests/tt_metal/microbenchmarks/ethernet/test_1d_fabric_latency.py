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

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def profile_results(
    line_size,
    latency_measurement_worker_line_index,
    latency_ping_message_size_bytes,
    latency_ping_burst_size,
    latency_ping_burst_count,
    add_upstream_fabric_congestion_writers,
    num_downstream_fabric_congestion_writers,
    congestion_writers_message_size,
    congestion_writers_use_mcast,
):
    freq = get_device_freq() / 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    main_test_body_string = "WAIT-FOR-ALL-SEMAPHORES"
    setup.timerAnalysis = {
        main_test_body_string: {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": main_test_body_string},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": main_test_body_string},
        },
    }
    devices_data = import_log_run_stats(setup)
    print(f"{list(devices_data['devices'].keys())}")
    devices = list(devices_data["devices"].keys())

    # MAIN-TEST-BODY
    # device = devices[latency_measurement_worker_line_index]
    device = devices[0]
    print(f"keys: {devices_data['devices'][device]['cores']['DEVICE'].keys()}")
    latency_avg_ns = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"][
        "Average"
    ]
    latency_max_ns = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"][
        "Max"
    ]
    latency_min_ns = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"][
        "Min"
    ]
    count = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"]["Count"]

    return latency_avg_ns, latency_min_ns, latency_max_ns, count


def run_latency_test(
    line_size,
    latency_measurement_worker_line_index,
    latency_ping_message_size_bytes,
    latency_ping_burst_size,
    latency_ping_burst_count,
    add_upstream_fabric_congestion_writers,
    num_downstream_fabric_congestion_writers,
    congestion_writers_message_size,
    congestion_writers_use_mcast,
    expected_mean_latency_ns,
    expected_min_latency_ns,
    expected_max_latency_ns,
):
    logger.warning("removing file profile_log_device.csv")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/ttnn/unit_tests_ttnn_1d_fabric_latency \
                {line_size} \
                {latency_measurement_worker_line_index} \
                {latency_ping_message_size_bytes} \
                {latency_ping_burst_size} \
                {latency_ping_burst_count} \
                {int(add_upstream_fabric_congestion_writers)} \
                {num_downstream_fabric_congestion_writers} \
                {congestion_writers_message_size} \
                {int(congestion_writers_use_mcast)}"
    rc = os.system(cmd)
    if rc != 0:
        if os.WEXITSTATUS(rc) == 1:
            pytest.skip("Skipping test because it only works with T3000")
            return
        logger.info("Error in running the test")
        assert False

    latency_avg_ns, latency_min_ns, latency_max_ns, count = profile_results(
        line_size,
        latency_measurement_worker_line_index,
        latency_ping_message_size_bytes,
        latency_ping_burst_size,
        latency_ping_burst_count,
        add_upstream_fabric_congestion_writers,
        num_downstream_fabric_congestion_writers,
        congestion_writers_message_size,
        congestion_writers_use_mcast,
    )
    logger.info("latency_ns: {} ns", latency_avg_ns)
    allowable_delta = expected_mean_latency_ns * 0.15
    print(f"latency_min_ns: {latency_min_ns}")
    print(f"latency_max_ns: {latency_max_ns}")
    print(f"count: {count}")

    assert (
        expected_mean_latency_ns - allowable_delta <= latency_avg_ns
        and latency_avg_ns <= expected_mean_latency_ns + allowable_delta
    )
    assert latency_min_ns <= expected_min_latency_ns * 1.1
    assert latency_max_ns <= expected_max_latency_ns * 1.05


#####################################
##        Multicast Tests
#####################################


# 1D All-to-All Multicast
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize(
    "latency_measurement_worker_line_index,expected_mean_latency_ns,expected_min_latency_ns,expected_max_latency_ns",
    [
        (0, 20000, 20000, 23500),
        (1, 10000, 9100, 19000),
        (2, 8500, 7600, 20700),
        (3, 6700, 6200, 8800),
        (4, 5000, 4700, 5300),
        (5, 3300, 3000, 3700),
        (6, 1690, 1550, 1850),
    ],
)
@pytest.mark.parametrize("latency_ping_burst_size", [1])
@pytest.mark.parametrize("latency_ping_burst_count", [200])
@pytest.mark.parametrize("add_upstream_fabric_congestion_writers", [False])
@pytest.mark.parametrize("num_downstream_fabric_congestion_writers", [0])
@pytest.mark.parametrize("congestion_writers_message_size", [0])
@pytest.mark.parametrize("congestion_writers_use_mcast", [False])
@pytest.mark.parametrize("latency_ping_message_size_bytes", [0])
def test_1D_fabric_latency_on_uncongested_fabric_minimal_packet_size(
    line_size,
    latency_measurement_worker_line_index,
    latency_ping_message_size_bytes,
    latency_ping_burst_size,
    latency_ping_burst_count,
    add_upstream_fabric_congestion_writers,
    num_downstream_fabric_congestion_writers,
    congestion_writers_message_size,
    congestion_writers_use_mcast,
    expected_mean_latency_ns,
    expected_min_latency_ns,
    expected_max_latency_ns,
):
    run_latency_test(
        line_size,
        latency_measurement_worker_line_index,
        latency_ping_message_size_bytes,
        latency_ping_burst_size,
        latency_ping_burst_count,
        add_upstream_fabric_congestion_writers,
        num_downstream_fabric_congestion_writers,
        congestion_writers_message_size,
        congestion_writers_use_mcast,
        expected_mean_latency_ns,
        expected_min_latency_ns,
        expected_max_latency_ns,
    )


# @pytest.mark.parametrize("line_size", [8])
# @pytest.mark.parametrize("latency_measurement_worker_line_index", [0])  # ,1,2,3,4,5,6])
# @pytest.mark.parametrize("latency_ping_burst_size", [1])
# @pytest.mark.parametrize("latency_ping_burst_count", [1])  # 000])
# @pytest.mark.parametrize("add_upstream_fabric_congestion_writers", [False])
# @pytest.mark.parametrize("num_downstream_fabric_congestion_writers", [0])
# @pytest.mark.parametrize("congestion_writers_message_size", [0])
# @pytest.mark.parametrize("congestion_writers_use_mcast", [False])
# @pytest.mark.parametrize("latency_ping_message_size_bytes,expected_mean_latency_ns", [(0, 1000)])  # , (512, 1300)])
# def test_1D_fabric_latency_on_congested_fabric(
#     line_size,
#     latency_measurement_worker_line_index,
#     latency_ping_message_size_bytes,
#     latency_ping_burst_size,
#     latency_ping_burst_count,
#     add_upstream_fabric_congestion_writers,
#     num_downstream_fabric_congestion_writers,
#     congestion_writers_message_size,
#     congestion_writers_use_mcast,
#     expected_mean_latency_ns,
# ):
#     run_latency_test(
#         line_size,
#         latency_measurement_worker_line_index,
#         latency_ping_message_size_bytes,
#         latency_ping_burst_size,
#         latency_ping_burst_count,
#         add_upstream_fabric_congestion_writers,
#         num_downstream_fabric_congestion_writers,
#         congestion_writers_message_size,
#         congestion_writers_use_mcast,
#         expected_mean_latency_ns,
#     )
