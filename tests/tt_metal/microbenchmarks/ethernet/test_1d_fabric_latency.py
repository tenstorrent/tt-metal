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
    freq_hz = get_device_freq() * 1000 * 1000
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
    devices = list(devices_data["devices"].keys())

    # MAIN-TEST-BODY
    # device = devices[latency_measurement_worker_line_index]
    device = devices[0]
    latency_avg_cycles = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"][
        "Average"
    ]
    latency_max_cycles = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"][
        "Max"
    ]
    latency_min_cycles = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"][
        "Min"
    ]
    Hz_per_GHz = 1e9
    latency_avg_ns = latency_avg_cycles / (freq_hz / Hz_per_GHz)
    latency_max_ns = latency_max_cycles / (freq_hz / Hz_per_GHz)
    latency_min_ns = latency_min_cycles / (freq_hz / Hz_per_GHz)
    count = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][main_test_body_string]["stats"]["Count"]

    return latency_avg_ns, latency_min_ns, latency_max_ns, count


def run_latency_test(
    topology,
    line_size,
    latency_measurement_worker_line_index,
    latency_ping_message_size_bytes,
    latency_ping_burst_size,
    latency_ping_burst_count,
    add_upstream_fabric_congestion_writers,
    num_downstream_fabric_congestion_writers,
    congestion_writers_message_size,
    congestion_writers_use_mcast,
    enable_fused_payload_with_sync,
    expected_mean_latency_ns,
    expected_min_latency_ns,
    expected_max_latency_ns,
    expected_avg_hop_latency_ns,
):
    logger.warning("removing file profile_log_device.csv")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    cmd = f"TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/ttnn/unit_tests_ttnn_1d_fabric_latency \
                {line_size} \
                {latency_measurement_worker_line_index} \
                {latency_ping_message_size_bytes} \
                {latency_ping_burst_size} \
                {latency_ping_burst_count} \
                {int(add_upstream_fabric_congestion_writers)} \
                {num_downstream_fabric_congestion_writers} \
                {congestion_writers_message_size} \
                {int(congestion_writers_use_mcast)} \
                {int(enable_fused_payload_with_sync)} \
                {topology}"
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
    num_hops = (line_size - 1 - latency_measurement_worker_line_index) * 2 if topology != "ring" else line_size
    avg_hop_latency = latency_avg_ns / num_hops
    logger.info("latency_ns: {} ns", latency_avg_ns)
    allowable_delta = expected_mean_latency_ns * 0.05
    print(f"latency_min_ns: {latency_min_ns}")
    print(f"latency_max_ns: {latency_max_ns}")
    print(f"count: {count}")
    print(f"avg_hop_latency: {avg_hop_latency}")

    min_max_latency_lower_bound_threshold_percent = 0.93
    lower_bound_threshold_percent = 0.95
    upper_bound_threshold_percent = 1.05
    min_max_latency_upper_bound_threshold_percent = 1.08

    assert latency_avg_ns <= expected_mean_latency_ns + allowable_delta
    assert latency_min_ns <= expected_min_latency_ns * min_max_latency_upper_bound_threshold_percent
    assert latency_max_ns <= expected_max_latency_ns * min_max_latency_upper_bound_threshold_percent
    assert avg_hop_latency <= expected_avg_hop_latency_ns * upper_bound_threshold_percent

    is_under_avg_lower_bound = latency_avg_ns <= expected_mean_latency_ns - allowable_delta
    is_under_min_lower_bound = latency_min_ns <= expected_min_latency_ns * min_max_latency_lower_bound_threshold_percent
    is_under_max_lower_bound = latency_max_ns <= expected_max_latency_ns * min_max_latency_lower_bound_threshold_percent
    is_under_avg_hop_lower_bound = avg_hop_latency <= expected_avg_hop_latency_ns * lower_bound_threshold_percent

    if is_under_avg_lower_bound or is_under_min_lower_bound or is_under_max_lower_bound or is_under_avg_hop_lower_bound:
        logger.warning(
            f"Some measured values were under (better) than the expected values (including margin). Please update targets accordingly."
        )
        assert expected_mean_latency_ns - allowable_delta <= latency_avg_ns
        assert expected_min_latency_ns * min_max_latency_lower_bound_threshold_percent <= latency_min_ns
        assert expected_max_latency_ns * min_max_latency_lower_bound_threshold_percent <= latency_max_ns
        assert expected_avg_hop_latency_ns * lower_bound_threshold_percent <= avg_hop_latency


#####################################
##        Multicast Tests
#####################################


# 1D All-to-All Multicast
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize(
    "latency_ping_message_size_bytes,latency_measurement_worker_line_index,enable_fused_payload_with_sync, expected_mean_latency_ns,expected_min_latency_ns,expected_max_latency_ns,expected_avg_hop_latency_ns",
    [
        (0, 0, False, 10625, 10300, 11000, 760),
        (0, 1, False, 9000, 8680, 9430, 750),
        (4096, 1, False, 15750, 15500, 16200, 1310),
        (0, 2, False, 7550, 7240, 7840, 755),
        (0, 3, False, 6160, 5850, 6580, 770),
        (0, 4, False, 4680, 4450, 4975, 780),
        (0, 5, False, 3160, 2975, 3420, 790),
        (0, 6, False, 1520, 1420, 1680, 760),
        (16, 6, False, 1520, 1400, 1550, 760),
        (16, 6, True, 1535, 1425, 1700, 770),
        (1024, 6, False, 2000, 1820, 2150, 1000),
        (2048, 6, False, 2240, 2150, 2290, 1120),
        (4096, 6, False, 2600, 2520, 2770, 1300),
    ],
)
@pytest.mark.parametrize("latency_ping_burst_size", [1])
@pytest.mark.parametrize("latency_ping_burst_count", [200])
@pytest.mark.parametrize("add_upstream_fabric_congestion_writers", [False])
@pytest.mark.parametrize("num_downstream_fabric_congestion_writers", [0])
@pytest.mark.parametrize("congestion_writers_message_size", [0])
@pytest.mark.parametrize("congestion_writers_use_mcast", [False])
def test_1D_line_fabric_latency_on_uncongested_fabric(
    line_size,
    latency_measurement_worker_line_index,
    enable_fused_payload_with_sync,
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
    expected_avg_hop_latency_ns,
):
    run_latency_test(
        "linear",
        line_size,
        latency_measurement_worker_line_index,
        latency_ping_message_size_bytes,
        latency_ping_burst_size,
        latency_ping_burst_count,
        add_upstream_fabric_congestion_writers,
        num_downstream_fabric_congestion_writers,
        congestion_writers_message_size,
        congestion_writers_use_mcast,
        enable_fused_payload_with_sync,
        expected_mean_latency_ns,
        expected_min_latency_ns,
        expected_max_latency_ns,
        expected_avg_hop_latency_ns,
    )


# 1D All-to-All Multicast
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize(
    "latency_ping_message_size_bytes,latency_measurement_worker_line_index,enable_fused_payload_with_sync, expected_mean_latency_ns,expected_min_latency_ns,expected_max_latency_ns,expected_avg_hop_latency_ns",
    [
        (0, 0, False, 3320, 2880, 3520, 805),
        (16, 0, False, 3130, 2840, 3400, 780),
        (16, 0, True, 3170, 2860, 3420, 790),
        (1024, 0, False, 3920, 3580, 4310, 975),
        (2048, 0, False, 4470, 4220, 4730, 1115),
        (4096, 0, False, 5310, 5050, 5700, 1330),
    ],
)
@pytest.mark.parametrize("latency_ping_burst_size", [1])
@pytest.mark.parametrize("latency_ping_burst_count", [62])
@pytest.mark.parametrize("add_upstream_fabric_congestion_writers", [False])
@pytest.mark.parametrize("num_downstream_fabric_congestion_writers", [0])
@pytest.mark.parametrize("congestion_writers_message_size", [0])
@pytest.mark.parametrize("congestion_writers_use_mcast", [False])
def test_1D_ring_fabric_latency_on_uncongested_fabric(
    line_size,
    latency_measurement_worker_line_index,
    enable_fused_payload_with_sync,
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
    expected_avg_hop_latency_ns,
):
    run_latency_test(
        "ring",
        line_size,
        latency_measurement_worker_line_index,
        latency_ping_message_size_bytes,
        latency_ping_burst_size,
        latency_ping_burst_count,
        add_upstream_fabric_congestion_writers,
        num_downstream_fabric_congestion_writers,
        congestion_writers_message_size,
        congestion_writers_use_mcast,
        enable_fused_payload_with_sync,
        expected_mean_latency_ns,
        expected_min_latency_ns,
        expected_max_latency_ns,
        expected_avg_hop_latency_ns,
    )
