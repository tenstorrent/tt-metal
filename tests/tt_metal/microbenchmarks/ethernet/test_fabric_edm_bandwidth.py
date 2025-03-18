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


def profile_results(zone_name, packets_per_src_chip, line_size, packet_size):
    freq = get_device_freq() / 1000.0
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    setup.timerAnalysis = {
        zone_name: {
            "across": "device",
            "type": "session_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": zone_name},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": zone_name},
        },
    }
    devices_data = import_log_run_stats(setup)
    devices = list(devices_data["devices"].keys())

    # MAIN-TEST-BODY
    main_loop_cycles = []
    for device in devices:
        main_loop_cycle = devices_data["devices"][device]["cores"]["DEVICE"]["analysis"][zone_name]["stats"]["Average"]
        main_loop_cycles.append(main_loop_cycle)

    traffic_streams_through_boundary = line_size / 2
    total_byte_sent = packets_per_src_chip * traffic_streams_through_boundary * packet_size
    bandwidth = total_byte_sent / max(main_loop_cycles)

    return bandwidth


def run_fabric_edm(
    is_unicast,
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    ring_topology,
    expected_bw,
):
    logger.warning("removing file profile_log_device.csv")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/ttnn/unit_tests_ttnn_fabric_edm \
                {num_mcasts} \
                {num_unicasts} \
                {num_links} \
                {num_op_invocations} \
                {int(line_sync)} \
                {line_size} \
                {packet_size} \
                {int(ring_topology)}"
    rc = os.system(cmd)
    if rc != 0:
        if os.WEXITSTATUS(rc) == 1:
            pytest.skip("Skipping test because it only works with T3000")
            return
        logger.info("Error in running the test")
        assert False

    zone_name_inner = "MAIN-WRITE-UNICAST-ZONE" if is_unicast else "MAIN-WRITE-MCAST-ZONE"
    zone_name_main = "MAIN-TEST-BODY"
    packets_per_src_chip = num_mcasts, num_unicasts

    num_messages = num_mcasts + num_unicasts
    bandwidth_inner_loop = profile_results(zone_name_inner, num_messages, line_size, packet_size)
    bandwidth = profile_results(zone_name_main, num_messages, line_size, packet_size)
    logger.info("bandwidth_inner_loop: {} B/c", bandwidth_inner_loop)
    logger.info("bandwidth: {} B/c", bandwidth)
    assert expected_bw - 0.07 <= bandwidth <= expected_bw + 0.07


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("num_links, expected_bw", [(1, 6.78), (2, 6.65)])
def test_fabric_edm_mcast_ring_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    expected_bw,
):
    run_fabric_edm(
        False,
        num_mcasts,
        num_unicasts,
        num_links,
        num_op_invocations,
        line_sync,
        line_size,
        packet_size,
        True,
        expected_bw,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("num_links, expected_bw", [(1, 6.72), (2, 5.87)])
def test_fabric_edm_mcast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    expected_bw,
):
    run_fabric_edm(
        False,
        num_mcasts,
        num_unicasts,
        num_links,
        num_op_invocations,
        line_sync,
        line_size,
        packet_size,
        False,
        expected_bw,
    )


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("num_links, expected_bw", [(1, 8.60), (2, 7.78)])
def test_fabric_edm_unicast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    expected_bw,
):
    run_fabric_edm(
        True,
        num_mcasts,
        num_unicasts,
        num_links,
        num_op_invocations,
        line_sync,
        line_size,
        packet_size,
        False,
        expected_bw,
    )
