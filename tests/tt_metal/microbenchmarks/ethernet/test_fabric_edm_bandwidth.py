# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from enum import Enum
from loguru import logger
import pytest
import csv
from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


# Python enum mirroring test_fabric_edm_common.hpp
class FabricTestMode(Enum):
    Linear = 0
    HalfRing = 1
    FullRing = 2
    SaturateChipToChipRing = 3


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def profile_results(
    zone_name, packets_per_src_chip, line_size, packet_size, fabric_mode, disable_sends_for_interior_workers
):
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

    if fabric_mode == FabricTestMode.FullRing:
        traffic_streams_through_boundary = line_size - 1
    elif fabric_mode == FabricTestMode.SaturateChipToChipRing:
        traffic_streams_through_boundary = 3
    else:
        traffic_streams_through_boundary = line_size / 2
        if disable_sends_for_interior_workers:
            traffic_streams_through_boundary = 1
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
    fabric_mode,
    expected_bw,
    disable_sends_for_interior_workers,
):
    logger.warning("removing file profile_log_device.csv")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    cmd = f"TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/ttnn/unit_tests_ttnn_fabric_edm \
                {num_mcasts} \
                {num_unicasts} \
                {num_links} \
                {num_op_invocations} \
                {int(line_sync)} \
                {line_size} \
                {packet_size} \
                {fabric_mode.value} \
                {int(disable_sends_for_interior_workers)}"
    rc = os.system(cmd)
    if rc != 0:
        if os.WEXITSTATUS(rc) == 1:
            pytest.skip("Skipping test because it only works with T3000")
            return
        logger.info("Error in running the test")
        assert False

    zone_name_inner = "MAIN-WRITE-UNICAST-ZONE" if is_unicast else "MAIN-WRITE-MCAST-ZONE"
    zone_name_main = "MAIN-TEST-BODY"

    num_messages = num_mcasts + num_unicasts
    bandwidth_inner_loop = profile_results(
        zone_name_inner, num_messages, line_size, packet_size, fabric_mode, disable_sends_for_interior_workers
    )
    bandwidth = profile_results(
        zone_name_main, num_messages, line_size, packet_size, fabric_mode, disable_sends_for_interior_workers
    )
    logger.info("bandwidth_inner_loop: {} B/c", bandwidth_inner_loop)
    logger.info("bandwidth: {} B/c", bandwidth)
    assert expected_bw - 0.07 <= bandwidth <= expected_bw + 0.07


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("line_size, num_links, expected_bw", [(4, 1, 8.23), (4, 2, 8.17)])
def test_fabric_edm_mcast_half_ring_bw(
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
        FabricTestMode.HalfRing,
        expected_bw,
        False,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("line_size, num_links, expected_bw", [(4, 1, 5.81), (4, 2, 5.75), (8, 1, 4.46)])
def test_fabric_edm_mcast_full_ring_bw(
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
        FabricTestMode.FullRing,
        expected_bw,
        False,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("line_size, num_links, expected_bw", [(4, 1, 6.72), (4, 2, 6.54)])
def test_fabric_edm_mcast_saturate_chip_to_chip_ring_bw(
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
        FabricTestMode.SaturateChipToChipRing,
        expected_bw,
        False,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("line_size, num_links, expected_bw", [(4, 1, 7.75), (4, 2, 7.75)])
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
        FabricTestMode.Linear,
        expected_bw,
        False,
    )


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("num_links, expected_bw", [(1, 11.03), (2, 11.03)])
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
        FabricTestMode.Linear,
        expected_bw,
        False,
    )


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("num_links, expected_bw", [(1, 9.74), (2, 9.63)])
def test_fabric_edm_unicast_multiproducer_multihop_bw(
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
        FabricTestMode.Linear,
        expected_bw,
        False,
    )


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("num_links, expected_bw", [(1, 9.35), (2, 9.26)])
def test_fabric_edm_unicast_single_producer_multihop_bw(
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
        FabricTestMode.Linear,
        expected_bw,
        True,
    )
