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
from tabulate import tabulate
import pandas as pd

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


def summarize_to_csv(
    test_name,
    packet_size,
    line_size,
    num_links,
    disable_sends_for_interior_workers,
    unidirectional,
    bandwidth,
    packets_per_second,
):
    """Write test results to a CSV file organized by packet size"""
    csv_path = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/bandwidth_summary.csv")

    # Create header if file doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Test Name",
                    "Packet Size",
                    "Line Size",
                    "Num Links",
                    "Disable Interior Workers",
                    "Unidirectional",
                    "Bandwidth (B/c)",
                    "Packets/Second",
                ]
            )

    # Append results
    with open(csv_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                test_name,
                packet_size,
                line_size,
                num_links,
                disable_sends_for_interior_workers,
                unidirectional,
                bandwidth,
                packets_per_second,
            ]
        )


def read_golden_results(
    test_name, packet_size, line_size, num_links, disable_sends_for_interior_workers, unidirectional
):
    """Print a summary table of all test results by packet size"""
    csv_path = os.path.join(
        os.environ["TT_METAL_HOME"], "tests/tt_metal/microbenchmarks/ethernet/fabric_edm_bandwidth_golden.csv"
    )

    if not os.path.exists(csv_path):
        logger.warning("No golden data found")
        return 0, 0

    df = pd.read_csv(csv_path)
    df = df.replace({float("nan"): None})
    results = df[
        (df["Test Name"] == test_name)
        & (df["Packet Size"] == packet_size)
        & (df["Line Size"] == line_size)
        & (df["Num Links"] == num_links)
        & (df["Disable Interior Workers"] == disable_sends_for_interior_workers)
        & (df["Unidirectional"] == unidirectional)
    ]

    return results["Bandwidth (B/c)"].values[0], results["Packets/Second"].values[0]


def profile_results(
    zone_name,
    packets_per_src_chip,
    line_size,
    packet_size,
    fabric_mode,
    disable_sends_for_interior_workers,
    unidirectional=False,
):
    freq_hz = get_device_freq() * 1000.0 * 1000.0
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
        if unidirectional:
            traffic_streams_through_boundary = 1
    total_packets_sent = packets_per_src_chip * traffic_streams_through_boundary
    total_byte_sent = total_packets_sent * packet_size
    bandwidth = total_byte_sent / max(main_loop_cycles)
    packets_per_second = total_packets_sent / max(main_loop_cycles) * freq_hz

    return bandwidth, packets_per_second


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
    disable_sends_for_interior_workers,
    unidirectional=False,
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
                {int(disable_sends_for_interior_workers)} \
                {int(unidirectional)}"
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
    bandwidth_inner_loop, packets_per_second_inner_loop = profile_results(
        zone_name_inner,
        num_messages,
        line_size,
        packet_size,
        fabric_mode,
        disable_sends_for_interior_workers,
        unidirectional=unidirectional,
    )
    bandwidth, packets_per_second = profile_results(
        zone_name_main,
        num_messages,
        line_size,
        packet_size,
        fabric_mode,
        disable_sends_for_interior_workers,
        unidirectional=unidirectional,
    )
    logger.info("bandwidth_inner_loop: {} B/c", bandwidth_inner_loop)
    logger.info("bandwidth: {} B/c", bandwidth)
    logger.info("packets_per_second_inner_loop: {} pps", packets_per_second_inner_loop)
    logger.info("packets_per_second: {} pps", packets_per_second)
    mega_packets_per_second = packets_per_second / 1000000

    # Add summary to CSV
    test_name = f"{'unicast' if is_unicast else 'mcast'}_{fabric_mode.name}"
    summarize_to_csv(
        test_name,
        packet_size,
        line_size,
        num_links,
        disable_sends_for_interior_workers,
        unidirectional,
        bandwidth,
        packets_per_second,
    )
    expected_bw, expected_pps = read_golden_results(
        test_name, packet_size, line_size, num_links, disable_sends_for_interior_workers, unidirectional
    )
    expected_Mpps = expected_pps / 1000000 if expected_pps is not None else None
    bw_threshold = 0.07
    if packet_size <= 2048 and fabric_mode != FabricTestMode.Linear:
        bw_threshold = 0.12
    assert (
        expected_bw - bw_threshold <= bandwidth <= expected_bw + bw_threshold
    ), f"Bandwidth mismatch. expected: {expected_bw} B/c, actual: {bandwidth} B/c"
    if expected_Mpps is not None:
        assert (
            expected_Mpps - 0.01 <= mega_packets_per_second <= expected_Mpps + 0.01
        ), f"Packets per second mismatch. expected: {expected_Mpps} Mpps, actual: {mega_packets_per_second} Mpps"


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("line_size, num_links", [(4, 1), (4, 2)])
def test_fabric_edm_mcast_half_ring_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4chip_one_link_mcast_full_ring_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_8chip_one_link_edm_mcast_full_ring_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4_chip_one_link_mcast_saturate_chip_to_chip_ring_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4chip_one_link_mcast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4chip_one_link_bidirectional_single_producer_mcast_bw(
    num_mcasts, num_unicasts, num_links, num_op_invocations, line_sync, line_size, packet_size
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
        disable_sends_for_interior_workers=True,
        unidirectional=False,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4chip_one_link_unidirectional_single_producer_mcast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        disable_sends_for_interior_workers=True,
        unidirectional=True,
    )


@pytest.mark.parametrize("num_mcasts", [200000])
@pytest.mark.parametrize("num_unicasts", [0])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4chip_two_link_mcast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_one_link_non_forwarding_unicast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_two_link_non_forwarding_unicast_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_one_link_forwarding_unicast_multiproducer_multihop_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        False,
    )


@pytest.mark.parametrize("num_mcasts", [0])
@pytest.mark.parametrize("num_unicasts", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_one_link_forwarding_unicast_single_producer_multihop_bw(
    num_mcasts,
    num_unicasts,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
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
        True,
    )


def print_bandwidth_summary():
    """Print a summary table of all test results by packet size"""
    csv_path = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/bandwidth_summary.csv")

    if not os.path.exists(csv_path):
        logger.warning("No bandwidth summary data found")
        return

    df = pd.read_csv(csv_path)

    # Sort by test name and packet size
    df = df.sort_values(
        ["Test Name", "Packet Size", "Line Size", "Num Links", "Disable Interior Workers", "Unidirectional"]
    )

    # Format table with raw values
    table = tabulate(
        df,
        headers=[
            "Test Name",
            "Packet Size",
            "Line Size",
            "Num Links",
            "Disable Interior Workers",
            "Unidirectional",
            "Bandwidth (B/c)",
            "Packets/Second",
        ],
        tablefmt="grid",
        floatfmt=".2f",
    )
    logger.info("\nBandwidth Test Results:\n{}", table)


@pytest.fixture(scope="session", autouse=True)
def print_summary_at_end(request):
    """Print bandwidth summary at end of session"""
    # Delete old CSV file at start
    csv_path = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/bandwidth_summary.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
        logger.info("Removed old bandwidth summary file")

    yield
    print_bandwidth_summary()
