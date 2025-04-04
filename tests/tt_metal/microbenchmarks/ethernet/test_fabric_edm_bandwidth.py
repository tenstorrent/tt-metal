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
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


# Python enum mirroring test_fabric_edm_common.hpp
class FabricTestMode(Enum):
    Linear = 0
    HalfRing = 1
    FullRing = 2
    SaturateChipToChipRing = 3
    RingAsLinear = 4


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
    *,
    noc_message_type,
    senders_are_unidirectional,
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
                    "Noc Message Type",
                    "Packet Size",
                    "Line Size",
                    "Num Links",
                    "Disable Interior Workers",
                    "Unidirectional",
                    "Senders Are Unidirectional",
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
                noc_message_type,
                packet_size,
                line_size,
                num_links,
                disable_sends_for_interior_workers,
                unidirectional,
                senders_are_unidirectional,
                bandwidth,
                packets_per_second,
            ]
        )


def read_golden_results(
    test_name,
    packet_size,
    line_size,
    num_links,
    disable_sends_for_interior_workers,
    unidirectional,  # traffic at fabric level
    *,
    noc_message_type,
    senders_are_unidirectional=False,  # coming out of any given worker
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
        & (df["Noc Message Type"] == noc_message_type)
        & (df["Packet Size"] == packet_size)
        & (df["Line Size"] == line_size)
        & (df["Num Links"] == num_links)
        & (df["Disable Interior Workers"] == disable_sends_for_interior_workers)
        & (df["Unidirectional"] == unidirectional)
        & (df["Senders Are Unidirectional"] == senders_are_unidirectional)
    ]

    if len(results["Bandwidth (B/c)"]) == 0 or len(results["Packets/Second"]) == 0:
        logger.error(
            f"No golden data found for tests_name={test_name} noc_message_type={noc_message_type} packet_size={packet_size} line_size={line_size} num_links={num_links} disable_sends_for_interior_workers={disable_sends_for_interior_workers} unidirectional={unidirectional}"
        )
        assert (
            len(results["Bandwidth (B/c)"]) == 0 and len(results["Packets/Second"]) == 0
        ), "Golden data may be incorrect or corrupted. One of `Bandwidth (B/c)` or `Packets/Second` was missing but not both. Either both should be present or both should be missing."
        return 0, 0

    bandwidth = results["Bandwidth (B/c)"].values[0]
    pps = results["Packets/Second"].values[0]

    return float(bandwidth) if bandwidth is not None else None, float(pps) if pps is not None else None


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
    bytes_per_GB = 1000000000
    bandwidth_GB_s = (bandwidth * freq_hz) / bytes_per_GB
    logger.info("main_loop_cycles: {} ", max(main_loop_cycles))
    logger.info("bandwidth: {} GB/s", bandwidth)

    return bandwidth, packets_per_second


def run_fabric_edm(
    *,
    is_unicast,
    num_messages,
    noc_message_type,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_mode,
    disable_sends_for_interior_workers,
    unidirectional=False,
    senders_are_unidirectional=False,
):
    logger.warning("removing file profile_log_device.csv")
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    enable_persistent_kernel_cache()
    cmd = f"TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/ttnn/unit_tests_ttnn_fabric_edm \
                {int(is_unicast)} \
                {noc_message_type} \
                {num_messages} \
                {num_links} \
                {num_op_invocations} \
                {int(line_sync)} \
                {line_size} \
                {packet_size} \
                {fabric_mode.value} \
                {int(disable_sends_for_interior_workers)} \
                {int(unidirectional)} \
                {int(senders_are_unidirectional)}"
    rc = os.system(cmd)

    disable_persistent_kernel_cache()
    if rc != 0:
        if os.WEXITSTATUS(rc) == 1:
            pytest.skip("Skipping test because it only works with T3000")
            return
        logger.info("Error in running the test")
        assert False

    zone_name_inner = "MAIN-TEST-BODY"
    zone_name_main = "MAIN-TEST-BODY"

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
        noc_message_type=noc_message_type,
        senders_are_unidirectional=senders_are_unidirectional,
    )
    expected_bw, expected_pps = read_golden_results(
        test_name,
        packet_size,
        line_size,
        num_links,
        disable_sends_for_interior_workers,
        unidirectional,
        noc_message_type=noc_message_type,
        senders_are_unidirectional=senders_are_unidirectional,
    )
    bw_threshold_general = 0.07
    pps_threshold_general = 0.01
    if packet_size <= 2048 and fabric_mode != FabricTestMode.Linear:
        bw_threshold_general = 0.12
    ## These seem to be a little more noisy so for now we widen the threshold to have test stability
    bw_threshold_fused_write_atomic = 0.14
    pps_threshold_fused_write_atomic = 0.03
    use_general_threshold = (
        noc_message_type != "noc_fused_unicast_write_flush_atomic_inc"
        and noc_message_type != "noc_fused_unicast_write_no_flush_atomic_inc"
    )

    bw_threshold = bw_threshold_general if use_general_threshold else bw_threshold_fused_write_atomic
    pps_threshold = pps_threshold_general if use_general_threshold else pps_threshold_fused_write_atomic

    expected_Mpps = expected_pps / 1000000 if expected_pps is not None else None
    assert (
        expected_bw - bw_threshold <= bandwidth <= expected_bw + bw_threshold
    ), f"Bandwidth mismatch. expected: {expected_bw} B/c, actual: {bandwidth} B/c"
    if expected_Mpps is not None:
        assert (
            expected_Mpps - expected_pps <= mega_packets_per_second <= expected_Mpps + expected_pps
        ), f"Packets per second mismatch. expected: {expected_Mpps} Mpps, actual: {mega_packets_per_second} Mpps"


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("packet_size", [4096])
@pytest.mark.parametrize("line_size, num_links", [(4, 1), (4, 2)])
def test_fabric_edm_mcast_half_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.HalfRing,
        disable_sends_for_interior_workers=False,
        unidirectional=False,
        senders_are_unidirectional=False,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4chip_one_link_mcast_full_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.FullRing,
        disable_sends_for_interior_workers=False,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [8])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_8chip_one_link_edm_mcast_full_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.FullRing,
        disable_sends_for_interior_workers=False,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
def test_fabric_4_chip_one_link_mcast_saturate_chip_to_chip_ring_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=FabricTestMode.SaturateChipToChipRing,
        disable_sends_for_interior_workers=False,
    )


# expected_Mpps = expected millions of packets per second
@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_4chip_one_link_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        unidirectional=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_4chip_one_link_bidirectional_single_producer_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        unidirectional=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_4chip_one_link_unidirectional_single_producer_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        unidirectional=True,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [2, 3, 4])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_4chip_two_link_mcast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=False,
        num_messages=num_messages,
        noc_message_type="noc_unicast_write",
        num_links=num_links,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_non_forwarding_unicast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [2])
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_two_link_non_forwarding_unicast_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_forwarding_unicast_multiproducer_multihop_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_forwarding_unicast_single_producer_multihop_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        unidirectional=False,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_forwarding_unicast_unidirectional_single_producer_multihop_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type="noc_unicast_write",
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        unidirectional=True,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("noc_message_type", ["noc_unicast_flush_atomic_inc", "noc_unicast_no_flush_atomic_inc"])
@pytest.mark.parametrize("packet_size", [16])
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_forwarding_unicast_single_producer_multihop_atomic_inc_bw(
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    noc_message_type,
    packet_size,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=True,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type=noc_message_type,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=True,
        senders_are_unidirectional=True,
    )


@pytest.mark.parametrize("num_messages", [200000])
@pytest.mark.parametrize("num_op_invocations", [1])
@pytest.mark.parametrize("line_sync", [True])
@pytest.mark.parametrize("line_size", [4])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("is_unicast", [False, True])
@pytest.mark.parametrize("disable_sends_for_interior_workers", [False, True])
@pytest.mark.parametrize("packet_size", [16, 2048, 4096])
@pytest.mark.parametrize("unidirectional", [False, True])
@pytest.mark.parametrize(
    "noc_message_type", ["noc_fused_unicast_write_flush_atomic_inc", "noc_fused_unicast_write_no_flush_atomic_inc"]
)
@pytest.mark.parametrize("fabric_test_mode", [FabricTestMode.Linear, FabricTestMode.RingAsLinear])
def test_fabric_one_link_multihop_fused_write_atomic_inc_bw(
    is_unicast,
    num_messages,
    num_links,
    num_op_invocations,
    line_sync,
    line_size,
    packet_size,
    disable_sends_for_interior_workers,
    noc_message_type,
    unidirectional,
    fabric_test_mode,
):
    run_fabric_edm(
        is_unicast=is_unicast,
        num_messages=num_messages,
        num_links=num_links,
        noc_message_type=noc_message_type,
        num_op_invocations=num_op_invocations,
        line_sync=line_sync,
        line_size=line_size,
        packet_size=packet_size,
        fabric_mode=fabric_test_mode,
        disable_sends_for_interior_workers=disable_sends_for_interior_workers,
        unidirectional=unidirectional,
        senders_are_unidirectional=True,
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
        [
            "Test Name",
            "Noc Message Type",
            "Packet Size",
            "Line Size",
            "Num Links",
            "Disable Interior Workers",
            "Unidirectional",
            "Senders Are Unidirectional",
        ]
    )

    # Format table with raw values
    table = tabulate(
        df,
        headers=[
            "Test Name",
            "Noc Message Type",
            "Packet Size",
            "Line Size",
            "Num Links",
            "Disable Interior Workers",
            "Unidirectional",
            "Senders Are Unidirectional",
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
