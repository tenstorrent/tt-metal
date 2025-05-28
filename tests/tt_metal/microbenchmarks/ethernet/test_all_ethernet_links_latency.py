# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger
import pytest
from tests.tt_metal.microbenchmarks.ethernet.test_all_ethernet_links_common import (
    process_profile_results,
    write_results_to_csv,
)

from models.utility_functions import is_single_chip

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

FILE_NAME = PROFILER_LOGS_DIR / "test_all_ethernet_links_latency.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


def run_erisc_write_worker_latency(
    benchmark_type,
    num_packets,
    packet_size,
    channel_count,
    disable_trid,
    num_iterations,
    packet_size_to_expected_latency,
    request,
):
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    num_iterations = request.config.getoption("--num-iterations") or num_iterations
    num_packets = request.config.getoption("--num-packets") or num_packets
    packet_size = request.config.getoption("--packet-size") or packet_size

    test_latency = 1

    ARCH_NAME = os.getenv("ARCH_NAME")

    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_all_ethernet_links \
                {benchmark_type} \
                {num_packets} \
                {packet_size} \
                {channel_count} \
                {test_latency} \
                {disable_trid} \
                {num_iterations} "
    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    process_profile_results(packet_size, num_packets, channel_count, benchmark_type, test_latency, num_iterations)
    avg_latency = write_results_to_csv(FILE_NAME, test_latency)
    logger.info(f"Sender latency {avg_latency} (ns)")
    if ARCH_NAME in packet_size_to_expected_latency:
        expected_latency_for_arch = packet_size_to_expected_latency[ARCH_NAME]
        if packet_size in expected_latency_for_arch:
            expected_latency = expected_latency_for_arch[packet_size]
            diff = expected_latency * 0.1
            expected_latency_lower_bound = expected_latency - diff
            expected_latency_upper_bound = expected_latency + diff
            assert expected_latency_lower_bound <= avg_latency <= expected_latency_upper_bound


# uni-direction test for eth-sender <---> eth-receiver
@pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [1])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096, 8192])
def test_erisc_latency_uni_dir(num_packets, packet_size, channel_count, num_iterations, request):
    packet_size_to_expected_latency = {
        "wormhole_b0": {
            16: 894.0,
            128: 911.0,
            256: 966.0,
            512: 984.0,
            1024: 1245.0,
            2048: 1479.0,
            4096: 1803.0,
            8192: 2451.0,
        }
    }
    benchmark_type_id = 0
    disable_trid = 0  # don't care in this case
    run_erisc_write_worker_latency(
        benchmark_type_id,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        packet_size_to_expected_latency,
        request,
    )


# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [1])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096, 8192])
def test_erisc_write_worker_latency_uni_dir(
    num_packets, packet_size, channel_count, disable_trid, num_iterations, request
):
    packet_size_to_expected_latency = {
        "wormhole_b0": {
            16: 984.0,
            128: 1002.0,
            256: 1019.0,
            512: 1074.0,
            1024: 1335.0,
            2048: 1609.0,
            4096: 2018.0,
            8192: 2811.0,
        }
    }
    benchmark_type_id = 2
    run_erisc_write_worker_latency(
        benchmark_type_id,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        packet_size_to_expected_latency,
        request,
    )
