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

FILE_NAME = PROFILER_LOGS_DIR / "test_all_ethernet_links_bandwidth.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)


def run_erisc_write_worker_bw(
    benchmark_type,
    num_packets,
    packet_size,
    channel_count,
    disable_trid,
    num_iterations,
    packet_size_to_expected_bw,
    request,
):
    os.system(f"rm -rf {os.environ['TT_METAL_HOME']}/generated/profiler/.logs/profile_log_device.csv")

    num_iterations = request.config.getoption("--num-iterations") or num_iterations
    num_packets = request.config.getoption("--num-packets") or num_packets
    packet_size = request.config.getoption("--packet-size") or packet_size

    test_latency = 0

    ARCH_NAME = os.getenv("ARCH_NAME")
    cmd = f"TT_METAL_DEVICE_PROFILER=1 \
            {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_all_ethernet_links \
                {benchmark_type} \
                {num_packets} \
                {packet_size} \
                {channel_count} \
                {test_latency} \
                {disable_trid} \
                {num_iterations}"
    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    process_profile_results(packet_size, num_packets, channel_count, benchmark_type, test_latency, num_iterations)
    avg_bw = write_results_to_csv(FILE_NAME, test_latency)
    logger.info(f"Sender bandwidth (GB/s) {avg_bw}")
    if ARCH_NAME in packet_size_to_expected_bw:
        expected_bw_for_arch = packet_size_to_expected_bw[ARCH_NAME]
        if packet_size in expected_bw_for_arch:
            expected_bw = expected_bw_for_arch[packet_size]
            expected_bw_lower_bound = expected_bw - 0.5
            expected_bw_upper_bound = expected_bw + 0.5
            assert expected_bw_lower_bound <= avg_bw <= expected_bw_upper_bound


##################################### No Worker BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver
@pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096, 8192])
def test_erisc_bw_uni_dir(num_packets, packet_size, channel_count, num_iterations, request):
    packet_size_to_expected_bw = {
        "wormhole_b0": {16: 0.28, 128: 2.25, 256: 4.39, 512: 8.35, 1024: 11.74, 2048: 11.84, 4096: 12.04, 8192: 12.07},
        "blackhole": {16: 0.11, 128: 0.97, 256: 1.88, 512: 3.65, 1024: 7.27, 2048: 12.97, 4096: 19.71, 8192: 24.16},
    }
    benchmark_type_id = 0
    disable_trid = 0  # don't care in this case
    run_erisc_write_worker_bw(
        benchmark_type_id,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        packet_size_to_expected_bw,
        request,
    )


# bi-direction test for eth-sender <---> eth-receiver
@pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096])
def test_erisc_bw_bi_dir(num_packets, packet_size, channel_count, num_iterations, request):
    packet_size_to_expected_bw = {
        "wormhole_b0": {
            16: 0.19,
            128: 1.59,
            256: 3.19,
            512: 6.39,
            1024: 10.9,
            2048: 11.4,
            4096: 11.82,
        },
        "blackhole": {
            16: 0.24,
            128: 1.94,
            256: 3.88,
            512: 7.75,
            1024: 15.46,
            2048: 12.89,
            4096: 18.75,
        },
    }
    benchmark_type_id = 1
    disable_trid = 0  # don't care in this case
    run_erisc_write_worker_bw(
        benchmark_type_id,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        packet_size_to_expected_bw,
        request,
    )


##################################### BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096, 8192])
def test_erisc_write_worker_bw_uni_dir(num_packets, packet_size, channel_count, disable_trid, num_iterations, request):
    packet_size_to_expected_bw = {
        "wormhole_b0": {16: 0.21, 128: 1.72, 256: 3.44, 512: 6.89, 1024: 11.73, 2048: 11.83, 4096: 12.04, 8192: 12.07},
        "blackhole": {16: 0.11, 128: 0.84, 256: 1.61, 512: 3.41, 1024: 6.52, 2048: 11.70, 4096: 19.10, 8192: 23.97},
    }
    benchmark_type_id = 2
    run_erisc_write_worker_bw(
        benchmark_type_id,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        packet_size_to_expected_bw,
        request,
    )


# bi-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096])
def test_erisc_write_worker_bw_bi_dir(num_packets, packet_size, channel_count, disable_trid, num_iterations, request):
    packet_size_to_expected_bw = {
        "wormhole_b0": {16: 0.13, 128: 1.03, 256: 2.08, 512: 4.15, 1024: 7.81, 2048: 11.40, 4096: 11.82},
        "blackhole": {16: 0.16, 128: 1.35, 256: 2.69, 512: 5.39, 1024: 10.79, 2048: 21.47, 4096: 18.70},
    }
    benchmark_type_id = 3
    run_erisc_write_worker_bw(
        benchmark_type_id,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        packet_size_to_expected_bw,
        request,
    )


##################################### No Transaction ID BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [256])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("disable_trid", [1])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096, 8192])
def test_erisc_write_worker_bw_uni_dir_no_trid(
    num_packets, packet_size, channel_count, disable_trid, num_iterations, request
):
    packet_size_to_expected_bw = {
        "wormhole_b0": {16: 0.18, 128: 1.71, 256: 3.81, 512: 7.72, 1024: 11.32, 2048: 11.83, 4096: 12.04, 8192: 12.07},
        "blackhole": {16: 0.11, 128: 0.90, 256: 1.82, 512: 3.71, 1024: 7.37, 2048: 12.62, 4096: 20.04, 8192: 24.09},
    }
    benchmark_type_id = 2
    run_erisc_write_worker_bw(
        benchmark_type_id,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        packet_size_to_expected_bw,
        request,
    )


# bi-direction test for eth-sender <---> eth-receiver ---> worker
@pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [1000])
@pytest.mark.parametrize("channel_count", [16])
@pytest.mark.parametrize("disable_trid", [1])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096])
def test_erisc_write_worker_bw_bi_dir_no_trid(
    num_packets, packet_size, channel_count, disable_trid, num_iterations, request
):
    packet_size_to_expected_bw = {
        "wormhole_b0": {
            16: 0.10,
            128: 0.87,
            256: 1.99,
            512: 4.47,
            1024: 9.43,
            2048: 11.00,
            4096: 11.82,
        },
        "blackhole": {
            16: 0.20,
            128: 1.60,
            256: 3.24,
            512: 6.49,
            1024: 12.95,
            2048: 14.46,
            4096: 19.02,
        },
    }
    benchmark_type_id = 3
    run_erisc_write_worker_bw(
        benchmark_type_id,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        packet_size_to_expected_bw,
        request,
    )
