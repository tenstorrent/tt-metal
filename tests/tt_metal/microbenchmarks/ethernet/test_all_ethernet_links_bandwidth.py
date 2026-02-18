# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shutil
from collections import defaultdict

from loguru import logger
from tests.tt_metal.microbenchmarks.ethernet.test_all_ethernet_links_common import (
    process_profile_results,
    write_results_to_csv,
)

# from models.common.utility_functions import is_single_chip

from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

FILE_NAME = PROFILER_LOGS_DIR / "test_all_ethernet_links_bandwidth.csv"

if os.path.exists(FILE_NAME):
    os.remove(FILE_NAME)

# Accumulate BW results per test function for summary tables
# Key: test_name -> {(packet_size, channel_count): avg_bw}
_bw_summary = defaultdict(dict)
_atexit_registered = False


def _print_summary_tables():
    for test_name in sorted(_bw_summary.keys()):
        results = _bw_summary[test_name]
        if not results:
            continue
        packet_sizes = sorted(set(ps for ps, _ in results.keys()))
        channel_counts = sorted(set(cc for _, cc in results.keys()))

        col_w = 10
        header = f"{'pkt_size':>{col_w}}" + "".join(f"{cc:>{col_w}}" for cc in channel_counts)
        sep = "-" * len(header)

        logger.info(f"\n{'=' * len(header)}")
        logger.info(f"{test_name} — BW (GB/s)  rows=packet_size  cols=channel_count")
        logger.info(f"{'=' * len(header)}")
        logger.info(header)
        logger.info(sep)
        for ps in packet_sizes:
            row = f"{ps:>{col_w}}"
            for cc in channel_counts:
                val = results.get((ps, cc))
                row += f"{val:>{col_w}.2f}" if val is not None else f"{'---':>{col_w}}"
            logger.info(row)
    logger.info("")


def run_erisc_write_worker_bw_batch(
    benchmark_type,
    num_packets,
    packet_sizes,
    channel_counts,
    disable_trid,
    num_iterations,
    packet_size_to_expected_bw,
    request,
):
    # Clean up old config-specific profiler files
    for f in PROFILER_LOGS_DIR.glob("profile_log_device_ps*_ch*.csv"):
        os.remove(f)

    num_iterations = request.config.getoption("--num-iterations") or num_iterations
    num_packets = request.config.getoption("--num-packets") or num_packets
    ps_override = request.config.getoption("--packet-size")
    if ps_override:
        packet_sizes = [ps_override]

    test_latency = 0

    ps_csv = ",".join(str(p) for p in packet_sizes)
    cc_csv = ",".join(str(c) for c in channel_counts)

    cmd = (
        f"TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1"
        f" {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_all_ethernet_links"
        f" {benchmark_type}"
        f" {num_packets}"
        f" {ps_csv}"
        f" {cc_csv}"
        f" {test_latency}"
        f" {disable_trid}"
        f" {num_iterations}"
    )
    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    global _atexit_registered
    test_name = request.node.name
    if not _atexit_registered:
        atexit.register(_print_summary_tables)
        _atexit_registered = True

    ARCH_NAME = os.getenv("ARCH_NAME")

    # Process each config's profiler file
    for ps in packet_sizes:
        for cc in channel_counts:
            config_file = PROFILER_LOGS_DIR / f"profile_log_device_ps{ps}_ch{cc}.csv"
            if not config_file.exists():
                logger.warning(f"Missing profiler file for ps={ps}, ch={cc}: {config_file}")
                continue
            shutil.copy(config_file, profiler_log_path)
            process_profile_results(ps, num_packets, cc, benchmark_type, test_latency, num_iterations)
            avg_bw = write_results_to_csv(FILE_NAME, test_latency)
            logger.info(f"packet_size={ps} channel_count={cc} => BW (GB/s) {avg_bw}")

            _bw_summary[test_name][(ps, cc)] = avg_bw

            if ARCH_NAME in packet_size_to_expected_bw:
                expected_bw_for_arch = packet_size_to_expected_bw[ARCH_NAME]
                if ps in expected_bw_for_arch:
                    expected_bw = expected_bw_for_arch[ps]
                    expected_bw_lower_bound = expected_bw - 0.5
                    expected_bw_upper_bound = expected_bw + 0.5
                    assert expected_bw_lower_bound <= avg_bw <= expected_bw_upper_bound, (
                        f"BW {avg_bw} outside expected range [{expected_bw_lower_bound}, {expected_bw_upper_bound}] "
                        f"for packet_size={ps}, channel_count={cc}"
                    )


##################################### No Worker BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver
# @pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
def test_erisc_bw_uni_dir(request):
    packet_size_to_expected_bw = {
        "wormhole_b0": {16: 0.28, 128: 2.25, 256: 4.39, 512: 8.35, 1024: 11.74, 2048: 11.84, 4096: 12.04, 8192: 12.07},
        "blackhole": {16: 0.11, 128: 0.97, 256: 1.88, 512: 3.65, 1024: 7.27, 2048: 12.97, 4096: 19.71, 8192: 24.16},
    }
    run_erisc_write_worker_bw_batch(
        benchmark_type=0,
        num_packets=256,
        packet_sizes=[16, 128, 256, 512, 1024, 2048, 4096, 8192],
        channel_counts=[4, 8, 12, 16, 20, 24, 28, 32],
        disable_trid=0,
        num_iterations=1,
        packet_size_to_expected_bw=packet_size_to_expected_bw,
        request=request,
    )


# bi-direction test for eth-sender <---> eth-receiver
# @pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
def test_erisc_bw_bi_dir(request):
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
    run_erisc_write_worker_bw_batch(
        benchmark_type=1,
        num_packets=1000,
        packet_sizes=[16, 128, 256, 512, 1024, 2048, 4096],
        channel_counts=[4, 8, 12, 16, 20, 24, 28, 32],
        disable_trid=0,
        num_iterations=1,
        packet_size_to_expected_bw=packet_size_to_expected_bw,
        request=request,
    )


##################################### BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver ---> worker
# @pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
def test_erisc_write_worker_bw_uni_dir(request):
    packet_size_to_expected_bw = {
        "wormhole_b0": {16: 0.21, 128: 1.72, 256: 3.44, 512: 6.89, 1024: 11.73, 2048: 11.83, 4096: 12.04, 8192: 12.07},
        "blackhole": {16: 0.11, 128: 0.84, 256: 1.61, 512: 3.41, 1024: 6.52, 2048: 11.70, 4096: 19.10, 8192: 23.97},
    }
    run_erisc_write_worker_bw_batch(
        benchmark_type=2,
        num_packets=256,
        packet_sizes=[16, 128, 256, 512, 1024, 2048, 4096, 8192],
        channel_counts=[4, 8, 12, 16, 20, 24, 28, 32],
        disable_trid=0,
        num_iterations=1,
        packet_size_to_expected_bw=packet_size_to_expected_bw,
        request=request,
    )


# bi-direction test for eth-sender <---> eth-receiver ---> worker
# @pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
def test_erisc_write_worker_bw_bi_dir(request):
    packet_size_to_expected_bw = {
        "wormhole_b0": {16: 0.13, 128: 1.03, 256: 2.08, 512: 4.15, 1024: 7.81, 2048: 11.40, 4096: 11.82},
        "blackhole": {16: 0.16, 128: 1.35, 256: 2.69, 512: 5.39, 1024: 10.79, 2048: 21.47, 4096: 18.70},
    }
    run_erisc_write_worker_bw_batch(
        benchmark_type=3,
        num_packets=1000,
        packet_sizes=[16, 128, 256, 512, 1024, 2048, 4096],
        channel_counts=[4, 8, 12, 16, 20, 24, 28, 32],
        disable_trid=0,
        num_iterations=1,
        packet_size_to_expected_bw=packet_size_to_expected_bw,
        request=request,
    )


##################################### Dual-ERISC Bi-Directional BW test (BH only) ####################################
# bi-direction test using ERISC_0 as sender and ERISC_1 as receiver per eth core
def test_erisc_bw_dual_erisc_bi_dir(request):
    packet_size_to_expected_bw = {}  # no expected values yet, establish baseline
    run_erisc_write_worker_bw_batch(
        benchmark_type=9,  # DualEriscBiDir
        num_packets=1000,
        packet_sizes=[16, 128, 256, 512, 1024, 2048, 4096],
        channel_counts=[4, 8, 12, 16, 20, 24, 28, 32],
        disable_trid=0,
        num_iterations=1,
        packet_size_to_expected_bw=packet_size_to_expected_bw,
        request=request,
    )


##################################### No Transaction ID BW test #######################################################
# uni-direction test for eth-sender <---> eth-receiver ---> worker
# @pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
def test_erisc_write_worker_bw_uni_dir_no_trid(request):
    packet_size_to_expected_bw = {
        "wormhole_b0": {16: 0.18, 128: 1.71, 256: 3.81, 512: 7.72, 1024: 11.32, 2048: 11.83, 4096: 12.04, 8192: 12.07},
        "blackhole": {16: 0.11, 128: 0.90, 256: 1.82, 512: 3.71, 1024: 7.37, 2048: 12.62, 4096: 20.04, 8192: 24.09},
    }
    run_erisc_write_worker_bw_batch(
        benchmark_type=2,
        num_packets=256,
        packet_sizes=[16, 128, 256, 512, 1024, 2048, 4096, 8192],
        channel_counts=[4, 8, 12, 16, 20, 24, 28, 32],
        disable_trid=1,
        num_iterations=1,
        packet_size_to_expected_bw=packet_size_to_expected_bw,
        request=request,
    )


# bi-direction test for eth-sender <---> eth-receiver ---> worker
# @pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
def test_erisc_write_worker_bw_bi_dir_no_trid(request):
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
    run_erisc_write_worker_bw_batch(
        benchmark_type=3,
        num_packets=1000,
        packet_sizes=[16, 128, 256, 512, 1024, 2048, 4096],
        channel_counts=[4, 8, 12, 16, 20, 24, 28, 32],
        disable_trid=1,
        num_iterations=1,
        packet_size_to_expected_bw=packet_size_to_expected_bw,
        request=request,
    )
