# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

from loguru import logger
import pytest
from tests.tt_metal.microbenchmarks.ethernet.test_all_ethernet_links_common import (
    process_profile_results,
    write_results_to_csv,
)

# from models.common.utility_functions import is_single_chip

from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

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
# @pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [1])
@pytest.mark.parametrize("channel_count", [2])  # 6])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16])  # , 128, 256, 512, 1024, 2048, 4096, 8192])
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
# @pytest.mark.skipif(is_single_chip(), reason="Unsupported on single chip systems")
@pytest.mark.parametrize("num_packets", [1])
@pytest.mark.parametrize("channel_count", [2])  # 6])
@pytest.mark.parametrize("disable_trid", [0])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16])  # , 128, 256, 512, 1024, 2048, 4096, 8192])
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


# full round-trip: tensix -> eth -> eth -> tensix -> eth -> eth -> tensix
@pytest.mark.parametrize("num_packets", [100])
@pytest.mark.parametrize("channel_count", [2])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("packet_size", [16, 128, 256, 512, 1024, 2048, 4096, 8192])
def test_tensix_eth_eth_tensix_latency_uni_dir(num_packets, packet_size, channel_count, num_iterations, request):
    packet_size_to_expected_latency = {}
    benchmark_type_id = 8
    disable_trid = 0
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


##################################### Core Sweep Latency test ################################################


def _print_sender_sweep_table(results, packet_size, sender_noc, receiver_noc, receiver_core):
    """Print a heatmap table of sender core sweep results (receiver fixed)."""
    if not results:
        logger.warning("No core sweep results to display")
        return
    max_x = max(x for x, _ in results.keys())
    max_y = max(y for _, y in results.keys())
    col_w = 8
    yx_label = "y\\x"
    header = f"{yx_label:>{col_w}}" + "".join(f"{x:>{col_w}}" for x in range(max_x + 1))
    logger.info(
        f"\nSender Core Sweep Latency (ns) — packet_size={packet_size}, "
        f"sender_noc=NOC{sender_noc}, receiver_noc=NOC{receiver_noc}, "
        f"receiver_tensix=({receiver_core[0]},{receiver_core[1]})"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for y in range(max_y + 1):
        row = f"{y:>{col_w}}"
        for x in range(max_x + 1):
            val = results.get((x, y))
            row += f"{val:>{col_w}.0f}" if val is not None else f"{'---':>{col_w}}"
        logger.info(row)


def _print_receiver_sweep_table(results, packet_size, sender_noc, receiver_noc, sender_core):
    """Print a heatmap table of receiver core sweep results (sender fixed)."""
    if not results:
        return
    max_x = max(x for x, _ in results.keys())
    max_y = max(y for _, y in results.keys())
    col_w = 8
    yx_label = "y\\x"
    header = f"{yx_label:>{col_w}}" + "".join(f"{x:>{col_w}}" for x in range(max_x + 1))
    logger.info(
        f"\nReceiver Core Sweep Latency (ns) — packet_size={packet_size}, "
        f"sender_noc=NOC{sender_noc}, receiver_noc=NOC{receiver_noc}, "
        f"sender_tensix=({sender_core[0]},{sender_core[1]})"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for y in range(max_y + 1):
        row = f"{y:>{col_w}}"
        for x in range(max_x + 1):
            val = results.get((x, y))
            row += f"{val:>{col_w}.0f}" if val is not None else f"{'---':>{col_w}}"
        logger.info(row)


def _parse_sweep_results(packet_size, channel_count, benchmark_type, num_packets, test_latency, num_iterations):
    """Parse profiler CSVs with _s{sx}_{sy}_r{rx}_{ry} naming into nested dict."""
    # Returns {(sx, sy): {(rx, ry): avg_latency_ns}}
    all_results = {}
    for config_file in sorted(
        PROFILER_LOGS_DIR.glob(f"profile_log_device_ps{packet_size}_ch{channel_count}_s*_r*.csv")
    ):
        name = config_file.stem
        # e.g. "profile_log_device_ps16_ch2_s3_5_r7_2"
        s_part = name.split("_s")[1]  # "3_5_r7_2"
        s_coords, r_part = s_part.split("_r")  # "3_5", "7_2"
        sx, sy = int(s_coords.split("_")[0]), int(s_coords.split("_")[1])
        rx, ry = int(r_part.split("_")[0]), int(r_part.split("_")[1])

        shutil.copy(config_file, profiler_log_path)
        process_profile_results(packet_size, num_packets, channel_count, benchmark_type, test_latency, num_iterations)
        avg_latency = write_results_to_csv(FILE_NAME, test_latency)

        if (sx, sy) not in all_results:
            all_results[(sx, sy)] = {}
        all_results[(sx, sy)][(rx, ry)] = avg_latency

    return all_results


def _run_sweep_binary(
    benchmark_type,
    num_packets,
    packet_size,
    channel_count,
    disable_trid,
    num_iterations,
    sweep_cores,
    noc_index,
    receiver_noc_index,
):
    """Run the C++ test binary with sweep parameters."""
    # Clean old per-core profiler files
    for f in PROFILER_LOGS_DIR.glob("profile_log_device_ps*_ch*_s*_r*.csv"):
        os.remove(f)

    test_latency = 1
    cmd = (
        f"TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1"
        f" {os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/ethernet/test_all_ethernet_links"
        f" {benchmark_type}"
        f" {num_packets}"
        f" {packet_size}"
        f" {channel_count}"
        f" {test_latency}"
        f" {disable_trid}"
        f" {num_iterations}"
        f" {sweep_cores}"
        f" {noc_index}"
        f" {receiver_noc_index}"
    )
    rc = os.system(cmd)
    assert rc == 0, "Test binary failed"


def run_sender_sweep(
    benchmark_type,
    num_packets,
    packet_size,
    channel_count,
    disable_trid,
    num_iterations,
    noc_index,
    receiver_noc_index,
    request,
):
    """Sweep sender cores (receiver fixed at closest to eth). One table per NOC config."""
    num_iterations = request.config.getoption("--num-iterations") or num_iterations
    num_packets = request.config.getoption("--num-packets") or num_packets
    packet_size = request.config.getoption("--packet-size") or packet_size

    _run_sweep_binary(
        benchmark_type,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        1,
        noc_index,
        receiver_noc_index,
    )

    all_results = _parse_sweep_results(packet_size, channel_count, benchmark_type, num_packets, 1, num_iterations)

    # For sweep_cores=1, receiver is fixed — all sender entries have the same single receiver.
    # Print one flat table of sender core latencies.
    for sender_core in sorted(all_results.keys()):
        for receiver_core, latency in all_results[sender_core].items():
            pass  # just get the receiver_core (same for all)
    # Flatten: {(sx, sy): latency}
    flat_results = {}
    fixed_receiver = None
    for sender_core, receiver_map in all_results.items():
        for receiver_core, latency in receiver_map.items():
            flat_results[sender_core] = latency
            fixed_receiver = receiver_core
    if flat_results:
        _print_sender_sweep_table(flat_results, packet_size, noc_index, receiver_noc_index, fixed_receiver)

    return all_results


def run_receiver_sweep(
    benchmark_type,
    num_packets,
    packet_size,
    channel_count,
    disable_trid,
    num_iterations,
    noc_index,
    receiver_noc_index,
    request,
):
    """Sweep receiver cores (sender fixed at closest to eth). One table of receiver core latencies."""
    num_iterations = request.config.getoption("--num-iterations") or num_iterations
    num_packets = request.config.getoption("--num-packets") or num_packets
    packet_size = request.config.getoption("--packet-size") or packet_size

    _run_sweep_binary(
        benchmark_type,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        3,
        noc_index,
        receiver_noc_index,
    )

    all_results = _parse_sweep_results(packet_size, channel_count, benchmark_type, num_packets, 1, num_iterations)

    # For sweep_cores=3, sender is fixed — all entries have the same single sender.
    # Flatten: {(rx, ry): latency}
    flat_results = {}
    fixed_sender = None
    for sender_core, receiver_map in all_results.items():
        fixed_sender = sender_core
        for receiver_core, latency in receiver_map.items():
            flat_results[receiver_core] = latency
    if flat_results:
        _print_receiver_sweep_table(flat_results, packet_size, noc_index, receiver_noc_index, fixed_sender)

    return all_results


def run_extended_sweep(
    benchmark_type,
    num_packets,
    packet_size,
    channel_count,
    disable_trid,
    num_iterations,
    noc_index,
    receiver_noc_index,
    request,
):
    """Sweep both sender and receiver cores (N^2). One table per sender core."""
    num_iterations = request.config.getoption("--num-iterations") or num_iterations
    num_packets = request.config.getoption("--num-packets") or num_packets
    packet_size = request.config.getoption("--packet-size") or packet_size

    _run_sweep_binary(
        benchmark_type,
        num_packets,
        packet_size,
        channel_count,
        disable_trid,
        num_iterations,
        2,
        noc_index,
        receiver_noc_index,
    )

    all_results = _parse_sweep_results(packet_size, channel_count, benchmark_type, num_packets, 1, num_iterations)

    # Print one table per sender core
    for sender_core in sorted(all_results.keys()):
        _print_receiver_sweep_table(all_results[sender_core], packet_size, noc_index, receiver_noc_index, sender_core)

    return all_results


# Default: sweep sender cores, receiver fixed closest to eth
# ETH is locked to NOC_0, tensix uses NOC_1 (opposite)
@pytest.mark.parametrize("packet_size", [16, 256, 1024, 4096])
def test_erisc_write_worker_latency_core_sweep(packet_size, request):
    """Sweep sender cores with receiver fixed at closest-to-eth core.

    ETH kernels are locked to NOC_0 (ERISC constraint). Tensix kernels should use
    NOC_1 (opposite) so eth→tensix and tensix→eth writes travel in opposite directions.
    """
    run_sender_sweep(
        benchmark_type=8,  # TensixEthEthTensixUniDir
        num_packets=100,
        packet_size=packet_size,
        channel_count=2,
        disable_trid=0,
        num_iterations=1,
        noc_index=1,  # tensix uses NOC_1 (opposite of eth NOC_0)
        receiver_noc_index=1,  # both tensix on NOC_1
        request=request,
    )


# Sweep receiver cores, sender fixed at closest to eth
@pytest.mark.parametrize("packet_size", [16, 256, 1024, 4096])
def test_erisc_write_worker_latency_receiver_sweep(packet_size, request):
    """Sweep receiver cores with sender fixed at closest-to-eth core."""
    run_receiver_sweep(
        benchmark_type=8,  # TensixEthEthTensixUniDir
        num_packets=100,
        packet_size=packet_size,
        channel_count=2,
        disable_trid=0,
        num_iterations=1,
        noc_index=1,  # tensix uses NOC_1 (opposite of eth NOC_0)
        receiver_noc_index=1,  # both tensix on NOC_1
        request=request,
    )


# Extended: sweep both sender and receiver cores independently
@pytest.mark.extended
@pytest.mark.parametrize("packet_size", [16, 4096])
@pytest.mark.parametrize("noc_index", [0, 1])
@pytest.mark.parametrize("receiver_noc_index", [0, 1])
def test_erisc_write_worker_latency_extended_sweep(packet_size, noc_index, receiver_noc_index, request):
    """Sweep both sender and receiver cores (N^2) with configurable NOCs."""
    run_extended_sweep(
        benchmark_type=8,  # TensixEthEthTensixUniDir
        num_packets=100,
        packet_size=packet_size,
        channel_count=2,
        disable_trid=0,
        num_iterations=1,
        noc_index=noc_index,
        receiver_noc_index=receiver_noc_index,
        request=request,
    )
