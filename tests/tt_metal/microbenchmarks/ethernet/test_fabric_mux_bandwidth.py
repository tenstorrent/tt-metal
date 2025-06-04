# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from loguru import logger
import pytest
import pandas as pd
import numpy as np
import csv
from tabulate import tabulate
import shutil

OUTPUT_FILE_DIR = os.path.join(os.environ["TT_METAL_HOME"], "generated")
OUTPUT_FILE_NAME = "fabric_mux_bandwidth.csv"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_FILE_DIR, OUTPUT_FILE_NAME)

LOG_FILE_NAME = "fabric_mux_bandwidth_temp.txt"
LOG_FILE_PATH = os.path.join(OUTPUT_FILE_DIR, LOG_FILE_NAME)

GOLDEN_FILE_DIR = os.path.join(os.environ["TT_METAL_HOME"], "tests/tt_metal/microbenchmarks/ethernet")
GOLDEN_FILE_NAME = "fabric_mux_bandwidth_golden.csv"
GOLDEN_FILE_PATH = os.path.join(GOLDEN_FILE_DIR, GOLDEN_FILE_NAME)

BW_THRESHOLD = 0.1

HEADERS = [
    "Test name",
    "Num full size channels",
    "Num header only channels",
    "Num buffers full size channel",
    "Num buffers header only channel",
    "Num packets",
    "Packet payload size bytes",
    "Num full size channel iters",
    "Num iters b/w teardown checks",
    "Bandwidth (B/c)",
]


def read_golden_results(test_params):
    if not os.path.exists(GOLDEN_FILE_PATH):
        logger.warning("No golden data file found")
        return 0

    df = pd.read_csv(GOLDEN_FILE_PATH)
    golden_result = df[np.logical_and.reduce([df[x] == y for x, y in test_params.items()])]["Bandwidth (B/c)"].values
    if len(golden_result) == 0:
        logger.error(f"No matching golden data found")
        return None

    expected_bw = golden_result[0]
    return float(expected_bw)


def summarize_to_csv(test_params, current_bw):
    # if the file doesnt exist, create one and write the header
    if not os.path.exists(OUTPUT_FILE_PATH):
        with open(OUTPUT_FILE_PATH, "w") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

    # append results
    with open(OUTPUT_FILE_PATH, "a") as f:
        writer = csv.writer(f)
        results = test_params.copy()
        results["Bandwidth (B/c)"] = current_bw
        output = [results[x] for x in HEADERS]
        writer.writerow(output)


def process_results(test_params, current_bw):
    expected_bw = read_golden_results(test_params)

    summarize_to_csv(test_params, current_bw)

    if expected_bw is None:
        assert 0, "Probably a new test, please update the golden file accrodingly"
    else:
        assert (
            expected_bw - BW_THRESHOLD <= current_bw <= expected_bw + BW_THRESHOLD
        ), f"Bandwidth mismatch. expected: {expected_bw} B/c, actual: {current_bw} B/c, test params: {test_params}"


def run_mux_benchmark_test(
    test_name,
    num_full_size_channels,
    num_header_only_channels,
    num_buffers_full_size_channel,
    num_buffers_header_only_channel,
    num_packets,
    packet_payload_size_bytes,
    num_full_size_channel_iters,
    num_iters_between_teardown_checks,
):
    cmd = f"{os.environ['TT_METAL_HOME']}/build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_mux_bandwidth \
            --log_file {LOG_FILE_PATH} \
            --test_name {test_name} \
            --num_full_size_channels {num_full_size_channels} \
            --num_header_only_channels {num_header_only_channels} \
            --num_buffers_full_size_channel {num_buffers_full_size_channel} \
            --num_buffers_header_only_channel {num_buffers_header_only_channel} \
            --num_packets {num_packets} \
            --packet_payload_size_bytes {packet_payload_size_bytes} \
            --num_full_size_channel_iters {num_full_size_channel_iters} \
            --num_iters_between_teardown_checks {num_iters_between_teardown_checks}"
    logger.info(f"Running command: {cmd}")

    rc = os.system(cmd)
    if rc != 0:
        logger.info("Error in running the test")
        assert False

    with open(LOG_FILE_PATH, "r") as f:
        current_bw = float(f.read())

    test_params = {}
    test_params["Test name"] = test_name
    test_params["Num full size channels"] = num_full_size_channels
    test_params["Num header only channels"] = num_header_only_channels
    test_params["Num buffers full size channel"] = num_buffers_full_size_channel
    test_params["Num buffers header only channel"] = num_buffers_header_only_channel
    test_params["Num packets"] = num_packets
    test_params["Packet payload size bytes"] = packet_payload_size_bytes
    test_params["Num full size channel iters"] = num_full_size_channel_iters
    test_params["Num iters b/w teardown checks"] = num_iters_between_teardown_checks

    process_results(test_params, current_bw)


@pytest.mark.parametrize("num_full_size_channels", [1, 2, 4, 8])
@pytest.mark.parametrize("num_header_only_channels", [0])
@pytest.mark.parametrize("num_buffers_full_size_channel", [8])
@pytest.mark.parametrize("num_buffers_header_only_channel", [0])
@pytest.mark.parametrize("num_packets", [10000])
@pytest.mark.parametrize("packet_payload_size_bytes", [4096])
@pytest.mark.parametrize("num_full_size_channel_iters", [1])
@pytest.mark.parametrize("num_iters_between_teardown_checks", [32])
def test_mux_bw_full_size_channels(
    num_full_size_channels,
    num_header_only_channels,
    num_buffers_full_size_channel,
    num_buffers_header_only_channel,
    num_packets,
    packet_payload_size_bytes,
    num_full_size_channel_iters,
    num_iters_between_teardown_checks,
):
    test_name = "full_size_channels"
    run_mux_benchmark_test(
        test_name,
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channel,
        num_buffers_header_only_channel,
        num_packets,
        packet_payload_size_bytes,
        num_full_size_channel_iters,
        num_iters_between_teardown_checks,
    )


@pytest.mark.parametrize("num_full_size_channels", [8])
@pytest.mark.parametrize("num_header_only_channels", [0])
@pytest.mark.parametrize("num_buffers_full_size_channel", [8])
@pytest.mark.parametrize("num_buffers_header_only_channel", [0])
@pytest.mark.parametrize("num_packets", [10000])
@pytest.mark.parametrize("packet_payload_size_bytes", [16, 1024, 2048, 4096])
@pytest.mark.parametrize("num_full_size_channel_iters", [1])
@pytest.mark.parametrize("num_iters_between_teardown_checks", [32])
def test_mux_bw_full_size_channel_packet_size(
    num_full_size_channels,
    num_header_only_channels,
    num_buffers_full_size_channel,
    num_buffers_header_only_channel,
    num_packets,
    packet_payload_size_bytes,
    num_full_size_channel_iters,
    num_iters_between_teardown_checks,
):
    test_name = "full_size_channel_packet_size"
    run_mux_benchmark_test(
        test_name,
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channel,
        num_buffers_header_only_channel,
        num_packets,
        packet_payload_size_bytes,
        num_full_size_channel_iters,
        num_iters_between_teardown_checks,
    )


@pytest.mark.parametrize("num_full_size_channels", [8])
@pytest.mark.parametrize("num_header_only_channels", [0])
@pytest.mark.parametrize("num_buffers_full_size_channel", [1, 2, 4, 8])
@pytest.mark.parametrize("num_buffers_header_only_channel", [0])
@pytest.mark.parametrize("num_packets", [10000])
@pytest.mark.parametrize("packet_payload_size_bytes", [4096])
@pytest.mark.parametrize("num_full_size_channel_iters", [1])
@pytest.mark.parametrize("num_iters_between_teardown_checks", [32])
def test_mux_bw_full_size_channel_buffers(
    num_full_size_channels,
    num_header_only_channels,
    num_buffers_full_size_channel,
    num_buffers_header_only_channel,
    num_packets,
    packet_payload_size_bytes,
    num_full_size_channel_iters,
    num_iters_between_teardown_checks,
):
    test_name = "full_size_channel_buffers"
    run_mux_benchmark_test(
        test_name,
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channel,
        num_buffers_header_only_channel,
        num_packets,
        packet_payload_size_bytes,
        num_full_size_channel_iters,
        num_iters_between_teardown_checks,
    )


@pytest.mark.parametrize("num_full_size_channels", [8])
@pytest.mark.parametrize("num_header_only_channels", [0])
@pytest.mark.parametrize("num_buffers_full_size_channel", [8])
@pytest.mark.parametrize("num_buffers_header_only_channel", [0])
@pytest.mark.parametrize("num_packets", [10000])
@pytest.mark.parametrize("packet_payload_size_bytes", [4096])
@pytest.mark.parametrize("num_full_size_channel_iters", [1])
@pytest.mark.parametrize("num_iters_between_teardown_checks", [16, 32, 64, 128])
def test_mux_bw_full_size_channel_teardown_iters(
    num_full_size_channels,
    num_header_only_channels,
    num_buffers_full_size_channel,
    num_buffers_header_only_channel,
    num_packets,
    packet_payload_size_bytes,
    num_full_size_channel_iters,
    num_iters_between_teardown_checks,
):
    test_name = "full_size_channel_teardown_iters"
    run_mux_benchmark_test(
        test_name,
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channel,
        num_buffers_header_only_channel,
        num_packets,
        packet_payload_size_bytes,
        num_full_size_channel_iters,
        num_iters_between_teardown_checks,
    )


@pytest.mark.parametrize("num_full_size_channels", [1, 2, 4, 8])
@pytest.mark.parametrize("num_header_only_channels", [1, 2, 4, 8])
@pytest.mark.parametrize("num_buffers_full_size_channel", [8])
@pytest.mark.parametrize("num_buffers_header_only_channel", [8])
@pytest.mark.parametrize("num_packets", [10000])
@pytest.mark.parametrize("packet_payload_size_bytes", [4096])
@pytest.mark.parametrize("num_full_size_channel_iters", [1])
@pytest.mark.parametrize("num_iters_between_teardown_checks", [32])
def test_mux_bw_both_channel_types(
    num_full_size_channels,
    num_header_only_channels,
    num_buffers_full_size_channel,
    num_buffers_header_only_channel,
    num_packets,
    packet_payload_size_bytes,
    num_full_size_channel_iters,
    num_iters_between_teardown_checks,
):
    test_name = "both_channel_types"
    run_mux_benchmark_test(
        test_name,
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channel,
        num_buffers_header_only_channel,
        num_packets,
        packet_payload_size_bytes,
        num_full_size_channel_iters,
        num_iters_between_teardown_checks,
    )


@pytest.fixture(scope="session", autouse=True)
def setup(request):
    clear_dir = False
    if not os.path.exists(OUTPUT_FILE_DIR):
        clear_dir = True
        os.mkdir(OUTPUT_FILE_DIR)

    # Delete old CSV file at start
    if os.path.exists(OUTPUT_FILE_PATH):
        os.remove(OUTPUT_FILE_PATH)
        logger.info("Removed old bandwidth summary file")

    # create the log file if doesnt exist
    if not os.path.exists(LOG_FILE_PATH):
        f = open(LOG_FILE_PATH, "x")
        f.close()
        logger.info("Created log file")

    yield

    # clear the log file
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)

    # remove the directory if it was created during the test
    if clear_dir and os.path.exists(OUTPUT_FILE_DIR):
        shutil.rmtree(OUTPUT_FILE_DIR)
