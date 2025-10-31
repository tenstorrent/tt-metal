# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from helpers.profiler import Profiler
from helpers.test_config import ProfilerBuild, run_test


def assert_marker(
    entry, expected_marker, expected_file_suffix, expected_line, expected_id
):
    assert (
        entry["marker"] == expected_marker
    ), f"Expected marker = '{expected_marker}', got {entry['marker']}"
    assert entry["file"].endswith(
        expected_file_suffix
    ), f"Expected file to end with '{expected_file_suffix}', got {entry['file']}"
    assert (
        entry["line"] == expected_line
    ), f"Expected line = {expected_line}, got {entry['line']}"
    assert (
        entry["marker_id"] == expected_id
    ), f"Expected marker_id = {expected_id}, got {entry['marker_id']}"


def test_profiler_primitives():

    test_config = {
        "testname": "profiler_primitives_test",
    }

    run_test(test_config, profiler_build=ProfilerBuild.Yes)

    runtime = Profiler.get_data(test_config["testname"])

    # ZONE_SCOPED - Get first ZONE type entry from UNPACK thread
    zones = runtime.unpack().zones().marker("TEST_ZONE").frame()
    assert len(zones) > 0, "Expected at least one TEST_ZONE entry"
    zone = zones.iloc[0]

    assert_marker(
        zone,
        "TEST_ZONE",
        "profiler_primitives_test.cpp",
        17,
        42158,
    )
    assert (
        zone["timestamp"] > 0
    ), f"Expected zone timestamp > 0, got {zone['timestamp']}"
    assert zone["duration"] > 0, f"Expected zone duration > 0, got {zone['duration']}"

    # TIMESTAMP - Get first TIMESTAMP type entry from MATH thread
    timestamps = runtime.math().timestamps().marker("TEST_TIMESTAMP").frame()
    assert len(timestamps) == 1, "Expected exactly one TEST_TIMESTAMP entry"
    timestamp = timestamps.iloc[0]

    assert_marker(
        timestamp,
        "TEST_TIMESTAMP",
        "profiler_primitives_test.cpp",
        26,
        28111,
    )
    assert (
        timestamp["timestamp"] > 0
    ), f"Expected timestamp > 0, got {timestamp['timestamp']}"
    assert pd.isna(
        timestamp["duration"]
    ), f"Expected duration to be None/NaN, got {timestamp['duration']}"
    assert pd.isna(
        timestamp["data"]
    ), f"Expected timestamp data to be None/NaN, got {timestamp['data']}"

    # TIMESTAMP_DATA - Get first TIMESTAMP type entry from PACK thread with data
    data_timestamps = runtime.pack().timestamps().marker("TEST_TIMESTAMP_DATA").frame()
    assert len(data_timestamps) == 1, "Expected exactly one TEST_TIMESTAMP_DATA entry"
    timestamp_data = data_timestamps.iloc[0]

    assert_marker(
        timestamp_data,
        "TEST_TIMESTAMP_DATA",
        "profiler_primitives_test.cpp",
        35,
        18694,
    )
    assert (
        timestamp_data["timestamp"] > 0
    ), f"Expected timestamp > 0, got {timestamp_data['timestamp']}"
    assert pd.isna(
        timestamp_data["duration"]
    ), f"Expected duration to be None/NaN, got {timestamp_data['duration']}"
    assert (
        timestamp_data["data"] == 0xBADC0FFE0DDF00D
    ), f"Expected data = 0xBADC0FFE0DDF00D, got {hex(int(timestamp_data['data']))}"
