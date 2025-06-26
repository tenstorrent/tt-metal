# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


from helpers.profiler import Profiler
from helpers.test_config import ProfilerBuild, run_test


def assert_marker(
    marker_obj, expected_marker, expected_file_suffix, expected_line, expected_id
):
    assert (
        marker_obj.marker == expected_marker
    ), f"Expected marker = '{expected_marker}', got {marker_obj.marker}"
    assert marker_obj.file.endswith(
        expected_file_suffix
    ), f"Expected file to end with '{expected_file_suffix}', got {marker_obj.file}"
    assert (
        marker_obj.line == expected_line
    ), f"Expected line = {expected_line}, got {marker_obj.line}"
    assert (
        marker_obj.id == expected_id
    ), f"Expected id = {expected_id}, got {marker_obj.id}"


def test_profiler_primitives():

    test_config = {
        "testname": "profiler_primitives_test",
    }

    run_test(test_config, profiler_build=ProfilerBuild.Yes)

    runtime = Profiler.get_data(test_config["testname"])

    # ZONE_SCOPED
    zone = runtime.unpack[0]
    assert_marker(
        zone.full_marker,
        "TEST_ZONE",
        "profiler_primitives_test.cpp",
        17,
        42158,
    )
    assert zone.start > 0, f"Expected zone.start > 0, got {zone.start}"
    assert zone.end > zone.start, f"Expected zone.end > {zone.start}, got {zone.end}"
    assert (
        zone.duration == zone.end - zone.start
    ), f"Expected zone.duration = {zone.end - zone.start}, got {zone.duration}"

    # TIMESTAMP
    timestamp = runtime.math[0]
    assert_marker(
        timestamp.full_marker,
        "TEST_TIMESTAMP",
        "profiler_primitives_test.cpp",
        26,
        28111,
    )
    assert (
        timestamp.timestamp > 0
    ), f"Expected timestamp.timestamp > 0, got {timestamp.timestamp}"
    assert (
        timestamp.data is None
    ), f"Expected timestamp.date to be None, got {timestamp.data}"

    # TIMESTAMP_DATA
    timestamp_data = runtime.pack[0]
    assert_marker(
        timestamp_data.full_marker,
        "TEST_TIMESTAMP_DATA",
        "profiler_primitives_test.cpp",
        35,
        18694,
    )
    assert (
        timestamp_data.timestamp > 0
    ), f"Expected timestamp_data.timestamp > 0, got {timestamp_data.timestamp}"
    assert (
        timestamp_data.data == 0xBADC0FFE0DDF00D
    ), f"Expected timestamp_data.data = 0xBADC0FFE0DDF00D, got {hex(timestamp_data.data)}"
