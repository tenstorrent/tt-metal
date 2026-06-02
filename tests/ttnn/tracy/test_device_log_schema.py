# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from models.common.utility_functions import skip_for_blackhole
from tracy.common import PROFILER_DEVICE_SIDE_LOG, generate_logs_folder
from tracy.device_log_schema import (
    DEVICE_LOG_COLUMN_HEADERS,
    DeviceLogSchemaError,
    parse_device_arch_metadata,
    parse_profile_log_device_csv,
    validate_profile_log_device_arch_and_headers,
    validate_profile_log_device_csv,
)
from tracy.process_device_log import extract_device_info, import_device_profile_log
from tracy.process_model_log import get_latest_ops_log_filename, get_profiler_folder, run_device_profiler

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "profile_log_device_minimal.csv"

PROFILER_SUBDIR = "DeviceLogSchema"
# Written by C++ under TT_METAL_PROFILER_DIR set by tracy -o; not generated/profiler/.logs.
DEVICE_LOG_PATH = generate_logs_folder(get_profiler_folder(PROFILER_SUBDIR)) / PROFILER_DEVICE_SIDE_LOG


def test_parse_device_arch_metadata_from_fixture():
    metadata = parse_device_arch_metadata(FIXTURE_PATH)
    assert metadata.arch == "wormhole_b0"
    assert metadata.freq_mhz == 1000
    assert metadata.max_compute_cores == 64


def test_extract_device_info_matches_schema_parser():
    arch, freq, max_cores = extract_device_info(FIXTURE_PATH)
    assert arch == "wormhole_b0"
    assert freq == 1000
    assert max_cores == 64


def test_validate_profile_log_device_arch_and_headers_accepts_fixture():
    metadata = validate_profile_log_device_arch_and_headers(FIXTURE_PATH)
    assert metadata.arch == "wormhole_b0"


def test_validate_profile_log_device_csv_accepts_fixture():
    validate_profile_log_device_csv(FIXTURE_PATH)


def test_parse_profile_log_device_csv_returns_named_columns():
    df = parse_profile_log_device_csv(FIXTURE_PATH)
    assert list(df.columns) == DEVICE_LOG_COLUMN_HEADERS
    assert len(df) == 10


def test_import_device_profile_log_accepts_fixture():
    devices_data = import_device_profile_log(FIXTURE_PATH)
    assert devices_data["deviceInfo"]["arch"] == "wormhole_b0"
    assert devices_data["deviceInfo"]["freq"] == 1000
    assert 0 in devices_data["devices"]
    assert (1, 1) in devices_data["devices"][0]["cores"]


def test_validate_rejects_missing_arch_line(tmp_path):
    bad_log = tmp_path / "bad.csv"
    bad_log.write_text(
        "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, "
        "run host ID, trace id, trace id counter, zone name, type, source line, source file, meta data\n"
        "0,1,1,BRISC,1,100,0,1,-1,-1,BRISC-FW,begin,1,a.cc,\n"
    )
    with pytest.raises(DeviceLogSchemaError, match="ARCH metadata"):
        validate_profile_log_device_arch_and_headers(bad_log)


def test_validate_rejects_wrong_column_header(tmp_path):
    headers = ", ".join(DEVICE_LOG_COLUMN_HEADERS)
    headers = headers.replace("core_x", "coreX", 1)
    bad_log = tmp_path / "bad.csv"
    bad_log.write_text(
        f"ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64\n"
        f"{headers}\n"
        f"0,1,1,BRISC,1,100,0,1,-1,-1,BRISC-FW,begin,1,a.cc,\n"
    )
    with pytest.raises(DeviceLogSchemaError, match="Column headers"):
        validate_profile_log_device_arch_and_headers(bad_log)


def test_validate_rejects_wrong_field_count(tmp_path):
    header_line = ", ".join(DEVICE_LOG_COLUMN_HEADERS)
    bad_log = tmp_path / "bad.csv"
    bad_log.write_text(
        f"ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64\n"
        f"{header_line}\n"
        f"0,1,1,BRISC,1,100,0,1,-1,-1,BRISC-FW,begin,1,a.cc\n"
    )
    with pytest.raises(DeviceLogSchemaError, match="has 14 fields"):
        validate_profile_log_device_csv(bad_log)


@skip_for_blackhole()
@pytest.mark.timeout(600)
def test_device_log_written_under_tracy_profiler_dir():
    """End-to-end: tracy -r runs a short TTNN test; C++ writes profile_log_device.csv under -o/.logs."""
    run_device_profiler(
        'pytest "tests/ttnn/tracy/test_trace_runs.py::test_with_ops"',
        PROFILER_SUBDIR,
    )

    assert DEVICE_LOG_PATH.is_file(), f"Missing device log at {DEVICE_LOG_PATH}"

    metadata = validate_profile_log_device_arch_and_headers(DEVICE_LOG_PATH)
    assert metadata.arch
    assert metadata.freq_mhz > 0

    report_device_log = get_latest_ops_log_filename(PROFILER_SUBDIR).parent / PROFILER_DEVICE_SIDE_LOG
    assert report_device_log.is_file(), f"Missing copied device log in report folder: {report_device_log}"
    validate_profile_log_device_arch_and_headers(report_device_log)
