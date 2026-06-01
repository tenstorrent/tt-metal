# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tracy.device_log_schema import (
    DEVICE_LOG_COLUMN_HEADERS,
    DeviceLogSchemaError,
    parse_profile_log_device_csv,
    validate_profile_log_device_csv,
)
from tracy.process_device_log import import_device_profile_log

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "profile_log_device_minimal.csv"


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
        validate_profile_log_device_csv(bad_log)


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
        validate_profile_log_device_csv(bad_log)


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
