# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Contract for profile_log_device.csv (device-side profiler output).

The file is written by the tt-metal device profiler (C++), not by Tracy. Tracy copies
it into performance report folders alongside ops_perf_results_*.csv.

Line 1 (ARCH metadata) is the primary consumer for tools that only need chip identity
and frequency (e.g. TTNN Visualizer). Lines 2+ are the column header and per-zone rows.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import NamedTuple, Union

import pandas as pd

# Must match writeCSVHeader() in tt_metal/impl/profiler/profiler.cpp.
DEVICE_LOG_COLUMN_HEADERS = [
    "PCIe slot",
    "core_x",
    "core_y",
    "RISC processor type",
    "timer_id",
    "time[cycles since reset]",
    "data",
    "run host ID",
    "trace id",
    "trace id counter",
    "zone name",
    "type",
    "source line",
    "source file",
    "meta data",
]

ARCH_LINE_PATTERN = re.compile(r"^ARCH:\s*.+,\s*CHIP_FREQ\[MHz\]:\s*\d+,\s*Max Compute Cores:\s*\d+\s*$")

GRAYSKULL_ARCH_LINE_PREFIX = "Chip clock is at "

PathLike = Union[str, Path]


class DeviceLogSchemaError(ValueError):
    """Raised when profile_log_device.csv does not match the expected schema."""


class DeviceArchMetadata(NamedTuple):
    arch: str
    freq_mhz: int
    max_compute_cores: int | None


def _read_text_lines(log_path: Path) -> list[str]:
    if not log_path.is_file():
        raise DeviceLogSchemaError(f"Device log not found: {log_path}")

    with log_path.open("r", newline="") as log_file:
        lines = log_file.readlines()

    if not lines:
        raise DeviceLogSchemaError(f"Device log is empty: {log_path}")

    return lines


def parse_device_arch_metadata(log_path: PathLike) -> DeviceArchMetadata:
    """Parse line 1 of profile_log_device.csv (ARCH / legacy Grayskull metadata)."""
    arch_line = _read_text_lines(Path(log_path))[0].rstrip("\n")

    if GRAYSKULL_ARCH_LINE_PREFIX in arch_line:
        return DeviceArchMetadata(arch="grayskull", freq_mhz=1200, max_compute_cores=None)

    if "ARCH" not in arch_line:
        raise DeviceLogSchemaError(f"Unrecognized ARCH metadata line: {arch_line!r}")

    parts = [segment.strip() for segment in arch_line.split(",")]
    if len(parts) < 3:
        raise DeviceLogSchemaError(f"ARCH metadata line has too few fields: {arch_line!r}")

    arch = parts[0].split(":")[-1].strip()
    freq_mhz = int(parts[1].split(":")[-1].strip())
    max_compute_cores = int(parts[2].split(":")[-1].strip())
    return DeviceArchMetadata(arch=arch, freq_mhz=freq_mhz, max_compute_cores=max_compute_cores)


def validate_arch_line(arch_line: str) -> None:
    arch_line = arch_line.rstrip("\n")
    if GRAYSKULL_ARCH_LINE_PREFIX in arch_line:
        return
    if not ARCH_LINE_PATTERN.match(arch_line):
        raise DeviceLogSchemaError(
            "ARCH metadata line must match "
            "'ARCH: <arch>, CHIP_FREQ[MHz]: <freq>, Max Compute Cores: <n>' "
            f"or legacy Grayskull format; got: {arch_line!r}"
        )


def validate_column_headers(header_line: str) -> list[str]:
    reader = csv.reader([header_line])
    headers = [field.strip() for field in next(reader)]
    if headers != DEVICE_LOG_COLUMN_HEADERS:
        raise DeviceLogSchemaError(
            "Column headers do not match the device profiler contract. "
            f"Expected: {DEVICE_LOG_COLUMN_HEADERS}; got: {headers}"
        )
    return headers


def validate_profile_log_device_arch_and_headers(log_path: PathLike) -> DeviceArchMetadata:
    """Validate ARCH metadata and column headers only (fast; no data-row scan)."""
    path = Path(log_path)
    lines = _read_text_lines(path)

    validate_arch_line(lines[0])
    if len(lines) < 2:
        raise DeviceLogSchemaError(f"Device log missing column header line: {path}")

    validate_column_headers(lines[1].rstrip("\n"))
    return parse_device_arch_metadata(path)


def validate_profile_log_device_csv(log_path: PathLike, *, require_data_rows: bool = True) -> None:
    """Validate ARCH metadata, headers, and optionally every data row."""
    validate_profile_log_device_arch_and_headers(log_path)

    path = Path(log_path)
    lines = _read_text_lines(path)
    headers = [field.strip() for field in next(csv.reader([lines[1].rstrip("\n")]))]

    data_rows = list(csv.reader(lines[2:]))
    if require_data_rows and not data_rows:
        raise DeviceLogSchemaError(f"Device log has no data rows: {path}")

    expected_field_count = len(DEVICE_LOG_COLUMN_HEADERS)
    for row_index, row in enumerate(data_rows, start=3):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != expected_field_count:
            raise DeviceLogSchemaError(
                f"Row {row_index} in {path} has {len(row)} fields; expected {expected_field_count}"
            )

        _validate_typed_fields(row, headers, row_index, path)


def _validate_typed_fields(row: list[str], headers: list[str], row_index: int, log_path: Path) -> None:
    field_map = dict(zip(headers, row))

    for int_column in ("PCIe slot", "core_x", "core_y", "time[cycles since reset]"):
        value = field_map[int_column].strip()
        if value == "":
            raise DeviceLogSchemaError(f"Row {row_index} in {log_path}: {int_column} is empty")
        try:
            int(value)
        except ValueError as exc:
            raise DeviceLogSchemaError(
                f"Row {row_index} in {log_path}: {int_column} must be an integer, got {value!r}"
            ) from exc

    for trace_column in ("trace id", "trace id counter"):
        value = field_map[trace_column].strip()
        if value == "":
            continue
        try:
            int(value)
        except ValueError as exc:
            raise DeviceLogSchemaError(
                f"Row {row_index} in {log_path}: {trace_column} must be an integer or empty, got {value!r}"
            ) from exc


def parse_profile_log_device_csv(log_path: PathLike, *, validate: bool = True) -> pd.DataFrame:
    """Load profile_log_device.csv as a DataFrame with named columns."""
    path = Path(log_path)
    if validate:
        validate_profile_log_device_arch_and_headers(path)

    df = pd.read_csv(path, skiprows=1, header=0, na_filter=False)
    df.columns = [str(column).strip() for column in df.columns]
    if list(df.columns) != DEVICE_LOG_COLUMN_HEADERS:
        raise DeviceLogSchemaError(
            f"Parsed headers do not match contract. Expected {DEVICE_LOG_COLUMN_HEADERS}; got {list(df.columns)}"
        )

    df["trace id"] = pd.to_numeric(df["trace id"], errors="coerce").fillna(-1).astype(int)
    df["trace id counter"] = pd.to_numeric(df["trace id counter"], errors="coerce").fillna(-1).astype(int)
    return df
