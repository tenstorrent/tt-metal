# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for profile_log_device.csv schema (ARCH metadata + column headers).

Other tools (e.g. TTNN Visualizer) use this schema to parse profile_log_device.csv.
"""

import os
import subprocess
from pathlib import Path

import pytest

from models.common.utility_functions import skip_for_blackhole
from tracy.common import (
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_LOGS_DIR,
    TT_METAL_HOME,
    clear_profiler_runtime_artifacts,
    generate_logs_folder,
)
from tracy.device_log_schema import (
    DEVICE_LOG_COLUMN_HEADERS,
    DeviceLogSchemaError,
    parse_profile_log_device_csv,
    validate_profile_log_device_arch_and_headers,
    validate_profile_log_device_csv,
)
from tracy.process_device_log import extract_device_info, import_device_profile_log
from tracy.process_model_log import get_profiler_folder, run_device_profiler

# Same binary used by test_custom_cycle_count in test_device_profiler.py.
PROFILER_EXAMPLE = TT_METAL_HOME / "build/programming_examples/profiler/test_custom_cycle_count"
TRACY_MATMUL_COMMAND = (
    "pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py"
    "::test_run_matmul_test[BFLOAT16-input_shapes0]"
)
TRACY_OUTPUT_NAME = "DeviceLogSchema"


def _device_contention_message() -> str | None:
    """Return a message if another process likely holds the device."""
    lock_files = list(Path("/tmp").glob("CHIP_IN_USE*"))
    if lock_files:
        return (
            "Device PCIe lock is held. Kill the blocking process and retry. "
            f"Lock files: {[str(p) for p in lock_files]}"
        )

    try:
        proc = subprocess.run(
            ["pgrep", "-af", "capture-release|python3 -m tracy|python -m tracy"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    lines = [
        line
        for line in proc.stdout.splitlines()
        if line.strip() and str(os.getpid()) not in line and "pgrep" not in line
    ]
    if lines:
        preview = "\n  ".join(lines[:5])
        return "Device may be held by a stale Tracy capture session. " f"Kill these processes and retry:\n  {preview}"
    return None


def _generate_profile_log_device_csv(tmp_path: Path, timeout_s: int = 90) -> Path:
    """Run a minimal profiler workload and return the generated CSV path."""
    contention = _device_contention_message()
    if contention:
        pytest.fail(contention)

    clear_profiler_runtime_artifacts()

    if PROFILER_EXAMPLE.is_file():
        workload_log = tmp_path / "test_custom_cycle_count.log"
        cmd = f"cd {TT_METAL_HOME} && TT_METAL_DEVICE_PROFILER=1 {PROFILER_EXAMPLE}"
        with workload_log.open("w") as log_f:
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    timeout=timeout_s,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                tail = workload_log.read_text()[-4000:]
                pytest.fail(
                    "Profiler example timed out opening or using the device. "
                    "Another process may hold the PCIe lock; check for stale "
                    f"`tracy` / `capture-release` sessions.\nLog tail:\n{tail}"
                )
        assert proc.returncode == 0, f"Profiler example failed:\n{workload_log.read_text()}"
        return PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

    run_device_profiler(
        TRACY_MATMUL_COMMAND,
        TRACY_OUTPUT_NAME,
        op_support_count=1000,
    )
    return generate_logs_folder(get_profiler_folder(TRACY_OUTPUT_NAME)) / PROFILER_DEVICE_SIDE_LOG


def _validate_generated_device_log(csv_path: Path) -> None:
    metadata = validate_profile_log_device_arch_and_headers(csv_path)
    assert metadata.arch
    assert metadata.freq_mhz > 0
    assert metadata.max_compute_cores is not None
    assert metadata.max_compute_cores > 0

    validate_profile_log_device_csv(csv_path)

    arch, freq, max_cores = extract_device_info(csv_path)
    assert arch == metadata.arch
    assert freq == metadata.freq_mhz
    assert max_cores == metadata.max_compute_cores

    df = parse_profile_log_device_csv(csv_path)
    assert list(df.columns) == DEVICE_LOG_COLUMN_HEADERS
    assert len(df) > 0

    devices_data = import_device_profile_log(csv_path)
    assert devices_data["deviceInfo"]["arch"] == metadata.arch
    assert devices_data["deviceInfo"]["freq"] == metadata.freq_mhz
    assert 0 in devices_data["devices"]


@pytest.mark.timeout(600)
@skip_for_blackhole()
def test_profile_log_device_csv_from_device_profiler(tmp_path):
    """Generate profile_log_device.csv via device profiler and validate schema."""
    csv_path = _generate_profile_log_device_csv(tmp_path)
    assert csv_path.is_file(), f"Expected device log at {csv_path}"
    _validate_generated_device_log(csv_path)


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
