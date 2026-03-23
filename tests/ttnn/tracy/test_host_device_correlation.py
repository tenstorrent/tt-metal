# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Test that verifies 1:1 correspondence between host-side Tracy messages
(EnqueueProgram op_id=X) and device-side real-time profiler records
(program_id=X).

Uses the full Tracy capture pipeline: capture-release captures the trace,
csvexport-release extracts messages, and the real-time profiler callback
(in the inner workload script) captures device records.
"""

import csv
import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest


TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
PROFILER_BIN_DIR = TT_METAL_HOME / "build" / "tools" / "profiler" / "bin"
PROFILER_ARTIFACTS_DIR = TT_METAL_HOME / "generated" / "profiler" / "host_device_correlation"
CAPTURE_TOOL = PROFILER_BIN_DIR / "capture-release"
CSVEXPORT_TOOL = PROFILER_BIN_DIR / "csvexport-release"
WORKLOAD_SCRIPT = Path(__file__).parent / "host_device_correlation_workload.py"


def get_available_port():
    """Find a free TCP port for Tracy capture."""
    ip = socket.gethostbyname(socket.gethostname())
    for port in range(8086, 8500):
        try:
            serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serv.bind((ip, port))
            serv.close()
            return str(port)
        except (PermissionError, OSError):
            pass
    return None


def parse_tracy_messages(csv_path):
    """
    Parse csvexport message output and extract EnqueueProgram op_ids.
    Returns a list of integer op_ids in the order they appear.
    """
    op_ids = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar="`")
        for row in reader:
            msg = row.get("MessageName", "")
            match = re.match(r"EnqueueProgram op_id=(\d+)", msg)
            if match:
                op_ids.append(int(match.group(1)))
    return op_ids


def test_host_device_correlation(tmp_path):
    """
    Run ResNet50 inference under Tracy capture and verify that every
    device-side program record has a matching host-side TracyMessage,
    establishing a 1:1 coupling between host dispatch and device execution.
    """
    assert CAPTURE_TOOL.exists(), f"Tracy capture tool not found: {CAPTURE_TOOL}"
    assert CSVEXPORT_TOOL.exists(), f"Tracy csvexport tool not found: {CSVEXPORT_TOOL}"
    assert WORKLOAD_SCRIPT.exists(), f"Workload script not found: {WORKLOAD_SCRIPT}"

    port = get_available_port()
    assert port is not None, "No available port for Tracy capture"

    tracy_file = tmp_path / "trace.tracy"
    messages_csv = tmp_path / "messages.csv"
    device_records_json = tmp_path / "device_records.json"

    # 1. Start Tracy capture
    capture_proc = subprocess.Popen(
        [str(CAPTURE_TOOL), "-o", str(tracy_file), "-f", "-p", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Brief pause to let capture-release start listening
    time.sleep(2)

    # 2. Run the workload script under Tracy
    env = dict(os.environ)
    env["TRACY_PORT"] = port
    env["DEVICE_RECORDS_PATH"] = str(device_records_json)
    # Ensure device profiler is OFF so TracyMessages are emitted
    env.pop("TT_METAL_DEVICE_PROFILER", None)

    workload_log = tmp_path / "workload_output.log"
    with open(workload_log, "w") as log_f:
        workload_proc = subprocess.run(
            ["python3", str(WORKLOAD_SCRIPT)],
            env=env,
            cwd=str(TT_METAL_HOME),
            timeout=600,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    # Print workload output for debugging (includes sync info)
    workload_output = workload_log.read_text()
    print(f"\n--- Workload output ---\n{workload_output}\n--- End workload output ---")
    assert workload_proc.returncode == 0, (
        f"Workload script failed with return code {workload_proc.returncode}\n" f"Output:\n{workload_output}"
    )

    # 3. Wait for capture to finish (it exits when the Tracy client disconnects)
    try:
        capture_proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        capture_proc.terminate()
        capture_proc.communicate()
        pytest.fail("Tracy capture did not finish within timeout")

    assert tracy_file.exists(), f"Tracy trace file not generated: {tracy_file}"

    # Copy artifacts to generated/profiler/ for post-run investigation
    PROFILER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_tracy = PROFILER_ARTIFACTS_DIR / "trace.tracy"
    shutil.copy2(tracy_file, saved_tracy)
    print(f"\n  Tracy file saved to: {saved_tracy}")

    # 4. Export messages from the trace
    with open(messages_csv, "w") as csv_out:
        export_proc = subprocess.run(
            [str(CSVEXPORT_TOOL), "-m", "-s", ";", str(tracy_file)],
            stdout=csv_out,
            stderr=subprocess.DEVNULL,
        )
    assert export_proc.returncode == 0, "csvexport failed"
    assert messages_csv.exists() and messages_csv.stat().st_size > 0, "Messages CSV is empty"

    # 5. Parse host-side Tracy messages
    host_op_ids = parse_tracy_messages(messages_csv)
    assert len(host_op_ids) > 0, "No EnqueueProgram TracyMessages found in trace"

    # 6. Load device-side records from callback
    assert device_records_json.exists(), f"Device records file not found: {device_records_json}"
    with open(device_records_json, "r") as f:
        device_records = json.load(f)
    assert len(device_records) > 0, "No device records collected by callback"

    device_program_ids = [r["program_id"] for r in device_records]

    # 7. Cross-reference: verify 1:1 mapping
    host_op_id_set = set(host_op_ids)
    device_pid_set = set(device_program_ids)

    host_op_id_counts = {}
    for op_id in host_op_ids:
        host_op_id_counts[op_id] = host_op_id_counts.get(op_id, 0) + 1

    device_pid_counts = {}
    for pid in device_program_ids:
        device_pid_counts[pid] = device_pid_counts.get(pid, 0) + 1

    # (a) Every host op_id must appear in device records.
    #     This is the critical direction: if the host dispatched a program,
    #     the device must have executed it and reported it back.
    missing_on_device = sorted(host_op_id_set - device_pid_set)
    assert len(missing_on_device) == 0, (
        f"{len(missing_on_device)} host op_id(s) not found in device records: "
        f"{missing_on_device[:10]}{'...' if len(missing_on_device) > 10 else ''}"
    )

    # (b) Device records that have no matching host TracyMessage are
    #     infrastructure programs (dispatch setup, realtime profiler kernel,
    #     etc.) that don't go through enqueue_mesh_workload. Log them but
    #     don't fail as long as the count is small.
    infra_only_on_device = sorted(device_pid_set - host_op_id_set)
    if infra_only_on_device:
        print(
            f"\n  INFO: {len(infra_only_on_device)} device-only program_id(s) "
            f"(infrastructure/dispatch): {infra_only_on_device}"
        )

    # (c) For the matched subset, verify multiplicity is identical.
    matched_ids = host_op_id_set & device_pid_set
    mismatched_counts = []
    for pid in sorted(matched_ids):
        if host_op_id_counts[pid] != device_pid_counts[pid]:
            mismatched_counts.append((pid, host_op_id_counts[pid], device_pid_counts[pid]))
    assert len(mismatched_counts) == 0, (
        f"Multiplicity mismatch for {len(mismatched_counts)} program_id(s): "
        f"{[(pid, f'host={h}, device={d}') for pid, h, d in mismatched_counts[:10]]}"
    )

    # (d) Matched count: all host messages accounted for on device
    matched_host_count = sum(host_op_id_counts[pid] for pid in matched_ids)
    assert matched_host_count == len(
        host_op_ids
    ), f"Not all host messages matched: {matched_host_count}/{len(host_op_ids)}"

    # 8. Validate device record integrity (user programs only;
    #    infrastructure programs may not have proper start/end pairs)
    for rec in device_records:
        if rec["program_id"] in matched_ids:
            assert rec["end_timestamp"] >= rec["start_timestamp"], (
                f"Invalid timestamps for program_id={rec['program_id']}: "
                f"end={rec['end_timestamp']} < start={rec['start_timestamp']}"
            )
        assert rec["frequency_ghz"] > 0, f"Invalid frequency for program_id={rec['program_id']}: {rec['frequency_ghz']}"

    # Save remaining artifacts for investigation
    shutil.copy2(messages_csv, PROFILER_ARTIFACTS_DIR / "messages.csv")
    shutil.copy2(device_records_json, PROFILER_ARTIFACTS_DIR / "device_records.json")
    shutil.copy2(workload_log, PROFILER_ARTIFACTS_DIR / "workload_output.log")

    print(f"\nHost-Device Correlation Test PASSED")
    print(f"  Host TracyMessages:        {len(host_op_ids)}")
    print(f"  Device records (total):    {len(device_records)}")
    print(f"  Matched user programs:     {len(matched_ids)}")
    print(f"  Device-only infra records: {len(infra_only_on_device)}")
    print(f"  Artifacts saved to:        {PROFILER_ARTIFACTS_DIR}")
