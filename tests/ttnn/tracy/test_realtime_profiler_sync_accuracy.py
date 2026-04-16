# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end accuracy test for the real-time profiler host/device clock sync.

The mesh device emits paired Tracy events on every sync cycle:
  - A host-side Tracy message (`SYNC_CHECK` at init, `FINISH_SYNC` per mid-run)
    captured on the host CPU just before the PCIe write.
  - A device-side TT-device GPU zone `SYNC_CHECK` placed via
    `PushSyncCheckMarker(device_time, ...)`.

The visual distance between the host marker and the device zone on the Tracy
timeline is the end-to-end calibration error.  If the sync is working it should
be dominated by the H2D PCIe write latency (~1-3µs).

This test runs a short matmul workload under Tracy capture, exports the
resulting trace with `csvexport-release`, pairs each host message with the
next device `SYNC_CHECK` zone by timestamp, and asserts every pair is within
±10µs.
"""

import csv
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
PROFILER_ARTIFACTS_DIR = TT_METAL_HOME / "generated" / "profiler" / "realtime_profiler_sync_accuracy"
CAPTURE_TOOL = PROFILER_BIN_DIR / "capture-release"
CSVEXPORT_TOOL = PROFILER_BIN_DIR / "csvexport-release"
WORKLOAD_SCRIPT = Path(__file__).parent / "realtime_profiler_sync_workload.py"

SYNC_DIFF_THRESHOLD_NS = 10_000  # ±10 µs
# How close (in ns) a device SYNC_CHECK zone has to be to a host sync message
# to be considered the matching pair. If we don't find a zone within this
# window it means something went wrong (calibration shifted by a lot, a host
# message was dropped, etc.) and the test should fail.
PAIRING_WINDOW_NS = 100_000  # 100 µs

HOST_SYNC_MESSAGES = {"SYNC_CHECK", "FINISH_SYNC"}


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


def parse_host_sync_messages(csv_path):
    """Parse csvexport -m output; return list of (name, ns_since_start)."""
    out = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            name, ns = row[0], row[1]
            if name in HOST_SYNC_MESSAGES:
                out.append((name, int(ns)))
    return out


def parse_device_sync_zones(csv_path):
    """
    Parse csvexport -u output and return list of ns_since_start values for
    every device-side `SYNC_CHECK` GPU zone, sorted ascending.
    """
    out = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row.get("name", "") == "SYNC_CHECK" and row.get("src_file", "") == "sync_check":
                try:
                    out.append(int(row["ns_since_start"]))
                except (KeyError, ValueError):
                    continue
    out.sort()
    return out


def pair_host_to_device(host_msgs, device_zones):
    """
    Pair each host SYNC_CHECK/FINISH_SYNC message with the device SYNC_CHECK
    zone closest to it in time (within PAIRING_WINDOW_NS). Returns a list of
    (host_name, host_ns, device_ns, diff_ns) tuples where diff_ns = device_ns
    - host_ns (negative means device is before host, positive means after).
    """
    if not device_zones:
        return [(n, t, None, None) for (n, t) in host_msgs]

    pairs = []
    for name, h_ns in host_msgs:
        # Binary search: find the index of the zone closest to h_ns.
        lo, hi = 0, len(device_zones) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if device_zones[mid] < h_ns:
                lo = mid + 1
            else:
                hi = mid
        candidates = []
        if lo < len(device_zones):
            candidates.append(device_zones[lo])
        if lo > 0:
            candidates.append(device_zones[lo - 1])
        # Pick candidate with smallest absolute diff.
        best = min(candidates, key=lambda d: abs(d - h_ns))
        pairs.append((name, h_ns, best, best - h_ns))
    return pairs


def test_realtime_profiler_sync_accuracy(tmp_path):
    assert CAPTURE_TOOL.exists(), f"Tracy capture tool not found: {CAPTURE_TOOL}"
    assert CSVEXPORT_TOOL.exists(), f"Tracy csvexport tool not found: {CSVEXPORT_TOOL}"
    assert WORKLOAD_SCRIPT.exists(), f"Workload script not found: {WORKLOAD_SCRIPT}"

    port = get_available_port()
    assert port is not None, "No available port for Tracy capture"

    tracy_file = tmp_path / "trace.tracy"
    messages_csv = tmp_path / "messages.csv"
    zones_csv = tmp_path / "zones.csv"

    # 1. Start Tracy capture.
    capture_proc = subprocess.Popen(
        [str(CAPTURE_TOOL), "-o", str(tracy_file), "-f", "-p", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    # 2. Run the workload under Tracy.
    env = dict(os.environ)
    env["TRACY_PORT"] = port
    env.pop("TT_METAL_DEVICE_PROFILER", None)

    workload_log = tmp_path / "workload_output.log"
    try:
        with open(workload_log, "w") as log_f:
            proc = subprocess.run(
                ["python3", str(WORKLOAD_SCRIPT)],
                env=env,
                cwd=str(TT_METAL_HOME),
                timeout=300,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
    finally:
        try:
            capture_proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            capture_proc.terminate()
            capture_proc.communicate()

    log_text = workload_log.read_text()
    print(f"\n--- Workload output (tail) ---\n{log_text[-4000:]}\n--- End workload output ---")
    assert proc.returncode == 0, f"Workload script failed (rc={proc.returncode})"
    assert tracy_file.exists(), f"Tracy trace file not generated: {tracy_file}"

    # 3. Export host messages and full zone list (CPU + GPU) from the trace.
    with open(messages_csv, "w") as out:
        subprocess.run(
            [str(CSVEXPORT_TOOL), "-m", "-s", ";", str(tracy_file)],
            stdout=out,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    with open(zones_csv, "w") as out:
        subprocess.run(
            [str(CSVEXPORT_TOOL), "-u", "-s", ";", str(tracy_file)],
            stdout=out,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    # 4. Persist artifacts for post-mortem investigation.
    PROFILER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tracy_file, PROFILER_ARTIFACTS_DIR / "trace.tracy")
    shutil.copy2(messages_csv, PROFILER_ARTIFACTS_DIR / "messages.csv")
    shutil.copy2(zones_csv, PROFILER_ARTIFACTS_DIR / "zones.csv")
    shutil.copy2(workload_log, PROFILER_ARTIFACTS_DIR / "workload_output.log")

    # 5. Parse host messages + device GPU zones.
    host_msgs = parse_host_sync_messages(messages_csv)
    device_zones = parse_device_sync_zones(zones_csv)

    assert len(host_msgs) > 0, (
        "No host SYNC_CHECK/FINISH_SYNC Tracy messages found. " "Real-time profiler may not have been initialized."
    )
    assert len(device_zones) > 0, (
        "No device-side SYNC_CHECK GPU zones found. Either the device never "
        "responded to the sync handshake, or csvexport is not emitting GPU zones."
    )

    # 6. Pair each host message with the nearest device zone.
    pairs = pair_host_to_device(host_msgs, device_zones)
    assert len(pairs) == len(host_msgs)

    print(f"\nMatched {len(pairs)} host sync message(s) against {len(device_zones)} " f"device SYNC_CHECK zone(s).")
    for name, h_ns, d_ns, diff in pairs:
        if d_ns is None:
            print(f"  {name:>12} @ {h_ns:>14,}ns  ->  NO MATCH")
        else:
            print(f"  {name:>12} @ {h_ns:>14,}ns  ->  {d_ns:>14,}ns  diff={diff:+.0f}ns")

    # 7. Fail if any pair is missing or outside the tight accuracy window.
    missing = [(n, h) for (n, h, d, _diff) in pairs if d is None]
    assert not missing, (
        f"{len(missing)} host sync message(s) had no device zone within " f"{PAIRING_WINDOW_NS}ns: {missing[:5]}"
    )

    far = [(n, h, d, diff) for (n, h, d, diff) in pairs if abs(diff) > PAIRING_WINDOW_NS]
    assert not far, (
        f"{len(far)} pair(s) outside pairing window ±{PAIRING_WINDOW_NS}ns — "
        f"likely a dropped message or gross mis-calibration: {far[:5]}"
    )

    bad = [(n, h, d, diff) for (n, h, d, diff) in pairs if abs(diff) > SYNC_DIFF_THRESHOLD_NS]
    assert not bad, f"{len(bad)}/{len(pairs)} sync check pair(s) exceeded ±{SYNC_DIFF_THRESHOLD_NS}ns:\n" + "\n".join(
        f"  {n} host={h}ns device={d}ns diff={diff:+.0f}ns" for (n, h, d, diff) in bad
    )

    diffs = [diff for (_n, _h, _d, diff) in pairs]
    print(
        f"\nAll {len(pairs)} sync check pairs within ±{SYNC_DIFF_THRESHOLD_NS}ns "
        f"(min={min(diffs):+.0f}ns, max={max(diffs):+.0f}ns, "
        f"mean={sum(diffs)/len(diffs):+.0f}ns)."
    )
