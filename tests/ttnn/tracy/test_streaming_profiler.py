# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming profiler end-to-end test over the DRISC relay path.

Runs the ``test_streaming_profiler_zones`` workload with ``TT_METAL_STREAMING_PROFILER=1`` and
``TT_METAL_STREAMING_PROFILER_TRACY=1`` under a connected ``tracy-capture``, and checks that the relays go
resident at bring-up and that the capture holds device zones from every RISC of every workload core, read back with
``tracy-csvexport -u``. Needs a Blackhole box with DRAM programmable cores; device work runs in a subprocess so the
pytest parent never takes the PCIe lock.
"""

from __future__ import annotations

import csv
import io
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tools.tracy.common import PROFILER_ARTIFACTS_DIR, PROFILER_BIN_DIR, TT_METAL_HOME

CAPTURE_TOOL = PROFILER_BIN_DIR / "tracy-capture"
WORKLOAD_BIN = Path(TT_METAL_HOME) / "build_Release" / "programming_examples" / "test_streaming_profiler_zones"
CSV_EXPORT = PROFILER_BIN_DIR / "tracy-csvexport"
RISCS_PER_CORE = 5
ARTIFACTS = PROFILER_ARTIFACTS_DIR / "streaming_profiler_tests"


def _free_port() -> str:
    ip = socket.gethostbyname(socket.gethostname())
    for port in range(8086, 8500):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((ip, port))
            s.close()
            return str(port)
        except (PermissionError, OSError):
            continue
    raise RuntimeError("no free TCP port for tracy-capture")


def _device_lanes_with_zones(tracy_file: Path) -> int:
    """Distinct (core, RISC) threads carrying device zones. The Tracy sink stamps every device zone with the source
    file ``kernel_profiler``, which is what separates them from host zones in the export."""
    out = subprocess.run([str(CSV_EXPORT), "-u", str(tracy_file)], capture_output=True, text=True, timeout=300).stdout
    return len({row["thread"] for row in csv.DictReader(io.StringIO(out)) if row["src_file"] == "kernel_profiler"})


@pytest.mark.parametrize("gx,gy,iters", [(2, 2, 50)])
def test_streaming_profiler_zones_capture(gx, gy, iters):
    if not WORKLOAD_BIN.exists():
        pytest.skip(f"workload not built: {WORKLOAD_BIN} (build target test_streaming_profiler_zones)")
    if not CAPTURE_TOOL.exists():
        pytest.skip(f"tracy-capture not found: {CAPTURE_TOOL}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_tracy = ARTIFACTS / "streaming_profiler_zones.tracy"
    out_tracy.unlink(missing_ok=True)
    port = _free_port()

    cap = subprocess.Popen(
        [str(CAPTURE_TOOL), "-o", str(out_tracy), "-f", "-p", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # let tracy-capture start listening

    env = dict(os.environ)
    env["TRACY_PORT"] = port
    env[
        "TT_METAL_STREAMING_PROFILER"
    ] = "1"  # boot the module at bring-up (its own mode; excludes TT_METAL_DEVICE_PROFILER)
    env["TT_METAL_STREAMING_PROFILER_TRACY"] = "1"  # the Tracy sink is opt-in
    try:
        proc = subprocess.run(
            # --markers 1: nothing else in the tree emits PP_EVENT / Data payload records, so without it
            # those wire layouts go untested.
            [str(WORKLOAD_BIN), "--gx", str(gx), "--gy", str(gy), "--iters", str(iters), "--markers", "1"],
            env=env,
            cwd=str(TT_METAL_HOME),
            timeout=300,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            cap.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            cap.terminate()
            cap.communicate()

    log = proc.stdout + proc.stderr
    # Match only the role-agnostic substring of the residency log line: a guard tied to fuller wording
    # skips unconditionally the moment the message is reworded, leaving the test green and asserting nothing.
    if "not Blackhole" in log or "[streaming profiler] active on" not in log:
        pytest.skip("streaming profiler did not start the DRISC relay (not Blackhole / no DRAM programmable cores)")

    assert proc.returncode == 0, f"workload failed (rc={proc.returncode}):\n{log[-2000:]}"
    assert "active on 1 device(s)" in log, "streaming profiler did not report active"
    assert out_tracy.exists() and out_tracy.stat().st_size > 4096, "no/empty Tracy capture produced"

    if not CSV_EXPORT.exists():
        pytest.fail(
            f"tracy-csvexport not built at {CSV_EXPORT} -- the device-zone assertion cannot run and this test "
            f"would otherwise verify only that the capture exceeds 4096 bytes."
        )

    lanes = _device_lanes_with_zones(out_tracy)
    assert lanes >= gx * gy * RISCS_PER_CORE, f"expected >= {gx * gy * RISCS_PER_CORE} lanes with zones, got {lanes}"
