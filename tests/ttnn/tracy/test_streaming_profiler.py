# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming profiler end-to-end test over the DRISC relay path.

Runs the ``test_streaming_profiler_zones`` workload with ``TT_METAL_STREAMING_PROFILER=1`` and
``TT_METAL_STREAMING_PROFILER_TRACY=1`` under a connected ``tracy-capture``, and checks that the relays go
resident at bring-up and that the capture holds device zones across the workload's per-core contexts. Needs a
Blackhole box with DRAM programmable cores and a built ``tools/drisc_drain/tracy_ctx_inspect``; device work runs
in a subprocess so the pytest parent never takes the PCIe lock.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tools.tracy.common import PROFILER_ARTIFACTS_DIR, PROFILER_BIN_DIR, TT_METAL_HOME

CAPTURE_TOOL = PROFILER_BIN_DIR / "tracy-capture"
WORKLOAD_BIN = Path(TT_METAL_HOME) / "build_Release" / "programming_examples" / "test_streaming_profiler_zones"
CTX_INSPECT = Path(TT_METAL_HOME) / "tools" / "drisc_drain" / "tracy_ctx_inspect" / "tracy_ctx_inspect"
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


def _gpu_context_stats(tracy_file: Path) -> tuple[int, int]:
    """Return (num_gpu_contexts, num_contexts_with_zones) via tracy_ctx_inspect."""
    out = subprocess.run([str(CTX_INSPECT), str(tracy_file)], capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"GPU contexts:\s*(\d+)", out)
    n_ctx = int(m.group(1)) if m else 0
    n_with_zones = sum(1 for c in re.findall(r"count=(\d+)", out) if int(c) > 0)
    return n_ctx, n_with_zones


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
    env["TT_METAL_STREAMING_PROFILER"] = "1"  # boot the module at bring-up; implies TT_METAL_DEVICE_PROFILER
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
    if "not Blackhole" in log or "resident on logical" not in log:
        pytest.skip("streaming profiler did not start the DRISC relay (not Blackhole / no DRAM programmable cores)")

    assert proc.returncode == 0, f"workload failed (rc={proc.returncode}):\n{log[-2000:]}"
    assert "active on 1 device(s)" in log, "streaming profiler did not report active"
    assert out_tracy.exists() and out_tracy.stat().st_size > 4096, "no/empty Tracy capture produced"

    if not CTX_INSPECT.exists():
        pytest.fail(
            f"tracy_ctx_inspect not built at {CTX_INSPECT} -- the device-zone assertions cannot run and "
            f"this test would otherwise verify only that the capture exceeds 4096 bytes. Build it with:\n"
            f"  bash tools/drisc_drain/tracy_ctx_inspect/build.sh "
            f"$TT_METAL_HOME/tools/drisc_drain/tracy_ctx_inspect/tracy_ctx_inspect\n"
            f"(the output path must be ABSOLUTE -- build.sh cds into third_party/tracy first)"
        )

    n_ctx, n_with_zones = _gpu_context_stats(out_tracy)
    # All per-core contexts are pre-created, so the substantive check is how many captured zones.
    assert n_ctx >= gx * gy, f"too few GPU contexts: {n_ctx}"
    assert n_with_zones >= gx * gy, f"expected >= {gx * gy} contexts with zones, got {n_with_zones}"
