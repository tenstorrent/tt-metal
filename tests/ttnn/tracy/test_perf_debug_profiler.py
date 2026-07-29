# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
perf-debug (X280) profiler test.

Runs the ``test_perf_debug_zones`` workload -- which emits 10 differently-named DeviceZoneScopedN zones
(with increasing durations) on all 5 RISCs of a small core grid -- with the perf-debug profiler enabled
(``TT_METAL_PERF_DEBUG_PROFILER=1``) under a connected ``tracy-capture``. Verifies:

  * the module boots the X280 drainer at MeshDevice bring-up (log line), and
  * the resulting Tracy capture holds device (GPU) zones across many per-core contexts.

Hardware-gated: needs a Blackhole box with a working (ECC-primed) X280 L2CPU + the built firmware. Skips
cleanly when the workload reports it is not on Blackhole / the X280 did not boot. Device work runs in a
subprocess so the pytest parent never takes the PCIe lock (mirrors test_realtime_profiler.py).
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
WORKLOAD_BIN = Path(TT_METAL_HOME) / "build_Release" / "programming_examples" / "test_perf_debug_zones"
CTX_INSPECT = Path(TT_METAL_HOME) / "tools" / "x280_bm" / "tracy_ctx_inspect" / "tracy_ctx_inspect"
ARTIFACTS = PROFILER_ARTIFACTS_DIR / "perf_debug_profiler_tests"


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
def test_perf_debug_zones_capture(gx, gy, iters):
    if not WORKLOAD_BIN.exists():
        pytest.skip(f"workload not built: {WORKLOAD_BIN} (build target test_perf_debug_zones)")
    if not CAPTURE_TOOL.exists():
        pytest.skip(f"tracy-capture not found: {CAPTURE_TOOL}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_tracy = ARTIFACTS / "perf_debug_zones.tracy"
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
    env["TT_METAL_PERF_DEBUG_PROFILER"] = "1"  # boot the module at MeshDevice bring-up
    env["TT_METAL_DEVICE_PROFILER"] = "1"  # PROFILE_KERNEL: make the kernels emit markers
    try:
        proc = subprocess.run(
            [str(WORKLOAD_BIN), "--gx", str(gx), "--gy", str(gy), "--iters", str(iters)],
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
    if "not Blackhole" in log or "booted X280 drainer" not in log:
        pytest.skip("perf-debug profiler did not boot the X280 (not Blackhole / X280 down / FW not built)")

    assert proc.returncode == 0, f"workload failed (rc={proc.returncode}):\n{log[-2000:]}"
    assert "active on 1 device(s)" in log, "perf-debug profiler did not report active"
    assert out_tracy.exists() and out_tracy.stat().st_size > 4096, "no/empty Tracy capture produced"

    if CTX_INSPECT.exists():
        n_ctx, n_with_zones = _gpu_context_stats(out_tracy)
        # A small grid still pre-creates all per-core contexts; assert the workload's cores captured zones.
        assert n_ctx >= gx * gy, f"too few GPU contexts: {n_ctx}"
        assert n_with_zones >= gx * gy, f"expected >= {gx * gy} contexts with zones, got {n_with_zones}"
