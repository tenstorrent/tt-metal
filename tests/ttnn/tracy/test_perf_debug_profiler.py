# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
perf-debug profiler test (DRISC drain path).

Runs the ``test_perf_debug_zones`` workload -- which emits 10 differently-named DeviceZoneScopedN zones
(with increasing durations) on all 5 RISCs of a small core grid -- with the perf-debug profiler enabled
(``TT_METAL_STREAMING_PROFILER=1`` + ``TT_METAL_STREAMING_PROFILER_TRACY=1``) under a connected ``tracy-capture``. Verifies:

  * the module leaves DRISC drainers resident at MeshDevice bring-up ("DRISC FILLER/MOVER ... resident
    on logical (x,y)" log lines), and
  * the resulting Tracy capture holds device (GPU) zones across many per-core contexts.

Requires ``tools/drisc_drain/tracy_ctx_inspect`` to be built -- the second check is the substantive
one and the test fails rather than skipping it if the tool is missing.

Hardware-gated: needs a Blackhole box with DRAM programmable cores enabled. Skips cleanly when the
workload reports it is not on Blackhole / no DRISC was available. Device work runs in a
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
CTX_INSPECT = Path(TT_METAL_HOME) / "tools" / "drisc_drain" / "tracy_ctx_inspect" / "tracy_ctx_inspect"
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
    env["TT_METAL_STREAMING_PROFILER"] = "1"  # boot the module at bring-up; implies TT_METAL_DEVICE_PROFILER
    env["TT_METAL_STREAMING_PROFILER_TRACY"] = "1"  # the Tracy sink is opt-in; this test verifies via tracy-capture
    try:
        proc = subprocess.run(
            # --markers 1: the point-marker trio is opt-in (off by default so knee sweeps run a pure
            # zone stream), and PP_EVENT / Data payload records have no other emitter in the tree, so
            # the markers must be on here or that layout goes untested.
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
    # Match the ROLE-SPLIT log lines the profiler actually emits ("DRISC FILLER ... resident on
    # logical (x,y)" / "DRISC MOVER ... resident on logical (x,y)"), not the pre-role-split phrase
    # "DRISC drainer resident". That phrase has not existed in the source since the role split became
    # the default, so this guard skipped unconditionally on every box -- the test was permanently green
    # while asserting nothing. Gate on the role-agnostic substring so both roles satisfy it.
    if "not Blackhole" in log or "resident on logical" not in log:
        pytest.skip("perf-debug profiler did not start the DRISC drainer (not Blackhole / no DRAM programmable cores)")

    assert proc.returncode == 0, f"workload failed (rc={proc.returncode}):\n{log[-2000:]}"
    assert "active on 1 device(s)" in log, "perf-debug profiler did not report active"
    assert out_tracy.exists() and out_tracy.stat().st_size > 4096, "no/empty Tracy capture produced"

    # tracy_ctx_inspect is REQUIRED, not optional. The device-zone assertions below are the only thing
    # this test verifies that a broken drain path could not also satisfy -- without them the test
    # degrades to "the capture is larger than 4096 bytes", which a capture containing zero device zones
    # passes comfortably. Guarding them behind `if CTX_INSPECT.exists()` meant a missing dev tool
    # silently narrowed what the test checked, which is the same failure class as the skip-guard bug
    # this test just exhibited. Fail loudly with the build command instead of quietly checking less.
    if not CTX_INSPECT.exists():
        pytest.fail(
            f"tracy_ctx_inspect not built at {CTX_INSPECT} -- the device-zone assertions cannot run and "
            f"this test would otherwise verify only that the capture exceeds 4096 bytes. Build it with:\n"
            f"  bash tools/drisc_drain/tracy_ctx_inspect/build.sh "
            f"$TT_METAL_HOME/tools/drisc_drain/tracy_ctx_inspect/tracy_ctx_inspect\n"
            f"(the output path must be ABSOLUTE -- build.sh cds into third_party/tracy first)"
        )

    n_ctx, n_with_zones = _gpu_context_stats(out_tracy)
    # A small grid still pre-creates all per-core contexts; assert the workload's cores captured zones.
    assert n_ctx >= gx * gy, f"too few GPU contexts: {n_ctx}"
    assert n_with_zones >= gx * gy, f"expected >= {gx * gy} contexts with zones, got {n_with_zones}"
