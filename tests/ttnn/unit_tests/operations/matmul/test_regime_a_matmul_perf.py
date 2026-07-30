# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Golden PERFORMANCE regression suite for ttnn.experimental.regime_a_matmul (Blackhole).

Ten shapes chosen so every production code path is covered, and a regression in any of them trips a threshold:

  Mt          1, 2, 4, 8, 16
  reduction   linear chain AND ring reduce-scatter
  placement   bank-local AND in1-near (M-split) AND 2D mesh
  scale       7 us .. 228 us of device time; 24 .. 96 cores

Everything runs at DEFAULTS (config=None, no env override), so this measures what actually ships.

MEASUREMENT. Device time comes from the device-profiler CSV demuxed by run-host-id -- the method every number in
this op's optimisation campaign used -- not host wall, which would fold dispatch overhead into a 7 us shape. The
profiler CSV is only written when the device CLOSES (verified: absent after ttnn.synchronize_device and after
ttnn.ReadDeviceProfiler, present only after close_device), so each shape is measured in its own SUBPROCESS that
opens a device, runs the op and closes it. That also isolates shapes from one another. The measurement worker is
tools/mm_sweep/picker_gen/prod_sweep_worker.py, reused rather than duplicated so the demux logic has one source
of truth.

Because these tests spawn their own device they must NOT take the `device` fixture -- holding the device here
would stop the subprocess from opening it.

THRESHOLDS are the measured median device times on this board plus a margin sized to the measured noise floor:
8% for shapes under 30 us (iteration spread there reaches 12%) and 5% for the rest (spread <= 4%). Every
regression this campaign actually caught was 9-33%, so the margins are wide enough not to flake and tight enough
to catch anything real. Update a golden number only alongside a deliberate, measured perf change.

Run with:
    pytest tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul_perf.py
"""
import json
import os
import subprocess
import sys

import pytest

from models.common.utility_functions import is_blackhole

WORKER = "tools/mm_sweep/picker_gen/prod_sweep_worker.py"
NBLOCKS = 2  # worker runs 2 warmup + 12 timed per block; median over all 24 timed iterations
PCC_MIN = 0.999
TIMEOUT_S = 900

# shape -> (golden median device us on this board, margin fraction, what this case covers)
GOLDEN = {
    (32, 2048, 512): (7.32, 0.08, "Mt1 chain bank-local 64c; smallest, overhead-dominated"),
    (32, 6144, 9216): (227.89, 0.05, "Mt1 chain bank-local 24c; highest DRAM efficiency, 98% of peak"),
    (64, 2048, 1024): (12.16, 0.08, "Mt2 reduce-scatter bank-local 64c"),
    (128, 6144, 4608): (125.27, 0.05, "Mt4 chain bank-local 96c"),
    (256, 2048, 1024): (19.54, 0.08, "Mt8 reduce-scatter in1-near 64c; noisiest shape in the corpus"),
    (256, 6144, 768): (35.72, 0.05, "Mt8 chain mesh 96c"),
    (256, 15360, 768): (86.86, 0.05, "Mt8 reduce-scatter mesh 96c; deep K"),
    (256, 6144, 6144): (187.48, 0.05, "Mt8 reduce-scatter in1-near 96c; largest reduce-scatter shape"),
    (512, 6144, 2304): (109.95, 0.05, "Mt16 chain mesh 96c"),
    (512, 6144, 4608): (180.16, 0.05, "Mt16 chain mesh 96c; the one compute-floor-bound shape"),
}


def _measure(M, K, N):
    """Measure one shape in a fresh process; returns the worker's result dict."""
    root = os.environ.get("TT_METAL_HOME") or os.getcwd()
    worker = os.path.join(root, WORKER)
    assert os.path.exists(worker), f"measurement worker not found at {worker}"
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"  # required for the device-time CSV
    env.pop("TT_REGIME_A_DIAG_MASK", None)  # DEFAULTS ONLY
    proc = subprocess.run(
        [sys.executable, worker, str(M), str(K), str(N), str(NBLOCKS)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("SWEEP_JSON ")), None)
    assert line, (
        f"measurement worker produced no result for {M}x{K}x{N} (rc={proc.returncode}).\n"
        f"stdout tail:\n{proc.stdout[-1500:]}\nstderr tail:\n{proc.stderr[-1500:]}"
    )
    return json.loads(line[len("SWEEP_JSON ") :])


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("shape", list(GOLDEN.keys()), ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}")
def test_regime_a_matmul_golden_perf(shape):
    M, K, N = shape
    golden_us, margin, covers = GOLDEN[shape]
    limit_us = golden_us * (1.0 + margin)

    res = _measure(M, K, N)
    assert res.get("outcome") == "ok", f"{M}x{K}x{N} failed to run: {res.get('err')}"

    # correctness is checked on the same program the timing uses, so a perf pass cannot mask a numerical break
    pcc = res.get("pcc")
    assert pcc is not None and pcc >= PCC_MIN, f"{M}x{K}x{N} PCC {pcc} < {PCC_MIN} ({covers})"

    median = res["median_us"]
    assert median <= limit_us, (
        f"{M}x{K}x{N} PERF REGRESSION: {median:.2f} us > {limit_us:.2f} us "
        f"(golden {golden_us:.2f} us +{margin * 100:.0f}%). Covers: {covers}. "
        f"block medians={res.get('block_medians')} over {res.get('n_iters')} iterations"
    )
