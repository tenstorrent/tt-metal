# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Golden PERFORMANCE regression suite for ttnn.experimental.small_m_matmul (Blackhole).

Ten shapes chosen so every production code path is covered, and a regression in any of them trips a threshold:

  Mt          1, 2, 4, 8, 16
  reduction   linear chain AND ring reduce-scatter
  placement   bank-local AND in1-near (M-split) AND 2D mesh
  scale       7 us .. 228 us of device time; 24 .. 96 cores

Everything runs at DEFAULTS (config=None, no env override), so this measures what actually ships.

Correctness is asserted alongside every timing -- PCC AND a zero-non-finite count -- so a perf pass can never
mask a numerical break.

MEASUREMENT. Device time comes from the device-profiler CSV demuxed by run-host-id -- the method every number in
this op's optimisation campaign used -- not host wall, which would fold dispatch overhead into a 7 us shape. The
profiler CSV is only written when the device CLOSES (verified: absent after ttnn.synchronize_device and after
ttnn.ReadDeviceProfiler, present only after close_device), so each shape is measured in its own SUBPROCESS that
opens a device, runs the op and closes it. That also isolates shapes from one another. The measurement worker is
tools/mm_sweep/picker_gen/prod_sweep_worker.py, reused rather than duplicated so the demux logic has one source
of truth.

Because these tests spawn their own device they must NOT take the `device` fixture -- holding the device here
would stop the subprocess from opening it. For the same reason this module must be run in its OWN pytest
invocation: any module before it in the same session still holds a device, and every measurement would then
block in UMD cluster init until it timed out. A probe up front detects that and skips with that reason.

THRESHOLDS are measured median device times PLUS a margin sized to the measured noise floor: 8% for shapes
under 30 us (iteration spread there reaches 12%) and 5% for the rest (spread <= 4%). Every regression this
campaign actually caught was 9-33%, so the margins are wide enough not to flake and tight enough to catch
anything real. Update a golden number only alongside a deliberate, measured perf change.

GOLDENS ARE PER BOARD, keyed by the compute grid (`compute_with_storage_grid_size`) that the measurement
worker reports for the device it ran on. A golden
measured on one harvest configuration is NOT a threshold on another: the core budget feeds the picker's own
feasibility rule (8*Pk*Ns*Sm <= available cores), and DRAM/clock behaviour differs between a single-card dev
part and a Galaxy tray chip. This was not always keyed -- the original set was written as "on this board"
without naming it, and running it on a 12x10 Galaxy chip reported six false regressions of +5..+14% against
11x10 numbers. An unknown board SKIPS with instructions rather than inventing a verdict.

    11x10 (110 cores)  single-card dev part, where the optimisation campaign ran
    12x10 (120 cores)  bh-glx-120-c02u02 Galaxy tray chip

Run with:
    pytest tests/ttnn/unit_tests/operations/matmul/test_small_m_matmul_perf.py
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
# 240s per shape. The measured work is milliseconds (24 iterations of a 7-260us op); the rest is one device
# open, a JIT compile and one close -- ~60-90s for the largest shape here. 900s was ten minutes of nothing
# to say, and when the device was unavailable it burned the full budget PER SHAPE before failing.
TIMEOUT_S = 240

# What each shape exercises -- board-independent, so it lives outside the per-board tables.
COVERS = {
    (32, 2048, 512): "Mt1 chain bank-local 64c; smallest, overhead-dominated",
    (32, 6144, 9216): "Mt1 chain bank-local 24c; highest DRAM efficiency, 98% of peak",
    (64, 2048, 1024): "Mt2 reduce-scatter bank-local 64c",
    (128, 6144, 4608): "Mt4 chain bank-local 96c",
    (256, 2048, 1024): "Mt8 reduce-scatter in1-near 64c; noisiest shape in the corpus",
    (256, 6144, 768): "Mt8 chain mesh 96c",
    (256, 15360, 768): "Mt8 reduce-scatter mesh 80c; deep K",
    (256, 6144, 6144): "Mt8 reduce-scatter 96c; large square",
    (512, 6144, 2304): "Mt16 chain mesh 96c",
    (512, 6144, 4608): "Mt16 chain mesh 96c; the one compute-floor-bound shape",
}

# board -> {shape: (golden median device us, margin fraction)}. See GOLDENS ARE PER BOARD above.
GOLDEN = {
    # 11x10 / 110 cores -- the single-card dev part the optimisation campaign ran on.
    "11x10": {
        (32, 2048, 512): (7.32, 0.08),
        (32, 6144, 9216): (227.89, 0.05),
        (64, 2048, 1024): (12.16, 0.08),
        (128, 6144, 4608): (125.27, 0.05),
        (256, 2048, 1024): (19.54, 0.08),
        (256, 6144, 768): (35.72, 0.05),
        # Re-baselined 2026-07-31: the exhaustive resweep moved this pick from (6,1,2,2,3)/96c to
        # (5,1,2,4,3)/80c, a measured -5.0%. The old 86.86 threshold would still pass but no longer bounds
        # the current implementation tightly enough to catch a regression.
        (256, 15360, 768): (81.30, 0.05),
        # Re-baselined 2026-08-03: Tier 1 resweep moved this pick from (6,1,2,4,2) to (6,1,2,2,6),
        # a confirmed -5.0% (two relaunches). Old threshold 187.48 would no longer bound the impl.
        (256, 6144, 6144): (177.57, 0.05),
        (512, 6144, 2304): (109.95, 0.05),
        (512, 6144, 4608): (180.16, 0.05),
    },
    # 12x10 / 120 cores -- bh-glx-120-c02u02 Galaxy tray chip. Measured 2026-08-11 at defaults on
    # f9daaf52fcf (rebased onto main 5a934d9f884), block-median spread 0.3-1.8%, PCC >= 0.999986, zero
    # non-finite. The picked configs match the 11x10 board on every shape sampled, so the differences here are
    # board behaviour rather than the picker reacting to the wider grid.
    "12x10": {
        (32, 2048, 512): (7.76, 0.08),
        (32, 6144, 9216): (258.90, 0.05),
        (64, 2048, 1024): (13.53, 0.08),
        (128, 6144, 4608): (140.24, 0.05),
        (256, 2048, 1024): (19.48, 0.08),
        (256, 6144, 768): (37.24, 0.05),
        (256, 15360, 768): (85.81, 0.05),
        (256, 6144, 6144): (194.70, 0.05),
        (512, 6144, 2304): (112.63, 0.05),
        (512, 6144, 4608): (188.45, 0.05),
    },
}

SHAPES = list(COVERS.keys())


_DEVICE_FREE = []


def _device_available():
    """Can a SUBPROCESS open the device? Cached; probed once per module.

    Every measurement runs in its own process (the profiler CSV is only written on device close), so this
    module only works when nothing else holds the device. In a combined pytest session the modules that ran
    before it hold one via the `device` fixture, and each measurement then blocks in UMD cluster init until it
    hits TIMEOUT_S -- 900s per shape, ten shapes, all failing. A short probe turns that into one fast, honest
    skip that names the cause.
    """
    if not _DEVICE_FREE:
        root = os.environ.get("TT_METAL_HOME") or os.getcwd()
        code = "import ttnn;d=ttnn.open_device(device_id=0);ttnn.close_device(d);print('DEVICE_FREE')"
        try:
            p = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120)
            _DEVICE_FREE.append("DEVICE_FREE" in p.stdout)
        except subprocess.TimeoutExpired:
            _DEVICE_FREE.append(False)
    return _DEVICE_FREE[0]


def _measure(M, K, N):
    """Measure one shape in a fresh process; returns the worker's result dict."""
    root = os.environ.get("TT_METAL_HOME") or os.getcwd()
    worker = os.path.join(root, WORKER)
    assert os.path.exists(worker), f"measurement worker not found at {worker}"
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"  # required for the device-time CSV
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


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}")
def test_small_m_matmul_golden_perf(shape):
    M, K, N = shape
    covers = COVERS[shape]
    if not _device_available():
        pytest.skip(
            "the device is not available to a subprocess -- another test in this pytest session is holding it. "
            "Every measurement here runs in its own process, so this module must be run in its OWN pytest "
            "invocation: pytest tests/ttnn/unit_tests/operations/matmul/test_small_m_matmul_perf.py"
        )

    # Measure FIRST, then select the golden by the board the measurement actually ran on. An earlier version
    # queried the board up front by opening a device here; that fails whenever this module shares a pytest
    # session with tests using the session-scoped `device` fixture (the device is still held), and it turned
    # that failure into a SKIP -- silently disabling the whole perf gate in exactly the full-suite run CI does.
    # Taking the board from the worker's own result cannot mask a failure and needs no second device open.
    res = _measure(M, K, N)
    assert res.get("outcome") == "ok", (
        f"{M}x{K}x{N} failed to run: {res.get('err')}. If this says the device could not be opened, another "
        f"test in this pytest session is holding it -- run this module in its own invocation."
    )
    board = res.get("board")
    assert board, f"{M}x{K}x{N}: worker reported no board key (stale worker?)"
    if board not in GOLDEN or shape not in GOLDEN[board]:
        pytest.skip(
            f"no perf golden for board {board} (shape {M}x{K}x{N}). Perf goldens are board-specific; add a "
            f"'{board}' entry to GOLDEN by measuring this shape at defaults with "
            f"tools/mm_sweep/picker_gen/prod_sweep_worker.py, rather than comparing against another board."
        )
    golden_us, margin = GOLDEN[board][shape]
    limit_us = golden_us * (1.0 + margin)

    # correctness is checked on the same program the timing uses, so a perf pass cannot mask a numerical break
    pcc = res.get("pcc")
    assert pcc is not None and pcc >= PCC_MIN, f"{M}x{K}x{N} PCC {pcc} < {PCC_MIN} ({covers})"

    # FINITENESS, asserted separately from PCC. The reduce-scatter CB wrap bug corrupted ~0.015% of elements,
    # which as finite garbage would leave PCC at ~0.9999 and pass the threshold above; NaN only tripped PCC
    # because it poisons the whole correlation. Half these shapes exercise reduce-scatter, so the property has
    # to be checked here and not inferred from PCC.
    n_nonfinite = res.get("n_nonfinite")
    assert n_nonfinite is not None, f"{M}x{K}x{N} worker reported no non-finite count (stale worker?)"
    assert n_nonfinite == 0, f"{M}x{K}x{N} produced {n_nonfinite} non-finite output elements ({covers})"

    median = res["median_us"]
    assert median <= limit_us, (
        f"{M}x{K}x{N} PERF REGRESSION: {median:.2f} us > {limit_us:.2f} us "
        f"(golden {golden_us:.2f} us +{margin * 100:.0f}% on board {board}). Covers: {covers}. "
        f"block medians={res.get('block_medians')} over {res.get('n_iters')} iterations"
    )
