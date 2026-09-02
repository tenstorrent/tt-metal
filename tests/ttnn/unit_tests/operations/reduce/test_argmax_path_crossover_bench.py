# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Trace-replay measurement of the Blackhole accelerated argmax paths (RVV, SFPU).

WHAT THIS MEASURES
------------------
Per-op **device** time of ``ttnn.argmax``'s accelerated paths as a function of
the core count they are given, so two questions can be answered from data:

1. **How few cores does each path need?** A fused argmax epilogue would share
   the compute grid with the LM-head matmul it follows, so it can only ever be
   handed a slice of that grid rather than all of it. What decides whether such
   a fusion is possible is therefore how few cores a path needs to reach a
   given latency, not the best time it can post with the whole grid to itself.
   So the report leads with **cores-to-match** (the fewest cores at which one
   path reaches the other's best time, and the fewest at which either reaches
   a fixed latency target), carries the RVV-vs-SFPU ratio **at every core
   count** rather than only at the defaults, and prints a `cores x time` column
   as a proxy for how much of the machine a result costs.
2. How far does each path actually scale, and where does it saturate? The
   factories ship DIFFERENT defaults, both capped by the grid and by
   ``w_tiles``: ``ceil(sqrt(1.5 * w_tiles))`` for the SFPU path (flat in H)
   and ``ceil(sqrt(w_tiles * (H + 2)) / 3)`` for the RVV path (per-row), see
   argmax_{sfpu,rvv}_tile_program_factory.cpp. Those are models fitted to
   measurements, so they are worth re-checking. This sweep prints the true optimum, the saturation
   knee, and any core count where adding cores made things *slower*, next to
   what the heuristic picks.
3. How much of the wall-clock cost an eager caller sees is host dispatch rather
   than device work. Every point is measured BOTH ways -- eager and traced --
   and every derived headline (cores-to-match included) is reported in both
   flavours, because an eager-only conclusion and a trace-only conclusion can
   disagree.

HOW TO RUN IT
-------------
Manual-only: this directory is swept wholesale by CI, so every test here is
gated behind ``TTNN_RUN_ARGMAX_CROSSOVER_BENCH=1`` and skips without it. On a
Blackhole host::

    export TTNN_RUN_ARGMAX_CROSSOVER_BENCH=1
    pytest tests/ttnn/unit_tests/operations/reduce/test_argmax_path_crossover_bench.py -s --timeout=7200

Add ``-k core_scaling`` for the perf sweep alone, ``-k trace_safety`` for the
replay-stability check alone. Then paste the markdown block it prints into the PR.

METHODOLOGY
-----------
Timing is **trace capture + replay**, which is the only mechanism here that
reports device time without a profiler: the whole dispatch command stream is
recorded once and replayed from device DRAM, so the per-op host cost that
dominates eager dispatch of a ~50 us kernel is not in the measured window.
Pattern copied from ``tests/ttnn/unit_tests/benchmarks/test_benchmark.py:200``
(run_matmul_measurement: warm up, ``begin_trace_capture``, N ops, ``end_trace_capture``,
time one ``execute_trace`` + ``synchronize_device``, divide by N) and
``tests/ttnn/unit_tests/base_functionality/test_single_device_trace.py:21``
(pre-allocate the activation with ``allocate_tensor_on_device``, push new data
in with ``copy_host_to_device_tensor``, replay, read the output back).

- **N ops per trace.** A trace holds N back-to-back copies of the SAME argmax
  call, and per-op time is (one replay) / N. N is chosen per point to keep a
  replay near ``TARGET_REPLAY_US`` (bounded by ``N_OPS_MIN`` / ``N_OPS_MAX``) and
  is printed in the table, so nothing about the divisor is implicit.
- **Replay launch overhead is measured, not assumed.** A 10-op vs 100-op
  two-point check on this box put the fixed per-replay cost at ~1-4 us, i.e.
  under 1% of an N=100 replay. It is left in the number rather than subtracted;
  ``test_argmax_trace_launch_overhead`` re-measures it and prints it.
- **A vacuous trace fails loudly.** After capture, a *different* input is copied
  into the pre-allocated device tensor and the trace is replayed; the output must
  equal the golden for that NEW input. A trace that captured nothing, or that
  replayed stale results, cannot pass -- it would still be holding the answer for
  the input that was resident at capture time.
- Warm-up (JIT compile + program cache fill) always happens BEFORE capture, so
  capture records the steady-state program.

CAVEATS
-------
- Trace replay is a *lower* bound on what a real caller sees; a model that
  dispatches argmax eagerly pays the host cost too. Both numbers are printed
  precisely so the choice of bound is the reader's.
- Per-op time inside a trace includes the on-device gap between back-to-back
  program dispatches. That gap is real steady-state cost, but it means a single
  isolated argmax is not necessarily this fast.
- Only ``min`` over ``REPLAYS`` is reported. The floor is the stable statistic
  here; the mean drags in unrelated host/DRAM noise.
- ``torch.randn`` emits no NaN, denormal or signed zero, so the SFPU path's
  documented special-value divergence is out of scope and the two paths must
  agree exactly, ties included. That is asserted at every point.

NOT A REGRESSION GATE. Nothing about timing is asserted; timings are the output.
Correctness *is* asserted, at every point.
"""

import math
import os
import statistics
import time

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher

# Manual-only benchmark. tests/pipeline_reorg/ttnn_sanity_tests.yaml sweeps this whole directory with
# `pytest --timeout 300 ... tests/ttnn/unit_tests/benchmarks ... -xv`, and a full run of this file takes
# hours, so every test in it is gated behind an opt-in env var. Same arrangement as
# GEMM_FLOPS_BENCHMARK_ENV in test_benchmark.py alongside it.
ARGMAX_CROSSOVER_BENCH_ENV = "TTNN_RUN_ARGMAX_CROSSOVER_BENCH"

pytestmark = [
    pytest.mark.skipif(
        os.getenv(ARGMAX_CROSSOVER_BENCH_ENV) != "1",
        reason=f"Benchmark is manual-only; set {ARGMAX_CROSSOVER_BENCH_ENV}=1 to run",
    ),
    run_for_blackhole("the accelerated argmax paths (RVV, SFPU) are Blackhole-only"),
    skip_with_watcher("Watcher perturbs kernel timing; a scaling curve measured under it is not the real one."),
    skip_with_llk_assert("LLK asserts perturb kernel timing."),
]

# Verification-only forced entries: ttnn.argmax picks a path on its own and takes no argument that
# names one, so a per-path measurement has to pin the path here. These are bound only under the
# private module and never fall back -- a path that cannot serve a case raises. See
# ttnn/cpp/ttnn/operations/reduction/argmax/argmax_force.hpp.
_FORCE = {
    "RVV": ttnn._ttnn.operations.reduction.argmax_force_rvv,
    "SFPU": ttnn._ttnn.operations.reduction.argmax_force_sfpu,
}
_PATHS = ("RVV", "SFPU")

# Reduction widths and rows-per-tile-row. The two V values are the ones tabulated in argmax.cpp;
# H spans the kSfpuMinRows boundary (shipped at 32): 1 and 8 are below it, 32 is at it.
V_SWEEP = (32768, 262144)
H_SWEEP = (1, 8, 32)

# Explicit core counts to pin with sub_core_grids. `None` means "no sub_core_grids", i.e. whatever the
# shipped heuristic in that path's program factory picks -- which is NOT the same formula for the two
# paths; see _default_num_cores.
CORE_SWEEP = (1, 2, 4, 8, 16, 32, 64, 111, None)

# Ops recorded into one trace. Chosen per point so a single replay lands near TARGET_REPLAY_US: large
# N amortizes the fixed replay-launch cost for cheap configurations, small N keeps a 40 ms/op
# configuration from turning one replay into four seconds.
TARGET_REPLAY_US = 10_000.0
N_OPS_MIN = 10
N_OPS_MAX = 100

# Replays of a captured trace; the reported figure is the minimum. One extra replay runs first and is
# discarded (it also serves as the vacuous-trace check).
REPLAYS = 3

# Eager dispatches per timed batch, and batches; same min-of-batches rule. Adaptive like N_OPS.
EAGER_BATCHES = 3
# Samples for the latency-mode eager number (one dispatch, one synchronize, repeat). Min is reported.
EAGER_ISOLATED_SAMPLES = 20
# Dispatches before any measurement, eager or captured. The first dispatch of a fresh program pays
# binary transfer to the cores on top of the steady-state cost. Precedent: WARMUP/ITERS in
# _topk_route_cells_bench.py and warmup_iters in models/perf/device_perf_utils.py.
WARMUP = 3

# The trace region is a DRAM carve-out sized for the recorded command streams. 100 dispatches of a
# 111-core program is the worst case here; 512 MB out of the p150's 32 GB clears it with room over.
TRACE_REGION_SIZE = 512 * 1024 * 1024

# The constant this file's scaling numbers feed into, mirrored from argmax.cpp for the report only.
# Nothing here asserts against it -- the threshold decision is not this file's to make.
SHIPPED_SFPU_MIN_ROWS = 32

# Fixed latency budgets the cores-to-reach table answers against. Spread so that the cheap V and the
# expensive V both land somewhere interesting; unreachable budgets print as "-".
LATENCY_TARGETS_US = (25.0, 50.0, 100.0, 200.0)

# A point counts as saturated once it is within this fraction of the path's best over the sweep --
# the knee, not the minimum, since the minimum is often one noisy point past the knee.
SATURATION_TOL = 0.05

# Adding cores must not make things slower. Anything worse than this fraction relative to the next
# smaller core count in the sweep is flagged as a negative-scaling point rather than averaged away.
REGRESSION_TOL = 0.05

_TRACE_DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True
)


# ---------------------------------------------------------------------------
# Fixtures for input staging
# ---------------------------------------------------------------------------


def _core_grid(n: int, grid_x: int) -> ttnn.CoreRangeSet:
    """``n`` cores as row-major single-core ranges over a ``grid_x``-wide compute grid."""
    ranges = []
    for i in range(n):
        c = ttnn.CoreCoord(i % grid_x, i // grid_x)
        ranges.append(ttnn.CoreRange(c, c))
    return ttnn.CoreRangeSet(ranges)


def _default_num_cores(path: str, v: int, h: int, grid_cores: int) -> int:
    """The shipped heuristic, mirrored from argmax_{rvv,sfpu}_tile_program_factory.cpp so the report can
    name the core count a call with no ``sub_core_grids`` actually gets.

    The two factories do NOT use the same formula, because their cost models differ: the SFPU pass is
    flat in H, so its optimum only tracks the reduction width, while the RVV scan is per row and its
    optimum grows with H as well. Mirroring one formula for both would mislabel every RVV `default` row.
    """
    w_tiles = math.ceil(v / 32)
    if path == "RVV":
        want = math.ceil(math.sqrt(w_tiles * (h + 2)) / 3)
    else:
        want = math.ceil(math.sqrt(1.5 * w_tiles))
    return min(want, grid_cores, w_tiles)


def _golden_indices(arr: torch.Tensor) -> torch.Tensor:
    """First-occurrence argmax over the last dim, flattened. The paths resolve ties to the smallest
    index (see _ref_argmax_row in test_argmax_rvv.py), and numpy's argmax on a boolean array is
    documented to return the first True -- torch's tie behaviour is not documented, so numpy it is."""
    a = arr.to(torch.float32).numpy().reshape(-1, arr.shape[-1])
    return torch.from_numpy((a == a.max(axis=-1, keepdims=True)).argmax(axis=-1)).to(torch.int64)


def _indices(tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor).flatten().to(torch.int64)


class _Staging:
    """A device-resident input slot plus two distinct host payloads and their goldens.

    The slot is allocated once, outside any trace, so a captured trace binds a stable address and new
    data can be pushed into it between replays -- the arrangement in
    tests/ttnn/unit_tests/base_functionality/test_single_device_trace.py:23.
    """

    def __init__(self, device, h: int, v: int, seed: int = 0):
        self.h, self.v = h, v
        shape = (1, 1, h, v)
        self.device_input = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        self.host = []
        self.golden = []
        for k in (0, 1):
            torch.manual_seed(2026 + seed + h * 131 + v + 7919 * k)
            t = torch.randn(shape, dtype=torch.bfloat16)
            self.host.append(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
            self.golden.append(_golden_indices(t))
        # Distinct payloads are what makes the vacuous-trace check bite: if the two goldens were
        # equal, replaying a stale result would pass.
        assert not torch.equal(self.golden[0], self.golden[1]), (
            f"the two staged inputs at H={h} V={v} happen to have identical argmax indices, so a "
            "trace that replayed a stale result would not be caught; change the seed"
        )
        self.load(0)

    def load(self, k: int) -> None:
        ttnn.copy_host_to_device_tensor(self.host[k], self.device_input)

    def close(self) -> None:
        ttnn.deallocate(self.device_input)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _pick_n(est_us: float, lo: int, hi: int) -> int:
    if est_us <= 0:
        return hi
    return int(max(lo, min(hi, round(TARGET_REPLAY_US / est_us))))


def _estimate_us(device, run_fn) -> float:
    """One rough eager wall-clock timing, used only to size N. Over-estimates device time (it carries
    host dispatch), which errs toward smaller N -- the safe direction for run time."""
    for _ in range(WARMUP):
        ttnn.deallocate(run_fn())
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(3):
        ttnn.deallocate(run_fn())
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / 3


def _measure_eager_us(device, run_fn, n: int) -> float:
    """Min over EAGER_BATCHES of (wall time of ``n`` dispatches + one synchronize) / n. Includes per-op
    host dispatch by construction; that inclusion is the point of the eager-vs-trace column."""
    best = None
    for _ in range(EAGER_BATCHES):
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(run_fn())
        ttnn.synchronize_device(device)
        us = (time.perf_counter() - t0) * 1e6 / n
        best = us if best is None else min(best, us)
    return best


def _measure_eager_isolated_us(device, run_fn, n: int) -> float:
    """Min over ``n`` samples of ONE dispatch followed immediately by ``synchronize_device``.

    This is latency-mode eager: nothing is in flight to hide the host behind, so the full per-op host
    dispatch cost is exposed. `_measure_eager_us` is throughput-mode -- it queues n dispatches before
    synchronizing, so host work overlaps device work and the host cost mostly disappears. The gap
    between the two is what an isolated caller pays and a pipelined one does not.
    """
    best = None
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.deallocate(run_fn())
        ttnn.synchronize_device(device)
        us = (time.perf_counter() - t0) * 1e6
        best = us if best is None else min(best, us)
    return best


def _measure_trace_us(device, run_fn, staging: _Staging, n_ops: int, label: str) -> float:
    """Min over REPLAYS of (one ``execute_trace`` + ``synchronize_device``) / ``n_ops``.

    Asserts, before timing anything, that a replay against an input the trace has never seen produces
    the golden for that input. That is what separates a real measurement from a trace that captured
    nothing and replays in microseconds.
    """
    # Warm up on the loaded payload so capture records a cached, already-compiled program.
    for _ in range(WARMUP):
        ttnn.deallocate(run_fn())
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = [run_fn() for _ in range(n_ops)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        # Swap in the payload the capture never ran against, then replay.
        staging.load(1)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        for position in (0, n_ops - 1):
            got = _indices(outputs[position])
            assert torch.equal(got, staging.golden[1]), (
                f"{label}: trace replay op[{position}] did not compute the golden for the input pushed "
                f"in after capture ({int((got != staging.golden[1]).sum())} of {got.numel()} rows wrong). "
                "Either the trace captured no work, or it replayed a stale result -- the timing below "
                "would be meaningless."
            )

        best = None
        for _ in range(REPLAYS):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            us = (time.perf_counter() - t0) * 1e6
            best = us if best is None else min(best, us)
    finally:
        ttnn.release_trace(device, tid)
        for out in outputs:
            ttnn.deallocate(out)
        staging.load(0)
    return best / n_ops


def _assert_program_cache_active(device) -> None:
    assert device.num_program_cache_entries() > 0, (
        "device program cache is empty after warmup dispatches; without it every dispatch is a cache "
        "miss and neither the eager nor the captured numbers are the steady state"
    )


# ---------------------------------------------------------------------------
# The core-count scaling sweep
# ---------------------------------------------------------------------------


@_TRACE_DEVICE_PARAMS
@pytest.mark.timeout(7200)
def test_argmax_path_core_scaling(device):
    """Sweep explicit core counts and each path's default for both paths at every (V, H), timed
    with trace replay and, for the same point, eagerly."""
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    logger.info(f"compute grid {grid.x}x{grid.y} = {grid_cores} cores")

    rows = []
    for v in V_SWEEP:
        for h in H_SWEEP:
            staging = _Staging(device, h, v)
            try:
                eager_indices = {}
                for path in _PATHS:
                    force = _FORCE[path]
                    eager_indices[path] = _indices(force(staging.device_input, dim=-1, keepdim=True))
                    assert torch.equal(eager_indices[path], staging.golden[0]), (
                        f"{path} at V={v} H={h} disagrees with the first-occurrence golden on "
                        f"{int((eager_indices[path] != staging.golden[0]).sum())} of "
                        f"{staging.golden[0].numel()} rows"
                    )
                # For randn data no special value is in play, so the two paths must agree exactly.
                assert torch.equal(eager_indices["RVV"], eager_indices["SFPU"]), (
                    f"RVV and SFPU disagree on indices at V={v} H={h} for ordinary random data "
                    "(no NaN/denormal/signed zero involved, so they must match exactly)"
                )
                _assert_program_cache_active(device)

                for n_cores in CORE_SWEEP:
                    if n_cores is not None and n_cores > grid_cores:
                        continue
                    sub_grid = None if n_cores is None else _core_grid(n_cores, grid.x)
                    for path in _PATHS:
                        force = _FORCE[path]

                        def run(force=force, sub_grid=sub_grid):
                            return force(staging.device_input, dim=-1, keepdim=True, sub_core_grids=sub_grid)

                        label = f"{path} V={v} H={h} cores={n_cores or 'default'}"
                        est = _estimate_us(device, run)
                        n_ops = _pick_n(est, N_OPS_MIN, N_OPS_MAX)
                        trace_us = _measure_trace_us(device, run, staging, n_ops, label)
                        n_eager = _pick_n(est, N_OPS_MIN, N_OPS_MAX)
                        eager_us = _measure_eager_us(device, run, n_eager)
                        eager_iso_us = _measure_eager_isolated_us(device, run, EAGER_ISOLATED_SAMPLES)
                        rows.append(
                            {
                                "v": v,
                                "h": h,
                                "path": path,
                                "cores": n_cores,
                                # The number of cores actually used: the factories clamp to w_tiles and
                                # to the grid, and the `default` row has no requested count at all.
                                "cores_used": n_cores
                                if n_cores is not None
                                else _default_num_cores(path, v, h, grid_cores),
                                "n_ops": n_ops,
                                "trace_us": trace_us,
                                "eager_us": eager_us,
                                "eager_iso_us": eager_iso_us,
                            }
                        )
                        logger.info(
                            f"{label}: trace {trace_us:.1f} us/op (N={n_ops}), eager-pipelined "
                            f"{eager_us:.1f} us/op, eager-isolated {eager_iso_us:.1f} us, "
                            f"delta(pipelined) {eager_us - trace_us:+.1f} us, "
                            f"delta(isolated) {eager_iso_us - trace_us:+.1f} us"
                        )
            finally:
                staging.close()

    _report_scaling(rows, grid_cores)


# ---------------------------------------------------------------------------
# Replay-launch overhead, measured rather than assumed
# ---------------------------------------------------------------------------


@_TRACE_DEVICE_PARAMS
@pytest.mark.timeout(1800)
def test_argmax_trace_launch_overhead(device):
    """Two-point check of the fixed per-replay cost that the scaling sweep leaves inside its per-op
    figure: time a 10-op trace and a 100-op trace of the same call, solve for the intercept."""
    grid = device.compute_with_storage_grid_size()
    lines = ["", "### Fixed cost of one `execute_trace` + `synchronize_device`", ""]
    lines.append("| V | H | path | cores | 10-op replay (us) | 100-op replay (us) | per-op (us) | launch (us) |")
    lines.append("|---:|---:|:---|---:|---:|---:|---:|---:|")
    for v, h, n_cores in ((262144, 1, 111), (32768, 8, 16)):
        staging = _Staging(device, h, v)
        try:
            sub_grid = _core_grid(n_cores, grid.x)
            for path in _PATHS:
                force = _FORCE[path]

                def run(force=force):
                    return force(staging.device_input, dim=-1, keepdim=True, sub_core_grids=sub_grid)

                label = f"{path} V={v} H={h} cores={n_cores}"
                t10 = _measure_trace_us(device, run, staging, 10, label) * 10
                t100 = _measure_trace_us(device, run, staging, 100, label) * 100
                per_op = (t100 - t10) / 90
                launch = t10 - 10 * per_op
                lines.append(
                    f"| {v} | {h} | {path} | {n_cores} | {t10:.1f} | {t100:.1f} | {per_op:.2f} | {launch:.1f} |"
                )
                logger.info(f"{label}: per-op {per_op:.2f} us, replay launch {launch:.1f} us")
        finally:
            staging.close()
    lines.append("")
    lines.append(
        "`launch` is the intercept the scaling table does not subtract. Divided over the N it uses "
        f"(up to {N_OPS_MAX}) it is the per-op error bar on every trace figure printed here."
    )
    lines.append("")
    block = "\n".join(lines)
    print(block)
    logger.info(block)


# ---------------------------------------------------------------------------
# Trace safety: does replay stay correct, many times over?
# ---------------------------------------------------------------------------

# Shapes that stress the multicore paths' cross-core handshake hardest under replay:
#  - ragged: w_tiles not divisible by the core count, so slices differ in width;
#  - multi-pass: H > 32, so the per-pass credit/semaphore flow runs more than once per dispatch;
#  - both at once, and one wide default-core-count case for good measure.
# (H, V, cores or None for default, note)
SAFETY_SHAPES = (
    (1, 1056, 7, "ragged: 33 w_tiles over 7 cores (4 or 5 each), single pass"),
    (80, 1056, 7, "ragged AND multi-pass: 33 w_tiles over 7 cores, 3 tile-row passes"),
    (80, 32768, None, "multi-pass at the default core count: 1024 w_tiles, 3 passes"),
    (32, 262144, 111, "widest single pass, 8192 w_tiles over 111 cores"),
)
SAFETY_REPLAYS = 60


@_TRACE_DEVICE_PARAMS
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("h, v, n_cores, note", SAFETY_SHAPES, ids=lambda x: str(x) if not isinstance(x, str) else "")
def test_argmax_trace_safety(device, path, h, v, n_cores, note):
    """Capture once, replay many times with the input alternating between two payloads, and require
    the golden for whichever payload is resident every single time.

    Alternating is what makes this a safety test rather than a smoke test: a replay that reused a
    stale result, or a path whose semaphores were left un-reset by the previous replay, produces
    the wrong payload's answer (or hangs) rather than a benign repeat.
    """
    grid = device.compute_with_storage_grid_size()
    if n_cores is not None and n_cores > grid.x * grid.y:
        pytest.skip(f"grid has {grid.x * grid.y} cores, need {n_cores}")
    sub_grid = None if n_cores is None else _core_grid(n_cores, grid.x)
    force = _FORCE[path]
    staging = _Staging(device, h, v, seed=11)
    try:

        def run():
            return force(staging.device_input, dim=-1, keepdim=True, sub_core_grids=sub_grid)

        for _ in range(WARMUP):
            ttnn.deallocate(run())
        ttnn.synchronize_device(device)

        tid = ttnn.begin_trace_capture(device, cq_id=0)
        output = run()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        try:
            for i in range(SAFETY_REPLAYS):
                k = i % 2
                staging.load(k)
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                got = _indices(output)
                assert torch.equal(got, staging.golden[k]), (
                    f"{path} {note}: replay {i} (payload {k}) returned the wrong indices on "
                    f"{int((got != staging.golden[k]).sum())} of {got.numel()} rows. "
                    f"Matches the OTHER payload's golden: {torch.equal(got, staging.golden[1 - k])}."
                )
        finally:
            ttnn.release_trace(device, tid)
            ttnn.deallocate(output)
        logger.info(f"{path} trace-safe over {SAFETY_REPLAYS} alternating replays -- {note}")
    finally:
        staging.close()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pick(rows, v, h, path, cores):
    for r in rows:
        if r["v"] == v and r["h"] == h and r["path"] == path and r["cores"] == cores:
            return r
    return None


def _series(rows, v, h, path):
    """This path's points at this shape, ordered by the core count actually used. The `default` row
    is dropped: it duplicates whichever explicit count the heuristic happens to land on, and keeping it
    would put two entries at the same x."""
    sub = [r for r in rows if r["v"] == v and r["h"] == h and r["path"] == path and r["cores"] is not None]
    return sorted(sub, key=lambda r: r["cores_used"])


def _cores_to_reach(rows, v, h, path, key, target_us):
    """Fewest cores at which this path gets at or under ``target_us``. None if it never does."""
    for r in _series(rows, v, h, path):
        if r[key] <= target_us:
            return r
    return None


def _saturation(rows, v, h, path, key):
    """(knee row, best row, [negative-scaling rows]).

    knee = the fewest cores within SATURATION_TOL of the best time over the sweep. Negative scaling =
    any point more than REGRESSION_TOL slower than the next smaller core count measured."""
    series = _series(rows, v, h, path)
    if not series:
        return None, None, []
    best = min(series, key=lambda r: r[key])
    knee = next(r for r in series if r[key] <= best[key] * (1.0 + SATURATION_TOL))
    regressions = [cur for prev, cur in zip(series, series[1:]) if cur[key] > prev[key] * (1.0 + REGRESSION_TOL)]
    return knee, best, regressions


def _shapes(rows):
    return [(v, h) for v in sorted({r["v"] for r in rows}) for h in sorted({r["h"] for r in rows if r["v"] == v})]


def _report_headline(lines, rows, key, flavour):
    """Cores-to-match: the fewest cores at which each path reaches the other's best. See this
    module's docstring for why that, and not best-case latency, is the number the fusion argument
    rests on."""
    lines.append(f"##### Cores-to-match ({flavour})")
    lines.append("")
    lines.append("| V | H | path | opponent's best | cores to match it | its time there | verdict |")
    lines.append("|---:|---:|:---|:---|---:|---:|:---|")
    for v, h in _shapes(rows):
        for path in _PATHS:
            other = "SFPU" if path == "RVV" else "RVV"
            mine, theirs = _series(rows, v, h, path), _series(rows, v, h, other)
            if not mine or not theirs:
                continue
            their_best = min(theirs, key=lambda r: r[key])
            hit = _cores_to_reach(rows, v, h, path, key, their_best[key])
            target = f"{other} {their_best[key]:.1f} us @ {their_best['cores_used']} cores"
            if hit is None:
                lines.append(
                    f"| {v} | {h} | {path} | {target} | never | {min(mine, key=lambda r: r[key])[key]:.1f} "
                    f"(best) | {path} never reaches it on this grid |"
                )
            else:
                factor = their_best["cores_used"] / hit["cores_used"]
                verdict = (
                    f"**{path} needs {hit['cores_used']} cores to beat {other}'s best "
                    f"({their_best['cores_used']} cores) -- {factor:.0f}x fewer**"
                    if factor > 1.0
                    else f"{path} needs {hit['cores_used']} cores, {other} needs {their_best['cores_used']}"
                )
                lines.append(
                    f"| {v} | {h} | {path} | {target} | **{hit['cores_used']}** | {hit[key]:.1f} | {verdict} |"
                )
    lines.append("")
    lines.append(f"##### Cores to reach a fixed latency budget ({flavour})")
    lines.append("")
    lines.append("| V | H | path | " + " | ".join(f"<= {t:.0f} us" for t in LATENCY_TARGETS_US) + " |")
    lines.append("|---:|---:|:---|" + "---:|" * len(LATENCY_TARGETS_US))
    for v, h in _shapes(rows):
        for path in _PATHS:
            cells = []
            for target in LATENCY_TARGETS_US:
                hit = _cores_to_reach(rows, v, h, path, key, target)
                cells.append("-" if hit is None else str(hit["cores_used"]))
            lines.append(f"| {v} | {h} | {path} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("`-` means no core count in the sweep gets the path under that budget.")
    lines.append("")


def _report_saturation(lines, rows, key, flavour):
    lines.append(f"##### Saturation and negative scaling ({flavour})")
    lines.append("")
    lines.append("| V | H | path | knee (cores) | knee time | best (cores) | best time | adding cores HURTS at |")
    lines.append("|---:|---:|:---|---:|---:|---:|---:|:---|")
    for v, h in _shapes(rows):
        for path in _PATHS:
            knee, best, regressions = _saturation(rows, v, h, path, key)
            if knee is None:
                continue
            if regressions:
                hurt = ", ".join(f"{r['cores_used']} ({r[key]:.1f} us)" for r in regressions)
                hurt = f"**{hurt}**"
            else:
                hurt = "-"
            lines.append(
                f"| {v} | {h} | {path} | {knee['cores_used']} | {knee[key]:.1f} | {best['cores_used']} | "
                f"{best[key]:.1f} | {hurt} |"
            )
    lines.append("")
    lines.append(
        f"knee = fewest cores within {SATURATION_TOL:.0%} of that path's best over the sweep. A bolded "
        f"`adding cores HURTS` cell is a core count more than {REGRESSION_TOL:.0%} slower than the next "
        "smaller one measured -- negative scaling, worth a look rather than an average."
    )
    lines.append("")


def _report_scaling(rows, grid_cores: int) -> None:
    """Print one markdown block, ready to paste into a PR comment."""
    lines = ["", "### ttnn.argmax path scaling vs core count (Blackhole, BFLOAT16, TILE, last-dim)", ""]
    lines.append(
        f"Input `[1, 1, H, V]`, `dim=-1`, `keepdim=True`. Paths pinned through the forced entries in "
        f"`argmax_force.hpp`; core counts pinned with `sub_core_grids`, `default` = no `sub_core_grids`, "
        f"i.e. that path's own shipped heuristic -- `ceil(sqrt(1.5 * w_tiles))` for SFPU and "
        f"`ceil(sqrt(w_tiles * (H + 2)) / 3)` for RVV, each capped by the grid and by `w_tiles`, so the two "
        f"`default` cells in a row are generally DIFFERENT core counts. Compute grid: {grid_cores} cores."
    )
    lines.append("")
    lines.append(
        f"**trace** = one `execute_trace` of a trace holding N back-to-back argmax ops, divided by N "
        f"(min of {REPLAYS} replays); N is per-point, sized to keep a replay near "
        f"{TARGET_REPLAY_US / 1000:.0f} ms, and printed. **eager** = wall clock over the same number of "
        f"ordinary dispatches plus one `synchronize_device`, divided by that number (min of "
        f"{EAGER_BATCHES} batches). Both after {WARMUP} warm-up dispatches. Every captured trace is "
        f"replayed once against an input it never saw at capture time and checked against that input's "
        f"golden before any timing is taken, so a trace that captured nothing fails instead of "
        f"reporting a fast lie."
    )
    lines.append("")

    lines.append("#### Headline: how few cores does each path need?")
    lines.append("")
    lines.append(
        "Cores-to-match, not latency at the top of the grid, is the headline; see the module "
        "docstring of `test_argmax_path_crossover_bench.py` for why a fused argmax epilogue cannot "
        "count on having the whole grid. Both flavours are printed because an eager-only answer and "
        "a trace-only answer can disagree."
    )
    lines.append("")
    _report_headline(lines, rows, "trace_us", "trace replay, device time")
    _report_headline(lines, rows, "eager_us", "eager dispatch, wall clock")

    lines.append("#### Per-core-count detail")
    lines.append("")
    lines.append(
        "`SFPU/RVV` > 1 means RVV is faster by that factor at the same core count. Watching it decay "
        "toward 1 as cores are added is the point: it says the two paths are converging on a shared "
        "floor, so a comparison taken only at the top of the grid measures the floor, not the paths. "
        "`cores x us` is core count times latency -- a proxy for how much of the machine the result "
        "costs, which is what a fused epilogue has to budget for."
    )
    lines.append("")
    for v, h in _shapes(rows):
        lines.append(f"##### V = {v}, H = {h}")
        lines.append("")
        lines.append(
            "| cores | N/trace (R/S) | RVV trace | SFPU trace | SFPU/RVV trace | RVV eager | SFPU eager | "
            "SFPU/RVV eager | RVV cores x us | SFPU cores x us |"
        )
        lines.append("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n_cores in CORE_SWEEP:
            r_rvv = _pick(rows, v, h, "RVV", n_cores)
            r_sfpu = _pick(rows, v, h, "SFPU", n_cores)
            if r_rvv is None or r_sfpu is None:
                continue
            name = (
                str(n_cores) if n_cores is not None else f"default (R {r_rvv['cores_used']} / S {r_sfpu['cores_used']})"
            )
            lines.append(
                f"| {name} | {r_rvv['n_ops']}/{r_sfpu['n_ops']} | {r_rvv['trace_us']:.1f} | "
                f"{r_sfpu['trace_us']:.1f} | {r_sfpu['trace_us'] / r_rvv['trace_us']:.2f}x | "
                f"{r_rvv['eager_us']:.1f} | {r_sfpu['eager_us']:.1f} | "
                f"{r_sfpu['eager_us'] / r_rvv['eager_us']:.2f}x | "
                f"{r_rvv['cores_used'] * r_rvv['trace_us']:.0f} | "
                f"{r_sfpu['cores_used'] * r_sfpu['trace_us']:.0f} |"
            )
        lines.append("")

    lines.append("#### Where each path stops benefiting from more cores")
    lines.append("")
    _report_saturation(lines, rows, "trace_us", "trace replay, device time")
    _report_saturation(lines, rows, "eager_us", "eager dispatch, wall clock")

    lines.append("#### Optimal core count vs the shipped heuristic (device time, trace replay)")
    lines.append("")
    lines.append(
        "| V | H | path | best cores | best (us) | default cores | default (us) | default / best | "
        "left on the table |"
    )
    lines.append("|---:|---:|:---|---:|---:|---:|---:|---:|---:|")
    for v, h in _shapes(rows):
        for path in _PATHS:
            series = _series(rows, v, h, path)
            default = _pick(rows, v, h, path, None)
            if not series or default is None:
                continue
            best = min(series, key=lambda r: r["trace_us"])
            ratio = default["trace_us"] / best["trace_us"] if best["trace_us"] > 0 else float("inf")
            lines.append(
                f"| {v} | {h} | {path} | {best['cores_used']} | {best['trace_us']:.1f} | "
                f"{default['cores_used']} | {default['trace_us']:.1f} | {ratio:.2f}x | "
                f"{default['trace_us'] - best['trace_us']:+.1f} us |"
            )
    lines.append("")

    lines.append("#### Host dispatch overhead: is the plateau host cost or device cost?")
    lines.append("")
    lines.append(
        "Three numbers per point. **trace** = trace replay, no per-op host dispatch at all. "
        "**eager pipelined** = N ordinary dispatches queued back-to-back with one `synchronize_device` "
        "at the end, so host work overlaps device work. **eager isolated** = one dispatch, one "
        "`synchronize_device`, repeated -- nothing in flight to hide the host behind, so this is the "
        "full latency an unpipelined caller pays."
    )
    lines.append("")
    lines.append(
        "If the plateau these paths run into at high core counts were host dispatch overhead, "
        "`pipelined - trace` would be large and roughly constant while the trace column kept falling. "
        "If it is near zero, the plateau is device time and the paths have genuinely stopped scaling."
    )
    lines.append("")
    lines.append(
        "| V | H | path | cores | trace (us) | eager pipelined | pipelined - trace | eager isolated | "
        "isolated - trace |"
    )
    lines.append("|---:|---:|:---|---:|---:|---:|---:|---:|---:|")
    for v, h in _shapes(rows):
        for path in _PATHS:
            for n_cores in CORE_SWEEP:
                r = _pick(rows, v, h, path, n_cores)
                if r is None:
                    continue
                name = str(n_cores) if n_cores is not None else f"default ({r['cores_used']})"
                lines.append(
                    f"| {v} | {h} | {path} | {name} | {r['trace_us']:.1f} | {r['eager_us']:.1f} | "
                    f"{r['eager_us'] - r['trace_us']:+.1f} | {r['eager_iso_us']:.1f} | "
                    f"{r['eager_iso_us'] - r['trace_us']:+.1f} |"
                )
    lines.append("")
    deltas = [r["eager_us"] - r["trace_us"] for r in rows]
    iso_deltas = [r["eager_iso_us"] - r["trace_us"] for r in rows]
    lines.append(
        f"Across all {len(rows)} points: `pipelined - trace` spans "
        f"{min(deltas):+.1f} to {max(deltas):+.1f} us (median {statistics.median(deltas):+.1f}); "
        f"`isolated - trace` spans {min(iso_deltas):+.1f} to {max(iso_deltas):+.1f} us "
        f"(median {statistics.median(iso_deltas):+.1f})."
    )
    lines.append("")

    lines.append("#### Winner at the default core count (what `ttnn.argmax` would actually deliver)")
    lines.append("")
    lines.append("| V | H | RVV cores | SFPU cores | RVV (us) | SFPU (us) | SFPU/RVV | winner |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|:---|")
    for v, h in _shapes(rows):
        r_rvv = _pick(rows, v, h, "RVV", None)
        r_sfpu = _pick(rows, v, h, "SFPU", None)
        if r_rvv is None or r_sfpu is None:
            continue
        rvv, sfpu = r_rvv["trace_us"], r_sfpu["trace_us"]
        winner = "**SFPU**" if sfpu < rvv else "RVV"
        lines.append(
            f"| {v} | {h} | {r_rvv['cores_used']} | {r_sfpu['cores_used']} | {rvv:.1f} | {sfpu:.1f} | "
            f"{sfpu / rvv:.2f}x | {winner} |"
        )
    lines.append("")
    lines.append(
        f"Shipped `kSfpuMinRows` = {SHIPPED_SFPU_MIN_ROWS}. Nothing above is asserted; this file "
        "measures. It does assert, at every point, that RVV and SFPU return identical indices and that "
        "both match a first-occurrence golden, so the timings compare two paths that computed the "
        "same answer."
    )
    lines.append("")
    block = "\n".join(lines)
    print(block)
    logger.info(block)
