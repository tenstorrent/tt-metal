# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks the MoE block op tests share: determinism folded on the device, and device time per op from the real-time
program profiler (no Tracy build, no CSVs).

* DeterminismMarker: the first call's outputs kept on the device; every later call's outputs compared to them with
  ttnn.ne (as bf16: max has no uint32 path) and folded into one running max - one scalar a chip goes to the host at
  the end (tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa.py test_sparse_sdpa_determinism's pattern).
* op_device_times: the device time of each op of a run, the op found by a path in its kernels' sources (the programs a
  run dispatches, merged over the chips: the slowest chip's time).
* PERF_RUNS / check_perf: every perf number is the median of PERF_RUNS (5) measurements, logged on every run (a
  passing CI run's log shows it too) and checked against its baseline - a number in the test's own Python table (one
  machine's: each test runs on one machine type). A slowdown past the margin fails with the baseline, the limit and
  the measurement; a missing baseline only logs a reminder to add it, a speedup past the margin one to lower it."""

import os
import statistics
import sys
from pathlib import Path

import pytest
import torch

from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))  # (the repo root: tests.ttnn.profiling)
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged  # noqa: E402

# the MoE block's ops, each by a path its program's kernel sources contain
BLOCK_OPS = {
    "moe_head": "moe_block/moe_head/device/kernels",
    "sparse_dispatch_2d": "moe_block/sparse_dispatch_2d/device/kernels",
    "routed_experts_combine": "moe_block/routed_experts_combine/device/kernels",
    "moe_tail": "moe_block/moe_tail/device/kernels",
}


class DeterminismMarker:
    """Bit-identity of named outputs across calls: each chip's shard compared as raw bits on the host (a float compare
    would take -0 for +0 and fail an unchanged NaN)."""

    _BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.ref = None
        self.differ = []
        self.calls = 0

    def _bits(self, t):
        # (each chip's shard: a mesh composer cannot concat every layout)
        multi = hasattr(self.mesh_device, "get_num_devices") and self.mesh_device.get_num_devices() > 1
        shards = ttnn.get_device_tensors(t) if multi else [t]
        host = [ttnn.to_torch(s) for s in shards]
        return [h.view(self._BITS[h.dtype]) if h.dtype in self._BITS else h for h in host]

    def add(self, outputs: dict):
        """One call's outputs (name -> device tensor); the first call's are the reference."""
        bits = {name: self._bits(t) for name, t in outputs.items()}
        self.calls += 1
        if self.ref is None:
            self.ref = bits
            return
        for name, shards in bits.items():
            if not all(torch.equal(a, b) for a, b in zip(self.ref[name], shards)):
                self.differ.append(f"{name} (call {self.calls})")

    def assert_identical(self, what: str):
        assert self.calls > 1, f"{what}: determinism needs at least two calls"
        assert not self.differ, f"{what}: outputs differ between calls (over {self.calls} calls): {self.differ[:8]}"
        logger.info(f"{what}: bit-identical over {self.calls} calls")
        self.ref = None


class MeshBarrier:
    """An all-gather over each mesh axis, put ahead of the measured ops in a trace: every chip starts them together.
    Without it the trace's first cross-chip op takes the chips' start skew as its slowest chip's time (the MoE block's
    head: 230 us traced vs 202 us, 877 us after 56 ms of matmuls on each chip - GLM 8x4 5120, Nov 5); with it the skew
    lands in the barrier ("other"). Needs a bank of L1_SMALL for its semaphores."""

    def __init__(self, mesh_device):
        import torch

        self.tok = ttnn.from_torch(
            torch.zeros(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def __call__(self):
        for axis in (0, 1):
            ttnn.deallocate(ttnn.all_gather(self.tok, dim=3, cluster_axis=axis))

    def close(self):
        ttnn.deallocate(self.tok)


def op_device_times(mesh_device, run_fn, ops: dict = BLOCK_OPS):
    """run_fn's programs on the real-time profiler: {op: [device ns of each of its programs, in dispatch order]}
    (each program's slowest chip), plus the programs that matched no op under "other"."""
    _, per_program = profile_realtime_program_merged(mesh_device, run_fn)
    out = {op: [] for op in ops}
    out["other"] = []
    for entry in per_program.values():
        srcs = [s.replace("\\", "/") for s in entry["kernel_sources"]]
        op = next((op for op, path in ops.items() if any(path in s for s in srcs)), "other")
        out[op].append(entry["duration_ns"])
    return out


def traced_op_device_times(mesh_device, trace_id, replays: int, ops: dict = BLOCK_OPS, after_replay=None):
    """Each of `replays` executions of a captured trace on its own profiler window (a trace's programs keep one runtime
    id, so replays in one window merge into one record): {op: [device ns a replay]}. after_replay(): e.g. fold the
    replay's outputs into a DeterminismMarker (outside the timed program set: its own programs land in "other")."""
    out = {op: [] for op in ops}
    out["other"] = []
    for _ in range(PERF_WARMUP):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    for _ in range(replays):

        def one():
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            if after_replay is not None:
                after_replay()

        t = op_device_times(mesh_device, one, ops)
        for op, v in t.items():
            out[op].extend(v)
    return out


def median_us(ns_list):
    return statistics.median(ns_list) / 1e3 if ns_list else float("nan")


# Absolute slack on top of the relative margin: a short op's run-to-run spread is a few us whatever its length (GLM 8x4
# 10240's ~80 us tail: 78 - 86 us between runs), which a 5% margin alone does not hold.
SLACK_US = 10.0


def perf_failure(
    label: str, measured_us: float, baseline_us, margin: float = 0.05, slack_us: float = SLACK_US, where: str = ""
):
    """Logs one perf number against its baseline and returns the failure message if it is slower than the limit
    (baseline * (1 + margin) + slack_us), else None. baseline_us None: no baseline yet - logged with a reminder to add
    it (where: the table entry to add), not a failure. A run faster than baseline * (1 - margin) - slack_us passes
    with a reminder to lower the baseline."""
    on = f"on {machine()}"
    if baseline_us is None:
        logger.warning(
            f"PERF {label}: {measured_us:.1f} us {on} - NO BASELINE, not gated; add it: {where} = {measured_us:.1f}"
        )
        return None
    upper, lower = baseline_us * (1 + margin) + slack_us, baseline_us * (1 - margin) - slack_us
    delta = measured_us / baseline_us - 1
    numbers = (
        f"measured {measured_us:.1f} us, baseline {baseline_us:.1f} us ({delta:+.1%}), "
        f"limit {upper:.1f} us (baseline +{margin:.0%} +{slack_us:.0f} us) {on}"
    )
    if measured_us > upper:
        logger.error(f"PERF {label}: SLOWER THAN THE LIMIT - {numbers}")
        return f"PERF {label}: slower than its limit - {numbers} ({where})"
    logger.info(f"PERF {label}: ok - {numbers}")
    if measured_us < lower:
        logger.warning(f"PERF {label}: faster than baseline -{margin:.0%} -{slack_us:.0f} us - lower it: {where}")
    return None


def check_perf(label: str, measured_us: float, baseline_us, margin: float = 0.05, slack_us: float = SLACK_US, where=""):
    """perf_failure's check of one number: fails the test on a slowdown past the limit."""
    failure = perf_failure(label, measured_us, baseline_us, margin, slack_us, where)
    assert failure is None, failure


def assert_no_perf_failures(failures):
    """A test with several numbers: each logged and checked first (perf_failure), then every failure in one message."""
    failures = [f for f in failures if f]
    assert not failures, f"{len(failures)} perf number(s) slower than their limit:\n" + "\n".join(failures)


# Every perf measurement: the median of this many runs (traced replays, eager calls or host-timed repetitions).
PERF_RUNS = int(os.environ.get("MOE_PERF_RUNS", "5"))
# (traced perf: this many untimed replays first - MOE_PERF_WARMUP, default none)
PERF_WARMUP = int(os.environ.get("MOE_PERF_WARMUP", "0"))


def machine() -> str:
    """The machine a perf number was measured on (logged with it): the cluster type (BLACKHOLE_GALAXY, P150_X8, ...)."""
    try:
        return str(ttnn.cluster.get_cluster_type()).split(".")[-1]
    except Exception:  # noqa: BLE001
        return "unknown"


def median_of_runs(measure, runs: int = PERF_RUNS):
    """measure() run `runs` times: (median, every run's value) - for numbers measured a run at a time (host time)."""
    values = [measure() for _ in range(runs)]
    return statistics.median(values), values
