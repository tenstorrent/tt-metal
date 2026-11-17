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
* assert_not_slower: a perf gate that fails on a slowdown past the margin and only reports a speedup.
* PERF_RUNS / PerfBaselines: every perf number is the median of PERF_RUNS (5) measurements, checked against its
  baseline in a JSON file beside the test, keyed by the machine (ttnn.cluster.get_cluster_type()) and a label;
  MOE_PERF_RECORD=1 writes the measured medians as the baselines instead of checking them."""

import json
import os
import statistics
import sys
from pathlib import Path

import pytest

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
    """Bit-identity of named outputs across calls, kept on the device."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.ref = None
        self.marker = None
        self.calls = 0

    def add(self, outputs: dict):
        """One call's outputs (name -> device tensor); the first call's are cloned as the reference."""
        if self.ref is None:
            self.ref = {name: ttnn.clone(t) for name, t in outputs.items()}
            self.calls = 1
            return
        for name, t in outputs.items():
            m = ttnn.max(ttnn.ne(self.ref[name], t, dtype=ttnn.bfloat16))  # (0 iff bit-identical)
            self.marker = m if self.marker is None else ttnn.maximum(self.marker, m)
        self.calls += 1

    def assert_identical(self, what: str):
        assert self.marker is not None, f"{what}: determinism needs at least two calls"
        # (max returns a 0-d tensor a chip: read each chip's scalar - a mesh composer cannot concat 0-d tensors)
        multi = hasattr(self.mesh_device, "get_num_devices") and self.mesh_device.get_num_devices() > 1
        shards = ttnn.get_device_tensors(self.marker) if multi else [self.marker]
        mismatch = max(float(ttnn.to_torch(t).float().max()) for t in shards)
        assert mismatch == 0.0, f"{what}: outputs differ between calls (over {self.calls} calls)"
        logger.info(f"{what}: bit-identical over {self.calls} calls")
        for t in self.ref.values():
            ttnn.deallocate(t)


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


def assert_not_slower(label: str, measured_us: float, expected_us: float, margin: float, slack_us: float = SLACK_US):
    """Fails past expected * (1 + margin) + slack_us; a run faster than expected * (1 - margin) - slack_us passes with
    a note to re-baseline."""
    upper, lower = expected_us * (1 + margin) + slack_us, expected_us * (1 - margin) - slack_us
    logger.info(f"PERF {label}: {measured_us:.1f} us (expected {expected_us:.1f} +{margin:.0%} +{slack_us:.0f} us)")
    assert measured_us <= upper, f"{label}: {measured_us:.1f} us is slower than {upper:.1f} us"
    if measured_us < lower:
        logger.warning(f"PERF {label}: {measured_us:.1f} us is faster than {lower:.1f} us - lower the baseline")


# Every perf measurement: the median of this many runs (traced replays, eager calls or host-timed repetitions).
PERF_RUNS = int(os.environ.get("MOE_PERF_RUNS", "5"))
# (traced perf: this many untimed replays first - MOE_PERF_WARMUP, default none)
PERF_WARMUP = int(os.environ.get("MOE_PERF_WARMUP", "0"))
RECORD = os.environ.get("MOE_PERF_RECORD", "0") == "1"


def machine() -> str:
    """The baselines' machine key: the cluster type (BLACKHOLE_GALAXY, P150_X8 (LoudBox), ...)."""
    try:
        return str(ttnn.cluster.get_cluster_type()).split(".")[-1]
    except Exception:  # noqa: BLE001
        return "unknown"


class PerfBaselines:
    """Perf baselines in a JSON file: {machine: {label: us}}. check() gates a measured median against its baseline
    (assert_not_slower); with MOE_PERF_RECORD=1 it stores the measurement instead. A label without a baseline on this
    machine fails - every perf case carries one (record it on the machine the test runs on)."""

    def __init__(self, path):
        self.path = Path(path)

    def _load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def expected(self, label: str):
        return self._load().get(machine(), {}).get(label)

    def check(self, label: str, measured_us: float, margin: float = 0.05, slack_us: float = SLACK_US):
        if RECORD:
            data = self._load()
            data.setdefault(machine(), {})[label] = round(float(measured_us), 1)
            data = {m: dict(sorted(v.items())) for m, v in sorted(data.items())}
            self.path.write_text(json.dumps(data, indent=1) + "\n")
            logger.info(f"PERF {label}: {measured_us:.1f} us recorded as the {machine()} baseline in {self.path.name}")
            return
        expected = self.expected(label)
        if expected is None:
            pytest.fail(
                f"no {machine()} baseline for {label} in {self.path.name}: measured {measured_us:.1f} us "
                "(MOE_PERF_RECORD=1 records it)"
            )
        assert_not_slower(label, measured_us, expected, margin, slack_us)


def median_of_runs(measure, runs: int = PERF_RUNS):
    """measure() run `runs` times: (median, every run's value) - for numbers measured a run at a time (host time)."""
    values = [measure() for _ in range(runs)]
    return statistics.median(values), values
