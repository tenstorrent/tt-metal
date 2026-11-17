# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.routed_experts_combine (the flat routed experts + the sparse combine, one op) on a
Galaxy: accuracy, determinism and device time, each its own test, plus a re-routing test. Every run has the combine's
snap on (q 2^-14).

The op's inputs are the real MoE head's and dispatch's outputs (the routing steered through the router's bias), run
once ahead of the op and kept: the tests run and time the op alone. Cases, routings, weights and the upstream ops:
block_stages.py.

* test_experts_combine_accuracy: SAMPLE tokens a chip against torch (each token's K experts on its own dispatched row,
  bfp4 weights, weighted by the head's gate weights), PCC over all of them and per chip.
* test_experts_combine_determinism: CALLS calls on the same inputs, bit-identical on the device.
* test_experts_combine_reroute: routing A, then B, then A again on the same op and buffers - each right, and the second
  A bit-identical to the first (nothing of a call's routing left in the next).
* test_experts_combine_perf: PERF_RUNS (5) traced replays (a barrier ahead of the op in the trace: every chip starts
  it together), the op's slowest chip, median, against its baseline in experts_combine_perf_baselines.json.
"""

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_stages import (  # noqa: E402
    CALLS,
    CASE_PARAMS,
    PARAMS,
    BlockStages,
    host_weights,  # noqa: F401 (the module's host weight cache fixture)
    skip_unless_galaxy,
)
from op_checks import (  # noqa: E402
    BLOCK_OPS,
    DeterminismMarker,
    PERF_RUNS,
    MeshBarrier,
    PerfBaselines,
    median_us,
    traced_op_device_times,
)

OPS = {"routed_experts_combine": BLOCK_OPS["routed_experts_combine"]}
BASELINES = PerfBaselines(Path(__file__).with_name("experts_combine_perf_baselines.json"))


# ---------------------------------------------------------------------------------------------------- the tests
@PARAMS
@pytest.mark.timeout(1800)
def test_experts_combine_accuracy(case, routing, mesh_device, device_params, host_weights):
    skip_unless_galaxy()
    op = BlockStages(mesh_device, case, host_weights)
    try:
        op.route(routing)
        op.check(op(), f"{case.id} {routing}")
    finally:
        op.close()


@PARAMS
@pytest.mark.timeout(1800)
def test_experts_combine_determinism(case, routing, mesh_device, device_params, host_weights):
    skip_unless_galaxy()
    op = BlockStages(mesh_device, case, host_weights)
    try:
        op.route(routing)
        det = DeterminismMarker(mesh_device)
        for _ in range(CALLS):
            det.add({"y_own": op()})  # (y_own: one persistent buffer, the marker cloned the first call's)
        det.assert_identical(f"experts + combine {case.id} {routing}")
    finally:
        op.close()


@CASE_PARAMS
@pytest.mark.timeout(1800)
def test_experts_combine_reroute(case, routing, mesh_device, device_params, host_weights):
    skip_unless_galaxy()
    op = BlockStages(mesh_device, case, host_weights)
    try:
        seen = []
        for r in ("uniform", "zipf", "uniform"):
            op.route(r)
            seen.append(op.check(op(), f"{case.id} reroute {len(seen)}: {r}"))
        assert torch.equal(seen[0], seen[2]), f"{case.id}: uniform after zipf differs from the first uniform"
        logger.info(f"{case.id}: uniform -> zipf -> uniform, the second uniform bit-identical to the first")
    finally:
        op.close()


@PARAMS
@pytest.mark.timeout(1800)
def test_experts_combine_perf(case, routing, mesh_device, device_params, host_weights):
    skip_unless_galaxy()
    op = BlockStages(mesh_device, case, host_weights)
    barrier = MeshBarrier(mesh_device)
    try:
        op.route(routing)
        op()  # (warm: program build)
        barrier()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        barrier()
        op()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # (a warm replay)
        t = traced_op_device_times(mesh_device, tid, PERF_RUNS, OPS)["routed_experts_combine"]
        ttnn.release_trace(mesh_device, tid)
    finally:
        barrier.close()
        op.close()
    us = median_us(t)
    label = f"experts + combine {case.id} {routing}"
    logger.info(f"PERF {label}: median {us:.1f} us (min {min(t) / 1e3:.1f}, max {max(t) / 1e3:.1f}) x{len(t)}")
    BASELINES.check(label, us, case.perf_margin)
