# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""routed_experts_combine (the flat routed experts + the combine) with the flat expert's x / h data formats
(FlatLayoutOptions.x_bf16 / h_bf16, bfp8 by default) on a Galaxy, each (x, h) combination on the real head + dispatch
outputs (block_stages.py, uniform routing):

* test_experts_combine_xh_accuracy: SAMPLE tokens a chip against torch - PCC and the largest absolute difference
  (XH_ACCURACY lines).
* test_experts_combine_xh_perf: the op's device time on a traced replay (a barrier ahead of the op), WARMUP replays
  first, then the median of RUNS replays (XH_PERF lines; the op's slowest chip). No baseline."""

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_stages import (  # noqa: E402
    MESHES,
    MODELS,
    PCC,
    SAMPLE,
    BlockStages,
    Case,
    host_weights,  # noqa: F401 (the module's host weight cache fixture)
    skip_unless_galaxy,
)
from op_checks import BLOCK_OPS, MeshBarrier, median_us, traced_op_device_times  # noqa: E402

OPS = {"routed_experts_combine": BLOCK_OPS["routed_experts_combine"]}
WARMUP = 25  # (at least 20 warm-up replays before any measured one)
RUNS = 10
XH = [(False, False), (True, False), (False, True), (True, True)]
CASES = [Case("glm53", "8x4", 5120), Case("kimi_k3", "8x4", 5120), Case("dsv41_flash", "8x4", 5120)]


def _xh_id(x, h):
    return f"x{'bf16' if x else 'bfp8'}-h{'bf16' if h else 'bfp8'}"


def _params():
    out = []
    for c in CASES:
        shape, make_params, topology = MESHES[c.mesh]
        dp = make_params(**MODELS[c.model].mesh_params())
        mark = pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topology)
        for xh in XH:
            out.append(pytest.param(c, xh, shape, dp, marks=mark, id=f"{c.id}-{_xh_id(*xh)}"))
    return out


PARAMS = pytest.mark.parametrize("case, xh, mesh_device, device_params", _params(), indirect=["mesh_device",
                                                                                                "device_params"])


def _stages(mesh_device, case, xh, host_weights):
    layout = ttnn.experimental.moe_block.FlatLayoutOptions(x_bf16=xh[0], h_bf16=xh[1])
    op = BlockStages(mesh_device, case, host_weights, layout=layout)
    op.route("uniform")
    return op


@PARAMS
@pytest.mark.timeout(1800)
def test_experts_combine_xh_accuracy(case, xh, mesh_device, device_params, host_weights):
    skip_unless_galaxy()
    op = _stages(mesh_device, case, xh, host_weights)
    try:
        y = op()
        ids, ws, xin = op.routing_on_host()
        got = op.y_on_host(y)
        assert torch.isfinite(got).all(), f"{case.id} {_xh_id(*xh)}: non-finite outputs"
        g = torch.Generator().manual_seed(77)
        pick = torch.cat([me * op.S + torch.randperm(op.S, generator=g)[:SAMPLE] for me in range(op.N)])
        want = op.reference(ids[pick], ws[pick], xin[pick])
        ok, pcc = comp_pcc(want, got[pick], PCC)
        diff = (want - got[pick]).abs()
        logger.info(
            f"XH_ACCURACY experts+combine {case.id} {_xh_id(*xh)}: pcc {float(pcc):.6f}, max abs diff "
            f"{diff.max().item():.5f}, mean abs diff {diff.mean().item():.6f} over {len(pick)} tokens"
        )
        assert ok, f"{case.id} {_xh_id(*xh)}: pcc {pcc} < {PCC}"
    finally:
        op.close()


@PARAMS
@pytest.mark.timeout(1800)
def test_experts_combine_xh_perf(case, xh, mesh_device, device_params, host_weights):
    skip_unless_galaxy()
    op = _stages(mesh_device, case, xh, host_weights)
    barrier = MeshBarrier(mesh_device)
    try:
        op()  # (program build)
        barrier()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        barrier()
        op()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        for _ in range(WARMUP):  # (warm-up replays)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        t = traced_op_device_times(mesh_device, tid, RUNS, OPS)["routed_experts_combine"]
        ttnn.release_trace(mesh_device, tid)
    finally:
        barrier.close()
        op.close()
    us = median_us(t)
    logger.info(
        f"XH_PERF experts+combine {case.id} {_xh_id(*xh)}: median {us:.1f} us (min {min(t) / 1e3:.1f}, "
        f"max {max(t) / 1e3:.1f}) over {len(t)} replays after {WARMUP} warm-up"
    )
