# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""(local scratch, not committed) GLM-5.3 experts + combine on the LoudBox across M: the traced routed_experts_combine
time (median of PERF_RUNS replays, slowest chip), uniform routing, and its DRAM / math utilization at the mean M.

    scripts/run_safe_pytest.sh --run-all tests/.../routed_experts_combine/zz_scratch_util_sweep.py -s
"""

import sys
from pathlib import Path

import pytest
from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_stages as bs  # noqa: E402
from block_stages import BlockStages, host_weights, skip_unless_galaxy  # noqa: E402, F401
from op_checks import PERF_RUNS, median_us, traced_op_device_times  # noqa: E402
from test_experts_combine_checks import MeshBarrier  # noqa: E402

OPS = {"routed_experts_combine": "moe_block/routed_experts_combine/device/kernels"}
H, I = 6144, 2048
W_BYTES = 3 * H * I / 1024 * 576  # an expert's bfp4 weights
PEAK = 4096 * 1.35e9  # LoFi FLOP / s a core
N_COMPUTE = 96  # the flat expert's gate/up + down cores


def _params():
    out = []
    for p in bs._params((None,)):
        c = p.values[0]
        if c.model in ("glm53_lb8", "glm53_lb16"):
            out.append(pytest.param(*p.values, marks=p.marks, id=c.id))
    return out


@pytest.mark.parametrize("case, routing, mesh_device, device_params", _params(), indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(1800)
def test_util_sweep(case, routing, mesh_device, device_params, host_weights):
    skip_unless_galaxy()
    epc = 8 if case.model == "glm53_lb8" else 16
    M = case.chunk * 8 // (8 * epc)  # (uniform: chunk x top-8 rows over 8 x epc experts... a chip owns chunk rows)
    op = BlockStages(mesh_device, case, host_weights)
    try:
        op.route("uniform")
        op()
        barrier = MeshBarrier(mesh_device)
        try:
            barrier()
            ttnn.synchronize_device(mesh_device)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            barrier()
            op()
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            t = traced_op_device_times(mesh_device, tid, PERF_RUNS, OPS)["routed_experts_combine"]
            ttnn.release_trace(mesh_device, tid)
        finally:
            barrier.close()
        us = median_us(t)
        dram = epc * W_BYTES / (us * 1e-6) / 512e9
        math = epc * M * 6 * H * I / (us * 1e-6) / (PEAK * N_COMPUTE)
        logger.info(
            f"[util] COMBINE glm53 e{epc} chunk {case.chunk} M {M}: {us:.1f} us, DRAM (weights) {dram:.1%}, math {math:.1%}"
        )
    finally:
        op.close()


def _rec_params():
    return [p for p in _params() if p.values[0].model == "glm53_lb16" and p.values[0].chunk in (5120, 6144)]


@pytest.mark.parametrize("case, routing, mesh_device, device_params", _rec_params(), indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(1800)
def test_decoupled(case, routing, mesh_device, device_params, host_weights):
    """Experts + combine with the hand-off cut (TEMPORARY probes, ZZ_REC_ADDR): the first call records every row
    trailer the combine's ports and local rings receive; every later call replays them - the ports / local rings take
    rows as soon as their slots are free, the y writers hand nothing off: the experts and the combine as two
    independent ops on the same grid."""
    import os

    import torch

    skip_unless_galaxy()
    P = 1 + 40 * 4096
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    rec = ttnn.from_torch(
        torch.zeros(P, 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep,
    )  # fmt: skip
    mode_host = torch.zeros(1, 16, dtype=torch.int32)
    mode_host[0, 0] = 1  # record
    mode = ttnn.from_torch(
        mode_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep,
    )  # fmt: skip
    os.environ["ZZ_REC_ADDR"] = str(rec.buffer_address())
    os.environ["ZZ_REC_MODE_ADDR"] = str(mode.buffer_address())
    epc = 16
    M = case.chunk * 8 // (8 * epc)
    op = BlockStages(mesh_device, case, host_weights)
    try:
        op.route("uniform")
        op()  # (record)
        ttnn.synchronize_device(mesh_device)
        mode_host[0, 0] = 2  # replay from here on
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(mode_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), mode
        )
        recorded = ttnn.to_torch(ttnn.get_device_tensors(rec)[0]).to(torch.int64)
        nz = int((recorded[1:] != 0).any(1).sum())
        op()  # (a replay outside the trace)
        barrier = MeshBarrier(mesh_device)
        try:
            barrier()
            ttnn.synchronize_device(mesh_device)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            barrier()
            op()
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            t = traced_op_device_times(mesh_device, tid, PERF_RUNS, OPS)["routed_experts_combine"]
            ttnn.release_trace(mesh_device, tid)
        finally:
            barrier.close()
        logger.info(
            f"[util] DECOUPLED glm53 e{epc} chunk {case.chunk} M {M}: {median_us(t):.1f} us (trailers recorded on chip 0: {nz})"
        )
    finally:
        op.close()
        os.environ.pop("ZZ_REC_ADDR", None)
        os.environ.pop("ZZ_REC_MODE_ADDR", None)
