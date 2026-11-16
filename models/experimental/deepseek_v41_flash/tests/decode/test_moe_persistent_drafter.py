# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's MoE stages served by the target's persistent MoE (layers 40-42 after the target's layers).

Same Galaxy layout as ``test_moe_persistent.py``: the 3x4 MoE submesh at row 5, the target's 4-row (verify block)
requester at row 4 (in turn) and the drafter's at row 3 (on demand, see fe_ctrl.cpp). The server holds the three
random target layers of that test and the checkpoint's three drafter stages (:class:`DSparkServerMoE`: 128 experts,
top-3, the shared expert). Requests alternate, target then drafter, each only once the previous one's output is back
(the on-demand contract). Every row of every output is checked against an fp32 torch reference of its layer
(``test_moe_persistent._reference``: sqrtsoftplus router ranked on the biased scores, the renormalized top-k, the
shared expert; any top-k a routing tie allows).

``test_moe_persistent_drafter_perf``: each drafter stage's request alone, traced on the drafter's mesh and replayed back
to back (us per request), and a target request for comparison.
"""

import contextlib
import gc
import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode import test_moe_persistent
from models.experimental.deepseek_v41_flash.tests.decode.test_moe_persistent import (
    PARENT_MESH,
    PCC,
    REQUEST_ROWS,
    _load_layers,
    _pcc,
    _reference,
    _setup,
    _single_senders,
    _snapshot,
)
from models.experimental.deepseek_v41_flash.tests.decode.test_moe_persistent_rows import _block
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import (
    DSPARK_LAYER_BASE,
    PersistentMoE,
    build_dspark_server_layers,
    untile_rows,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

ROWS = 4
# Biased-score gap within which the drafter's top-3 may be picked either way (see test_moe_persistent_drafter).
DRAFT_ROUTING_TIE = 0.06
# The target's verify block requester (in turn) and the drafter's (on demand), as indices into REQUEST_ROWS.
TARGET_ROW, DRAFT_ROW = 4, 3
# (target layer, drafter stage) per step.
STEPS = ((0, 0), (1, 1), (2, 2), (1, 0), (0, 2), (2, 1))
MIXES = (0.3, 1.0, 0.0, 0.6, 0.15, 0.3)


def _server(mesh_device):
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    loader = DeepseekV4WeightLoader(_snapshot())
    start = time.perf_counter()
    drafter = build_dspark_server_layers(config, loader, moe_mesh, WeightCache(CACHE_DIR))
    logger.info(f"drafter stages loaded in {time.perf_counter() - start:.1f}s")
    # The references read the drafter's router back from the MoE mesh, which the launch then owns: read it now.
    for moe in drafter.values():
        _drafter_reference_weights(moe)
    served = {**dict(enumerate(moes)), **drafter}
    meshes = [request_meshes[REQUEST_ROWS.index(TARGET_ROW)], request_meshes[REQUEST_ROWS.index(DRAFT_ROW)]]
    moe_op = PersistentMoE(
        moe_mesh,
        meshes,
        served,
        socket_core,
        request_core,
        _single_senders([TARGET_ROW, DRAFT_ROW]),
        rows=ROWS,
        on_demand=[1],
    )
    return moe_op, config, layers, moes, drafter


@contextlib.contextmanager
def _routing_tie(tie):
    saved = test_moe_persistent.ROUTING_TIE
    test_moe_persistent.ROUTING_TIE = tie
    try:
        yield
    finally:
        test_moe_persistent.ROUTING_TIE = saved


def _drafter_reference_weights(moe):
    """A drafter stage's torch reference weights, its router as the server holds it: the bf8 gate and the bf16
    shifted bias, so only the matmul's accumulation can still tip a near-tie (the band of ``_reference``)."""
    if not hasattr(moe, "_reference_cache"):
        weights = {k: v() if callable(v) else v for k, v in moe.reference_weights.items()}
        e = weights["gate.weight"].shape[0]
        gate = ttnn.to_torch(ttnn.get_device_tensors(moe.gate.fused_gate_weight)[0]).float()
        weights["gate.weight"] = gate.reshape(-1, gate.shape[-1])[:, :e].t().contiguous()
        bias = ttnn.to_torch(ttnn.get_device_tensors(moe.gate.ranking_bias())[0]).float()
        weights["gate.e_score_correction_bias"] = bias.reshape(-1)[:e]
        moe._reference_cache = weights
    return moe._reference_cache


def _run(moe_op, requester, layer, block):
    moe_op.upload(requester, block)
    out = moe_op.request(requester, layer)
    rows = untile_rows(PersistentMoE.row_of(out).float(), ROWS)
    ttnn.deallocate(out)
    return rows


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_drafter(mesh_device):
    moe_op, config, layers, moes, drafter = _server(mesh_device)
    g = torch.Generator().manual_seed(0)
    report, failed = [], False
    try:
        for step, (layer, stage) in enumerate(STEPS):
            for requester, lid in ((0, layer), (1, DSPARK_LAYER_BASE + stage)):
                block = _block(config.hidden_size, MIXES[step % len(MIXES)], g)
                out = _run(moe_op, requester, lid, block)
                if requester == 0:
                    provider, weights = layers[layer]
                    experts = moes[layer].experts
                else:
                    moe = drafter[lid]
                    provider = moe.expert_provider
                    weights = _drafter_reference_weights(moe)
                    experts = SimpleNamespace(
                        top_k=moe.server_top_k,
                        limit=moe.experts.limit,
                        routed_scaling_factor=moe.experts.routed_scaling_factor,
                        routing_eps=moe.experts.routing_eps,
                    )
                pccs = []
                # The server's router sums its K-slice partials in bf16: at the drafter's scores (~3-4) a pick can
                # flip within ~0.05 of the 3rd/4th boundary, so the drafter's tie band is wider than the target's.
                tie = DRAFT_ROUTING_TIE if requester == 1 else test_moe_persistent.ROUTING_TIE
                with _routing_tie(tie):
                    for r in range(ROWS):
                        candidates, _ = _reference(block[r], provider, weights, config, experts)
                        pccs.append(max(_pcc(c, out[r]) for c in candidates))
                failed |= min(pccs) < PCC
                report.append(
                    f"step {step} {'target' if requester == 0 else 'drafter'} layer {lid:2d}: per-row PCC "
                    + " ".join(f"{p:.5f}" for p in pccs)
                )
    finally:
        moe_op.stop()
    logger.info("\n" + "\n".join(report))
    assert not failed, f"a row's PCC vs torch below {PCC}:\n" + "\n".join(report)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 32 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_drafter_perf(mesh_device):
    """us per request, traced and replayed back to back on one requester: each drafter stage (row 3), then a target
    layer (row 4). Each traced output must equal its eager run's."""
    moe_op, config, layers, moes, drafter = _server(mesh_device)
    steps = int(os.environ.get("MOE_PERF_STEPS", 100))
    g = torch.Generator().manual_seed(1)
    results = []
    try:
        cases = [(1, lid) for lid in sorted(drafter)] + [(0, 0)]
        for requester, lid in cases:
            mesh = moe_op.request_meshes[requester]
            block = _block(config.hidden_size, 0.3, g)
            moe_op.upload(requester, block)
            eager = moe_op.request(requester, lid)
            ref = ttnn.to_torch(ttnn.get_device_tensors(eager)[0])
            ttnn.deallocate(eager)
            inp = moe_op.request_input(requester)
            moe_op.upload(requester, block, inp)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = moe_op.request(requester, lid, inp)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(steps):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            us = (time.perf_counter() - start) / steps * 1e6
            same = all(torch.equal(ttnn.to_torch(t), ref) for t in ttnn.get_device_tensors(out))
            results.append(
                f"{'drafter' if requester else 'target'} layer {lid:2d}: {us:7.1f} us/request (traced == eager: {same})"
            )
            ttnn.release_trace(mesh, tid)
            ttnn.deallocate(out)
            ttnn.deallocate(inp)
            assert same, results[-1]
    finally:
        moe_op.stop()
        gc.collect()
    logger.info("\n" + "\n".join(results))
