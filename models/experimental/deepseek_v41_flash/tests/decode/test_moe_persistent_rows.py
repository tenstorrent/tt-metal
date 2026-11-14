# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The persistent TP12 MoE on DSpark verify blocks: every request carries ``ROWS`` token rows (the token and its
drafts), each routed on its own, through one pass of the op.

Same setup as ``test_moe_persistent.py`` (three random layers on the 3x4 MoE submesh at row 5, TP4 requesters
above it), with :class:`PersistentMoE` built for ``rows=ROWS``: the union of a block's selections (up to
``ROWS * top_k`` experts) is fetched once and applied to every row with that row's routing weight.

A block's rows are ``base * (1 - mix) + noise_r * mix`` (normalized): ``mix = 1`` gives independent rows (the
largest union, ~``ROWS * top_k``), ``mix = 0`` identical ones (a union of ``top_k``), in between some sharing --
consecutive real tokens route to overlapping experts. Each row is checked against the torch reference.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe_persistent_rows.py
"""

import gc
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_moe_persistent import (
    PARENT_MESH,
    PCC,
    REQUEST_ROWS,
    STEPS,
    _load_layers,
    _pcc,
    _reference,
    _setup,
    _single_senders,
    _snapshot,
)
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import PersistentMoE, untile_rows

ROWS = int(os.environ.get("MOE_ROWS", 4))
# Per step, how independent the block's rows are (see the module docstring).
MIXES = (1.0, 0.0, 0.3, 0.6, 1.0, 0.15, 0.45, 1.0, 0.0)


def _block(hidden, mix, generator):
    base = torch.randn(hidden, generator=generator)
    rows = []
    for _ in range(ROWS):
        row = base * (1.0 - mix) + torch.randn(hidden, generator=generator) * mix
        rows.append(row / row.norm() * hidden**0.5)
    return torch.stack(rows).to(torch.bfloat16).float()


def _selection(x, weights, config, k):
    """The top-k of row ``x`` as the reference ranks it."""
    gate_w = weights["gate.weight"] / getattr(config, "gate_temp", 1.0)
    biased = torch.sqrt(torch.nn.functional.softplus(gate_w @ x)) + weights["gate.e_score_correction_bias"]
    return set(biased.topk(k).indices.tolist())


def _union(block, weights, config, k):
    return len(set().union(*(_selection(x, weights, config, k) for x in block)))


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_rows(mesh_device):
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    num_requesters = len(request_meshes)
    k = moes[0].experts.top_k

    g = torch.Generator().manual_seed(0)
    blocks = [_block(config.hidden_size, MIXES[s % len(MIXES)], g) for s in range(len(STEPS))]
    references = [
        [_reference(x, *layers[layer], config, moes[layer].experts) for x in blocks[s]] for s, layer in enumerate(STEPS)
    ]
    unions = [_union(blocks[s], layers[layer][1], config, k) for s, layer in enumerate(STEPS)]

    moe_op = PersistentMoE(
        moe_mesh, request_meshes, moes, socket_core, request_core, _single_senders(REQUEST_ROWS), rows=ROWS
    )
    outs = []
    try:
        for first in range(0, len(STEPS), num_requesters):
            round_steps = range(first, min(first + num_requesters, len(STEPS)))
            pending = []
            for s in round_steps:
                moe_op.upload(s % num_requesters, blocks[s])
                pending.append(moe_op.request(s % num_requesters, STEPS[s]))
            for out in pending:
                outs.append(untile_rows(PersistentMoE.row_of(out).float(), ROWS))
                ttnn.deallocate(out)
    finally:
        moe_op.stop()

    if os.environ.get("MOE_ROWS_DUMP"):
        torch.save({"blocks": blocks, "outs": outs, "references": references}, os.environ["MOE_ROWS_DUMP"])
    report = []
    failed = False
    for s, layer in enumerate(STEPS):
        pccs = []
        for r in range(ROWS):
            candidates, margin = references[s][r]
            pccs.append(max(_pcc(c, outs[s][r]) for c in candidates))
        failed |= min(pccs) < PCC
        report.append(
            f"step {s} layer {layer} mix {MIXES[s % len(MIXES)]:.2f} union {unions[s]:2d}: per-row PCC vs torch "
            + " ".join(f"{p:.5f}" for p in pccs)
        )
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
def test_moe_persistent_rows_perf(mesh_device):
    """Per-block latency of the persistent op on ``ROWS``-row blocks, with 1 requester and then 3, traced and
    replayed back to back with no host I/O, at each ``MOE_ROWS_MIXES`` (comma-separated, default ``1,0.3,0``).

    ``MOE_PERF_WARMUP`` / ``MOE_PERF_STEPS`` override the step counts; ``MOE_ROWS_REQUESTERS`` (e.g. ``1``) the
    requester counts. Each traced (requester, layer)'s output must match the eager run's on the same block bit for
    bit."""
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    hidden = config.hidden_size
    k = moes[0].experts.top_k
    warmup = int(os.environ.get("MOE_PERF_WARMUP", 20))
    steps = int(os.environ.get("MOE_PERF_STEPS", 200))
    mixes = [float(m) for m in os.environ.get("MOE_ROWS_MIXES", "1,0.3,0").split(",")]
    requester_counts = [int(n) for n in os.environ.get("MOE_ROWS_REQUESTERS", f"1,{len(request_meshes)}").split(",")]
    order = [STEPS[i % len(STEPS)] for i in range(warmup + steps)]
    results = []
    for mix in mixes:
        g = torch.Generator().manual_seed(1)
        blocks = [_block(hidden, mix, g) for _ in order]
        mean_union = sum(_union(blocks[i], layers[order[i]][1], config, k) for i in range(len(order))) / len(order)
        for num_requesters in requester_counts:
            meshes = request_meshes[-num_requesters:]
            moe_op = PersistentMoE(
                moe_mesh,
                meshes,
                moes,
                socket_core,
                request_core,
                _single_senders(REQUEST_ROWS[-num_requesters:]),
                rows=ROWS,
            )
            name = f"rows {ROWS} mix {mix:.2f} (union {mean_union:4.1f}), {num_requesters} req"
            mismatched = []
            try:
                eager_outs = []
                for first in range(0, len(order) - len(order) % num_requesters, num_requesters):
                    pending = []
                    for i in range(first, min(first + num_requesters, len(order))):
                        moe_op.upload(i % num_requesters, blocks[i])
                        pending.append(moe_op.request(i % num_requesters, order[i]))
                    for out in pending:
                        eager_outs.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1))
                        ttnn.deallocate(out)

                firsts = {}
                for i, layer in enumerate(order):
                    firsts.setdefault((i % num_requesters, layer), i)
                inputs, tids, trace_outs = {}, {}, {}
                for (j, layer), first in firsts.items():
                    inputs[j, layer] = moe_op.request_input(j)
                    moe_op.upload(j, blocks[first], inputs[j, layer])
                    tids[j, layer] = ttnn.begin_trace_capture(meshes[j], cq_id=0)
                    trace_outs[j, layer] = moe_op.request(j, layer, inputs[j, layer])
                    ttnn.end_trace_capture(meshes[j], tids[j, layer], cq_id=0)
                for mesh in meshes:
                    ttnn.synchronize_device(mesh)

                def replay(indices):
                    for i in indices:
                        j = i % num_requesters
                        ttnn.execute_trace(meshes[j], tids[j, order[i]], cq_id=0, blocking=False)
                    for mesh in meshes:
                        ttnn.synchronize_device(mesh)

                replay(range(warmup))
                start = time.perf_counter()
                replay(range(warmup, len(order)))
                per_block = (time.perf_counter() - start) / steps
                results.append(
                    f"{name:<44} traced, device only: {per_block * 1e6:8.1f} us/block, "
                    f"{per_block * 1e6 / ROWS:7.1f} us/token"
                )
                for (j, layer), first in firsts.items():
                    copies = PersistentMoE.rows_of(trace_outs[j, layer])
                    mismatched += [
                        (j, layer, c) for c, copy in enumerate(copies) if not torch.equal(copy, eager_outs[first])
                    ]
                for key in tids:
                    ttnn.release_trace(meshes[key[0]], tids[key])
                for t in list(trace_outs.values()) + list(inputs.values()):
                    ttnn.deallocate(t)
            finally:
                moe_op.stop()
            assert not mismatched, f"{name}: traced (requester, layer, chip) outputs {mismatched} differ from eager"
            # PersistentMoE's relaunch closure holds a reference cycle: collect it so its L1 (router columns,
            # replica, scratch) is free for the next instance's program.
            del moe_op
            gc.collect()
            ttnn.synchronize_device(moe_mesh)
    logger.info(f"MoE TP12 {ROWS}-row blocks, layers interleaved {STEPS}:\n" + "\n".join(results))
