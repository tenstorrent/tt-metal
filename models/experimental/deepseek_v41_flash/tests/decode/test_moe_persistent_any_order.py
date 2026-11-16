# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The persistent TP12 MoE taking its requesters' rows in any order.

Every MoE chip's control core routes whichever requester's page is in, so the chips may take the requesters' rows in
different orders (persistent.h): a chip's only input from the others is their router candidates, kept per requester,
and its router stage never waits on them. These tests check that:

* a row's output does not depend on the order: every request is run once alone (the reference) and then again with
  every requester's requests in flight together, and the two must match bit for bit;
* the requesters need not take turns: they send different numbers of requests (one sends a single request, one
  none at all) back to back, in a shuffled interleaving, and all of them complete. Under the old strict turn
  (``TT_FE_FIXED_ORDER=1``) the requesters with more requests would wait forever for the turn of the one with fewer;
* every output matches the torch reference (PCC), as in ``test_moe_persistent.py``.

Same setup as ``test_moe_persistent.py``: three random layers at V4.1's shapes on the 3x4 MoE submesh at row 5, TP4
requesters above it. ``ROWS`` 1 is the one-row request, 4 the DSpark verify block.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe_persistent_any_order.py

``MOE_ANY_ORDER_ROWS`` (comma-separated mesh rows, default ``2,3,4``) picks the requesters' rows, e.g. ``0,1,2,3,4``
for five, the five-instance layout.
"""

import os
import random

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_moe_persistent import (
    MOE_OFFSET,
    MOE_SUBMESH,
    PARENT_MESH,
    PCC,
    REQUEST_SUBMESH,
    _load_layers,
    _pcc,
    _reference,
    _snapshot,
)
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import (
    PersistentMoE,
    default_socket_core,
    untile_rows,
)

REQUEST_ROWS = tuple(int(r) for r in os.environ.get("MOE_ANY_ORDER_ROWS", "2,3,4").split(","))
# Requests per requester (by position; requesters past the list send 2): uneven on purpose, one with a single
# request and one with none.
COUNTS = (9, 1, 5, 0, 3)


def _setup(mesh_device):
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    moe_mesh = mesh_device.create_submesh(ttnn.MeshShape(*MOE_SUBMESH), ttnn.MeshCoordinate(*MOE_OFFSET))
    request_meshes = [
        mesh_device.create_submesh(ttnn.MeshShape(*REQUEST_SUBMESH), ttnn.MeshCoordinate(row, 0))
        for row in REQUEST_ROWS
    ]
    core = default_socket_core(moe_mesh)
    single_senders = [row == MOE_OFFSET[0] - 1 for row in REQUEST_ROWS]
    return moe_mesh, request_meshes, load_config(_snapshot()), core, single_senders


def _plan(num_requesters, num_layers, rows, hidden):
    """``(requests, order)``: per requester its requests ``(layer, block)``, ``block`` ``[rows, D]``; and a shuffled
    interleaving of ``(requester, index)`` that keeps each requester's own order."""
    g = torch.Generator().manual_seed(0)
    rng = random.Random(0)
    requests = []
    for j in range(num_requesters):
        count = COUNTS[j] if j < len(COUNTS) else 2
        mine = []
        for _ in range(count):
            block = torch.randn(rows, hidden, generator=g)
            block = (block / block.norm(dim=-1, keepdim=True) * hidden**0.5).to(torch.bfloat16).float()
            mine.append((rng.randrange(num_layers), block))
        requests.append(mine)
    order = [(j, i) for j, mine in enumerate(requests) for i in range(len(mine))]
    rng.shuffle(order)
    # Each requester's requests in its own order (a requester has one in flight at a time anyway).
    seen = [0] * num_requesters
    ordered = []
    for j, _ in order:
        ordered.append((j, seen[j]))
        seen[j] += 1
    return requests, ordered


def _output(out, rows):
    row = PersistentMoE.row_of(out).float()
    return untile_rows(row, rows) if rows > 1 else row.reshape(1, -1)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("rows", [1, 4], ids=["rows1", "rows4"])
@torch.no_grad()
def test_moe_persistent_any_order(mesh_device, rows):
    if os.environ.get("TT_FE_FIXED_ORDER") == "1":
        pytest.skip("TT_FE_FIXED_ORDER=1 takes the requesters in strict turn, which this test's uneven counts hang")
    moe_mesh, request_meshes, config, socket_core, single_senders = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    num_requesters = len(request_meshes)
    requests, order = _plan(num_requesters, len(layers), rows, config.hidden_size)
    logger.info(
        f"{num_requesters} requesters (rows {REQUEST_ROWS}), {rows}-row requests, counts "
        f"{[len(mine) for mine in requests]}, interleaving {order}"
    )

    moe_op = PersistentMoE(moe_mesh, request_meshes, moes, socket_core, socket_core, single_senders, rows=rows)
    solo = {}
    together = {}
    try:
        # 1. The reference: each request alone, nothing else in flight.
        for j, mine in enumerate(requests):
            for i, (layer, block) in enumerate(mine):
                tensor = moe_op.request_input(j)
                moe_op.upload(j, block, tensor)
                out = moe_op.request(j, layer, tensor)
                solo[j, i] = _output(out, rows)
                ttnn.deallocate(out)
                ttnn.deallocate(tensor)

        # 2. Every requester's requests back to back, in the shuffled interleaving, all in flight before any is read.
        inputs = {}
        outs = {}
        for j, i in order:
            layer, block = requests[j][i]
            inputs[j, i] = moe_op.request_input(j)
            moe_op.upload(j, block, inputs[j, i])
            outs[j, i] = moe_op.request(j, layer, inputs[j, i])
        for key, out in outs.items():
            together[key] = _output(out, rows)
            ttnn.deallocate(out)
            ttnn.deallocate(inputs[key])
    finally:
        moe_op.stop()

    report = []
    failed = False
    for j, mine in enumerate(requests):
        for i, (layer, block) in enumerate(mine):
            same = torch.equal(solo[j, i], together[j, i])
            pccs = []
            for r in range(rows):
                candidates, _ = _reference(block[r], *layers[layer], config, moes[layer].experts)
                pccs.append(max(_pcc(c, together[j, i][r]) for c in candidates))
            failed |= not same or min(pccs) < PCC
            report.append(
                f"requester {j} request {i} layer {layer}: {'bit-identical' if same else 'DIFFERS'} alone vs "
                f"together, per-row PCC vs torch " + " ".join(f"{p:.5f}" for p in pccs)
            )
    logger.info("\n" + "\n".join(report))
    assert (
        not failed
    ), f"a request's output depends on the other requesters' (alone vs together) or is below PCC {PCC}:\n" + "\n".join(
        report
    )
