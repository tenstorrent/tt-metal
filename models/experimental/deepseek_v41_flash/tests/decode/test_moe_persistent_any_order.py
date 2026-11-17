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

The mixed-rows tests (one launch for 4-row blocks serving one-row requests too) run one-row requests natively -- the
plain row on the wire, 1x32 tiles through the op -- and skip under ``DEEPSEEK_V41_MOE_NATIVE_ONE_ROW=0`` (the padded
mode, which they do not cover).
"""

import gc
import os
import random
import time

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
    STEPS,
    _load_layers,
    _pcc,
    _reference,
    _snapshot,
)
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import (
    NATIVE_ONE_ROW,
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


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_mixed_rows(mesh_device):
    """One launch serving one-row and 4-row requests together (``PersistentMoE(rows=1, block_rows=4)``, mixed): every
    requester sends an uneven, shuffled run of requests, each one row or a 4-row block at random. Each output must
    match the same request run alone bit for bit, and torch. A one-row request runs natively (1x32 tiles through the
    op, switching tile shapes with the 4-row ones around it), so it also checks that neither the shape switches nor
    another row leak into it."""
    if not NATIVE_ONE_ROW:
        pytest.skip("DEEPSEEK_V41_MOE_NATIVE_ONE_ROW=0 (padded one-row requests) is not covered by these tests")
    if os.environ.get("TT_FE_FIXED_ORDER") == "1":
        pytest.skip("TT_FE_FIXED_ORDER=1 takes the requesters in strict turn, which this test's uneven counts hang")
    moe_mesh, request_meshes, config, socket_core, single_senders = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    num_requesters = len(request_meshes)
    block = 4
    requests, order = _plan(num_requesters, len(layers), block, config.hidden_size)
    rng = random.Random(1)
    # Each request one row (its block's row 0) or the whole block.
    rows_of = {(j, i): rng.choice((1, block)) for j, mine in enumerate(requests) for i in range(len(mine))}
    logger.info(f"{num_requesters} requesters, one-row requests native, rows per request {rows_of}")

    moe_op = PersistentMoE(
        moe_mesh, request_meshes, moes, socket_core, socket_core, single_senders, rows=1, block_rows=block
    )
    assert moe_op.mixed, "the mixed-rows server is off (DEEPSEEK_V41_MOE_MIXED_ROWS=0?)"

    def send(j, i):
        rows = rows_of[j, i]
        layer, data = requests[j][i]
        moe_op.use_rows(rows)
        tensor = moe_op.request_input(j)
        moe_op.upload(j, data[:rows], tensor)
        return tensor, moe_op.request(j, layer, tensor)

    solo, together = {}, {}
    try:
        for j, mine in enumerate(requests):
            for i in range(len(mine)):
                tensor, out = send(j, i)
                solo[j, i] = _output(out, rows_of[j, i])
                ttnn.deallocate(out)
                ttnn.deallocate(tensor)
        pending = {key: send(*key) for key in order}
        for key, (tensor, out) in pending.items():
            together[key] = _output(out, rows_of[key])
            ttnn.deallocate(out)
            ttnn.deallocate(tensor)
    finally:
        moe_op.stop()

    report = []
    failed = False
    for j, mine in enumerate(requests):
        for i, (layer, data) in enumerate(mine):
            rows = rows_of[j, i]
            same = torch.equal(solo[j, i], together[j, i])
            pccs = []
            for r in range(rows):
                candidates, _ = _reference(data[r], *layers[layer], config, moes[layer].experts)
                pccs.append(max(_pcc(c, together[j, i][r]) for c in candidates))
            failed |= not same or min(pccs) < PCC
            report.append(
                f"requester {j} request {i} layer {layer} rows {rows}: {'bit-identical' if same else 'DIFFERS'} alone "
                f"vs together, per-row PCC vs torch " + " ".join(f"{p:.5f}" for p in pccs)
            )
    logger.info("\n" + "\n".join(report))
    assert not failed, "a mixed-rows request differs alone vs together, or is below PCC:\n" + "\n".join(report)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 32 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_mixed_rows_perf(mesh_device):
    """Per-request latency on the mixed server (one launch for 4-row blocks, ``PersistentMoE(rows=1,
    block_rows=4)``): one-row requests, 4-row requests, and the two alternating, with 1 requester and then all of
    them, each (requester, layer, rows) request traced once and replayed back to back with no host I/O (device only).
    One-row requests run natively (1x32 tiles).

    ``MOE_PERF_WARMUP`` / ``MOE_PERF_STEPS`` override the step counts, ``MOE_MIXED_REQUESTERS`` (e.g. ``1``) the
    requester counts. Each traced request's output must match its eager run's bit for bit."""
    if not NATIVE_ONE_ROW:
        pytest.skip("DEEPSEEK_V41_MOE_NATIVE_ONE_ROW=0 (padded one-row requests) is not covered by these tests")
    moe_mesh, request_meshes, config, socket_core, single_senders = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    hidden = config.hidden_size
    block = 4
    warmup = int(os.environ.get("MOE_PERF_WARMUP", 20))
    steps = int(os.environ.get("MOE_PERF_STEPS", 200))
    requester_counts = [int(n) for n in os.environ.get("MOE_MIXED_REQUESTERS", f"1,{len(request_meshes)}").split(",")]
    order = [STEPS[i % len(STEPS)] for i in range(warmup + steps)]
    g = torch.Generator().manual_seed(2)
    data = {}
    for layer in sorted(set(STEPS)):
        rows = torch.randn(block, hidden, generator=g)
        data[layer] = (rows / rows.norm(dim=-1, keepdim=True) * hidden**0.5).to(torch.bfloat16).float()
    mode = "native"
    # Per phase, the rows of request i.
    phases = {
        "1-row": lambda i: 1,
        f"{block}-row": lambda i: block,
        "alternating": lambda i: 1 if i % 2 == 0 else block,
    }
    results = []
    for num_requesters in requester_counts:
        meshes = request_meshes[-num_requesters:]
        moe_op = PersistentMoE(
            moe_mesh,
            meshes,
            moes,
            socket_core,
            socket_core,
            single_senders[-num_requesters:],
            rows=1,
            block_rows=block,
        )
        assert moe_op.mixed, "the mixed-rows server is off (DEEPSEEK_V41_MOE_MIXED_ROWS=0?)"
        mismatched = []
        inputs, tids, trace_outs, eager = {}, {}, {}, {}
        try:
            keys = {(i % num_requesters, layer, rows) for i, layer in enumerate(order) for rows in (1, block)}
            # The eager reference of every (requester, layer, rows) request, then its trace.
            for j, layer, rows in sorted(keys):
                moe_op.use_rows(rows)
                tensor = moe_op.request_input(j)
                moe_op.upload(j, data[layer][:rows], tensor)
                out = moe_op.request(j, layer, tensor)
                eager[j, layer, rows] = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)
                ttnn.deallocate(out)
                inputs[j, layer, rows] = tensor
            for j, layer, rows in sorted(keys):
                moe_op.use_rows(rows)
                tids[j, layer, rows] = ttnn.begin_trace_capture(meshes[j], cq_id=0)
                trace_outs[j, layer, rows] = moe_op.request(j, layer, inputs[j, layer, rows])
                ttnn.end_trace_capture(meshes[j], tids[j, layer, rows], cq_id=0)
            for mesh in meshes:
                ttnn.synchronize_device(mesh)

            def replay(indices, rows_of):
                for i in indices:
                    j = i % num_requesters
                    ttnn.execute_trace(meshes[j], tids[j, order[i], rows_of(i)], cq_id=0, blocking=False)
                for mesh in meshes:
                    ttnn.synchronize_device(mesh)

            for phase, rows_of in phases.items():
                replay(range(warmup), rows_of)
                start = time.perf_counter()
                replay(range(warmup, len(order)), rows_of)
                per_request = (time.perf_counter() - start) / steps
                results.append(
                    f"{phase:<12} requests, one-row {mode}, {num_requesters} req: traced, device only "
                    f"{per_request * 1e6:8.1f} us/request"
                )
            for key, out in trace_outs.items():
                copies = PersistentMoE.rows_of(out)
                mismatched += [(*key, c) for c, copy in enumerate(copies) if not torch.equal(copy, eager[key])]
            for key, tid in tids.items():
                ttnn.release_trace(meshes[key[0]], tid)
            for t in list(trace_outs.values()) + list(inputs.values()):
                ttnn.deallocate(t)
        finally:
            moe_op.stop()
        assert (
            not mismatched
        ), f"{num_requesters} req: traced (requester, layer, rows, chip) {mismatched} differ from eager"
        # PersistentMoE's relaunch closure holds a reference cycle: collect it so its L1 is free for the next one.
        del moe_op
        gc.collect()
        ttnn.synchronize_device(moe_mesh)
    logger.info(f"MoE TP12 mixed server ({block}-row launch), layers interleaved {STEPS}:\n" + "\n".join(results))


# test_moe_persistent_lockstep: rounds of every requester sending at once (MOE_LOCKSTEP_ROUNDS, default 120 = 40 per
# layer of the three), the shape of several model instances decoding the same prompt in step.
LOCKSTEP_ROUNDS = int(os.environ.get("MOE_LOCKSTEP_ROUNDS", "120"))


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("rows", [1, 4], ids=["rows1", "rows4"])
@pytest.mark.parametrize("same_layer", [True, False], ids=["same_layer", "other_layers"])
@pytest.mark.parametrize("identical", [True, False], ids=["identical_data", "own_data"])
@pytest.mark.parametrize("post", [False, True], ids=["plain_out", "hc_post"])
@torch.no_grad()
def test_moe_persistent_lockstep(mesh_device, rows, same_layer, identical, post):
    """Every requester sends at the same instant, round after round, as several model instances decoding one prompt in
        step do -- the case the model-level test (test_instances_dspark.py) hangs on unless the users' starts are staggered.
        Each round enqueues one request from every requester -- the same layer (``same_layer``) or each its own, with the
        same block (``identical``) or each its own -- and then reads them all, so the next round's pages land on the MoE
        chips together again and the chips may take them in different orders. On the mixed-rows server the model runs on
        (``rows=1, block_rows=4``). Every output must match the same request alone bit for bit. A hang here (the timeout) is
        the bug; the parametrizations say what it needs. ``post`` has every request run the hyper-connection post step fused
    (streams, post and comb, as every model request does), its output the new streams."""
    if os.environ.get("TT_FE_FIXED_ORDER") == "1":
        pytest.skip("TT_FE_FIXED_ORDER=1 takes the requesters in strict turn, not at once")
    moe_mesh, request_meshes, config, socket_core, single_senders = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    num_requesters, num_layers = len(request_meshes), len(layers)
    g = torch.Generator().manual_seed(0)

    def block():
        b = torch.randn(rows, config.hidden_size, generator=g)
        return (b / b.norm(dim=-1, keepdim=True) * config.hidden_size**0.5).to(torch.bfloat16).float()

    shared = block()
    blocks = [shared if identical else block() for _ in range(num_requesters)]
    layer_of = lambda round_, j: round_ % num_layers if same_layer else (round_ + j) % num_layers  # noqa: E731

    moe_op = PersistentMoE(
        moe_mesh, request_meshes, moes, socket_core, socket_core, single_senders, rows=1, block_rows=4
    )
    moe_op.use_rows(rows)
    if post and not moe_op.fuse_hc:
        pytest.skip("the HC post step is not fused (DEEPSEEK_V41_MOE_FUSE_HC=0)")
    hc = config.hc_mult
    inputs, hc_args = [], []
    for j in range(num_requesters):
        tensor = moe_op.request_input(j)
        moe_op.upload(j, blocks[j], tensor)
        inputs.append(tensor)
        if post:
            mesh = request_meshes[j]
            streams = (
                torch.randn(hc, config.hidden_size, generator=g)
                if rows == 1
                else torch.randn(hc, rows, config.hidden_size, generator=g)
            )

            def tile(t):
                return ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )

            hc_args.append(
                {
                    "streams": moe_op.block_streams(mesh, streams.to(torch.bfloat16).float()),
                    "post": tile(torch.rand(1, rows, hc, 1, generator=g)),
                    "comb": tile(torch.rand(1, rows, hc, hc, generator=g) / hc),
                }
            )
        else:
            hc_args.append({})

    def result(out):
        got = PersistentMoE.row_of(out).float() if post else _output(out, rows)
        ttnn.deallocate(out)
        return got

    solo = {}
    try:
        # The references: each (requester, layer) alone.
        for j in range(num_requesters):
            for layer in range(num_layers):
                solo[j, layer] = result(moe_op.request(j, layer, inputs[j], **hc_args[j]))
        logger.info(
            f"{num_requesters} requesters (rows {REQUEST_ROWS}), {rows}-row requests, "
            f"{'the same layer' if same_layer else 'different layers'}, "
            f"{'identical' if identical else 'their own'} blocks, {'HC post fused' if post else 'plain outputs'}: "
            f"{LOCKSTEP_ROUNDS} rounds all at once"
        )
        t0 = time.perf_counter()
        for round_ in range(LOCKSTEP_ROUNDS):
            outs = [moe_op.request(j, layer_of(round_, j), inputs[j], **hc_args[j]) for j in range(num_requesters)]
            for j, out in enumerate(outs):
                got = result(out)
                assert torch.equal(
                    got, solo[j, layer_of(round_, j)]
                ), f"round {round_}: requester {j} layer {layer_of(round_, j)} differs from the same request alone"
        logger.info(
            f"{LOCKSTEP_ROUNDS} lockstep rounds ok ({1e6 * (time.perf_counter() - t0) / LOCKSTEP_ROUNDS:.0f} us each)"
        )
    finally:
        moe_op.stop()
