# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/row_exchange``: the hybrid MoE row exchange (gather / reduce) as one fabric program per direction, on
every chip of a mesh: per-chip exactness against torch (the gather is data movement; the reduce sums in fp32 and rounds
once), the same after trace replays (the counters reset in-program), and the op's traced wall time.

    # the 8x4 (default GLM53_MESH_SHAPE=8,4, T = 4 users per row: the served decode point)
    pytest models/demos/glm53_flash/tests/custom_ops/test_row_exchange.py -s
    # the (2,4) block, 4 users per row (8 real rows, padding rows zero)
    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 \\
      TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
      pytest models/demos/glm53_flash/tests/custom_ops/test_row_exchange.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.row_exchange.op import TILE, RowExchange
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

ROWS, COLS = (int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("GLM53_ROWX_T", "4"))
H, E = 4096, 288
N_TIME = int(os.environ.get("GLM53_ROWX_STEPS", "20"))
MESH_PARAMS = [
    pytest.param(mesh_device_params(ROWS, COLS, trace_region_size=64_000_000), id=f"{ROWS}x{COLS}"),
]


def _mapper(mesh, h: torch.Tensor):
    """The mesh placement of a host tensor shaped [ROWS, 1, 32, W] (row shards, replicated over the columns) or
    [ROWS, COLS, 32, W] (one shard per chip): the GLMMesh upload's own mappers."""
    return mesh.shard_2d(0, 1 if (h.shape[1] == COLS and COLS > 1) else None)


def _chips(t: ttnn.Tensor) -> list[torch.Tensor]:
    """Per chip, the logical rows of ``t`` (the ops allocate T or rows * T logical rows, tile-padded)."""
    return [ttnn.to_torch(d).float().reshape(t.shape[-2], -1) for d in ttnn.get_device_tensors(t)]


def _replay_with_new_inputs(mesh, op, inputs, outputs, hosts, expect, rounds: int = 3) -> None:
    """A stale-staging race is invisible when every replay carries the same data: capture one call, then for a few
    rounds write fresh host data into the input buffers, replay, and check the outputs against ``expect(hosts)``
    (``hosts``: the per-input host tensors in the mesh placement the inputs were uploaded with; ``expect``: host
    tensors -> the per-chip expected outputs, one list per output)."""
    mesh_device = mesh.device
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op(inputs, outputs)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    g = torch.Generator().manual_seed(7)
    for rnd in range(rounds):
        fresh = [(torch.randn(h.shape, generator=g) * (h != 0)).bfloat16().float() for h in hosts]
        for dev, h in zip(inputs, fresh):
            host = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_mapper(mesh, h))
            ttnn.copy_host_to_device_tensor(host, dev)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        for out, exp_chips in zip(outputs, expect(fresh)):
            for i, (got, exp) in enumerate(zip(_chips(out), exp_chips)):
                err = (got - exp).abs()
                tol = exp.abs() * (1 / 128) + 1e-3
                assert bool(
                    (err <= tol).all()
                ), f"round {rnd} chip {i}: max err {err.max().item():.4f} (stale staging?)"
    ttnn.release_trace(mesh_device, tid)


def _timed(mesh_device, op, inputs, outputs) -> tuple[float, float]:
    """(eager wall, traced replay wall) per call in us; the outputs hold the last replay's result."""
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(N_TIME):
        op(inputs, outputs)
    ttnn.synchronize_device(mesh_device)
    eager = (time.perf_counter() - t0) / N_TIME * 1e6
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op(inputs, outputs)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(N_TIME):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    traced = (time.perf_counter() - t0) / N_TIME * 1e6
    ttnn.release_trace(mesh_device, tid)
    return eager, traced


@pytest.mark.parametrize("device_params", MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_gather(mesh_device):
    mesh = GLMMesh.from_device(mesh_device)
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(ROWS, 1, TILE, H)
    x[:, :, :T] = torch.randn(ROWS, 1, T, H, generator=g)
    w = torch.zeros(ROWS, 1, TILE, E)
    w[:, :, :T] = torch.rand(ROWS, 1, T, E, generator=g)
    x, w = x.bfloat16().float(), w.bfloat16().float()
    xd, wd = mesh.upload(x, ttnn.bfloat16, rows=0), mesh.upload(w, ttnn.bfloat16, rows=0)
    op = RowExchange(mesh, "gather", [H, E], T)
    logger.info(
        f"gather on {ROWS}x{COLS}, T={T}: {op.nchunks} chunks, piece slot {op.ps} B, {op.cpp} per packet, "
        f"{op.np} packets per peer, groups per chip {[len(v) for v in op.groups.values()][:4]}..."
    )
    xg, wg = op([xd, wd])
    exp_x = x[:, 0, :T].reshape(ROWS * T, H)
    exp_w = w[:, 0, :T].reshape(ROWS * T, E)
    for i, (cx, cw) in enumerate(zip(_chips(xg), _chips(wg))):
        assert torch.equal(cx, exp_x), f"chip {i}: x gather differs ({(cx != exp_x).sum().item()} elements)"
        assert torch.equal(cw, exp_w), f"chip {i}: routing gather differs ({(cw != exp_w).sum().item()} elements)"
    _replay_with_new_inputs(
        mesh,
        op,
        [xd, wd],
        [xg, wg],
        [x, w],
        lambda hs: [
            [hs[0][:, 0, :T].reshape(ROWS * T, H)] * (ROWS * COLS),
            [hs[1][:, 0, :T].reshape(ROWS * T, E)] * (ROWS * COLS),
        ],
    )
    eager, traced = _timed(mesh_device, op, [xd, wd], [xg, wg])
    for i, (cx, cw) in enumerate(zip(_chips(xg), _chips(wg))):
        assert torch.equal(cx, exp_x) and torch.equal(cw, exp_w), f"chip {i}: differs after the trace replays"
    logger.info(f"ROWX gather {ROWS}x{COLS} T={T}: eager {eager:.0f} us, traced {traced:.0f} us per call")


@pytest.mark.parametrize("device_params", MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_reduce(mesh_device):
    mesh = GLMMesh.from_device(mesh_device)
    g = torch.Generator().manual_seed(1)
    y = torch.zeros(ROWS, 1, TILE, H)
    y[:, :, : ROWS * T] = torch.randn(ROWS, 1, ROWS * T, H, generator=g)
    y = y.bfloat16().float()
    yd = mesh.upload(y, ttnn.bfloat16, rows=0)
    op = RowExchange(mesh, "reduce", [H], T)
    (out,) = op([yd])
    chips = _chips(out)
    for r in range(ROWS):
        exp = y[:, 0, r * T : (r + 1) * T].sum(dim=0).bfloat16().float()  # over the source rows, fp32
        for c in range(COLS):
            got = chips[r * COLS + c]
            err = (got - exp).abs()
            tol = exp.abs() * (1 / 128) + 1e-3  # one bf16 ulp: the device's fp32 order may differ from torch's
            assert bool((err <= tol).all()), f"chip ({r},{c}): max err {err.max().item():.4f}"
    _replay_with_new_inputs(
        mesh,
        op,
        [yd],
        [out],
        [y],
        lambda hs: [
            [hs[0][:, 0, r * T : (r + 1) * T].sum(dim=0).bfloat16().float() for r in range(ROWS) for _ in range(COLS)]
        ],
    )
    eager, traced = _timed(mesh_device, op, [yd], [out])
    chips2 = _chips(out)
    for i, (a, b) in enumerate(zip(chips, chips2)):
        assert torch.equal(a, b), f"chip {i}: differs after the trace replays"
    logger.info(f"ROWX reduce {ROWS}x{COLS} T={T}: eager {eager:.0f} us, traced {traced:.0f} us per call")


@pytest.mark.parametrize("device_params", MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_allreduce(mesh_device):
    """The TP all-reduce over the columns of every mesh row: per chip the sum of its row's COLS partials."""
    mesh = GLMMesh.from_device(mesh_device)
    if COLS < 2 or COLS * T > TILE:
        pytest.skip(f"the all-reduce needs 2..{TILE // T} columns at T={T}")
    g = torch.Generator().manual_seed(2)
    y = torch.zeros(ROWS, COLS, TILE, H)
    y[:, :, :T] = torch.randn(ROWS, COLS, T, H, generator=g)
    y = y.bfloat16().float()
    yd = mesh.upload(y, ttnn.bfloat16, rows=0, cols=1)  # chip (r, c) holds y[r, c]
    op = RowExchange(mesh, "allreduce", [H], T)
    logger.info(
        f"allreduce on {ROWS}x{COLS}, T={T}: {op.nchunks} chunks, {op.cpo} per owner, {op.np} packets per peer per phase"
    )
    (out,) = op([yd])
    chips = _chips(out)
    for r in range(ROWS):
        exp = y[r, :, :T].sum(dim=0).bfloat16().float()
        for c in range(COLS):
            got = chips[r * COLS + c]
            err = (got - exp).abs()
            tol = exp.abs() * (1 / 128) + 1e-3
            assert bool((err <= tol).all()), f"chip ({r},{c}): max err {err.max().item():.4f}"
    _replay_with_new_inputs(
        mesh,
        op,
        [yd],
        [out],
        [y],
        lambda hs: [[hs[0][r, :, :T].sum(dim=0).bfloat16().float() for r in range(ROWS) for _ in range(COLS)]],
    )
    eager, traced = _timed(mesh_device, op, [yd], [out])
    chips2 = _chips(out)
    for i, (a, b) in enumerate(zip(chips, chips2)):
        assert torch.equal(a, b), f"chip {i}: differs after the trace replays"
    logger.info(f"ROWX allreduce {ROWS}x{COLS} T={T}: eager {eager:.0f} us, traced {traced:.0f} us per call")


@pytest.mark.parametrize("device_params", MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_model_sequence(mesh_device):
    """The decode MoE sequence under one trace with fresh data every replay: fabric gather -> a per-row scaled
    "expert" (eltwise) -> fabric combine -> add -> fabric TP all-reduce, checked against torch per chip and round.
    Isolates the chain gate's decode drift (gate84_B4_fabcombine / fabar) from the model's other ops."""
    mesh = GLMMesh.from_device(mesh_device)
    if COLS < 2 or COLS * T > TILE or ROWS * T > TILE:
        pytest.skip("needs the (8,4) decode shapes")
    g = torch.Generator().manual_seed(3)
    x = torch.zeros(ROWS, 1, TILE, H)
    x[:, :, :T] = torch.randn(ROWS, 1, T, H, generator=g)
    x = x.bfloat16().float()
    w = torch.zeros(ROWS, 1, TILE, E)
    w[:, :, :T] = torch.rand(ROWS, 1, T, E, generator=g)
    w = w.bfloat16().float()
    scale_host = (
        torch.tensor([0.5 + 0.25 * r for r in range(ROWS)]).view(ROWS, 1, 1, 1).expand(ROWS, 1, TILE, H).contiguous()
    )
    xd, wd = mesh.upload(x, ttnn.bfloat16, rows=0), mesh.upload(w, ttnn.bfloat16, rows=0)
    scale = mesh.upload(scale_host, ttnn.bfloat16, rows=0)  # row r's chips scale by 0.5 + 0.25 r
    gather = RowExchange(mesh, "gather", [H, E], T)
    reduce = RowExchange(mesh, "reduce", [H], T)
    allred = RowExchange(mesh, "allreduce", [H], T)

    def step():
        xg, wg = gather([xd, wd])
        y = ttnn.multiply(xg, scale)  # [1,1,32,H]: this chip's "expert" partial for every gathered user
        (comb,) = reduce([y])  # [1,1,T,H]: this row's users summed over the rows
        comb2 = ttnn.add(comb, comb)  # a stand-in for the shared expert add (keeps the shape [1,1,T,H])
        (out,) = allred([comb2])
        return out

    def expect(xh):
        # per row r: comb[t] = sum_r' scale_r' * x[r][t]; comb2 = 2 comb; all-reduce over 4 equal columns = 4 comb2
        s = sum(0.5 + 0.25 * r2 for r2 in range(ROWS))
        return [(8 * s * xh[r, 0, :T]).bfloat16().float() for r in range(ROWS) for _ in range(COLS)]

    out = step()
    for i, (got, exp) in enumerate(zip(_chips(out), expect(x))):
        err = (got - exp).abs()
        tol = exp.abs() * (3 / 128) + 1e-2  # three bf16 roundings along the chain
        assert bool((err <= tol).all()), f"eager chip {i}: max err {err.max().item():.4f}"
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = step()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    g2 = torch.Generator().manual_seed(11)
    for rnd in range(4):
        fresh = torch.zeros(ROWS, 1, TILE, H)
        fresh[:, :, :T] = torch.randn(ROWS, 1, T, H, generator=g2)
        fresh = fresh.bfloat16().float()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(fresh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh.shard_2d(0, None)), xd
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        for i, (got, exp) in enumerate(zip(_chips(out), expect(fresh))):
            err = (got - exp).abs()
            tol = exp.abs() * (3 / 128) + 1e-2
            assert bool(
                (err <= tol).all()
            ), f"round {rnd} chip {i}: max err {err.max().item():.4f} (stale data in the chain?)"
    ttnn.release_trace(mesh_device, tid)
    logger.info(
        f"ROWX sequence {ROWS}x{COLS} T={T}: gather -> scale -> combine -> add -> all-reduce exact over 4 fresh replays"
    )
