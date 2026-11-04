# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/moe_persistent``: the hybrid MoE block (row gather -> fused experts -> row combine) as ONE persistent
program per chip, on the (8,4) mesh at the model's decode shapes (T users per row, 288 experts over the 4 columns,
I_loc = 256): construction, bit-exactness against the three-op sequence (RowExchange gather, ExpertStreamFused,
RowExchange reduce), the torch golden, traced replays with fresh inputs, and the traced wall time of both forms.
    pytest models/demos/glm53_flash/tests/custom_ops/test_moe_persistent.py -s
"""
from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.expert_stream.op import (
    ExpertStream,
    bfp8_round,
    down_stream_layout,
    expert_stream_golden,
    fold_base_host,
    gate_up_stream_layout,
    upload_stream_weights,
)
from models.demos.glm53_flash.custom_ops.expert_stream.plan import e1_stream_knobs
from models.demos.glm53_flash.custom_ops.moe_persistent.op import MoePersistent, persistent_fused_groups
from models.demos.glm53_flash.custom_ops.row_exchange.op import TILE, RowExchange
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

ROWS, COLS = (int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("GLM53_ROWX_T", "4"))
E, K, I, H = 288, 4096, 256, 4096  # the model: 288 experts over the columns, the down's I slice over the rows
E_LOC = E // COLS
PICKS = 8
N_TIME = int(os.environ.get("GLM53_ROWX_STEPS", "20"))
MESH_PARAMS = [
    pytest.param(mesh_device_params(ROWS, COLS, trace_region_size=64_000_000), id=f"{ROWS}x{COLS}"),
]


def _chips(t: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float().reshape(t.shape[-2], -1) for d in ttnn.get_device_tensors(t)]


def _routing(rows: int, seed: int) -> torch.Tensor:
    """[rows, 1, 32, E] bf16: every row's T users pick PICKS random experts with random positive weights."""
    g = torch.Generator().manual_seed(seed)
    r = torch.zeros(rows, 1, TILE, E)
    for row in range(rows):
        for t in range(T):
            ids = torch.randperm(E, generator=g)[:PICKS]
            r[row, 0, t, ids] = torch.rand(PICKS, generator=g) + 0.2
    return r.bfloat16().float()


def _x(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(rows, 1, TILE, H)
    x[:, :, :T] = torch.randn(rows, 1, T, H, generator=g)
    return x.bfloat16().float()


@pytest.fixture(scope="module")
def weights():
    """Per column c its 72 experts' slices (random; the rows share them: the test checks the program, the row
    selection of the reduce is test_row_exchange's)."""
    g = torch.Generator().manual_seed(0)
    w13 = (torch.randn(COLS, E_LOC, K, 2 * I, generator=g) * 0.02).bfloat16()
    w2 = (torch.randn(COLS, E_LOC, I, H, generator=g) * 0.02).bfloat16()
    return w13, w2


def _expected(x: torch.Tensor, r: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor) -> list[list[torch.Tensor]]:
    """Per chip (row-major), the combined output of its row's T users: every chip of column c computes the gathered
    32 users against column c's experts; the reduce sums the column's ROWS identical partials."""
    xg = x[:, 0, :T].reshape(ROWS * T, H)
    rg = r[:, 0, :T].reshape(ROWS * T, E)
    y_col = []
    for c in range(COLS):
        _, y = expert_stream_golden(
            xg.bfloat16(),
            rg[:, c * E_LOC : (c + 1) * E_LOC].bfloat16(),
            bfp8_round(w13[c].float()),
            bfp8_round(w2[c].float()),
        )
        y_col.append(y)
    out = []
    for row in range(ROWS):
        for c in range(COLS):
            out.append((ROWS * y_col[c][row * T : (row + 1) * T]).bfloat16().float())
    return out


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_moe_persistent(mesh_device, weights):
    mesh = GLMMesh.from_device(mesh_device)
    if COLS < 2 or COLS * T > TILE or ROWS * T > TILE:
        pytest.skip("needs the (8,4) decode shapes")
    device = mesh.device
    w13_host, w2_host = weights
    col_mapper = mesh.shard_2d(None, 0)  # one column slice per column, replicated over the rows
    w13 = upload_stream_weights(
        device, torch.stack([gate_up_stream_layout(w.float()) for w in w13_host]), mesh_mapper=col_mapper
    )
    w2 = upload_stream_weights(
        device, torch.stack([down_stream_layout(w.float()) for w in w2_host]), mesh_mapper=col_mapper
    )
    knobs = e1_stream_knobs(os.environ)
    assert knobs["fused"] and knobs["routing_fold"], knobs
    groups = persistent_fused_groups(device, E_LOC, I, H)
    split_k = os.environ.get("GLM53_STREAM_SPLITK", "0") == "1"
    op = ExpertStream(
        device,
        E_LOC,
        K,
        I,
        H,
        fused=True,
        fused_groups=groups,
        routing_fold=True,
        scan_words=knobs["scan_words"],
        split_k=split_k,
    )
    logger.info(f"expert op: {groups} groups, split_k={split_k}")
    base = mesh.upload(fold_base_host(COLS, E_LOC), ttnn.uint16, cols=1, layout=ttnn.ROW_MAJOR_LAYOUT)
    x_host, r_host = _x(ROWS, 1), _routing(ROWS, 2)
    xd, rd = mesh.upload(x_host, ttnn.bfloat16, rows=0), mesh.upload(r_host, ttnn.bfloat16, rows=0)
    expect = _expected(x_host, r_host, w13_host, w2_host)

    # the three-op sequence (today's fabric path)
    gather = RowExchange(mesh, "gather", [H, E], T)
    reduce = RowExchange(mesh, "reduce", [H], T)
    xg, wg = gather([xd, rd], memory_config=ttnn.L1_MEMORY_CONFIG)
    act, out = op.alloc(device, act_l1=True)
    op.forward(xg, wg, w13, w2, act, out, base)
    (comb_ref,) = reduce([out])
    ref = _chips(comb_ref)
    for d, (got, exp) in enumerate(zip(ref, expect)):
        p = pcc(exp[:T], got[:T])
        assert p > 0.999, ("three-op sequence vs golden", d, p)

    # the persistent program
    t0 = time.perf_counter()
    pop = MoePersistent(mesh, op, T, H, E)
    logger.info(
        f"MoePersistent built in {time.perf_counter() - t0:.1f} s: {groups} groups, exchange core {pop.xchg_core}, "
        f"workers {[(w.x, w.y) for w in pop.gather.workers]}"
    )
    xg2, wg2 = pop.gather.allocate_outputs(ttnn.L1_MEMORY_CONFIG)
    act2, out2 = op.alloc(device, act_l1=True)
    (comb,) = pop.reduce.allocate_outputs(ttnn.DRAM_MEMORY_CONFIG)
    pop(xd, rd, w13, w2, act2, out2, base, xg2, wg2, comb)
    got = _chips(comb)
    for d in range(len(got)):
        assert torch.equal(got[d][:T], ref[d][:T]), (
            "persistent != three-op sequence",
            d,
            (got[d][:T] != ref[d][:T]).sum(),
        )
    logger.info("persistent program == three-op sequence on every chip (eager)")

    # traced: replays with fresh inputs, then the wall time of both forms
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    pop(xd, rd, w13, w2, act2, out2, base, xg2, wg2, comb)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for rnd in range(3):
        x_new, r_new = _x(ROWS, 10 + rnd), _routing(ROWS, 20 + rnd)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x_new, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh.shard_2d(0, None)), xd
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(r_new, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh.shard_2d(0, None)), rd
        )
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        got = _chips(comb)
        exp = _expected(x_new, r_new, w13_host, w2_host)
        for d in range(len(got)):
            p = pcc(exp[d][:T], got[d][:T])
            assert p > 0.999, ("traced replay vs golden", rnd, d, p)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N_TIME):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    per = (time.perf_counter() - t0) / N_TIME * 1e3
    ttnn.release_trace(device, tid)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    gather([xd, rd], [xg, wg])
    op.forward(xg, wg, w13, w2, act, out, base)
    reduce([out], [comb_ref])
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(N_TIME):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    per3 = (time.perf_counter() - t0) / N_TIME * 1e3
    ttnn.release_trace(device, tid)
    logger.info(
        f"MOE_PERSISTENT traced: persistent {per:.3f} ms / replay, three ops {per3:.3f} ms / replay (T={T}, {ROWS}x{COLS})"
    )
