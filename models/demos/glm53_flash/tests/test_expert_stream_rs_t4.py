# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode ledger C1 follow-up: the EP reduce-scatter combine (``GLM53_EP_COMBINE=rs``) at T % 32 != 0 users per mesh
row through the down writer's chunked output layout (``GLM53_EP_RS_REMAP=1``: ``custom_ops/expert_stream/op.py``
``ep_rs_chunk``, ``kernels/writer_down_dispatch.cpp``).

1. ``test_down_writer_row_chunks`` (one chip, after tests/custom_ops/test_expert_dispatch.py::test_expert_dispatch_ep_rm):
   the EP row-major dispatch ops with an explicit ``out_chunk`` vs the plain layout on the same inputs: chunk r rows
   0..chunk-1 are the plain rows r*chunk.. byte-exact, the pad rows exact zeros, the plain result at the golden's
   PCC; and the env-derived chunk on a 1-row mesh.
2. ``test_ep_moe_rs_remap_vs_op`` (a (2, 4) block, after tests/test_mesh_moe.py and tests/test_a2a_perf.py): the EP
   MoE layer at 4 (and 2) users per row with real layer-4 weights: ``out`` of the remapped run vs the plain run per
   chip (placement, zeros), the reduce-scatter combine vs the op combine (PCC >= 0.9999), the whole forward vs HF,
   the columns byte-identical, no L1 left behind, and the forward captured in a trace and replayed.

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 GLM53_MOE_LAYOUT=ep pytest models/demos/glm53_flash/tests/test_expert_stream_rs_t4.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.expert_stream.op import (
    ExpertDispatch,
    ep_base_tensor,
    ep_rm_inputs,
    ep_rs_chunk,
    expert_dispatch_golden_ep,
    mesh_rows,
)
from models.demos.glm53_flash.tests.custom_ops.test_expert_dispatch import DOWN_READERS, GROUPS
from models.demos.glm53_flash.tests.custom_ops.test_expert_dispatch import upload as upload_random_experts
from models.demos.glm53_flash.tests.test_mesh_moe import l1_high_water, l1_resident
from models.demos.glm53_flash.tests.test_moe import LAYER, realistic_mlp_input, weights  # noqa: F401 (fixture)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params
from models.demos.glm53_flash.tt.moe import TtMoE

ROWS, COLS = 2, 4
H = 4096


def _routing(t_valid: int, E_total: int, E_local: int, base: int, K: int, seed: int):
    """[t_valid, K] distinct global ids per token (the kernels count one slot per (token, expert)): every third token
    has no local expert (an exact zero output row), the others exactly one of [base, base + E_local); positive
    bf16-representable weights."""
    g = torch.Generator().manual_seed(seed)
    non_local = torch.tensor([e for e in range(E_total) if not base <= e < base + E_local])
    idx = torch.zeros(t_valid, K, dtype=torch.int64)
    for i in range(t_valid):
        pick = non_local[torch.randperm(len(non_local), generator=g)[:K]]
        if i % 3 != 2:
            pick[0] = base + (i * 7) % E_local
        idx[i] = pick
    wts = (torch.rand(t_valid, K, generator=g) + 0.2).bfloat16().float()
    return idx, wts


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "t_valid, chunk",
    [(8, 4), (32, 4), (4, 2)],
    ids=["block-4-per-row", "8x4-4-per-row", "block-2-per-row"],
)
def test_down_writer_row_chunks(mesh_device, t_valid, chunk, monkeypatch):
    """``t_valid = rows * chunk`` dispatched rows (the (2,4) block / the 8x4 at 4 users per row, the block at 2) on one
    chip: the same inputs through the plain layout and through ``out_chunk`` tokens per 32-row chunk."""
    monkeypatch.delenv("GLM53_EP_RS_REMAP", raising=False)
    I, E_total, E_local, base, K, T = 2048, 288, 36, 72, 8, 32
    w13, w2, w13_q, w2_q = upload_random_experts(mesh_device, E_local, I, seed=base)
    op = ExpertDispatch(
        mesh_device, E_local, 4096, I, H, T, groups=GROUPS, down_readers=DOWN_READERS, ep=True, ep_rm=True
    )
    x = torch.randn(t_valid, 4096).bfloat16()
    idx, wts = _routing(t_valid, E_total, E_local, base, K, seed=t_valid * 8 + chunk)
    _, y_ref, passes = expert_dispatch_golden_ep(x, idx, wts, base, E_local, w13_q, w2_q)
    xw, meta = ep_rm_inputs(x, idx, wts, mesh_device)
    bt = ep_base_tensor(base, mesh_device)
    # the plain layout: [1, 1, t_valid, H], dispatched row i at tile i // 32, row i % 32
    act, table, out = op.alloc(mesh_device, t_valid=t_valid)
    assert tuple(out.shape) == (1, 1, t_valid, H), tuple(out.shape)
    op.forward(xw, xw, meta, w13, w2, act, table, out, base=bt, t_valid=t_valid)
    y_plain = ttnn.to_torch(out).float().reshape(t_valid, H)
    p = pcc(y_ref, y_plain)
    # the chunked layout: [1, 1, n_chunks * 32, H], dispatched row i at chunk i // chunk, row i % chunk
    n_chunks = -(-t_valid // chunk)
    act2, table2, out2 = op.alloc(mesh_device, t_valid=t_valid, out_chunk=chunk)
    assert tuple(out2.shape) == (1, 1, n_chunks * 32, H), tuple(out2.shape)
    op.forward(xw, xw, meta, w13, w2, act2, table2, out2, base=bt, t_valid=t_valid, out_chunk=chunk)
    y_chunks = ttnn.to_torch(out2).float().reshape(n_chunks, 32, H)
    for r in range(n_chunks):
        n = min(chunk, t_valid - r * chunk)
        assert torch.equal(y_chunks[r, :n], y_plain[r * chunk : r * chunk + n]), f"chunk {r}: rows differ"
        assert not y_chunks[r, n:].any(), f"chunk {r}: pad rows not zero"
    routed = [i for i in range(t_valid) if i % 3 != 2]
    unrouted = [i for i in range(t_valid) if i % 3 == 2]
    assert not y_plain[unrouted].any(), "tokens without a local expert must be exact zero rows"
    assert bool(y_plain[routed].abs().sum(-1).gt(0).all()), "routed tokens must have non-zero rows"
    logger.info(
        f"t_valid {t_valid}, chunk {chunk}: {len(passes)} passes, plain out pcc {p:.6f} vs the golden; "
        f"{n_chunks} chunks placed byte-exact, pad rows zero"
    )
    assert p > 0.9999, p
    # the env-derived chunk: a 1-row mesh (this chip) has T_users = t_valid, remapped only below a tile
    assert mesh_rows(mesh_device) == 1
    assert ep_rs_chunk(mesh_device, t_valid) is None  # knob off
    monkeypatch.setenv("GLM53_EP_RS_REMAP", "1")
    assert ep_rs_chunk(mesh_device, t_valid) == (t_valid if t_valid % 32 else None)
    act3, table3, out3 = op.alloc(mesh_device, t_valid=t_valid)
    assert tuple(out3.shape) == (1, 1, 32 if t_valid % 32 else t_valid, H), tuple(out3.shape)
    for t in (act, table, out, act2, table2, out2, act3, table3, out3, xw, meta):
        ttnn.deallocate(t)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(ROWS, COLS, fabric_config=ttnn.FabricConfig.FABRIC_1D, trace_region_size=200_000_000)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("rows_T", [4, 2])
def test_ep_moe_rs_remap_vs_op(mesh_device, weights, rows_T, monkeypatch):
    """The EP MoE layer on the block at ``rows_T`` users per row (4 = the 8x4's 32-user point): the op combine on the
    plain expert output (today's path below a tile) against the reduce-scatter combine on the remapped output."""
    monkeypatch.delenv("GLM53_EP_RS_REMAP", raising=False)
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mlp = {k[len("mlp.") :]: v for k, v in raw.items() if "experts." not in k or "shared_experts" in k}
    mlp["experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    mlp["experts.down_proj"] = sd["experts.down_proj"]
    mesh = GLMMesh.from_device(mesh_device)
    t0 = time.perf_counter()
    moe = TtMoE(mesh_device, cfg, mlp, mesh=mesh, layout="ep")
    ex = moe.experts
    logger.info(f"EP MoE layer {LAYER} on {ROWS}x{COLS}: built in {time.perf_counter() - t0:.0f} s")
    x_all = realistic_mlp_input(snapshot, text, raw, ROWS * 32).reshape(1, 1, ROWS * 32, -1)
    xs = torch.cat([x_all[:, :, 32 * r : 32 * r + rows_T] for r in range(ROWS)], dim=2)  # row r: rows_T tokens
    with torch.no_grad():
        y_hf = hf(xs.reshape(1, ROWS * rows_T, -1)).reshape(1, 1, ROWS * rows_T, -1)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)  # [1,1,T,H] per chip, replicated over the columns
    w16, idx, _ = moe.router(tt_x, ep_rows=True)
    tok, meta = ex.dispatch_rows(tt_x, w16, idx)
    Td = tok.shape[1]
    assert Td == ROWS * rows_T, (Td, ROWS, rows_T)
    # ---- the op combine on the plain layout (the reference)
    ex.combine_mode = "op"
    out_op = ex.run_experts(tok, meta)
    assert tuple(out_op.shape) == (1, 1, Td, H), tuple(out_op.shape)
    y_op = ex.combine_rows(out_op, meta, rows_T)
    # ---- the reduce-scatter combine on the remapped layout
    monkeypatch.setenv("GLM53_EP_RS_REMAP", "1")
    ex.combine_mode = "rs"
    out_rs = ex.run_experts(tok, meta)
    assert tuple(out_rs.shape) == (1, 1, ROWS * 32, H), tuple(out_rs.shape)
    for i, (d_op, d_rs) in enumerate(zip(mesh.device_tensors(out_op), mesh.device_tensors(out_rs))):
        a = ttnn.to_torch(d_op).reshape(ROWS, rows_T, H)
        b = ttnn.to_torch(d_rs).reshape(ROWS, 32, H)
        assert torch.equal(b[:, :rows_T], a), f"chip {i}: chunk rows differ from the plain layout"
        assert not b[:, rows_T:].any(), f"chip {i}: pad rows not zero"
    y_rs = ex.combine_rows(out_rs, meta, rows_T)
    assert tuple(y_rs.shape) == tuple(y_op.shape) == (1, 1, rows_T, H), (tuple(y_rs.shape), tuple(y_op.shape))
    worst = 1.0
    for a, b in zip(mesh.device_tensors(y_op), mesh.device_tensors(y_rs)):
        worst = min(worst, pcc(ttnn.to_torch(a).float(), ttnn.to_torch(b).float()))
    logger.info(f"T={rows_T}/row: reduce-scatter combine (remapped out) vs op combine: worst pcc {worst:.6f}")
    assert worst >= 0.9999, worst
    for t in (w16, idx, tok, meta, out_op, y_op, out_rs, y_rs):
        ttnn.deallocate(t)
    # ---- the whole forward under rs + remap vs HF (as tests/test_mesh_moe.py), columns identical, no L1 left
    l1_marks = [l1_high_water(mesh_device)]
    shards = []
    for it in range(2):
        y = moe(tt_x)
        ttnn.synchronize_device(mesh_device)
        shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y)]
        worst = 1.0
        for r in range(ROWS):
            for c in range(COLS):
                worst = min(worst, pcc(y_hf[:, :, r * rows_T : (r + 1) * rows_T], shards[r * COLS + c][:, :, :rows_T]))
                if c:
                    assert torch.equal(shards[r * COLS + c], shards[r * COLS]), (r, c)
        logger.info(f"EP MoE (rs + remap) T={rows_T}/row iter {it}: worst pcc vs HF {worst:.6f}")
        assert worst > 0.995, worst
        ttnn.deallocate(y)
        l1_marks.append(l1_high_water(mesh_device))
    assert l1_marks[2] == l1_marks[1], f"the forward leaks {l1_marks[1] - l1_marks[2]} B of L1 per call"
    resident = l1_resident(moe, mesh)
    assert not resident, resident
    # ---- the forward as a captured trace, replayed (the layout and combine decisions are static per shape)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_tr = moe(tt_x)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    tr = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y_tr)]
    worst = min(pcc(a, b) for a, b in zip(shards, tr))
    same = all(torch.equal(a, b) for a, b in zip(shards, tr))
    logger.info(f"T={rows_T}/row: traced forward vs eager: worst pcc {worst:.6f}, byte-identical {same}")
    assert worst > 0.9999, worst
    ttnn.release_trace(mesh_device, tid)
