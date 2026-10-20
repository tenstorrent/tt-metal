# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""custom_ops/expert_stream phase 2: the per-expert dispatch ops (ExpertDispatch) for T gathered tokens (every
active expert's weights read once; on-device gather of each expert's routed rows; token sums from gathered slots),
vs the torch golden on bfp8-rounded weights, and as a captured trace replayed (device time per call).

    TT_VISIBLE_DEVICES=2 pytest models/demos/glm53_flash/tests/custom_ops/test_expert_dispatch.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.expert_stream.op import (
    ExpertDispatch,
    bfp4_round,
    bfp8_round,
    down_stream_layout,
    ep_base_tensor,
    ep_index_tiles,
    expert_dispatch_golden,
    expert_dispatch_golden_ep,
    gate_up_stream_layout,
    upload_stream_weights,
)
from models.demos.glm53_flash.tests.utils import pcc

K, H = 4096, 4096
GROUPS = int(os.environ.get("GLM53_STREAM_GROUPS", "12"))
DOWN_READERS = int(os.environ.get("GLM53_DISPATCH_DOWN_READERS", "8"))


def skewed_routing(tokens: int, E: int, picks: int, seed: int, popular: int = 0) -> torch.Tensor:
    """[tokens, E] bf16: `picks` experts per token drawn from a Zipf-like preference (as the real router: ~30 of 72
    active at 256 tokens, a few experts with many rows), plus `popular` experts routed by every token (multi-pass)."""
    g = torch.Generator().manual_seed(seed)
    pref = 1.0 / (torch.arange(E, dtype=torch.float) + 1.0) ** 1.3
    pref = pref[torch.randperm(E, generator=g)]
    r = torch.zeros(tokens, E)
    for t in range(tokens):
        ids = torch.multinomial(pref, picks, replacement=False, generator=g)
        r[t, ids] = torch.rand(picks, generator=g) + 0.2
    for e in range(popular):
        r[:, e] = torch.rand(tokens, generator=g) + 0.2
    return r.bfloat16()


def upload(mesh_device, E, I, seed=0, w13_dtype=ttnn.bfloat8_b):
    g = torch.Generator().manual_seed(seed)
    w13_host = (torch.randn(E, K, 2 * I, generator=g) * 0.02).bfloat16()
    w2_host = (torch.randn(E, I, H, generator=g) * 0.02).bfloat16()
    w13 = upload_stream_weights(mesh_device, gate_up_stream_layout(w13_host.float()), dtype=w13_dtype)
    w2 = upload_stream_weights(mesh_device, down_stream_layout(w2_host.float()))
    w13_q = (bfp4_round if w13_dtype == ttnn.bfloat4_b else bfp8_round)(w13_host.float())
    return w13, w2, w13_q, bfp8_round(w2_host.float())


def inputs(mesh_device, x, r):
    T, E = r.shape
    xd = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    rd = ttnn.from_torch(r.reshape(1, 1, T, E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    return xd, rd


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "T, I, E, picks, popular",
    [(64, 256, 72, 2, 0), (256, 256, 72, 2, 0), (256, 256, 72, 2, 2), (64, 1024, 72, 2, 1)],
    ids=["64-256", "256-256", "256-256-popular", "64-1024-popular"],
)
def test_expert_dispatch_vs_golden(mesh_device, T, I, E, picks, popular):
    w13, w2, w13_q, w2_q = upload(mesh_device, E, I)
    op = ExpertDispatch(mesh_device, E, K, I, H, T, groups=GROUPS, down_readers=DOWN_READERS)
    x = torch.randn(T, K).bfloat16()
    r = skewed_routing(T, E, picks, seed=T + I, popular=popular)
    act_ref, y_ref, passes = expert_dispatch_golden(x, r, w13_q, w2_q)
    active = int((r.abs().sum(0) > 0).sum())
    logger.info(
        f"T {T}, I {I}: {active} of {E} active, {len(passes)} passes (max {op.P}), gate_up {len(op.gate_up.assignments)} "
        f"cores x {op.gate_up.ppb} pairs, down {len(op.down.assignments)} cores x {op.down.cols} columns"
    )
    assert len(passes) <= op.P
    xd, rd = inputs(mesh_device, x, r)
    x_rm, counts = op.prepare(xd, rd)
    act, table, out = op.alloc(mesh_device)
    op.forward(x_rm, rd, counts, w13, w2, act, table, out)
    tab = ttnn.to_torch(table).reshape(op.P, -1).to(torch.int64)
    for k, (e, rows) in enumerate(passes):
        got = (int(tab[k, 0]), int(tab[k, 1]), tab[k, 2 : 2 + len(rows)].tolist())
        assert got == (len(rows), e, rows), (k, got, (len(rows), e, rows[:8]))
    a = ttnn.to_torch(act).float().reshape(op.P, 32, I)[: len(passes)]
    p_act = pcc(act_ref, a)
    y = ttnn.to_torch(out).float().reshape(T, H)
    p = pcc(y_ref, y)
    logger.info(
        f"T {T}, I {I}: table ok, act pcc {p_act:.6f}, out pcc {p:.6f}, max |diff| {(y_ref - y).abs().max():.4f} "
        f"(|y| max {y_ref.abs().max():.3f})"
    )
    assert p_act > 0.9999, p_act
    assert p > 0.9999, p


@pytest.mark.timeout(2400)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 64 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("T, I, E", [(256, 256, 72), (64, 1024, 72)], ids=["256-256", "64-1024"])
def test_expert_dispatch_trace(mesh_device, T, I, E):
    """The whole dispatch (layout conversion, counts, the two ops) captured as a trace and replayed."""
    w13, w2, w13_q, w2_q = upload(mesh_device, E, I)
    op = ExpertDispatch(mesh_device, E, K, I, H, T, groups=GROUPS, down_readers=DOWN_READERS)
    x = torch.randn(T, K).bfloat16()
    r = skewed_routing(T, E, 2, seed=11, popular=1)
    _, y_ref, passes = expert_dispatch_golden(x, r, w13_q, w2_q)
    active = int((r.abs().sum(0) > 0).sum())
    xd, rd = inputs(mesh_device, x, r)

    def run():
        x_rm, counts = op.prepare(xd, rd)
        act, table, out = op.alloc(mesh_device)
        op.forward(x_rm, rd, counts, w13, w2, act, table, out)
        for t_ in (x_rm, counts, act, table):
            ttnn.deallocate(t_)
        return out

    out = run()  # compile
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = run()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    steps = 20
    t0 = time.perf_counter()
    for _ in range(steps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = (time.perf_counter() - t0) / steps
    y = ttnn.to_torch(out).float().reshape(T, H)
    p = pcc(y_ref, y)
    logger.info(
        f"dispatch trace replay T {T}, I {I} ({active} of {E} active, {len(passes)} passes): {dt*1e6:.1f} us per call, "
        f"pcc vs golden {p:.6f}"
    )
    # the two ops separately (eager, warmed)
    x_rm, counts = op.prepare(xd, rd)
    act, table, out2 = op.alloc(mesh_device)
    for name, fn in (
        ("gate_up", lambda: op.gate_up(x_rm, rd, counts, w13, act, table)),
        ("down", lambda: op.down(act, table, counts, w2, out2)),
    ):
        fn()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        ttnn.synchronize_device(mesh_device)
        logger.info(f"  {name}: {(time.perf_counter() - t0) / 10 * 1e6:.1f} us per call (eager)")
    ttnn.release_trace(mesh_device, tid)
    assert p > 0.9999, p


def topk_lists(tokens: int, E_total: int, K: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, K] global expert ids (distinct per token, Zipf-skewed) and positive weights."""
    g = torch.Generator().manual_seed(seed)
    pref = 1.0 / (torch.arange(E_total, dtype=torch.float) + 1.0) ** 1.1
    pref = pref[torch.randperm(E_total, generator=g)]
    idx = torch.stack([torch.multinomial(pref, K, replacement=False, generator=g) for _ in range(tokens)])
    wts = (torch.rand(tokens, K, generator=g) + 0.2).bfloat16().float()
    return idx, wts


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "T, E_local, base",
    [(64, 9, 9), (256, 9, 27), (64, 36, 36), (256, 36, 72)],
    ids=["64-9", "256-9", "64-36", "256-36"],
)
def test_expert_dispatch_ep(mesh_device, T, E_local, base):
    """EP mode: this chip owns experts [base, base + E_local) at full width (I = 2048, 8 column pairs per bank);
    the routing arrives as top-k lists over all 288 experts (index + weight tiles) and the chip's expert base."""
    I, E_total, K = 2048, 288, 8
    w13, w2, w13_q, w2_q = upload(mesh_device, E_local, I, seed=base)
    op = ExpertDispatch(mesh_device, E_local, 4096, I, H, T, groups=GROUPS, down_readers=DOWN_READERS, ep=True)
    x = torch.randn(T, 4096).bfloat16()
    idx, wts = topk_lists(T, E_total, K, seed=T + base)
    act_ref, y_ref, passes = expert_dispatch_golden_ep(x, idx, wts, base, E_local, w13_q, w2_q)
    local = int(((idx >= base) & (idx < base + E_local)).sum())
    logger.info(
        f"EP T {T}, experts [{base}, {base + E_local}): {local} routed rows, {len(passes)} passes (max {op.P}), "
        f"down slots a {op.down.a_slots} w {op.down.w_slots}"
    )
    xd = ttnn.from_torch(x.reshape(1, 1, T, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    x_rm = ttnn.to_layout(xd, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    it, wt = ep_index_tiles(idx, wts, mesh_device, K)
    bt = ep_base_tensor(base, mesh_device)
    act, table, out = op.alloc(mesh_device)
    op.forward(x_rm, wt, it, w13, w2, act, table, out, base=bt)
    tab = ttnn.to_torch(table).reshape(op.P, -1).to(torch.int64)
    for k, (e, rows) in enumerate(passes):
        got = (int(tab[k, 0]), int(tab[k, 1]), tab[k, 2 : 2 + len(rows)].tolist())
        assert got == (len(rows), e, rows), (k, got, (len(rows), e, rows[:8]))
    a = ttnn.to_torch(act).float().reshape(op.P, 32, I)[: len(passes)]
    y = ttnn.to_torch(out).float().reshape(T, H)
    p_act, p = pcc(act_ref, a), pcc(y_ref, y)
    logger.info(f"EP T {T}, E_local {E_local}: table ok, act pcc {p_act:.6f}, out pcc {p:.6f}")
    assert p_act > 0.9999, p_act
    assert p > 0.9999, p


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_expert_dispatch_ep_bfp4(mesh_device):
    """The gate|up weights in bfp4 (576 B tiles; the dtype policy knob): EP mode, T = 64, 9 full-width experts,
    vs the torch golden on the bfp4-rounded weights."""
    T, E_local, base, I, K = 64, 9, 9, 2048, 8
    w13, w2, w13_q, w2_q = upload(mesh_device, E_local, I, seed=base, w13_dtype=ttnn.bfloat4_b)
    op = ExpertDispatch(
        mesh_device, E_local, 4096, I, H, T, groups=GROUPS, down_readers=DOWN_READERS, ep=True, w13_dtype=ttnn.bfloat4_b
    )
    x = torch.randn(T, 4096).bfloat16()
    idx, wts = topk_lists(T, 288, K, seed=T + base)
    act_ref, y_ref, passes = expert_dispatch_golden_ep(x, idx, wts, base, E_local, w13_q, w2_q)
    xd = ttnn.from_torch(x.reshape(1, 1, T, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    x_rm = ttnn.to_layout(xd, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    it, wt = ep_index_tiles(idx, wts, mesh_device, K)
    bt = ep_base_tensor(base, mesh_device)
    act, table, out = op.alloc(mesh_device)
    op.forward(x_rm, wt, it, w13, w2, act, table, out, base=bt)
    a = ttnn.to_torch(act).float().reshape(op.P, 32, I)[: len(passes)]
    y = ttnn.to_torch(out).float().reshape(T, H)
    p_act, p = pcc(act_ref, a), pcc(y_ref, y)
    logger.info(
        f"EP bfp4 gate|up T {T}, E_local {E_local}: act pcc {p_act:.6f}, out pcc {p:.6f} (vs bfp4-rounded golden)"
    )
    assert p_act > 0.9999, p_act
    assert p > 0.9999, p
