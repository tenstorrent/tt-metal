# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""custom_ops/expert_stream: the chip's routed-expert block (72 experts x [4096, 512] gate|up and [256, 4096] down,
the 8x4 hybrid slice) as DRAM-bank-streaming generic ops, vs the torch golden on bfp8-rounded weights, at 1 / 4 / 32
users per row (8 / 32 / 256 gathered tokens; 256 = 8 blocks of 32), eager and as a captured trace replayed 3x.

    TT_VISIBLE_DEVICES=2 pytest models/demos/glm53_flash/tests/custom_ops/test_expert_stream.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.expert_stream.op import (
    ExpertStream,
    bfp8_round,
    down_stream_layout,
    expert_stream_golden,
    gate_up_stream_layout,
    upload_stream_weights,
)
from models.demos.glm53_flash.tests.utils import pcc

E, K, I, H = 72, 4096, 256, 4096
GROUPS = int(os.environ.get("GLM53_STREAM_GROUPS", "12"))
DOWN_READERS = int(os.environ["GLM53_STREAM_DOWN_READERS"]) if os.environ.get("GLM53_STREAM_DOWN_READERS") else None
X_L1 = os.environ.get("GLM53_STREAM_X_L1", "0") == "1"  # the block's tokens in L1 (no DRAM re-reads per core)
ACT_L1 = os.environ.get("GLM53_STREAM_ACT_L1", "0") == "1"  # the activations between the two ops in L1
TRACE_PICKS = int(os.environ.get("GLM53_STREAM_TRACE_PICKS", "2"))  # routed experts per token in the trace test
MESH_TESTS = os.environ.get("GLM53_STREAM_MESH_TESTS", "0") == "1"  # the (1,4) / (2,4) variants (not run yet)


def routing_matrix(tokens: int, picks: int, seed: int) -> torch.Tensor:
    """[tokens, E] bf16: `picks` random experts per token with random positive weights (the real router gives the
    chip ~2 of a token's 8 experts; ``picks`` is the per-chip count)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.zeros(tokens, E)
    for t in range(tokens):
        ids = torch.randperm(E, generator=g)[:picks]
        r[t, ids] = torch.rand(picks, generator=g) + 0.2
    return r.bfloat16()


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator().manual_seed(0)
    w13 = (torch.randn(E, K, 2 * I, generator=g) * 0.02).bfloat16()
    w2 = (torch.randn(E, I, H, generator=g) * 0.02).bfloat16()
    return w13, w2


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("tokens, picks", [(8, 2), (32, 2), (32, 8), (256, 2)])
def test_expert_stream_vs_golden(mesh_device, weights, tokens, picks):
    w13_host, w2_host = weights
    w13 = upload_stream_weights(mesh_device, gate_up_stream_layout(w13_host.float()))
    w2 = upload_stream_weights(mesh_device, down_stream_layout(w2_host.float()))
    w13_q, w2_q = bfp8_round(w13_host.float()), bfp8_round(w2_host.float())
    op = ExpertStream(mesh_device, E, K, I, H, groups=GROUPS, down_readers=DOWN_READERS)
    logger.info(
        f"gate_up: {len(op.gate_up.assignments)} cores ({op.gate_up.groups} per bank), down: {len(op.down.assignments)} "
        f"cores ({op.down.readers} per bank, {op.down.cols} columns each)"
    )
    x_all = torch.randn(tokens, K).bfloat16()
    r_all = routing_matrix(tokens, picks, seed=tokens)
    act = ttnn.from_torch(torch.zeros(1, E, 32, I), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    out = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    worst = 1.0
    for b0 in range(0, tokens, 32):
        x = x_all[b0 : b0 + 32]
        r = r_all[b0 : b0 + 32]
        m = x.shape[0]
        x_pad = torch.zeros(32, K).bfloat16()
        x_pad[:m] = x
        r_pad = torch.zeros(32, E).bfloat16()
        r_pad[:m] = r
        xd = ttnn.from_torch(
            x_pad.reshape(1, 1, 32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        rd = ttnn.from_torch(
            r_pad.reshape(1, 1, 32, E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        op.forward(xd, rd, w13, w2, act, out)
        y = ttnn.to_torch(out).float().reshape(32, H)[:m]
        a = ttnn.to_torch(act).float().reshape(E, 32, I)[:, :m]
        act_ref, y_ref = expert_stream_golden(x, r, w13_q, w2_q)
        active = [e for e in range(E) if r[:, e].abs().sum() > 0]
        p_act = pcc(act_ref[active], a[active])
        p = pcc(y_ref, y)
        worst = min(worst, p)
        logger.info(
            f"tokens {tokens} block {b0 // 32} ({m} rows, {len(active)} active): act pcc {p_act:.6f}, out pcc {p:.6f}, "
            f"max |diff| {(y_ref - y).abs().max():.4f} (|y| max {y_ref.abs().max():.3f})"
        )
        assert p_act > 0.9999, p_act
        assert p > 0.9999, p
        ttnn.deallocate(xd)
        ttnn.deallocate(rd)
    logger.info(f"tokens {tokens} picks {picks}: worst block pcc {worst:.6f}")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 64 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_expert_stream_trace(mesh_device, weights):
    """One block captured as a trace and replayed: same output, and the dispatch-free device time per block."""
    w13_host, w2_host = weights
    w13 = upload_stream_weights(mesh_device, gate_up_stream_layout(w13_host.float()))
    w2 = upload_stream_weights(mesh_device, down_stream_layout(w2_host.float()))
    w13_q, w2_q = bfp8_round(w13_host.float()), bfp8_round(w2_host.float())
    op = ExpertStream(mesh_device, E, K, I, H, groups=GROUPS, down_readers=DOWN_READERS)
    x = torch.randn(32, K).bfloat16()
    r = routing_matrix(32, TRACE_PICKS, seed=7)  # 2 picks: ~43 of 72 active; 8: ~71
    xd = ttnn.from_torch(
        x.reshape(1, 1, 32, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG if X_L1 else ttnn.DRAM_MEMORY_CONFIG,
    )
    rd = ttnn.from_torch(r.reshape(1, 1, 32, E), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    act = ttnn.from_torch(
        torch.zeros(1, E, 32, I),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG if ACT_L1 else ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    op.forward(xd, rd, w13, w2, act, out)  # compile
    ttnn.synchronize_device(mesh_device)
    _, y_ref = expert_stream_golden(x, r, w13_q, w2_q)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op.forward(xd, rd, w13, w2, act, out)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    steps = 20
    t0 = time.perf_counter()
    for _ in range(steps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = (time.perf_counter() - t0) / steps
    y = ttnn.to_torch(out).float().reshape(32, H)
    p = pcc(y_ref, y)
    active = int((r.abs().sum(0) > 0).sum())
    logger.info(
        f"trace replay (groups {GROUPS}, down readers {DOWN_READERS}, x in {'L1' if X_L1 else 'DRAM'}, act in "
        f"{'L1' if ACT_L1 else 'DRAM'}): {dt*1e6:.1f} us "
        f"per block ({active} of {E} active), pcc vs golden {p:.6f}"
    )
    # the two ops separately (eager, warmed: dispatch-bound upper bound)
    for name, fn in (("gate_up", lambda: op.gate_up(xd, rd, w13, act)), ("down", lambda: op.down(act, rd, w2, out))):
        fn()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        ttnn.synchronize_device(mesh_device)
        logger.info(f"  {name}: {(time.perf_counter() - t0) / 10 * 1e6:.1f} us per call (eager)")
    ttnn.release_trace(mesh_device, tid)
    assert p > 0.9999, p


@pytest.mark.skipif(not MESH_TESTS, reason="multi-chip meshes were not opened in this pass; GLM53_STREAM_MESH_TESTS=1")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 4), (2, 4)], indirect=True)
def test_expert_stream_mesh(mesh_device, weights):
    """The same program on every chip of a mesh with that chip's expert slice, as in the hybrid placement: the
    weights sharded over the chips (chip d = the 72-expert pool rotated by 7 d), the block's tokens replicated,
    each chip's routing restricted to its own experts; checked per chip against the torch golden.
    NOT RUN YET: no multi-chip mesh was opened while this op was built (board note of 2026-10-20)."""
    D = mesh_device.get_num_devices()
    w13_host, w2_host = weights
    w13_d = [w13_host.roll(7 * d, dims=0).float() for d in range(D)]
    w2_d = [w2_host.roll(7 * d, dims=0).float() for d in range(D)]
    shard0 = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    w13 = upload_stream_weights(mesh_device, torch.stack([gate_up_stream_layout(w) for w in w13_d]), mesh_mapper=shard0)
    w2 = upload_stream_weights(mesh_device, torch.stack([down_stream_layout(w) for w in w2_d]), mesh_mapper=shard0)
    op = ExpertStream(mesh_device, E, K, I, H, groups=GROUPS, down_readers=DOWN_READERS)
    x = torch.randn(32, K).bfloat16()
    r_d = [routing_matrix(32, 2, seed=100 + d) for d in range(D)]
    xd = ttnn.from_torch(
        x.reshape(1, 1, 32, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG if X_L1 else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rd = ttnn.from_torch(
        torch.stack([r.reshape(1, 32, E) for r in r_d]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=shard0,
    )
    act, out = op.alloc(mesh_device, act_l1=ACT_L1)
    op.forward(xd, rd, w13, w2, act, out)
    ys = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float().reshape(D, 32, H)
    for d in range(D):
        _, y_ref = expert_stream_golden(x, r_d[d], bfp8_round(w13_d[d]), bfp8_round(w2_d[d]))
        p = pcc(y_ref, ys[d])
        active = int((r_d[d].abs().sum(0) > 0).sum())
        logger.info(f"mesh {tuple(mesh_device.shape)} chip {d} ({active} active): out pcc {p:.6f}")
        assert p > 0.9999, (d, p)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("I_loc, E_loc", [(1024, 72), (2048, 24)])
def test_expert_stream_widths(mesh_device, I_loc, E_loc):
    """The wider per-chip intermediate slices of the smaller meshes ((2,4) block: 1024 = 4 column pairs per bank;
    one chip: 2048 = 8 pairs; the down segments then span several NOC packets and more readers per bank), 32
    tokens, 2 routed experts per token, vs the torch golden."""
    g = torch.Generator().manual_seed(1)
    w13_host = (torch.randn(E_loc, K, 2 * I_loc, generator=g) * 0.02).bfloat16()
    w2_host = (torch.randn(E_loc, I_loc, H, generator=g) * 0.02).bfloat16()
    w13 = upload_stream_weights(mesh_device, gate_up_stream_layout(w13_host.float()))
    w2 = upload_stream_weights(mesh_device, down_stream_layout(w2_host.float()))
    w13_q, w2_q = bfp8_round(w13_host.float()), bfp8_round(w2_host.float())
    op = ExpertStream(mesh_device, E_loc, K, I_loc, H, groups=GROUPS, down_readers=DOWN_READERS)
    logger.info(
        f"I = {I_loc}: gate_up {op.gate_up.ppb} pairs per bank on {len(op.gate_up.assignments)} cores, down "
        f"{op.down.readers} readers per bank x {op.down.cols} columns, {op.down.page_bytes} B packets"
    )
    x = torch.randn(32, K).bfloat16()
    r = torch.zeros(32, E_loc)
    for t in range(32):
        ids = torch.randperm(E_loc, generator=g)[:2]
        r[t, ids] = torch.rand(2, generator=g) + 0.2
    r = r.bfloat16()
    xd = ttnn.from_torch(x.reshape(1, 1, 32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    rd = ttnn.from_torch(r.reshape(1, 1, 32, E_loc), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    act, out = op.alloc(mesh_device)
    op.forward(xd, rd, w13, w2, act, out)
    act_ref, y_ref = expert_stream_golden(x, r, w13_q, w2_q)
    a = ttnn.to_torch(act).float().reshape(E_loc, 32, I_loc)
    y = ttnn.to_torch(out).float().reshape(32, H)
    active = [e for e in range(E_loc) if r[:, e].abs().sum() > 0]
    p_act, p = pcc(act_ref[active], a[active]), pcc(y_ref, y)
    logger.info(f"I = {I_loc}, E = {E_loc} ({len(active)} active): act pcc {p_act:.6f}, out pcc {p:.6f}")
    assert p_act > 0.9999, p_act
    assert p > 0.9999, p
