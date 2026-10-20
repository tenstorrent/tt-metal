# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops.dsa_decode_state`` vs its torch golden: the model's shapes (Lq 1536, L 512, D 128, P 4), B = 1/4/32/40,
identity and permuted page tables, 6-step evolution of the in-place caches, and a captured-trace replay.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_dsa_decode_state.py -s
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.glm53_flash.custom_ops.dsa_decode_state.op import DsaDecodeState
from models.demos.glm53_flash.custom_ops.kda_decode.op import selector_tiles
from models.demos.glm53_flash.tests.utils import pcc

Lq, L, D, P = 1536, 512, 128, 4
EPS_RMS, EPS_LN = 1e-5, 1e-6
PCC = 0.9999
OFF = (0, Lq, Lq + L, Lq + L + D)
WS = Lq + L + 2 * D + 32


def _dev(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_case(B, seed, *, blocks_per_seq=4, pool_blocks=2, permute_pages=False):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    n_blocks = B * blocks_per_seq
    pt = torch.arange(n_blocks, dtype=torch.int32).reshape(B, blocks_per_seq)
    if permute_pages:
        pt = pt.flatten()[torch.randperm(n_blocks, generator=gen)].reshape(B, blocks_per_seq)
    x = dict(
        stem=r(1, 1, B, WS).to(torch.bfloat16).float(),
        latent=(0.5 * r(n_blocks, 1, 32, L)).to(torch.bfloat16).float(),
        pt=pt,
        positions=torch.randint(0, blocks_per_seq * 32, (B,), generator=gen, dtype=torch.int32),
        ring_k=torch.zeros(B, 1, 32, D),
        ring_g=torch.zeros(B, 1, 32, D),
        pool=(0.5 * r(B * pool_blocks, 1, 32, D)).to(torch.bfloat16).float(),
        wq=r(Lq) * 0.2 + 1.0,
        wkva=r(L) * 0.2 + 1.0,
        wk=r(D) * 0.2 + 1.0,
        bk=r(D) * 0.1,
        ape=r(P, D) * 0.5,
        pool_blocks=pool_blocks,
    )
    x["ring_k"][:, 0, :P] = r(B, P, D).to(torch.bfloat16).float()
    x["ring_g"][:, 0, :P] = r(B, P, D).to(torch.bfloat16).float()
    return x


def golden(x, B):
    return DsaDecodeState.golden(
        x["stem"][0, 0, :B],
        x["latent"],
        x["pt"],
        x["positions"],
        x["ring_k"],
        x["ring_g"],
        x["pool"],
        x["wq"],
        x["wkva"],
        x["wk"],
        x["bk"],
        x["ape"],
        Lq=Lq,
        L=L,
        D=D,
        P=P,
        pool_blocks=x["pool_blocks"],
        offsets=OFF,
        eps_rms=EPS_RMS,
        eps_ln=EPS_LN,
    )


def to_device(device, x, B):
    ape = torch.full((1, 1, 32, D), -1e9)
    ape[0, 0, :P] = x["ape"]
    return dict(
        stem=_dev(device, x["stem"].to(torch.bfloat16), ttnn.bfloat16),
        latent=_dev(device, x["latent"].to(torch.bfloat16), ttnn.bfloat16),
        pt=_dev(device, x["pt"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        positions=_dev(device, x["positions"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        ring_k=_dev(device, x["ring_k"].to(torch.bfloat16), ttnn.bfloat16),
        ring_g=_dev(device, x["ring_g"].to(torch.bfloat16), ttnn.bfloat16),
        pool=_dev(device, x["pool"].to(torch.bfloat16), ttnn.bfloat16),
        wq=_dev(device, x["wq"].reshape(1, 1, 1, -1)),
        wkva=_dev(device, x["wkva"].reshape(1, 1, 1, -1)),
        wk=_dev(device, x["wk"].reshape(1, 1, 1, -1)),
        bk=_dev(device, x["bk"].reshape(1, 1, 1, -1)),
        ape=_dev(device, ape),
        sel=_dev(device, selector_tiles().to(torch.bfloat16), ttnn.bfloat16),
        q=_dev(device, torch.zeros(1, 1, B, Lq).to(torch.bfloat16), ttnn.bfloat16),
    )


def make_op(pool_blocks):
    return DsaDecodeState(
        q_lora_rank=Lq,
        latent_dim=L,
        index_dim=D,
        pool_size=P,
        pool_blocks=pool_blocks,
        qa_offset=OFF[0],
        kva_offset=OFF[1],
        k_offset=OFF[2],
        g_offset=OFF[3],
        eps_rms=EPS_RMS,
        eps_ln=EPS_LN,
    )


def run(op, d):
    return op(
        d["stem"],
        d["latent"],
        d["pt"],
        d["positions"],
        d["ring_k"],
        d["ring_g"],
        d["pool"],
        d["wq"],
        d["wkva"],
        d["wk"],
        d["bk"],
        d["ape"],
        d["sel"],
        d["q"],
    )


def check(d, x, B, q_ref, lat_ref, rk_ref, rg_ref, pool_ref):
    q = ttnn.to_torch(d["q"]).float()[0, 0]
    lat = ttnn.to_torch(d["latent"]).float()
    rk = ttnn.to_torch(d["ring_k"]).float()
    rg = ttnn.to_torch(d["ring_g"]).float()
    pool = ttnn.to_torch(d["pool"]).float()
    assert q.shape == (B, Lq) and pcc(q_ref, q) > PCC
    assert pcc(lat_ref, lat) > PCC and pcc(pool_ref, pool) > PCC and pcc(rk_ref[:, 0, :P], rk[:, 0, :P]) > PCC
    torch.testing.assert_close(rg, rg_ref, rtol=0, atol=0)  # the raw gate row is moved, not computed
    for b in range(B):
        a = int(x["positions"][b])
        page = int(x["pt"][b, a // 32])
        others = [i for i in range(32) if i != a % 32]
        assert torch.equal(lat[page, 0, others], lat_ref[page, 0, others]), f"user {b}: untouched latent rows changed"
        assert (lat[page, 0, a % 32] - lat_ref[page, 0, a % 32]).abs().max() < 4e-2  # one bf16 ulp at |x| ~ 4
        p = a // P
        blk = b * x["pool_blocks"] + p // 32
        assert (pool[blk, 0, p % 32] - pool_ref[blk, 0, p % 32]).abs().max() < 4e-2


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B, permute", [(1, False), (4, False), (4, True), (32, True), (40, False)])
def test_dsa_decode_state_vs_golden(mesh_device, B, permute):
    x = make_case(B, seed=B + (7 if permute else 0), permute_pages=permute)
    refs = golden(x, B)
    d = to_device(mesh_device, x, B)
    run(make_op(x["pool_blocks"]), d)
    check(d, x, B, *refs)
    print(f"B={B} permute={permute}: ok")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dsa_decode_state_multi_step(mesh_device):
    B = 4
    x = make_case(B, seed=3)
    x["positions"] = torch.tensor([5, 40, 71, 100], dtype=torch.int32)
    d = to_device(mesh_device, x, B)
    op = make_op(x["pool_blocks"])
    ref = dict(x)
    gen = torch.Generator().manual_seed(99)
    for step in range(6):
        ref["stem"] = torch.randn(1, 1, B, WS, generator=gen).to(torch.bfloat16).float()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(ref["stem"].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), d["stem"]
        )
        q_ref, lat_ref, rk_ref, rg_ref, pool_ref = golden(ref, B)
        run(op, d)
        check(d, ref, B, q_ref, lat_ref, rk_ref, rg_ref, pool_ref)
        ref["latent"], ref["ring_k"], ref["ring_g"], ref["pool"] = (
            ttnn.to_torch(d[n]).float() for n in ("latent", "ring_k", "ring_g", "pool")
        )
        ref["positions"] = ref["positions"] + 1
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(ref["positions"], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), d["positions"]
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dsa_decode_state_trace_replay(mesh_device):
    """Inside a captured trace on persistent io tensors (positions advanced between replays through the persistent
    [B] vector, as the model does)."""
    B = 4
    x = make_case(B, seed=9)
    x["positions"] = torch.tensor([3, 36, 70, 101], dtype=torch.int32)
    d = to_device(mesh_device, x, B)
    op = make_op(x["pool_blocks"])
    ref = dict(x)
    run(op, d)  # compile
    q_ref, lat_ref, rk_ref, rg_ref, pool_ref = golden(ref, B)
    ref["latent"], ref["ring_k"], ref["ring_g"], ref["pool"] = lat_ref, rk_ref, rg_ref, pool_ref
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run(op, d)  # the capture run executes as well, at the same positions (idempotent rewrite of the same rows)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    q_ref, lat_ref, rk_ref, rg_ref, pool_ref = golden(ref, B)
    ref["latent"], ref["ring_k"], ref["ring_g"], ref["pool"] = lat_ref, rk_ref, rg_ref, pool_ref
    for _ in range(3):
        # the reference starts every replay from the device state (its written rows differ by bf16 rounding)
        ref["latent"], ref["ring_k"], ref["ring_g"], ref["pool"] = (
            ttnn.to_torch(d[n]).float() for n in ("latent", "ring_k", "ring_g", "pool")
        )
        ref["positions"] = ref["positions"] + 1
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(ref["positions"], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), d["positions"]
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        q_ref, lat_ref, rk_ref, rg_ref, pool_ref = golden(ref, B)
        check(d, ref, B, q_ref, lat_ref, rk_ref, rg_ref, pool_ref)
    ttnn.release_trace(mesh_device, tid)
    print("trace replay: ok")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_dsa_decode_state_perf(mesh_device, B):
    import time

    x = make_case(B, seed=1)
    d = to_device(mesh_device, x, B)
    op = make_op(x["pool_blocks"])
    for _ in range(3):
        run(op, d)
    ttnn.synchronize_device(mesh_device)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        run(op, d)
    ttnn.synchronize_device(mesh_device)
    print(f"dsa_decode_state B={B}: {(time.perf_counter() - t0) / n * 1e6:.0f} us/call (eager)")
