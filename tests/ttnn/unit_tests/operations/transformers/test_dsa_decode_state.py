# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of ttnn.transformer.dsa_decode_state (the fused DSA decode state maintenance) against a torch reference:
q_a RMSNorm, the latent append (RMSNorm'd kv_a row into the paged latent cache), the indexer ring update (LayerNorm'd k
and raw gate rows) and the pooled key of the current pool into the pool cache, all in place.

    pytest tests/ttnn/unit_tests/operations/transformers/test_dsa_decode_state.py -s
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

OP = ttnn.transformer.dsa_decode_state
Lq, L, D, P = 1536, 512, 128, 4
EPS_RMS, EPS_LN = 1e-5, 1e-6
PCC = 0.9999
OFF = {"qa": 0, "kva": Lq, "k": Lq + L, "g": Lq + L + D, "w": Lq + L + 2 * D}
WS = Lq + L + 2 * D + 32


def selectors():
    sel = torch.zeros(33, 32, 32)
    sel[torch.arange(32), 0, torch.arange(32)] = 1.0
    sel[32] = 1.0
    return sel.reshape(33 * 32, 32)


def _dev(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_case(B, seed, *, blocks_per_seq=4, pool_blocks=2, permute_pages=False):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float32)
    stem = r(1, 1, B, WS).to(torch.bfloat16).float()
    n_blocks = B * blocks_per_seq
    pt = torch.arange(n_blocks, dtype=torch.int32).reshape(B, blocks_per_seq)
    if permute_pages:
        pt = pt.flatten()[torch.randperm(n_blocks, generator=gen)].reshape(B, blocks_per_seq)
    positions = torch.randint(0, blocks_per_seq * 32, (B,), generator=gen, dtype=torch.int32)
    x = dict(
        stem=stem,
        latent=(0.5 * r(n_blocks, 1, 32, L)).to(torch.bfloat16).float(),
        pt=pt,
        positions=positions,
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
    # rings with the earlier in-pool rows filled (bf16, as the device keeps them)
    x["ring_k"][:, 0, :P] = r(B, P, D).to(torch.bfloat16).float()
    x["ring_g"][:, 0, :P] = r(B, P, D).to(torch.bfloat16).float()
    return x


def reference(x, B):
    """One step in torch; returns (q_resid, latent, ring_k, ring_g, pool) after the step (bf16-rounded like the device)."""
    stem = x["stem"][0, 0, :B]
    bf = lambda t: t.to(torch.bfloat16).float()
    qa = stem[:, OFF["qa"] : OFF["qa"] + Lq]
    q_resid = bf(qa * torch.rsqrt((qa * qa).mean(-1, keepdim=True) + EPS_RMS) * x["wq"])
    kva = stem[:, OFF["kva"] : OFF["kva"] + L]
    lat_row = bf(kva * torch.rsqrt((kva * kva).mean(-1, keepdim=True) + EPS_RMS) * x["wkva"])
    k = stem[:, OFF["k"] : OFF["k"] + D]
    mu, var = k.mean(-1, keepdim=True), k.var(-1, keepdim=True, unbiased=False)
    k_row = bf((k - mu) * torch.rsqrt(var + EPS_LN) * x["wk"] + x["bk"])
    g_row = stem[:, OFF["g"] : OFF["g"] + D]
    latent, ring_k, ring_g, pool = (x[n].clone() for n in ("latent", "ring_k", "ring_g", "pool"))
    for b in range(B):
        a = int(x["positions"][b])
        page = int(x["pt"][b, a // 32])
        latent[page, 0, a % 32] = lat_row[b]
        ring_k[b, 0, a % P] = k_row[b]
        ring_g[b, 0, a % P] = g_row[b]
        logits = ring_g[b, 0, :P] + x["ape"]  # [P, D]
        prob = torch.softmax(logits, dim=0)
        pooled = bf((prob * ring_k[b, 0, :P]).sum(0))
        p = a // P
        pool[b * x["pool_blocks"] + p // 32, 0, p % 32] = pooled
    return q_resid, latent, ring_k, ring_g, pool


def to_device(device, x):
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
        sel=_dev(device, selectors().to(torch.bfloat16), ttnn.bfloat16),
    )


def run(d, pool_blocks):
    return OP(
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
        OFF["qa"],
        OFF["kva"],
        OFF["k"],
        OFF["g"],
        Lq,
        L,
        D,
        P,
        pool_blocks,
        eps_rms=EPS_RMS,
        eps_ln=EPS_LN,
    )


def check(d, x, B, q_ref, lat_ref, rk_ref, rg_ref, pool_ref, q_out):
    q = ttnn.to_torch(q_out).float()[0, 0]
    assert q.shape == (B, Lq)
    assert_with_pcc(q_ref, q, PCC)
    lat = ttnn.to_torch(d["latent"]).float()
    rk = ttnn.to_torch(d["ring_k"]).float()
    rg = ttnn.to_torch(d["ring_g"]).float()
    pool = ttnn.to_torch(d["pool"]).float()
    # untouched rows must be bit-identical, written rows close
    assert_with_pcc(lat_ref, lat, PCC)
    assert_with_pcc(pool_ref, pool, PCC)
    torch.testing.assert_close(rg, rg_ref, rtol=0, atol=0)  # the raw gate row is moved, not computed
    assert_with_pcc(rk_ref[:, 0, :P], rk[:, 0, :P], PCC)
    for b in range(B):
        a = int(x["positions"][b])
        page = int(x["pt"][b, a // 32])
        err = (lat[page, 0, a % 32] - lat_ref[page, 0, a % 32]).abs().max().item()
        others = torch.cat([lat[page, 0, : a % 32], lat[page, 0, a % 32 + 1 :]])
        others_ref = torch.cat([lat_ref[page, 0, : a % 32], lat_ref[page, 0, a % 32 + 1 :]])
        assert torch.equal(others, others_ref), f"user {b}: untouched latent rows changed"
        assert err < 4e-2, f"user {b}: latent row error {err}"  # one bf16 ulp at |x| ~ 4
        p = a // P
        blk = b * x["pool_blocks"] + p // 32
        perr = (pool[blk, 0, p % 32] - pool_ref[blk, 0, p % 32]).abs().max().item()
        assert perr < 4e-2, f"user {b}: pooled row error {perr}"
    return q


@pytest.mark.parametrize("B, permute", [(1, False), (4, False), (4, True), (32, True), (40, False)])
def test_dsa_decode_state_vs_reference(device, B, permute):
    x = make_case(B, seed=B + (7 if permute else 0), permute_pages=permute)
    refs = reference(x, B)
    d = to_device(device, x)
    q_out = run(d, x["pool_blocks"])
    check(d, x, B, *refs, q_out)
    print(f"B={B} permute={permute}: ok")


def test_dsa_decode_state_multi_step(device):
    """Consecutive positions of one user fill a pool: the rings and the pooled key evolve as in the reference."""
    B = 4
    x = make_case(B, seed=3)
    x["positions"] = torch.tensor([5, 40, 71, 100], dtype=torch.int32)
    d = to_device(device, x)
    ref = dict(x)
    gen = torch.Generator().manual_seed(99)
    for step in range(6):
        ref["stem"] = torch.randn(1, 1, B, WS, generator=gen).to(torch.bfloat16).float()
        d["stem"] = _dev(device, ref["stem"].to(torch.bfloat16), ttnn.bfloat16)
        q_ref, lat_ref, rk_ref, rg_ref, pool_ref = reference(ref, B)
        q_out = run(d, x["pool_blocks"])
        check(d, ref, B, q_ref, lat_ref, rk_ref, rg_ref, pool_ref, q_out)
        ref["latent"], ref["ring_k"], ref["ring_g"], ref["pool"] = (
            ttnn.to_torch(d["latent"]).float(),  # carry the device state (bf16) as the reference's starting point
            ttnn.to_torch(d["ring_k"]).float(),
            ttnn.to_torch(d["ring_g"]).float(),
            ttnn.to_torch(d["pool"]).float(),
        )
        ref["positions"] = ref["positions"] + 1
        d["positions"] = _dev(device, ref["positions"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        print(f"step {step}: ok")


@pytest.mark.parametrize("B", [4, 32])
def test_dsa_decode_state_perf(device, B):
    import time

    x = make_case(B, seed=1)
    d = to_device(device, x)
    for _ in range(3):
        run(d, x["pool_blocks"])
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        run(d, x["pool_blocks"])
    ttnn.synchronize_device(device)
    print(f"dsa_decode_state B={B}: {(time.perf_counter() - t0) / n * 1e6:.0f} us/call (eager)")
