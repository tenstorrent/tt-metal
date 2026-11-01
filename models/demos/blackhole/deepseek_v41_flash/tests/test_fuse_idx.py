# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Indexer fused kernels (tt/fuse_attn.py idx_q / idx_k / idx_local / idx_final) vs the unfused chains of DSV41PrefillIndexer (prefill_sparse.py), one chip, C=512 rows."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import fuse_attn as FA
from models.demos.blackhole.deepseek_v41_flash.tt import pf_tune
from models.demos.blackhole.deepseek_v41_flash.tt.attention import pair_swap_matrix

C, COLS, TOPK = 512, 8, 512
SKIP = 0xFFFFFFFF


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def down(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def mk(md):
    rep = ttnn.ReplicateTensorToMesh(md)
    return lambda t, dt, lay: ttnn.from_torch(
        t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="p")], indirect=True)
@pytest.mark.parametrize("nvalid", [300, 2000])
@torch.no_grad()
def test_idx_post(mesh_device, nvalid):
    md = mesh_device
    up = mk(md)
    torch.manual_seed(3)
    Lc = 2048
    sc = (torch.randn(1, 1, C, Lc + 32) * 3).to(torch.bfloat16)
    tsc = up(sc, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    cid = up(torch.full((1, 1, 1, 1), 5.0), ttnn.float32, ttnn.TILE_LAYOUT)
    nv = up(torch.full((1, 1, 1, 1), nvalid).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    ids = ttnn.experimental.topk_large_indices(tsc, k=TOPK, valid_length_tensor=nv)
    # unfused chain (select_single)
    idf = ttnn.to_layout(ttnn.typecast(ids, ttnn.float32), ttnn.TILE_LAYOUT)
    valid = ttnn.lt(idf, float(Lc))
    idc = ttnn.typecast(ttnn.where(valid, idf, 0.0), ttnn.uint32)
    vals = ttnn.where(
        ttnn.typecast(valid, ttnn.bfloat16),
        ttnn.to_layout(ttnn.gather(tsc, 3, index=ttnn.to_layout(idc, ttnn.ROW_MAJOR_LAYOUT)), ttnn.TILE_LAYOUT),
        float("-inf"),
    )
    pg = ttnn.typecast(ttnn.add(ttnn.multiply(idf, float(COLS)), cid), ttnn.uint32)
    rv = down(ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT)).float()
    rp = down(ttnn.to_layout(pg, ttnn.ROW_MAJOR_LAYOUT)).to(torch.int64) & 0xFFFFFFFF
    fv, fp = FA.idx_local(tsc, ids, cid, Lc, COLS, C)
    gv = down(fv).float()
    gp = down(fp).to(torch.int64) & 0xFFFFFFFF
    ev, ep = torch.equal(rv, gv), torch.equal(rp, gp)
    nbad = int((rp != gp).sum())
    print(f"idx_local nvalid={nvalid}: vals equal {ev}, pg equal {ep} (mismatches {nbad} of {rp.numel()})", flush=True)
    if nbad:
        bad = (rp != gp).nonzero()[:5]
        for b in bad:
            print(
                "  ref",
                int(rp[tuple(b)]),
                "got",
                int(gp[tuple(b)]),
                "id",
                int(down(ids).to(torch.int64)[tuple(b)] & 0xFFFFFFFF),
            )
    assert ev and ep
    # ---- merge post-processing
    cv = (torch.randn(1, 1, C, COLS * TOPK) * 3).to(torch.bfloat16)
    cv[..., 3000:] = float("-inf")  # fewer valid candidates than 512 in some rows
    cp = torch.randint(0, 100000, (1, 1, C, COLS * TOPK))
    cp[..., 3000:] = SKIP
    tcv, tcp = up(cv, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), up(cp.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    pos = ttnn.experimental.topk_large_indices(tcv, k=TOPK)
    posf = ttnn.to_layout(ttnn.typecast(pos, ttnn.float32), ttnn.TILE_LAYOUT)
    posu = ttnn.typecast(ttnn.clamp(posf, max=float(COLS * TOPK)), ttnn.uint32)
    skip = up(torch.full((1, 1, C, 32), SKIP, dtype=torch.int64).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    cpe = ttnn.concat([tcp, skip], dim=3)
    ref = down(ttnn.gather(cpe, 3, index=ttnn.to_layout(posu, ttnn.ROW_MAJOR_LAYOUT))).to(torch.int64) & 0xFFFFFFFF
    got = down(FA.idx_final(pos, tcp, C)).to(torch.int64) & 0xFFFFFFFF
    print(f"idx_final: equal {torch.equal(ref, got)} (mismatches {int((ref != got).sum())})", flush=True)
    assert torch.equal(ref, got)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="p")], indirect=True)
@pytest.mark.parametrize("U", [2])
@torch.no_grad()
def test_idx_q_k(mesh_device, U):
    md = mesh_device
    up = mk(md)
    torch.manual_seed(4)
    P = pf_tune.p64(md)
    P128 = torch.zeros(128, 128)
    P128[64:, 64:] = pair_swap_matrix()
    dP = up(P128.reshape(1, 1, 128, 128).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    pos = torch.arange(U * C).float().view(U, 1, C, 1)
    inv = 1.0 / (10000.0 ** (torch.arange(32).float() / 32))
    ang = pos * inv.view(1, 1, 1, 32)
    cos, sin = ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)
    c = torch.ones(U, 1, C, 512)
    s = torch.zeros(U, 1, C, 512)
    c[..., 448:], s[..., 448:] = cos, sin
    tabs = (up(c, ttnn.bfloat16, ttnn.TILE_LAYOUT), up(s, ttnn.bfloat16, ttnn.TILE_LAYOUT), None)
    # q
    q = up((torch.randn(1, 1, U * C, 32 * 128) * 2).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    qrm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
    qh = ttnn.to_layout(ttnn.permute(ttnn.reshape(qrm, [U, C, 32, 128]), (0, 2, 1, 3)), ttnn.TILE_LAYOUT)
    cc = ttnn.slice(tabs[0], [0, 0, 0, 384], [U, 1, C, 512])
    ss = ttnn.slice(tabs[1], [0, 0, 0, 384], [U, 1, C, 512])
    ref = down(ttnn.addcmul(ttnn.multiply(qh, cc), ttnn.matmul(qh, dP, compute_kernel_config=ckc), ss)).float()
    got = down(FA.idx_q(q, tabs, P, U, C)).float()
    print(
        f"idx_q: equal {torch.equal(ref, got)} max|d| {(ref - got).abs().max():.4g} pcc {pcc(ref, got):.7f}", flush=True
    )
    assert pcc(ref, got) > 0.9999
    # k
    k = up((torch.randn(U, 1, C, 128) * 2).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    ref = down(ttnn.addcmul(ttnn.multiply(k, cc), ttnn.matmul(k, dP, compute_kernel_config=ckc), ss)).float()
    got = down(FA.idx_k(k, tabs, P, U, C)).float()
    print(
        f"idx_k: equal {torch.equal(ref, got)} max|d| {(ref - got).abs().max():.4g} pcc {pcc(ref, got):.7f}", flush=True
    )
    assert pcc(ref, got) > 0.9999
