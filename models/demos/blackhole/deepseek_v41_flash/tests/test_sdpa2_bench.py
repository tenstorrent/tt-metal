# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""sdpa2: single-chip microbenchmark / accuracy test of the decode sparse_sdpa variants with a REALISTIC index pattern
(window = 128 contiguous ring rows of the user, 512 indexer picks scattered over page-table placed compressed rows, rows of one user consecutive).

Env: SS2_ROWS (8 | 12), SS2_NQ (rows per user; default 4 for 8 rows, 6 for 12), SS2_REPS (20), SS2_VARIANTS (comma list, default 'cur'),
SS2_NV ('full' | 'rand'), SS2_ITERS (stress iterations, default 1), SS2_CHECK (1: PCC vs torch reference + vs 'cur').
Variants: 'cur' = tt/sparse_split.py sparse_sdpa_split (DSV41_SPARSE_SPLIT=5 production, G from SS2_G (5)); others registered in VARIANTS.
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import sparse_split as SS

DP = {"l1_small_size": 16384}
R_POOL, D, H, TOPK, WIN, RING = 120000, 512, 32, 640, 128, 160


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - a.mean()) @ (b - b.mean()) / ((a - a.mean()).norm() * (b - b.mean()).norm() + 1e-30))


def _variants():
    v = {}

    def cur(q, kv, idx, scale, ckc, sink):
        return SS.sparse_sdpa_split(q, kv, idx, D, scale, 128, int(os.environ.get("SS2_GCUR", "5")), ckc, sink)

    v["cur"] = cur
    try:
        from models.demos.blackhole.deepseek_v41_flash.tt import sdpa2 as S2

        v.update(S2.VARIANTS)
    except ImportError:
        pass
    return v


def make_idx(S, nq, g, nv_mode):
    """[1,1,S,TOPK] int32 (-1 = sentinel) following paged_kv_step: window (ring rows) then compressed picks."""
    users = S // nq
    PAGE_ROWS, NPAGES = 32, 3000
    idx = torch.full((1, 1, S, TOPK), -1, dtype=torch.int64)
    ring_base = NPAGES * PAGE_ROWS
    for u in range(users):
        pos0 = 8000 + 37 * u
        n_ent = (pos0 + nq) // 4  # compressed entries (ratio 4)
        pt = torch.randperm(NPAGES, generator=g)[: (n_ent * 4) // 128 + 2]
        base_pick = torch.randperm(n_ent - 1, generator=g)[:512]
        for t in range(nq):
            r = u * nq + t
            p = pos0 + t
            win = [ring_base + u * RING + (q % RING) for q in range(p - WIN + 1, p + 1)]
            pk = base_pick.clone()
            sw = torch.randint(0, 512, (50,), generator=g)
            pk[sw] = torch.randint(0, n_ent - 1, (50,), generator=g)
            rows = [int(pt[(e * 4) // 128]) * PAGE_ROWS + ((e * 4) % 128) // 4 for e in pk.tolist()]
            allr = win + rows
            if nv_mode == "rand":
                allr = allr[: int(torch.randint(1, TOPK + 1, (1,), generator=g))]
            idx[0, 0, r, : len(allr)] = torch.tensor(allr)
    return idx


def torch_ref(q_t, kv_f, idx, sink, scale):
    """q_t [S,32,512] float, kv_f [R,512] float -> [S,32,512]"""
    S = q_t.shape[0]
    out = torch.zeros(S, H, D)
    for r in range(S):
        ids = idx[0, 0, r]
        ids = ids[ids >= 0]
        K = kv_f[ids]
        s = (q_t[r] @ K.T) * scale  # [32, nv]
        m = torch.maximum(s.max(-1).values, sink.float())
        e = torch.exp(s - m[:, None])
        den = e.sum(-1) + torch.exp(sink.float() - m)
        out[r] = (e @ K) / den[:, None]
    return out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="plain")], indirect=True)
@torch.no_grad()
def test_sdpa2_bench(mesh_device):
    md = mesh_device
    S = int(os.environ.get("SS2_ROWS", "8"))
    nq = int(os.environ.get("SS2_NQ", "4" if S == 8 else "6"))
    reps = int(os.environ.get("SS2_REPS", "20"))
    iters = int(os.environ.get("SS2_ITERS", "1"))
    nv_mode = os.environ.get("SS2_NV", "full")
    names = os.environ.get("SS2_VARIANTS", "cur").split(",")
    check = os.environ.get("SS2_CHECK", "1") == "1"
    V = _variants()
    g = torch.Generator().manual_seed(11)
    kv_t = (torch.randn(1, 1, R_POOL, D, generator=g) * 0.5).to(torch.float8_e4m3fn)
    kv_f = kv_t.float()[0, 0]
    up = lambda t, dt, lay: ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    kv = up(kv_t.float(), ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT)
    sink_t = torch.randn(1, 1, 1, H, generator=g).to(torch.bfloat16)
    sink = up(sink_t, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("SS2_FID", "HiFi4")),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    scale = 512**-0.5
    _ck = {}

    def CK(n):  # 'name@HiFi2' -> compute config with that fidelity
        if "@" not in n:
            return ckc
        f = n.split("@")[1]
        if f not in _ck:
            _ck[f] = ttnn.init_device_compute_kernel_config(
                md.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, f),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        return _ck[f]

    worst = {n: 1.0 for n in names}
    worst_vs_cur = {n: 1.0 for n in names}
    for it in range(iters):
        idx_t = make_idx(S, nq, g, nv_mode)
        qq = torch.zeros(S, H, D)
        qq[:, :9] = torch.randn(S, 9, D, generator=g) * float(os.environ.get("SS2_QSCALE", "2"))
        qq = qq.to(torch.bfloat16)
        q = up(qq.reshape(1, S, H, D), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        idx = up(idx_t.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ql = ttnn.to_memory_config(q, ttnn.L1_MEMORY_CONFIG)
        il = ttnn.to_memory_config(idx, ttnn.L1_MEMORY_CONFIG)
        crs = lambda n_: ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(i, 9), ttnn.CoreCoord(i, 9)) for i in range(n_)]
        )
        qsh = ttnn.to_memory_config(
            q,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs(S), (32, 512), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        ish = ttnn.to_memory_config(
            idx,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs(S), (1, 640), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        IN = lambda n: (qsh, ish) if n.startswith("v2sh") else ((ql, il) if n.startswith("v2l1") else (q, idx))
        ref = torch_ref(qq.float(), kv_f, idx_t, sink_t.flatten(), scale) if check else None
        outs = {}
        for n in names:
            o = ttnn.to_torch(
                ttnn.get_device_tensors(V[n.split("@")[0]](IN(n)[0], kv, IN(n)[1], scale, CK(n), sink))[0]
            ).float()  # [1,H,S,512]
            outs[n] = o[0].permute(1, 0, 2)  # [S,H,512]
        for n in names:
            if check:
                worst[n] = min(worst[n], min(pcc(outs[n][r, :9], ref[r, :9]) for r in range(S)))
                if "cur" in outs:
                    worst_vs_cur[n] = min(
                        worst_vs_cur[n], min(pcc(outs[n][r, :9], outs["cur"][r, :9]) for r in range(S))
                    )
    for n in names:
        print(f"SS2_PCC variant={n} S={S} nq={nq} iters={iters}: vs torch {worst[n]:.6f} vs cur {worst_vs_cur[n]:.6f}")
    for n in names:
        f = lambda: V[n.split("@")[0]](IN(n)[0], kv, IN(n)[1], scale, CK(n), sink)
        f()
        ttnn.synchronize_device(md)
        t0 = time.time()
        for _ in range(reps):
            f()
        ttnn.synchronize_device(md)
        print(f"SS2_WALL variant={n} S={S}: {(time.time()-t0)/reps*1e6:.1f} us/call (eager wall)")
    for n in names:
        assert worst[n] > 0.999, (n, worst[n])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="plain")], indirect=True)
@torch.no_grad()
def test_gather_floor(mesh_device):
    """Gather-only floor: configs 'S:G:depth:nocs:reps:mode' in SS2_GB (comma list); each runs 3x (profile ops in call order)."""
    if "SS2_GB" not in os.environ:
        pytest.skip("SS2_GB not set")
    from models.demos.blackhole.deepseek_v41_flash.tt import sdpa2 as S2

    md = mesh_device
    g = torch.Generator().manual_seed(11)
    kv_t = (torch.randn(1, 1, R_POOL, D, generator=g) * 0.5).to(torch.float8_e4m3fn)
    up = lambda t, dt, lay: ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    kv = up(kv_t.float(), ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT)
    cfgs = os.environ.get("SS2_GB", "8:5:8:2:1:0").split(",")
    idxs = {}
    for c in cfgs:
        S, G, depth, nocs, reps, mode = [int(x) for x in c.split(":")]
        if S not in idxs:
            nq = 4 if S == 8 else 6
            idxs[S] = up(make_idx(S, nq, g, "full").to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        for _ in range(3):
            S2.gather_bench(kv, idxs[S], S, G, depth, nocs, reps, mode)
        ttnn.synchronize_device(md)
        print("SS2_GB_CALL", c)
