# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill collective micro-benchmarks on the 4x8 galaxy (fusedccl study). The 8-chip axis is mesh axis 1 (cluster_axis=1), the 4-chip axis is axis 0.

    FC_JOBS="rs:R=512,links=2;ag:R=512,links=2" pytest tests/test_fusedccl_bench.py -s

Every job prints one ``FCRES {json}`` line: us per op from a trace of NREP back-to-back ops (min of 3 replays), and the min PCC of the eager result against torch.
Jobs (comma separated key=value, defaults in the functions):
  rs     reduce_scatter_minimal_async of a [R, D] partial per chip over axis 1 (dim 2 = token scatter as the model's attention output, dim 3 = hidden scatter)
  ag     all_gather_async of [R/8, D] -> [R, D] over axis 1 (dim 2)
  ar     all_reduce composite of the model (rs dim 3 + ag dim 3, one worker)     [not used by the layer path any more]
  linrs  matmul wo_b-like ([R,K] x [K,N], bf16, minimal_matmul on the full grid) + rs, separate ops
  mmrs   minimal_matmul_strided_reduce_scatter_async (fused), config from mmrs_rules.pick_v23 unless mb/kb/nb/gy are given
  a2a    all_to_all_async_generic hidden shard -> own tokens ([R,D/8] -> [R/8,D])
"""

import json
import math
import os
import time

import pytest
import torch

import ttnn

D = 5120
NREP = int(os.environ.get("FC_NREP", "6"))
TOPO = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}


def _router_cfg(payload):
    c = ttnn._ttnn.fabric.FabricRouterConfig()
    c.max_packet_payload_size_bytes = payload
    return c


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else 1.0


class Sems:
    def __init__(self, md):
        g = md.compute_with_storage_grid_size()
        self.cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        mk = lambda: ttnn.create_global_semaphore(md, self.cores, 0)
        self.rs = [[mk() for _ in range(3)] for _ in range(2)]
        self.ag = [[mk() for _ in range(2)] for _ in range(2)]
        self.bar = [mk() for _ in range(2)]
        self.i = [0, 0, 0]
        ttnn.synchronize_device(md)

    def nxt(self, k, lst):
        v = lst[self.i[k]]
        self.i[k] ^= 1
        return v

    def r(self):
        return self.nxt(0, self.rs)

    def a(self):
        return self.nxt(1, self.ag)

    def b(self):
        return self.nxt(2, self.bar)


def shard(md, t, dims=(0, 1), dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=tuple(md.shape), dims=dims),
    )


def gather(md, t, dims=(0, 1)):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(md, mesh_shape=tuple(md.shape), dims=dims)).float()


def bench(md, fn, free=True):
    """fn() -> list of output tensors (kept alive by the caller until freed). Returns us/op (trace of NREP ops, min of 3)."""
    outs = fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        res = [fn() for _ in range(NREP)]
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / NREP * 1e6)
    ttnn.release_trace(md, tid)
    return min(ts), outs, res


def kv(s):
    d = {}
    for p in s.split(","):
        if p:
            k, v = p.split("=")
            d[k] = v
    return d


def ival(d, k, default):
    return int(d[k]) if k in d else default


def rs_kwargs(d, sems):
    kw = dict(
        multi_device_global_semaphore=sems.r(),
        num_links=ival(d, "links", 2),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=TOPO[d.get("topo", "ring")],
        cluster_axis=1,
        barrier_semaphore=sems.b(),
    )
    for k, n in (("w", "num_workers_per_link"), ("cps", "chunks_per_sync"), ("nb", "num_buffers_per_channel")):
        if k in d:
            kw[n] = int(d[k])
    return kw


def job_rs(md, d, sems):
    R, dim, Dm = ival(d, "R", 512), ival(d, "dim", 2), ival(d, "D", D)
    x = torch.randn(4, 8, R, Dm).bfloat16().float()
    tx = shard(md, x)
    f = lambda: [ttnn.experimental.reduce_scatter_minimal_async(tx, dim=dim, **rs_kwargs(d, sems))]
    us, outs, res = bench(md, f)
    got = gather(md, outs[0])  # [4, 8*R/8 ...]
    full = x.sum(dim=1)  # [4,R,D]
    exp = torch.stack(
        [
            full[:, c * (R // 8) : (c + 1) * (R // 8)] if dim == 2 else full[:, :, c * (Dm // 8) : (Dm // 8) * (c + 1)]
            for c in range(8)
        ],
        dim=1,
    )
    pcc = _pcc(got.reshape(exp.shape), exp)
    return dict(us=us, pcc=pcc, mb=4 * R * Dm * 2 / 1e6 / 4)


def job_ag(md, d, sems):
    R, Dm = ival(d, "R", 512), ival(d, "D", D)
    x = torch.randn(4, 8, R // 8, Dm).bfloat16().float()
    tx = shard(md, x)

    def f():
        kw = dict(
            dim=2,
            cluster_axis=1,
            mesh_device=md,
            topology=TOPO[d.get("topo", "ring")],
            multi_device_global_semaphore=sems.a(),
            num_links=ival(d, "links", 2),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=sems.b(),
        )
        for k, n in (("w", "num_workers_per_link"), ("cps", "chunks_per_sync"), ("nb", "num_buffers_per_channel")):
            if k in d:
                kw[n] = int(d[k])
        if d.get("bcast") == "1":
            kw["use_broadcast"] = True
        return [ttnn.experimental.all_gather_async(tx, **kw)]

    us, outs, res = bench(md, f)
    got = gather(md, outs[0])  # [4, 8, R, D]
    exp = torch.cat([x[:, j] for j in range(8)], dim=1)  # [4, R, D]
    pcc = _pcc(got, exp.unsqueeze(1).expand(-1, 8, -1, -1))
    return dict(us=us, pcc=pcc, mb=R * Dm * 2 / 1e6)


def job_a2a(md, d, sems):
    R, Dm = ival(d, "R", 512), ival(d, "D", D)
    x = torch.randn(4, 8, R, Dm // 8).bfloat16().float()
    tx = shard(md, x)
    f = lambda: [
        ttnn.experimental.all_to_all_async_generic(
            tx, in_dim=3, out_dim=2, num_links=ival(d, "links", 2), topology=TOPO[d.get("topo", "ring")], cluster_axis=1
        )
    ]
    us, outs, res = bench(md, f)
    return dict(us=us, pcc=-1)


def job_agmm(md, d, sems):
    """all_gather_minimal_matmul_async (gathers K = the hidden dim over axis 1, i.e. the activation is HIDDEN-sharded; our attention input is token-sharded, so this is the closest shape test):
    [R, K/8] per chip x [K, N] -> [R, N]; vs all_gather_async dim 3 + minimal_matmul (use_non_fused). Not our token-gather pattern.
    """
    R, K, N = ival(d, "R", 1024), ival(d, "K", 5120), ival(d, "N", 1792)
    g = md.compute_with_storage_grid_size()
    x = torch.randn(4 * R, K).bfloat16().float()
    w = (torch.randn(K, N) / math.sqrt(K)).bfloat16().float()
    tx = ttnn.from_torch(
        x,
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=tuple(md.shape), dims=[0, 1]),
    )
    tw = ttnn.from_torch(
        w,
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    buf = ttnn.from_torch(
        torch.zeros(R, K),
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    ck = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=ival(d, "mb", 8),
        K_block_size=ival(d, "kb", 8),
        N_block_size=ival(d, "nb_", 8),
        subblock_h=2,
        subblock_w=1,
        compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y - ival(d, "gm", 1)),
    )
    sem2 = [sems.a()[0], sems.a()[1]]

    def f():
        return [
            ttnn.experimental.all_gather_minimal_matmul_async(
                tx,
                tw,
                compute_kernel_config=ck,
                config=cfg,
                persistent_output_buffer=buf,
                multi_device_global_semaphore=sem2,
                num_links=ival(d, "links", 2),
                topology=TOPO[d.get("topo", "ring")],
                cluster_axis=1,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=ival(d, "w", 4),
                num_buffers_per_channel=ival(d, "nbuf", 48),
            )
        ]

    us, outs, res = bench(md, f)
    got = ttnn.to_torch(
        outs[0], mesh_composer=ttnn.ConcatMesh2dToTensor(md, mesh_shape=tuple(md.shape), dims=[0, 1])
    ).float()  # [4R, 8N]
    exp = x @ w
    pc = min(
        _pcc(got[r * R : (r + 1) * R, c * N : (c + 1) * N], exp[r * R : (r + 1) * R])
        for r in range(4)
        for c in range(8)
    )

    def f2():
        gth = ttnn.experimental.all_gather_async(
            ttnn.reshape(tx, [1, 1, R, K // 8]),
            dim=3,
            cluster_axis=1,
            mesh_device=md,
            topology=TOPO[d.get("topo", "ring")],
            multi_device_global_semaphore=sems.a(),
            num_links=ival(d, "links", 2),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=sems.b(),
        )
        return [gth, ttnn.experimental.minimal_matmul(gth, tw, compute_kernel_config=ck, config=cfg)]

    us2, _, _ = bench(md, f2)
    return dict(us=us, us_separate=us2, pcc=pc)


def _divisor_le(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _subblock(pm, pn, cap=4):
    best = (1, 1)
    for h in range(1, pm + 1):
        if pm % h:
            continue
        for w in range(1, pn + 1):
            if pn % w == 0 and h * w <= cap and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def mm_cfg(grid_xy, m_tiles, k_tiles, n_tiles):
    """2D mcast matmul config on a (gx, gy) grid (as tt/moe_overlap.py of the moeoverlap branch)."""
    gx, gy = grid_xy
    pm = math.ceil(m_tiles / gy)
    pn = math.ceil(n_tiles / gx)
    sh, sw = _subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=_divisor_le(k_tiles, 8),
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


def job_ovl(md, d, sems):
    """Sub-device overlap primitive: a collective (rs | ag) on sub-device 0 (rows 0..rows-1) next to an INDEPENDENT bf8 matmul ([R,K]x[K,N], 2D mcast config) on sub-device 1 (the rest),
    traced; reports collective alone, matmul alone (both on their sub-device), both issued back to back (concurrent if the sum is > the max), and the full-grid sequential references.
    """
    R, Dm = ival(d, "R", 512), ival(d, "D", D)
    K, N = ival(d, "K", 5120), ival(d, "N", 1792)
    rows = ival(d, "rows", 2)
    op = d.get("op", "rs")
    g = md.compute_with_storage_grid_size()
    ccl_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, rows - 1))})
    mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    if op == "rs":
        x = torch.randn(4, 8, R, Dm).bfloat16().float()
    else:
        x = torch.randn(4, 8, R // 8, Dm).bfloat16().float()
    tx = shard(md, x)
    a = torch.randn(4, 8, R, K).bfloat16().float()
    ta = shard(md, a)
    tw = shard(md, (torch.randn(4, 8, K, N) / math.sqrt(K)).bfloat16().float(), dtype=ttnn.bfloat8_b)
    ck = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def coll(sd):
        kw = dict(
            num_links=ival(d, "links", 2),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPO[d.get("topo", "ring")],
            cluster_axis=1,
        )
        if "w" in d:
            kw["num_workers_per_link"] = int(d["w"])
        if sd is not None:
            kw["subdevice_id"] = sd
        if op == "rs":
            return ttnn.experimental.reduce_scatter_minimal_async(
                tx, dim=2, multi_device_global_semaphore=sems.r(), barrier_semaphore=sems.b(), **kw
            )
        return ttnn.experimental.all_gather_async(
            tx, dim=2, mesh_device=md, multi_device_global_semaphore=sems.a(), barrier_semaphore=sems.b(), **kw
        )

    def mm(grid, sd):
        pc = mm_cfg(grid, R // 32, K // 32, N // 32)
        kw = dict(program_config=pc, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sd is not None:
            kw["sub_device_id"] = sd
        return ttnn.matmul(ta, tw, **kw)

    res = {}
    res["coll_full"], _, _ = bench(md, lambda: [coll(None)])
    res["mm_full"], _, _ = bench(md, lambda: [mm((g.x, g.y), None)])
    res["seq_full"], _, _ = bench(md, lambda: [coll(None), mm((g.x, g.y), None)])
    mgr = md.create_sub_device_manager([ttnn.SubDevice([ccl_cores]), ttnn.SubDevice([mm_cores])], 0)
    cid, mid = ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)
    md.load_sub_device_manager(mgr)
    md.set_sub_device_stall_group([cid, mid])
    try:
        res["coll_sd"], _, _ = bench(md, lambda: [coll(cid)])
        res["mm_sd"], _, _ = bench(md, lambda: [mm((g.x, g.y - rows), mid)])
        res["both_sd"], outs, _ = bench(md, lambda: [coll(cid), mm((g.x, g.y - rows), mid)])
        got = gather(md, outs[0])
        if op == "rs":
            exp = rs_expect(x.sum(dim=1), 2, R, Dm)
        else:
            exp = torch.cat([x[:, j] for j in range(8)], dim=1).unsqueeze(1).expand(-1, 8, -1, -1)
        res["pcc"] = _pcc(got.reshape(exp.shape), exp)
    finally:
        md.reset_sub_device_stall_group()
        md.clear_loaded_sub_device_manager()
    return res


def job_conv(md, d, sems):
    """to_layout cost: [rows, cols] TILE -> ROW_MAJOR (and back) for bf16 / uint32 (the candidate tensors of the indexer gather: [C, 8*512] after the all-gather)."""
    rows, cols = ival(d, "rows", 512), ival(d, "cols", 4096)
    dt = ttnn.uint32 if d.get("dt", "bf16") == "u32" else ttnn.bfloat16
    src = d.get("src", "tile")
    x = (
        torch.randint(0, 100, (4, 8, rows, cols))
        if dt == ttnn.uint32
        else torch.randn(4, 8, rows, cols).bfloat16().float()
    )
    tx = ttnn.from_torch(
        x,
        device=md,
        layout=ttnn.TILE_LAYOUT if src == "tile" else ttnn.ROW_MAJOR_LAYOUT,
        dtype=dt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=tuple(md.shape), dims=(0, 1)),
    )
    tgt = ttnn.ROW_MAJOR_LAYOUT if src == "tile" else ttnn.TILE_LAYOUT
    us, outs, res = bench(md, lambda: [ttnn.to_layout(tx, tgt)])
    return dict(us=us, pcc=-1)


def job_mmrows(md, d, sems):
    """[rows, K] x [K, N] bf8 weight, minimal_matmul default config, HiFi2: cost of a per-token projection on all R rows (replicated 8x today) vs on the R/8 own rows."""
    rows, K, N = ival(d, "rows", 1024), ival(d, "K", 5120), ival(d, "N", 1792)
    x = torch.randn(4, 8, rows, K).bfloat16().float()
    tx = shard(md, x)
    tw = ttnn.from_torch(
        (torch.randn(K, N) / math.sqrt(K)).bfloat16().float(),
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    ck = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    us, outs, res = bench(md, lambda: [ttnn.experimental.minimal_matmul(tx, tw, compute_kernel_config=ck)])
    return dict(us=us, pcc=-1)


def job_agrm(md, d, sems):
    """all-gather of a small [rows, cols] tensor per chip over axis 1 (dim 2), ROW_MAJOR vs TILE, bf16 vs uint32: the indexer candidate gathers ([512,512] RM, ~200 us in situ)."""
    rows, cols = ival(d, "rows", 512), ival(d, "cols", 512)
    lay = ttnn.ROW_MAJOR_LAYOUT if d.get("lay", "rm") == "rm" else ttnn.TILE_LAYOUT
    dt = ttnn.uint32 if d.get("dt", "bf16") == "u32" else ttnn.bfloat16
    x = (
        torch.randint(0, 100, (4, 8, rows, cols)).float()
        if dt == ttnn.uint32
        else torch.randn(4, 8, rows, cols).bfloat16().float()
    )
    tx = ttnn.from_torch(
        x.to(torch.int64) if dt == ttnn.uint32 else x,
        device=md,
        layout=lay,
        dtype=dt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=tuple(md.shape), dims=(0, 1)),
    )

    def f():
        return [
            ttnn.experimental.all_gather_async(
                tx,
                dim=ival(d, "dim", 2),
                cluster_axis=1,
                mesh_device=md,
                topology=TOPO[d.get("topo", "ring")],
                multi_device_global_semaphore=sems.a(),
                num_links=ival(d, "links", 2),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=sems.b(),
            )
        ]

    us, outs, res = bench(md, f)
    return dict(us=us, pcc=-1, in_MB=rows * cols * (4 if dt == ttnn.uint32 else 2) / 1e6)


def job_rshid(md, d, sems):
    """the unified MoE's column-partial reduction (reduce_scatter_tokens mode 'hidden'): generic ttnn.reduce_scatter over the hidden dim + all_to_all to own tokens.
    kind=gen (ttnn.reduce_scatter) | min (reduce_scatter_minimal_async dim 3); a2a topo/links separately (a2a_links)."""
    R, Dm = ival(d, "R", 512), ival(d, "D", D)
    x = torch.randn(4, 8, R, Dm).bfloat16().float()
    tx = shard(md, x)
    kind = d.get("kind", "gen")

    def f():
        if kind == "gen":
            r_ = ttnn.reduce_scatter(
                tx,
                dim=3,
                cluster_axis=1,
                num_links=ival(d, "links", 2),
                topology=TOPO[d.get("topo", "ring")],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            r_ = ttnn.experimental.reduce_scatter_minimal_async(tx, dim=3, **rs_kwargs(d, sems))
        o = ttnn.experimental.all_to_all_async_generic(
            r_, in_dim=3, out_dim=2, num_links=ival(d, "a2a_links", 2), topology=ttnn.Topology.Ring, cluster_axis=1
        )
        return [o]

    us, outs, res = bench(md, f)
    got = gather(md, outs[0])
    exp = rs_expect(x.sum(dim=1), 2, R, Dm)
    return dict(us=us, pcc=_pcc(got.reshape(exp.shape), exp))


def mm_ref(x, w):
    # x [4,8,R,K], w [8,K,N] -> per (row r): sum_j x[r,j] @ w[j]  -> [4,R,N]
    return torch.stack([sum(x[r, j] @ w[j] for j in range(8)) for r in range(4)])


def mm_setup(md, d):
    R, K, N = ival(d, "R", 512), ival(d, "K", 1024), ival(d, "N", D)
    x = torch.randn(4, 8, R, K).bfloat16().float()
    w = (torch.randn(8, K, N) / math.sqrt(K)).bfloat16().float()
    tx = shard(md, x)
    tw = shard(md, w.unsqueeze(1).unsqueeze(0).expand(4, -1, -1, -1, -1).reshape(4, 8, K, N), dims=(0, 1))
    return R, K, N, x, w, tx, tw


def rs_expect(full, dim, R, N):
    return torch.stack(
        [
            full[:, c * (R // 8) : (c + 1) * (R // 8)] if dim == 2 else full[:, :, c * (N // 8) : (c + 1) * (N // 8)]
            for c in range(8)
        ],
        dim=1,
    )


def job_linrs(md, d, sems):
    R, K, N, x, w, tx, tw = mm_setup(md, d)
    dim = ival(d, "dim", 2)
    ck = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    g = md.compute_with_storage_grid_size()
    mmc = ttnn.MinimalMatmulConfig(
        M_block_size=ival(d, "mb", 2),
        K_block_size=ival(d, "kb", 2),
        N_block_size=ival(d, "nb_", 8),
        subblock_h=2,
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y),
    )

    def f():
        if d.get("mm", "min") == "min":
            y = ttnn.experimental.minimal_matmul(tx, tw, compute_kernel_config=ck, config=mmc)
        else:
            y = ttnn.linear(tx, tw, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return [y, ttnn.experimental.reduce_scatter_minimal_async(y, dim=dim, **rs_kwargs(d, sems))]

    us, outs, res = bench(md, f)
    # time of the matmul alone
    us_mm, _, _ = bench(
        md,
        lambda: [ttnn.experimental.minimal_matmul(tx, tw, compute_kernel_config=ck, config=mmc)]
        if d.get("mm", "min") == "min"
        else [ttnn.linear(tx, tw, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)],
    )
    got = gather(md, outs[1])
    exp = rs_expect(mm_ref(x, w), dim, R, N)
    return dict(us=us, us_mm=us_mm, pcc=_pcc(got.reshape(exp.shape), exp))


def job_mmrs(md, d, sems):
    R, K, N, x, w, tx, tw = mm_setup(md, d)
    dim = ival(d, "dim", 3)
    sys_path = os.path.dirname(__file__)  # noqa: F841
    from models.tt_dit.utils.mmrs_rules import pick_v23

    p = pick_v23(R, K, N) or {"mm_grid": (12, 8), "blocks": (2, 2, 8), "subblock": (2, 2)}
    gx, gy = p["mm_grid"]
    gy = ival(d, "gy", gy)
    mb, kb, nb_ = ival(d, "mb", p["blocks"][0]), ival(d, "kb", p["blocks"][1]), ival(d, "nb_", p["blocks"][2])
    sh, sw = p["subblock"]
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb_,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )
    ck = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    g = md.compute_with_storage_grid_size()
    slots = g.x * g.y
    allc = sems.cores
    mk = lambda: ttnn.allocate_tensor_on_device(
        ttnn.Shape([slots, slots]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        md,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(allc, [1, slots], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    prog, cred = mk(), mk()
    mt_core = math.ceil(math.ceil(R / 32) / gy)
    blocks_core = math.ceil(mt_core / mb)
    win = None if blocks_core <= 1 else min(ival(d, "win", 2), blocks_core)
    if "win" in d and d["win"] == "0":
        win = None

    def f():
        kw = dict(
            num_links=ival(d, "links", 2),
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPO[d.get("topo", "ring")],
            cluster_axis=1,
            config=cfg,
            compute_kernel_config=ck,
            barrier_semaphore=sems.b(),
            chunk_width_in_mm_blocks=ival(d, "cw", 1),
            mm_progress_counters=prog,
            mm_credit_counters=cred,
            mm_window_blocks=win,
        )
        if "w" in d:
            kw["num_workers_per_link"] = int(d["w"])
        if "nbuf" in d:
            kw["num_buffers_per_channel"] = int(d["nbuf"])
        a, b = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            tx, tw, dim, sems.r(), ttnn.CoreCoord(0, gy), **kw
        )
        return [a, b]

    us, outs, res = bench(md, f)
    got = gather(md, outs[1])
    exp = rs_expect(mm_ref(x, w), dim, R, N)
    return dict(us=us, pcc=_pcc(got.reshape(exp.shape), exp), cfg=[gx, gy, mb, kb, nb_, win])


JOBS = dict(
    rs=job_rs,
    ag=job_ag,
    a2a=job_a2a,
    linrs=job_linrs,
    mmrs=job_mmrs,
    rshid=job_rshid,
    agrm=job_agrm,
    ovl=job_ovl,
    agmm=job_agmm,
    conv=job_conv,
    mmrows=job_mmrows,
)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            dict(
                {
                    "l1_small_size": 16384,
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                    "trace_region_size": 200000000,
                },
                **(
                    {"fabric_router_config": _router_cfg(int(os.environ["FC_ROUTER"]))}
                    if os.environ.get("FC_ROUTER")
                    else {}
                ),
            ),
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3000)
def test_fusedccl_bench(mesh_device):
    md = mesh_device
    sems = Sems(md)
    for job in os.environ["FC_JOBS"].split(";"):
        name, _, args = job.partition(":")
        d = kv(args)
        t0 = time.time()
        try:
            r = JOBS[name](md, d, sems)
            r.update(job=job, wall=round(time.time() - t0, 1))
        except Exception as e:  # report and go on; a hung CCL is not catchable (needs a board reset)
            r = dict(job=job, error=repr(e)[:300])
        print("FCRES " + json.dumps(r), flush=True)
        ttnn.synchronize_device(md)
