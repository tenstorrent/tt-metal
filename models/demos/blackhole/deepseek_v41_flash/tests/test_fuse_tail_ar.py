# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Step A of the fused decode collective tail: the one-program all-reduce (tt/fuse_tail_ar.py) against the lean reduce_scatter + all_gather pair, 4x8 galaxy.

FT_JOBS (comma list): check (PCC of the fused all-reduce rows 0..7 against the fp32 sum, every chip), time (traced chain us per op, fused and stock pair), stop (phase-1-only variant, check it does not hang).
FT_N (chain length, default 4), FT_SEED.
"""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.fuse_tail_ar import FuseTailAR

D = 5120
N = int(os.environ.get("FT_N", "4"))
JOBS = os.environ.get("FT_JOBS", "check,time").split(",")
IN_L1 = os.environ.get("FT_IN", "dram") == "l1"
LINKS = int(os.environ.get("FT_LINKS", "2"))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


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


def stock_pair(md, sems, part):
    sc = ttnn.experimental.reduce_scatter_minimal_async(
        part,
        dim=3,
        multi_device_global_semaphore=sems.nxt(0, sems.rs),
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        cluster_axis=1,
        barrier_semaphore=sems.nxt(2, sems.bar),
        num_workers_per_link=1,
    )
    return ttnn.experimental.all_gather_async(
        sc,
        dim=3,
        cluster_axis=1,
        mesh_device=md,
        topology=ttnn.Topology.Ring,
        multi_device_global_semaphore=sems.nxt(1, sems.ag),
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=sems.nxt(2, sems.bar),
        num_workers_per_link=1,
    )


def trace_us(md, fn, n):
    fn(1)
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    try:
        keep = fn(n)
    finally:
        ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(md, tid)
    return min(ts), keep


def per_iter(md, fn):
    t1, _ = trace_us(md, fn, 1)
    tn, keep = trace_us(md, fn, 1 + N)
    return round((tn - t1) / N, 1), round(t1, 1)


@pytest.fixture(scope="module")
def mesh_device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=200000000)
    yield md
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def make_part(md, seed, l1=False):
    g = torch.Generator().manual_seed(seed)
    host = torch.randn(4 * 32, 8 * D, generator=g).bfloat16()  # per chip [32, 5120]
    mc = ttnn.DRAM_MEMORY_CONFIG
    if l1:
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                [32, 640],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    part = ttnn.from_torch(
        host,
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mc,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=tuple(md.shape), dims=(0, 1)),
    )
    return host, ttnn.reshape(part, (1, 1, 32, D))


def golden(host):
    # [4*32, 8*D] -> per mesh row r: sum over the 8 column chips of the valid rows 0..7
    h = host.float().reshape(4, 32, 8, D)[:, :8]  # [r, row, c, D]
    return h.sum(dim=2)  # [r, 8, D]


@pytest.mark.timeout(3000)
def test_fuse_tail_ar(mesh_device):
    md = mesh_device
    seed = int(os.environ.get("FT_SEED", "1"))
    host, part = make_part(md, seed, IN_L1)
    _, part_dram = make_part(md, seed)
    gold = golden(host)
    stop = 1 if "stop" in JOBS else 0
    ar = FuseTailAR(md, stop=stop, in_l1=IN_L1, links=LINKS)
    res = {}
    if "check" in JOBS or "stop" in JOBS:
        out = ar(part)
        ttnn.synchronize_device(md)
        print("ft: first call done", flush=True)
        if "stop" in JOBS:
            return
        outs = ttnn.get_device_tensors(out)
        worst = 1.0
        for r in range(4):
            for c in range(8):
                got = FuseTailAR.decode(ttnn.to_torch(outs[r * 8 + c]).reshape(32, -1))
                p = pcc(got, gold[r])
                worst = min(worst, p)
                if p < 0.999:
                    print(f"ft: chip ({r},{c}) pcc {p:.5f}", flush=True)
        res["pcc_min"] = worst
        # repeat: second call (stale semaphores / buffers)
        for _ in range(3):
            ar(part)
        ttnn.synchronize_device(md)
        outs = ttnn.get_device_tensors(ar.out)
        got = FuseTailAR.decode(ttnn.to_torch(outs[0]).reshape(32, -1))
        res["pcc_after_repeat"] = pcc(got, gold[0])
    if "time" in JOBS:
        host2, part2 = make_part(md, seed + 7, IN_L1)
        gold2 = golden(host2)
        res["fused_us"], res["fused_n1_us"] = per_iter(md, lambda n: [ar(part2) for _ in range(n)])
        ttnn.synchronize_device(md)
        res["pcc_traced"] = min(
            pcc(FuseTailAR.decode(ttnn.to_torch(t).reshape(32, -1)), gold2[i // 8])
            for i, t in enumerate(ttnn.get_device_tensors(ar.out))
        )
        sems = Sems(md)
        res["stock_us"], res["stock_n1_us"] = per_iter(
            md, lambda n: [stock_pair(md, sems, part_dram) for _ in range(n)]
        )
    print("FTRES " + json.dumps(res), flush=True)


@pytest.mark.timeout(3000)
def test_fuse_tail_ar_prof(mesh_device):
    """FT_JOBS=prof: eager calls with the device profiler (TT_METAL_DEVICE_PROFILER=1 in the environment); the digest is read after 3 warm calls and after 3 measured calls."""
    md = mesh_device
    host, part = make_part(md, 1, IN_L1)
    ar = FuseTailAR(md, in_l1=IN_L1, links=LINKS)
    for _ in range(3):
        ar(part)
    ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    for _ in range(3):
        ar(part)
        ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    print("FTPROF done", flush=True)


@pytest.mark.timeout(3000)
def test_fuse_tail_ar_skew(mesh_device):
    """FT_JOBS=skew: FT_IT distinct inputs through a traced chain [random per-chip delay, all-reduce, copy of the result to DRAM]; every iteration's result is checked (buffer reuse / semaphore hazards under chip skew).
    FT_DELAY (max delay us, default 40), FT_REPLAYS (default 5)."""
    from models.demos.blackhole.deepseek_v41_flash.tt.fuse_tail_ar import delay_op

    md = mesh_device
    it = int(os.environ.get("FT_IT", "24"))
    maxd = int(os.environ.get("FT_DELAY", "40"))
    replays = int(os.environ.get("FT_REPLAYS", "5"))
    ar = FuseTailAR(md, in_l1=IN_L1, links=LINKS)
    parts, golds = [], []
    for i in range(it):
        h, p = make_part(md, 100 + i, IN_L1)
        parts.append(p)
        golds.append(golden(h))
    dummy = ttnn.from_torch(
        torch.zeros(32, 32),
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    g = torch.Generator().manual_seed(5)
    delays = [torch.randint(0, maxd + 1, (32,), generator=g).tolist() for _ in range(it)]

    def run():
        outs = []
        for i in range(it):
            delay_op(md, dummy, delays[i])
            o = ar(parts[i])
            outs.append(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG))
        return outs

    run()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    outs = run()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    worst = 1.0
    for rep in range(replays):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        for i, o in enumerate(outs):
            for idx, t in enumerate(ttnn.get_device_tensors(o)):
                p = pcc(FuseTailAR.decode(ttnn.to_torch(t).reshape(32, -1)), golds[i][idx // 8])
                worst = min(worst, p)
                if p < 0.999:
                    print(f"ft: replay {rep} iter {i} chip {idx} pcc {p:.4f}", flush=True)
                    break
    print("FTRES " + json.dumps(dict(skew_pcc_min=worst, it=it, delay=maxd)), flush=True)


def _up(md, t, dt=ttnn.float32):
    return ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


@pytest.mark.timeout(3000)
def test_fuse_tail_post(mesh_device):
    """FT_JOBS=post: all-reduce + mHC post in one program against the torch reference (PCC) and against stock RS + AG + mhc_expand3 (traced chain us per op). FT_T (5..8, default 8)."""
    from models.demos.blackhole.deepseek_v41_flash.tt import mhc_stream_pool
    from models.demos.blackhole.deepseek_v41_flash.tt.mhc_expand3 import mhc_expand3

    md = mesh_device
    T = int(os.environ.get("FT_T", "8"))
    host, part = make_part(md, 3, IN_L1)
    _, part_dram = make_part(md, 3)
    gold = golden(host)[:, :T]  # [4, T, D]
    torch.manual_seed(11)
    x = torch.randn(T, 1, 4, D)
    post = torch.rand(T, 1, 4, 1) * 2
    comb = torch.rand(T, 1, 4, 4)
    xt, pt, ct = _up(md, x), _up(md, post), _up(md, comb)
    buf = mhc_stream_pool.get(md, T, D, "ft", "attn")
    ar = FuseTailAR(md, in_l1=IN_L1, links=LINKS)
    res = {}
    ar.call_post(part, xt, ct, pt, buf)
    ttnn.synchronize_device(md)

    def ref_for(y):  # y [T, D] -> new streams [T, 4, D]
        return post[:, 0, :, :] * y[:, None, :] + torch.einsum("tij,tid->tjd", comb[:, 0], x[:, 0])

    worst = 1.0
    for idx, t in enumerate(ttnn.get_device_tensors(buf)):
        got = ttnn.to_torch(t)[:, 0].float()  # [T, 4, D]
        p = pcc(got, ref_for(gold[idx // 8]))
        worst = min(worst, p)
        if p < 0.999:
            print(f"ft: chip {idx} post pcc {p:.5f}", flush=True)
    res["post_pcc_min"] = worst
    if "time" in JOBS:
        for _ in range(3):
            ar.call_post(part, xt, ct, pt, buf)
        res["fused_post_us"], res["fused_post_n1_us"] = per_iter(
            md, lambda n: [ar.call_post(part, xt, ct, pt, buf) for _ in range(n)]
        )
        sems = Sems(md)
        yb = mhc_stream_pool.get(md, T, D, "ft", "stock")

        def stock(n):
            outs = []
            for _ in range(n):
                y = stock_pair(md, sems, part_dram)
                y = ttnn.reshape(y, (1, 1, T, D), (1, 1, 32, D))
                outs.append(mhc_expand3(y, xt, pt, ct, yb))
            return outs

        res["stock_post_us"], res["stock_post_n1_us"] = per_iter(md, stock)
    print("FTRES " + json.dumps(res), flush=True)


@pytest.mark.timeout(3000)
def test_fuse_tail_post_prof(mesh_device):
    """FT_JOBS=prof for the fused all-reduce + mHC post (device profiler digest after 3 warm + 3 measured eager calls)."""
    from models.demos.blackhole.deepseek_v41_flash.tt import mhc_stream_pool

    md = mesh_device
    T = 8
    host, part = make_part(md, 3, IN_L1)
    torch.manual_seed(11)
    xt, pt, ct = _up(md, torch.randn(T, 1, 4, D)), _up(md, torch.rand(T, 1, 4, 1)), _up(md, torch.rand(T, 1, 4, 4))
    buf = mhc_stream_pool.get(md, T, D, "ft", "attn")
    ar = FuseTailAR(md, in_l1=IN_L1, links=LINKS)
    for _ in range(3):
        ar.call_post(part, xt, ct, pt, buf)
    ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    for _ in range(3):
        ar.call_post(part, xt, ct, pt, buf)
        ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    print("FTPROF done", flush=True)


@pytest.mark.timeout(3000)
def test_stock_sum_order(mesh_device):
    """Reverse engineering of the summation order of the stock lean reduce_scatter + all_gather (ring, 2 links, 1 worker): for every tile column of the result, which candidate chain order
    (forward / backward ring, bf16 rounding after every add) reproduces the stock bits exactly. Prints the match fraction per candidate and the winning candidate per tile column.
    """
    md = mesh_device
    host, part = make_part(md, 21)
    sems = Sems(md)
    out = stock_pair(md, sems, part)
    ttnn.synchronize_device(md)
    outs = ttnn.get_device_tensors(out)
    h = host.reshape(4, 32, 8, D)[:, :8].to(torch.bfloat16)  # [r, row, c, D]
    res = {}
    for r in (0, 1):
        got = ttnn.to_torch(outs[r * 8]).reshape(32, -1)[:8].to(torch.bfloat16)  # [8, D]
        cands = {}
        for j in range(8):
            sl = slice(j * 640, (j + 1) * 640)
            for name, step in (("fwd", 1), ("bwd", -1)):
                order = [(j + step * k) % 8 for k in range(1, 8)] + [j]
                a = h[r, :, order[0], sl].float()
                for o in order[1:]:
                    a = (a + h[r, :, o, sl].float()).to(torch.bfloat16).float()
                cands.setdefault(name, torch.zeros(8, D))[:, sl] = a
            f32 = h[r, :, :, sl].float().sum(dim=1)
            cands.setdefault("fp32sum", torch.zeros(8, D))[:, sl] = f32.to(torch.bfloat16).float()
        g = got.float()
        for name, c in cands.items():
            eq = (c == g).reshape(8, 160, 32).float().mean(dim=(0, 2))  # per tile column: fraction of equal elements
            res[f"r{r}_{name}_elem_equal"] = round(float(eq.mean()), 4)
            res[f"r{r}_{name}_tile_slices_80"] = [round(float(eq[k * 20 : (k + 1) * 20].mean()), 2) for k in range(8)]
            if False:
                res[f"r{r}_{name}_first_tiles"] = [int(t) for t in eq.nonzero().flatten()[:40]]
    print("FTRES " + json.dumps(res), flush=True)


@pytest.mark.timeout(3000)
def test_sum_error(mesh_device):
    """Relative L2 error against the fp32 sum of the stock RS + AG and of the fused all-reduce (same bf16 inputs, heavy-tailed data like real activations)."""
    md = mesh_device
    g = torch.Generator().manual_seed(33)
    base = torch.randn(4 * 32, 8 * D, generator=g)
    scale = torch.exp(torch.randn(4 * 32, 8 * D, generator=g) * 1.5)  # log-normal magnitudes
    host = (base * scale).bfloat16()
    part = ttnn.from_torch(
        host,
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, mesh_shape=tuple(md.shape), dims=(0, 1)),
    )
    part = ttnn.reshape(part, (1, 1, 32, D))
    gold = golden(host)  # [4, 8, D] fp32 sum of the bf16 inputs
    sems = Sems(md)
    so = stock_pair(md, sems, part)
    ar = FuseTailAR(md)
    fo = ar(part)
    ttnn.synchronize_device(md)
    res = {}
    for r in range(4):
        s = ttnn.to_torch(ttnn.get_device_tensors(so)[r * 8]).reshape(32, -1)[:8].float()
        f = FuseTailAR.decode(ttnn.to_torch(ttnn.get_device_tensors(fo)[r * 8]).reshape(32, -1))
        rel = lambda a: float((a - gold[r]).norm() / gold[r].norm())
        res[f"r{r}"] = dict(
            stock_rel=round(rel(s), 6),
            fused_rel=round(rel(f), 6),
            stock_vs_fused=round(float((s - f).norm() / gold[r].norm()), 6),
        )
    print("FTRES " + json.dumps(res), flush=True)
