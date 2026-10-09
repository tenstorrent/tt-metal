# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""In-trace sampler alone (no model): real logits rows (GSM8K teacher-forced, dsv4-e2e-ref/logits_tf.pt) on the 4x8 mesh, against the CPU reference semantics
(exact top-k, then top-p nucleus over the renormalised top-k, draw = inverse CDF in vocabulary order at u). Also the cost of the sampler vs the candidate path and the greedy path, replayed from a trace.
"""

import time
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import VOCAB, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import DSV41DeviceSampler
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

LOGITS = "/mnt/tt-data/ssinghal/dsv4-e2e-ref/logits_tf.pt"


def ref_draw(l, T, tk, tp, u):
    p = torch.softmax(l.double() / T, 0)
    s, i = p.sort(descending=True)
    if tk > 0:
        s, i = s[:tk], i[:tk]
    s = s / s.sum()
    if tp < 1:
        s = s * ((s.cumsum(0) - s) < tp)
    q = torch.zeros_like(p)
    q[i] = s
    cum = q.cumsum(0)
    tok = int(torch.searchsorted(cum, torch.tensor(float(u)) * cum[-1], right=True).clamp(max=VOCAB - 1))
    return tok, q


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 << 20},
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_device_sampler(mesh_device):
    import os

    md = mesh_device
    rows, cols = tuple(md.shape)
    T = int(os.environ.get("SAMP_T", "8"))
    B = rows * T
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    smp = DSV41DeviceSampler(
        md, mc, ccl, T, J=int(os.environ.get("SAMP_J", "32")), levels=int(os.environ.get("SAMP_L", "4"))
    )
    prm = smp.alloc_params()
    L = torch.load(LOGITS).reshape(-1, VOCAB).float()
    g = torch.Generator().manual_seed(1)
    sel = torch.randperm(L.shape[0], generator=g)[:B]
    lg = L[sel]  # [B, V]
    logits = ttnn.from_torch(
        lg.reshape(1, 1, B, VOCAB),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, 3), mesh_shape=(rows, cols)),
    )
    cfgs = [
        (1.0, 0, 1.0),
        (1.0, 0, 0.95),
        (0.7, 0, 0.95),
        (1.0, 50, 0.95),
        (1.3, 0, 0.9),
        (1.0, 20, 1.0),
        (0.0, 1, 1.0),
        (1.0, 200, 0.9),
    ]
    temps = torch.tensor([cfgs[b % len(cfgs)][0] for b in range(B)])
    tks = torch.tensor([cfgs[b % len(cfgs)][1] for b in range(B)])
    tps = torch.tensor([cfgs[b % len(cfgs)][2] for b in range(B)])
    greedy = (temps == 0).float() + ((tks == 1).float())
    greedy = greedy.clamp(max=1)
    invT = torch.where(temps > 0, 1 / temps.clamp(min=1e-6), torch.ones(B))
    ud = torch.Generator().manual_seed(7)

    def run(u):
        smp.set_params(invT, tks, tps, u, greedy)
        return ttnn.to_torch(ttnn.get_device_tensors(smp.forward(logits, prm))[0]) if False else None

    # compile + capture
    smp.set_params(invT, tks, tps, torch.rand(B), greedy)
    out = smp.forward(logits, prm)
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    out = smp.forward(logits, prm)
    ttnn.end_trace_capture(md, tid, cq_id=0)

    def read():
        return torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(out)[r * cols]).reshape(-1) for r in range(rows)]
        ).long()

    N = int(os.environ.get("SAMP_N", "100"))
    bad = tot = outside = 0
    for it in range(N):
        u = torch.rand(B, generator=ud)
        smp.set_params(invT, tks, tps, u, greedy)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        got = read()
        for b in range(B):
            if greedy[b]:
                exp = int(lg[b].argmax())
                tot += 1
                bad += int(got[b]) != exp
                continue
            exp, q = ref_draw(lg[b], float(temps[b]), int(tks[b]), float(tps[b]), float(u[b]))
            tot += 1
            if int(got[b]) != exp:
                bad += 1
                outside += float(q[got[b]]) == 0.0
    print(
        f"SAMPLER_CHECK draws {tot}: token != reference (same u) {bad} ({100 * bad / max(tot, 1):.3f}%), of which outside the exact support {outside}",
        flush=True,
    )
    assert outside == 0 or outside / max(tot, 1) < 1e-3

    if os.environ.get("SAMP_PROF"):
        smp.prof = []
        smp.set_params(invT, tks, tps, torch.rand(B), greedy)
        smp.forward(logits, prm)
        smp.prof = []
        smp.forward(logits, prm)
        print("SAMPLER_PROF " + " ".join(f"{n}={1e3 * d:.2f}ms" for n, d in smp.prof), flush=True)
        smp.prof = None

    if os.environ.get("SAMP_STAGES"):
        res = []
        for st in ["greedy_part", "scale_exp", "search_k", "search_p", "prefix", None]:
            smp.stop = st
            smp.forward(logits, prm)
            ttnn.synchronize_device(md)
            tr = ttnn.begin_trace_capture(md, cq_id=0)
            smp.forward(logits, prm)
            ttnn.end_trace_capture(md, tr, cq_id=0)
            ttnn.execute_trace(md, tr, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(30):
                ttnn.execute_trace(md, tr, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            res.append((st or "all", (time.perf_counter() - t0) / 30 * 1e3))
        smp.stop = None
        print("SAMPLER_STAGES (cumulative trace ms) " + " ".join(f"{n}={v:.3f}" for n, v in res), flush=True)

    # cost: trace replays
    def timeit(fn_trace, n=50):
        ttnn.execute_trace(md, fn_trace, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, fn_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / n * 1e3

    ns = types.SimpleNamespace(cols=cols, md=md, _col_ids=None)
    invT_t = ttnn.from_torch(
        invT.reshape(1, 1, B, 1),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
    )
    DSV41DeviceHead.sample_global(ns, logits, mc, ccl)
    tg = ttnn.begin_trace_capture(md, cq_id=0)
    DSV41DeviceHead.sample_global(ns, logits, mc, ccl)
    ttnn.end_trace_capture(md, tg, cq_id=0)
    DSV41DeviceHead.topk_candidates(ns, logits, mc, ccl, 64, invT_t)
    tc = ttnn.begin_trace_capture(md, cq_id=0)
    DSV41DeviceHead.topk_candidates(ns, logits, mc, ccl, 64, invT_t)
    ttnn.end_trace_capture(md, tc, cq_id=0)
    print(
        f"SAMPLER_COST T={T} (B={B}) J={smp.J} levels={smp.levels}: greedy argmax {timeit(tg):.3f} ms, candidate top-64 + partition {timeit(tc):.3f} ms, in-trace sampler {timeit(tid):.3f} ms",
        flush=True,
    )


def exact_cdf_parts(l, T, tk, tp):
    """Exact sampler (float64) of one logits row: top-k then top-p over the tempered distribution, ties at a boundary kept (the in-trace sampler keeps every token above a value threshold). Returns the
    normalised kept distribution q [V] in vocabulary order; the exact draw at u is the first token whose cumulative q exceeds u.
    """
    s = (l.double() - l.double().max()) / T
    srt, _ = torch.sort(s, descending=True)
    keep = torch.ones_like(s, dtype=torch.bool)
    if tk > 0 and tk < s.numel():
        keep &= s >= srt[tk - 1]
    e = torch.exp(s) * keep
    if tp < 1.0:
        w = torch.exp(srt) * (srt >= (srt[tk - 1] if 0 < tk < s.numel() else srt[-1]))
        cum = w.cumsum(0)
        j = int(torch.searchsorted(cum, tp * w.sum(), right=False))
        keep &= s >= srt[min(j, s.numel() - 1)]
        e = torch.exp(s) * keep
    return e / e.sum()


@pytest.mark.parametrize("fast", [False, True], ids=["full", "topk"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 << 20},
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_device_sampler_distribution_vs_exact_sampler(mesh_device, fast):
    """Distribution check of the production sampler (J=16, 3 levels: the settings of the decode / verify traces) against the exact float64 sampler, on flat, near-flat, constant, ordinary and peaked rows
    and on real logits rows: every row of every replay draws with its own stratified uniform u_i = (i + 0.5) / N; the token must satisfy cdf_before(tok) - tol <= u_i <= cdf_after(tok) + tol of the EXACT
    kept distribution (tol (0.002) absorbs the finite resolution of the threshold search: tokens within range / J**levels of a top-k / top-p boundary), i.e. the empirical law equals the exact law to within
    tol in Kolmogorov distance. Constant rows (every logit tied): ties are kept, so the law is uniform over the vocabulary.
    """
    import os

    md = mesh_device
    rows, cols = tuple(md.shape)
    T = 8
    B = rows * T
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    smp = DSV41DeviceSampler(md, mc, ccl, T, J=16, levels=3, levels_k=int(os.environ.get("SAMP_LEVELS_K", "4")))
    prm = smp.alloc_params()
    g = torch.Generator().manual_seed(4)
    V = VOCAB
    rows_l = {
        "flat randn*0.05": torch.randn(V, generator=g) * 0.05,
        "flat randn*0.3": torch.randn(V, generator=g) * 0.3,
        "constant": torch.full((V,), 1.5),
        "randn*3": torch.randn(V, generator=g) * 3.0,
        "peaked randn*8": torch.randn(V, generator=g) * 8.0,
    }
    if os.path.exists(LOGITS):
        L = torch.load(LOGITS).reshape(-1, V).float()
        for i, j in enumerate(torch.randperm(L.shape[0], generator=g)[:3].tolist()):
            rows_l[f"real logits row {i}"] = L[j]
    cfgs = [(1.0, 0, 1.0), (1.0, 0, 0.95), (0.6, 20, 0.95), (1.0, 50, 1.0), (1.3, 0, 0.9)]
    if (
        fast
    ):  # the top-k path (``forward_topk``): rows with 0 < top_k <= kcand, exact kept set (constant rows: ties beyond the candidates are not kept)
        cfgs = [(1.0, 20, 1.0), (0.6, 20, 0.95), (1.0, 32, 0.9), (1.3, 5, 0.95), (1.0, 2, 1.0), (0.8, 32, 1.0)]
        rows_l.pop("constant")
    fwd = smp.forward_topk if fast else smp.forward
    N = int(os.environ.get("SAMP_DIST_N", "1024"))
    assert N % B == 0
    tol = float(os.environ.get("SAMP_DIST_TOL", "0.002"))
    worst = 0.0
    for name, lrow in rows_l.items():
        logits = ttnn.from_torch(
            lrow.reshape(1, 1, 1, V).expand(1, 1, B, V).contiguous(),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, 3), mesh_shape=(rows, cols)),
        )
        smp.set_params(torch.ones(B), torch.zeros(B), torch.ones(B), torch.rand(B), torch.zeros(B))
        out = fwd(logits, prm)
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        out = fwd(logits, prm)
        ttnn.end_trace_capture(md, tid, cq_id=0)
        for T_, tk, tp in cfgs:
            q = exact_cdf_parts(lrow, T_, tk, tp)
            cdf = q.cumsum(0)
            bad = 0
            dev = 0.0
            for rep in range(N // B):
                u = (torch.arange(B) + rep * B + 0.5) / N
                smp.set_params(
                    torch.full((B,), 1.0 / T_), torch.full((B,), float(tk)), torch.full((B,), tp), u, torch.zeros(B)
                )
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                got = torch.cat(
                    [ttnn.to_torch(ttnn.get_device_tensors(out)[r * cols]).reshape(-1) for r in range(rows)]
                ).long()
                after = cdf[got]
                before = after - q[got]
                d = torch.maximum(before - u.double(), u.double() - after).clamp(
                    min=0
                )  # distance of u from the token's CDF interval
                dev = max(dev, float(d.max()))
                bad += int((d > tol).sum())
            print(
                f"SAMPLER_DIST row '{name}' T={T_} top_k={tk} top_p={tp}: {N} draws, max CDF-interval deviation {dev:.4f}, draws beyond tol {bad}",
                flush=True,
            )
            worst = max(worst, dev)
            assert bad == 0, (name, T_, tk, tp, dev)
        ttnn.release_trace(md, tid)
        ttnn.deallocate(logits)
    print(f"SAMPLER_DIST worst deviation {worst:.4f} (tol {tol})", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 256 << 20},
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_device_sampler_cost(mesh_device):
    """Replay cost (ms per step, the trace alone) of the greedy sampler of the head (``sample_global``) and of the in-trace sampler cut after each stage, at T users per mesh row (SAMP_T, default 8 =
    batch 32; 32 = the verify rows of batch 32 at k = 3)."""
    import os
    import time

    md = mesh_device
    rows, cols = tuple(md.shape)
    T = int(os.environ.get("SAMP_T", "8"))
    B = rows * T
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    J, L, LK = (
        int(os.environ.get("SAMP_J", "16")),
        int(os.environ.get("SAMP_L", "3")),
        int(os.environ.get("SAMP_LK", "4")),
    )
    smp = DSV41DeviceSampler(md, mc, ccl, T, J=J, levels=L, levels_k=LK)
    prm = smp.alloc_params()
    head = DSV41DeviceHead.__new__(DSV41DeviceHead)
    head.md, head.cols = md, cols
    g = torch.Generator().manual_seed(1)
    logits = ttnn.from_torch(
        (torch.randn(1, 1, B, VOCAB, generator=g) * 3).float(),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, 3), mesh_shape=(rows, cols)),
    )
    smp.set_params(
        torch.full((B,), 1 / 0.6), torch.full((B,), 20.0), torch.full((B,), 0.95), torch.rand(B), torch.zeros(B)
    )

    def timed(fn, n=30):
        fn()  # compile
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        dt = (time.perf_counter() - t0) / n
        ttnn.release_trace(md, tid)
        return dt * 1e3

    res = {"greedy sample_global": timed(lambda: head.sample_global(logits, mc, ccl))}
    for stop in ("greedy_part", "scale_exp", "search_k", "search_p", "prefix", None):
        smp.stop = stop
        res[f"sampler up to {stop or 'end'}"] = timed(lambda: smp.forward(logits, prm))
    smp.stop = None
    res["top-k path (kcand=%d)" % smp.kcand] = timed(lambda: smp.forward_topk(logits, prm))
    print(
        f"SAMPLER_COST T={T} J={J} levels={L} levels_k={LK}: " + ", ".join(f"{k}: {v:.3f} ms" for k, v in res.items()),
        flush=True,
    )
