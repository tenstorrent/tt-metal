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
