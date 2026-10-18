# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt/shared_sampler.SharedSamplerAdapter (the shared TTSampling behind our sampler interface, DSV41_SHARED_SAMPLER=1) on the 4x8 mesh, inside a captured trace like the decode path:
  * greedy rows (temperature 0 / top_k 1) in a MIXED batch are the exact fp32 argmax (the token buffer the step trace wrote is kept for them), sampled rows lie in the top-32 / nucleus,
  * the padding of T users per mesh row to the 32 slots of the shared sampler does not leak (T = 1, 4, 8),
  * seeds: the same uniform gives the same token whatever slot / mesh row the user is in and across replays; different uniforms give different tokens,
  * the law of the draw (T = 4, real high-entropy row, temperature 1, top_p 0.95 / 1.0, no top_k = top-32) against the exact law of the top-32 truncation.
Env: SS_N (replays of the law test, default 150).
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_device_sampler import LOGITS, exact_cdf_parts
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import VOCAB
from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import sampling_rows
from models.demos.blackhole.deepseek_v41_flash.tt.shared_sampler import SharedSamplerAdapter
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

DEV_PARAMS = [
    pytest.param(
        {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 128 << 20},
        id="ring",
    )
]


def _setup(md, T):
    rows, cols = tuple(md.shape)
    sm = SharedSamplerAdapter(md, mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring), T)
    prm = sm.alloc_params()
    B = rows * T

    def to_dev(x):  # x [B, V] fp32 -> [1,1,B,V]: users over mesh rows, vocab over mesh columns
        return ttnn.from_torch(
            x.reshape(1, 1, B, VOCAB),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, 3), mesh_shape=(rows, cols)),
        )

    def tok_dev(t):  # [B] token buffer like the decoder's ``tok_dev`` ([B,1] uint32 row-major, row sharded)
        return ttnn.from_torch(
            t.reshape(-1, 1).to(torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
        )

    def read(t):
        return torch.cat([ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(-1) for r in range(rows)]).long()

    return sm, prm, B, to_dev, tok_dev, read


def _traced(md, sm, logits, base):
    """Capture the sampler (compile pass first) the way ``SamplerTraces`` does; returns (trace id, output buffer)."""
    out = ttnn.copy(sm.run(logits, sm.params, False, base_tok=base), base)  # compile pass (writes ``base``)
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    sm.run_out = sm.run(logits, sm.params, False, base_tok=base)
    ttnn.copy(sm.run_out, base)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    return tid


def _real_rows(n):
    L = torch.load(LOGITS).reshape(-1, VOCAB).float()
    ent = -(torch.softmax(L, 1) * torch.log_softmax(L, 1)).sum(1)
    order = torch.argsort(ent, descending=True)
    return L[order[: 200 : 200 // n][:n]]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("T", [1, 4, 8])
@torch.no_grad()
def test_greedy_mixed_and_padding(mesh_device, T):
    md = mesh_device
    sm, prm, B, to_dev, tok_dev, read = _setup(md, T)
    L = _real_rows(B)
    exact = L.argmax(1)
    logits = to_dev(L)
    base = tok_dev(exact)  # the step trace's exact greedy tokens
    tid = _traced(md, sm, logits, base)
    g = torch.Generator().manual_seed(5)
    n_bad = 0
    top32 = torch.topk(L, 32, dim=1).indices
    for it in range(6):
        # alternate rows: sampled (T=1.0, top_p .95, no top_k) / greedy by temperature 0 / greedy by top_k 1
        temps = torch.tensor([1.0 if (i % 3 == 0) else (0.0 if i % 3 == 1 else 0.8) for i in range(B)])
        ks = torch.tensor([0 if i % 3 == 0 else (5 if i % 3 == 1 else 1) for i in range(B)])
        ps = torch.full((B,), 0.95)
        sm.set_params(*sampling_rows(temps.tolist(), ks.tolist(), ps.tolist(), g, B))
        ttnn.copy_host_to_device_tensor(  # the step trace rewrites the buffer with the exact greedy tokens before each sampler replay
            ttnn.from_torch(
                exact.reshape(-1, 1).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
            ),
            base,
        )
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        got = read(base)
        for i in range(B):
            greedy = float(temps[i]) == 0.0 or int(ks[i]) == 1
            if greedy:
                n_bad += int(got[i] != exact[i])
            else:
                assert int(got[i]) in top32[i].tolist(), f"row {i}: sampled token outside the top-32 candidates"
    print(f"SHARED_SAMPLER_IT T={T}: greedy rows != exact argmax: {n_bad}", flush=True)
    assert n_bad == 0


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@torch.no_grad()
def test_seeds(mesh_device):
    md = mesh_device
    T = 4
    sm, prm, B, to_dev, tok_dev, read = _setup(md, T)
    row = _real_rows(1)[0]
    logits = to_dev(row.expand(B, VOCAB).contiguous())
    base = tok_dev(torch.zeros(B, dtype=torch.long))
    tid = _traced(md, sm, logits, base)
    temps, ks, ps = torch.ones(B), torch.zeros(B), torch.full((B,), 0.95)

    def run(u):
        sm.set_params(1 / temps, ks, ps, u, torch.zeros(B))
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        return read(base)

    # same uniform in every slot of every mesh row -> same token everywhere, and again on a replay
    u = torch.full((B,), 0.37)
    a, b = run(u), run(u)
    assert (a == a[0]).all(), f"same seed, different tokens across slots: {a.tolist()}"
    assert (a == b).all(), "same seed, different token on the next replay"
    # a request moves between slots: the seed of user 0 put in the last slot gives the same token
    u2 = torch.rand(B, generator=torch.Generator().manual_seed(1))
    c = run(u2)
    u3 = u2.clone()
    u3[0], u3[B - 1] = u2[B - 1], u2[0]
    d = run(u3)
    assert int(c[0]) == int(d[B - 1]) and int(c[B - 1]) == int(d[0]), (c.tolist(), d.tolist())
    # different uniforms -> not all equal over many draws (the row is high entropy)
    toks = torch.cat([run(torch.rand(B, generator=torch.Generator().manual_seed(100 + i))) for i in range(10)])
    print(f"SHARED_SAMPLER_IT seeds: {len(set(toks.tolist()))} distinct tokens in {toks.numel()} draws", flush=True)
    assert len(set(toks.tolist())) > 3


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@torch.no_grad()
def test_law(mesh_device):
    md = mesh_device
    T = 4
    N = int(os.environ.get("SS_N", "150"))
    sm, prm, B, to_dev, tok_dev, read = _setup(md, T)
    rows_ = _real_rows(3)
    base = tok_dev(torch.zeros(B, dtype=torch.long))
    ud = torch.Generator().manual_seed(7)
    for ri, row in enumerate(rows_):
        logits = to_dev(row.expand(B, VOCAB).contiguous())
        tid = _traced(md, sm, logits, base)
        for temp, tp in [(1.0, 1.0), (1.0, 0.95), (0.7, 0.95)]:
            # exact law of the TOP-32 truncation of the requested law (what 'no top_k' means for the shared sampler)
            q = exact_cdf_parts(row, temp, 32, tp)
            q_full = exact_cdf_parts(row, temp, 0, tp)
            cnt = torch.zeros(VOCAB, dtype=torch.float64)
            for it in range(N):
                u = torch.rand(B, generator=ud)
                sm.set_params(torch.full((B,), 1 / temp), torch.zeros(B), torch.full((B,), tp), u, torch.zeros(B))
                ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
                cnt += torch.bincount(read(base), minlength=VOCAB).double()
            emp = cnt / cnt.sum()
            tv32 = 0.5 * float((emp - q).abs().sum())
            tvf = 0.5 * float((emp - q_full).abs().sum())
            ent = float(-(q_full[q_full > 0] * q_full[q_full > 0].log()).sum())
            print(
                f"SHARED_SAMPLER_IT law row{ri} (entropy {ent:.2f}) T={temp} top_p={tp}: TV vs top-32 law {tv32:.4f}, TV vs requested (no top_k) law {tvf:.4f}",
                flush=True,
            )
        ttnn.release_trace(md, tid)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("T", [1, 4])
@torch.no_grad()
def test_step_cost(mesh_device, T):
    """Cost of one sampled decode step's sampler part as the decode path pays it: host ``set_params`` (changing seeds / uniforms every step), the sampler trace replay, and both together
    (set_params + replay + wait), shared adapter against our sampler (top-k 20 fast path / full vocabulary). ms per step.
    """
    import time

    from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import DSV41DeviceSampler

    md = mesh_device
    sm, prm, B, to_dev, tok_dev, read = _setup(md, T)
    ours = DSV41DeviceSampler(md, mesh_4x8(), sm.ccl, T, J=16, levels=3, levels_k=4)
    ours.alloc_params()
    row = _real_rows(1)[0]
    logits = to_dev(row.expand(B, VOCAB).contiguous())
    base = tok_dev(torch.zeros(B, dtype=torch.long))
    for fast in (True, False):  # compile passes of every variant before ANY trace exists
        ttnn.copy(ours.run(logits, ours.params, fast), base)
    ttnn.copy(sm.run(logits, sm.params, False, base_tok=base), base)
    ttnn.synchronize_device(md)
    traces = {}
    for name, fast in (("ours_topk20", True), ("ours_full", False)):
        t = ttnn.begin_trace_capture(md, cq_id=0)
        ttnn.copy(ours.run(logits, ours.params, fast), base)
        ttnn.end_trace_capture(md, t, cq_id=0)
        traces[name] = (t, fast)
    tid_s = ttnn.begin_trace_capture(md, cq_id=0)
    ttnn.copy(sm.run(logits, sm.params, False, base_tok=base), base)
    ttnn.end_trace_capture(md, tid_s, cq_id=0)
    traces["shared"] = (tid_s, False)
    g = torch.Generator().manual_seed(3)
    N = 60
    for name, (tid, fast) in traces.items():
        smp = sm if name == "shared" else ours
        kk = torch.full((B,), 20.0 if name != "ours_full" else 0.0)

        def prm_args():
            u = torch.rand(B, generator=g)
            return (torch.ones(B), kk, torch.full((B,), 0.95), u, torch.zeros(B))

        smp.set_params(*prm_args())
        t0 = time.perf_counter()
        for _ in range(N):
            smp.set_params(*prm_args())
        t_set = (time.perf_counter() - t0) / N * 1e3
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(N):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t_rep = (time.perf_counter() - t0) / N * 1e3
        t0 = time.perf_counter()
        for _ in range(N):
            smp.set_params(*prm_args())
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
        t_both = (time.perf_counter() - t0) / N * 1e3
        print(
            f"SAMPLER_STEP_COST T={T} {name}: set_params {t_set:.3f} ms, trace replay {t_rep:.3f} ms, set_params + replay + wait {t_both:.3f} ms",
            flush=True,
        )
