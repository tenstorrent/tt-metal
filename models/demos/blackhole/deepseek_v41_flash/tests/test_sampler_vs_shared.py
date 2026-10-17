# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Our in-trace sampler (tt/device_sampler.py, production settings J=16 / 3 levels) against the shared ``models.common.sampling.TTSampling`` (the sampler of
tt_transformers) on the same 4x8 mesh and the same real logits rows (GSM8K teacher-forced, dsv4-e2e-ref/logits_tf.pt).

Three measurements:
  SAMPLER_CMP perf     trace replay time of one sampling step (ours full-vocabulary path / ours top-k path / shared). The shared sampler works on 32 user slots
                       per mesh row (max_batch_size is rounded up to 32); ours on T users per row.
  SAMPLER_CMP greedy   argmax agreement on distinct real rows (exact torch argmax; ties in the logits are broken to the lowest index by both, so any mismatch is counted).
  SAMPLER_CMP dist     distribution fidelity: one real row replicated over all users, N replays with fresh randomness, empirical histogram against the exact float64 law of the
                       REQUESTED (temperature, top-k, top-p) (tests/test_device_sampler.exact_cdf_parts); total-variation distance, the mass drawn outside the exact support,
                       and the noise floor (the same number of exact CPU draws).
The shared sampler clamps top-k to max_top_k (32) and maps "no top-k" to 32; the distribution rows show what that does to the law.
Env: SAMP_T (users per row of our sampler, default 8), SAMP_N (replays per config, default 200), SAMP_ROWS (real rows, default 3).
"""

import os
import time
import types

import pytest
import torch

import ttnn
from models.common.sampling.tt_sampling import TTSampling
from models.demos.blackhole.deepseek_v41_flash.tests.test_device_sampler import LOGITS, exact_cdf_parts
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import VOCAB
from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import DSV41DeviceSampler
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CFGS = [  # (temperature, top_k (0 = off), top_p (1 = off))
    (1.0, 0, 1.0),
    (1.0, 0, 0.95),
    (0.7, 0, 0.95),
    (1.0, 20, 0.95),
    (1.0, 32, 1.0),
    (1.0, 50, 0.95),
    (1.0, 200, 0.9),
]


def tv_and_outside(counts, q):
    emp = counts / counts.sum()
    return 0.5 * float((emp - q).abs().sum()), float(emp[q == 0].sum())


def noise_floor(q, n, g):
    idx = torch.multinomial(q.float(), n, replacement=True, generator=g)
    return tv_and_outside(torch.bincount(idx, minlength=q.numel()).double(), q)[0]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 128 << 20},
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_sampler_vs_shared(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    T = int(os.environ.get("SAMP_T", "8"))
    N = int(os.environ.get("SAMP_N", "200"))
    NR = int(os.environ.get("SAMP_ROWS", "3"))
    B = rows * T
    BS = rows * 32  # shared sampler: 32 slots per mesh row
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)

    ours = DSV41DeviceSampler(md, mc, ccl, T, J=16, levels=3)
    prm = ours.alloc_params()
    args = types.SimpleNamespace(
        vocab_size=VOCAB,
        padded_vocab_size=VOCAB,
        cluster_shape=(rows, cols),
        max_batch_size=32,
        max_top_k=32,
        sampling_dp=rows,
        sampling_all_gather_axis=1,
    )
    shared = TTSampling(md, ccl, args)

    L = torch.load(LOGITS).reshape(-1, VOCAB).float()
    ent = -(torch.softmax(L, 1) * torch.log_softmax(L, 1)).sum(1)
    order = torch.argsort(ent)
    # informative rows only: the highest-entropy real rows (most teacher-forced rows are nearly deterministic) and two synthetic wide rows
    rows_list = [(f"real#{int(order[-1 - 40 * i])}", L[int(order[-1 - 40 * i])]) for i in range(NR)]
    gs = torch.Generator().manual_seed(3)
    rows_list += [
        ("synthetic normal(0,4)", 4 * torch.randn(VOCAB, generator=gs)),
        ("synthetic normal(0,6)", 6 * torch.randn(VOCAB, generator=gs)),
    ]

    def to_dev(x, dtype):  # x [Bx, V] -> [1,1,Bx,V], batch over mesh rows, vocab over mesh columns
        return ttnn.from_torch(
            x.reshape(1, 1, x.shape[0], VOCAB),
            device=md,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, 3), mesh_shape=(rows, cols)),
        )

    def reseed_shared(seed):
        g = torch.Generator().manual_seed(seed)
        host = ttnn.from_torch(
            torch.randint(0, 2**31 - 1, (BS,), generator=g).to(torch.uint32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=shared._param_dims, mesh_shape=shared.cluster_shape),
        )
        ttnn.copy_host_to_device_tensor(host, shared.seeds_tt_tensor)

    def read_ours(out):
        return torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(out)[r * cols]).reshape(-1) for r in range(rows)]
        ).long()

    def read_shared(out):
        o = out[0] if isinstance(out, tuple) else out
        return torch.cat(
            [ttnn.to_torch(ttnn.get_device_tensors(o)[r * cols]).reshape(-1)[:32] for r in range(rows)]
        ).long()

    def set_shared(temps, ks, ps):
        # ttnn.sampling takes the INVERSE temperature (format_sampling_params inverts it); greedy rows keep temp 1
        t = torch.as_tensor(temps, dtype=torch.float32)
        shared.reset_params(
            k=torch.as_tensor(ks),
            p=torch.as_tensor(ps, dtype=torch.float32),
            temp=torch.where(t > 0, 1.0 / t.clamp(min=1e-6), torch.ones_like(t)),
        )

    def set_ours(temps, ks, ps, u, greedy):
        invT = torch.where(temps > 0, 1 / temps.clamp(min=1e-6), torch.ones_like(temps))
        ours.set_params(invT, ks, ps, u, greedy)

    def trace_of(fn):
        fn()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        out = fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        return tid, out

    def timeit(tid, n=50):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t0) / n * 1e3

    # ---- greedy agreement + perf on distinct real rows
    g = torch.Generator().manual_seed(1)
    sel = torch.randperm(L.shape[0], generator=g)[: max(B, BS)]
    lg_o, lg_s = L[sel[:B]], L[sel[:BS]]
    logits_o = to_dev(lg_o, ttnn.float32)
    logits_s = to_dev(lg_s, ttnn.bfloat16)
    set_ours(torch.ones(B), torch.ones(B, dtype=torch.long), torch.ones(B), torch.rand(B), torch.ones(B))
    tid_o, out_o = trace_of(lambda: ours.forward(logits_o, prm))
    ttnn.execute_trace(md, tid_o, cq_id=0, blocking=True)
    got = read_ours(out_o)
    bad_o = int((got != lg_o.argmax(1)).sum())
    set_shared(torch.ones(BS), torch.ones(BS, dtype=torch.long), torch.ones(BS))
    reseed_shared(5)
    tid_s, out_s = trace_of(lambda: shared.forward(logits_s))
    ttnn.execute_trace(md, tid_s, cq_id=0, blocking=True)
    got = read_shared(out_s)
    # the shared sampler computes top-k on bf16 logits: compare with the argmax of the bf16-rounded row too
    exp_f = lg_s.argmax(1)
    exp_b = lg_s.to(torch.bfloat16).float().argmax(1)
    bad_s_f, bad_s_b = int((got != exp_f).sum()), int((got != exp_b).sum())
    print(
        f"SAMPLER_CMP greedy: ours {bad_o}/{B} != argmax; shared {bad_s_f}/{BS} != argmax(fp32 row), {bad_s_b}/{BS} != argmax(bf16 row)",
        flush=True,
    )

    # sampled perf: a mixed batch (temperature 1, top-p 0.95, no top-k) for the full-vocabulary path, top-k 20 for the top-k path
    ps_full = torch.full((B,), 0.95)
    set_ours(torch.ones(B), torch.zeros(B, dtype=torch.long), ps_full, torch.rand(B), torch.zeros(B))
    tf, _ = trace_of(lambda: ours.forward(logits_o, prm))
    t_full = timeit(tf)
    set_ours(torch.ones(B), torch.full((B,), 20), ps_full, torch.rand(B), torch.zeros(B))
    tk, _ = trace_of(lambda: ours.forward_topk(logits_o, prm))
    t_topk = timeit(tk)
    set_shared(torch.ones(BS), torch.full((BS,), 20), torch.full((BS,), 0.95))
    reseed_shared(6)
    ts, _ = trace_of(lambda: shared.forward(logits_s))
    t_shared = timeit(ts)
    print(
        f"SAMPLER_CMP perf (trace replay, ms/step): ours full-vocab ({T} users/row) {t_full:.3f}, ours top-k path {t_topk:.3f}, shared ({32} slots/row) {t_shared:.3f}",
        flush=True,
    )

    # ---- distribution fidelity on one replicated real row per config
    gnf = torch.Generator().manual_seed(99)
    for label, row in rows_list:
        for temp, tk_, tp_ in CFGS:
            q = exact_cdf_parts(row, temp, tk_, tp_)
            lo = to_dev(row.expand(B, VOCAB).contiguous(), ttnn.float32)
            ls = to_dev(row.expand(BS, VOCAB).contiguous(), ttnn.bfloat16)
            temps_o, ks_o, ps_o = torch.full((B,), temp), torch.full((B,), tk_, dtype=torch.long), torch.full((B,), tp_)
            set_ours(temps_o, ks_o, ps_o, torch.rand(B), torch.zeros(B))
            fast_cfg = 0 < tk_ <= 32
            tid_a, out_a = trace_of(lambda: ours.forward_topk(lo, prm) if fast_cfg else ours.forward(lo, prm))
            temps_s = torch.full((BS,), temp)
            set_shared(temps_s, torch.full((BS,), tk_ if tk_ > 0 else VOCAB, dtype=torch.long), torch.full((BS,), tp_))
            reseed_shared(11)
            tid_b, out_b = trace_of(lambda: shared.forward(ls))
            cnt_o = torch.zeros(VOCAB, dtype=torch.float64)
            cnt_s = torch.zeros(VOCAB, dtype=torch.float64)
            ud = torch.Generator().manual_seed(7)
            for it in range(N):
                set_ours(temps_o, ks_o, ps_o, (torch.arange(B) + torch.rand(B, generator=ud)) / B, torch.zeros(B))
                ttnn.execute_trace(md, tid_a, cq_id=0, blocking=False)
                cnt_o += torch.bincount(read_ours(out_a), minlength=VOCAB).double()
                reseed_shared(1000 + it)
                ttnn.execute_trace(md, tid_b, cq_id=0, blocking=False)
                cnt_s += torch.bincount(read_shared(out_b), minlength=VOCAB).double()
            tv_o, out_o_ = tv_and_outside(cnt_o, q)
            tv_s, out_s_ = tv_and_outside(cnt_s, q)
            fl = noise_floor(q, int(cnt_o.sum()), gnf)
            ent_ = float(-(q[q > 0] * q[q > 0].log()).sum())
            print(
                f"SAMPLER_CMP dist {label} (entropy {ent_:.2f}) T={temp} top_k={tk_} top_p={tp_}: "
                f"ours TV {tv_o:.4f} outside {out_o_:.4f} | shared TV {tv_s:.4f} outside {out_s_:.4f} | exact-draw noise floor TV {fl:.4f}",
                flush=True,
            )
