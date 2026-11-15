"""mtp-device stage 6 on a 1x8 slot: statistical test of the device draft-from-q rejection sampler (tt/hy4_spec_sample.py) on synthetic logits (V = 120832 sharded over the 8 columns).
Per scenario (p, q logits, tau): N draws of ONE round at depth 1 (d ~ q, accept u < p(d)/q(d), residual / bonus): the emitted token a must be distributed exactly as p. Also checks: q-sample marginal == q,
acceptance rate == sum_v min(p, q), bonus rows ~ p. Env: CALLS (default 1900) x T=32 draws, SCEN (names). Prints RES lines (chi2 z-scores over bins with expected count >= 5, TV vs the sampling-noise TV)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_sampler import Hy4DeviceSampler
from models.demos.blackhole.hy4_preview.tt.hy4_spec_sample import SpecSampleOps
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

V, T = 120832, 32


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def stats(counts, p, n):
    """chi-square over bins formed by merging tokens (in descending p) until the expected count is >= 20; TV vs the TV of an exact sampler."""
    order = torch.argsort(p, descending=True)
    pe, ce = (p * n)[order], counts[order]
    bins, cur_e, cur_c = [], 0.0, 0.0
    for e, c in zip(pe.tolist(), ce.tolist()):
        cur_e += e
        cur_c += c
        if cur_e >= 20:
            bins.append((cur_e, cur_c))
            cur_e = cur_c = 0.0
    if cur_e > 0 and bins:
        bins[-1] = (bins[-1][0] + cur_e, bins[-1][1] + cur_c)
    chi = sum((c - e) ** 2 / e for e, c in bins)
    dof = len(bins) - 1
    z = (chi - dof) / (2 * dof) ** 0.5
    tv = float(0.5 * (counts / n - p).abs().sum())
    sim = torch.multinomial(p, n, replacement=True)
    tv0 = float(0.5 * (torch.bincount(sim, minlength=len(p)).double() / n - p).abs().sum())
    return dict(chi2_z=round(z, 2), bins=dof + 1, tv=round(tv, 4), tv_noise=round(tv0, 4))


def main():
    calls = int(os.environ.get("CALLS", "1900"))
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        smp = Hy4DeviceSampler(md, mc, ccl, users_per_row=T)
        ops = SpecSampleOps(smp)
        g = torch.Generator().manual_seed(7)
        rep = ttnn.ReplicateTensorToMesh(md)
        mk = lambda t, dt=ttnn.float32: ttnn.from_torch(
            t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
        )
        shard_up = lambda lg: ttnn.from_torch(
            lg.reshape(1, 1, 1, V).repeat(1, 1, T, 1),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(md, dim=3),
        )
        scen = {
            "peaky_close": (torch.randn(V, generator=g) * 3.0, 0.5, 1.0),
            "flat_far": (torch.randn(V, generator=g) * 1.2, 1.5, 1.0),
            "cold_tau0.6": (torch.randn(V, generator=g) * 2.5, 0.8, 1 / 0.6),
        }
        zeros = mk(torch.zeros(1, 1, T, 1))
        ones = mk(torch.ones(1, 1, T, 1))
        for name, (lp, qnoise, invT_) in scen.items():
            lq = lp + torch.randn(V, generator=g) * qnoise
            p = torch.softmax(lp.double() * invT_, 0)
            q = torch.softmax(lq.double() * invT_, 0)
            acc_th = float(torch.minimum(p, q).sum())
            P_log, Q_log = shard_up(lp.float()), shard_up(lq.float())
            invT = mk(torch.full((1, 1, T, 1), invT_))
            cnt_a = torch.zeros(V, dtype=torch.long)
            cnt_d = torch.zeros(V, dtype=torch.long)
            cnt_b = torch.zeros(V, dtype=torch.long)
            acc_sum, n = 0.0, 0
            t0 = time.time()
            for c in range(calls):
                u = [mk(torch.rand(1, 1, T, 1, generator=g)) for _ in range(4)]
                d, q4 = ops.sample_q(Q_log, invT, u[0])
                a, acc = ops.sampled_tokens(P_log, q4, d, invT, u[1], u[2], u[3], zeros)
                ab, _ = ops.sampled_tokens(P_log, q4, d, invT, u[1], u[2], u[3], ones)  # bonus rows: a ~ p
                tk = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)[:T].round().long()
                cnt_a += torch.bincount(tk(a), minlength=V)
                cnt_d += torch.bincount(tk(d), minlength=V)
                cnt_b += torch.bincount(tk(ab), minlength=V)
                acc_sum += float(ttnn.to_torch(ttnn.get_device_tensors(acc)[0]).reshape(-1)[:T].sum())
                n += T
                for t in u + [d, q4, a, acc, ab]:
                    ttnn.deallocate(t)
            res(
                event="scenario",
                name=name,
                draws=n,
                sec=round(time.time() - t0, 1),
                emitted_vs_p=stats(cnt_a.double(), p, n),
                draft_vs_q=stats(cnt_d.double(), q, n),
                bonus_vs_p=stats(cnt_b.double(), p, n),
                accept_rate=round(acc_sum / n, 4),
                accept_theory=round(acc_th, 4),
            )
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
