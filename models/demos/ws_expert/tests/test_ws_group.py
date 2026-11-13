# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert with streamed weights (20-core groups, expert e on group e % G, weights landed by injector cores from
DRAM into 2-slot L1 rings). Every expert has its own weights; each group's last expert is checked against torch."""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, read_stamps, to_dev
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_group import GCfg, build


def run_group(device, cfg: GCfg, Ms, seed=0):
    assert len(Ms) == cfg.nexp and max(Ms) <= cfg.MMAX
    torch.manual_seed(seed)
    x = torch.randn(cfg.T + 1, cfg.H)
    x[cfg.T] = 0
    idx = torch.full((cfg.nexp, cfg.RMAX * 32), cfg.T, dtype=torch.int32)
    for e, M in enumerate(Ms):
        idx[e, :M] = torch.randperm(cfg.T)[:M].to(torch.int32)
    counts = torch.tensor(Ms, dtype=torch.int32).reshape(1, -1)
    wg = torch.randn(cfg.nexp, cfg.H, cfg.I) / cfg.H**0.5
    wu = torch.randn(cfg.nexp, cfg.H, cfg.I) / cfg.H**0.5
    wd = torch.randn(cfg.nexp, cfg.I, cfg.H) / cfg.I**0.5
    wgu = torch.cat([interleave_gu(wg[e], wu[e]) for e in range(cfg.nexp)], 0)  # [E * H, 2 I]
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    idx_t = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    counts_t = ttnn.from_torch(counts, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    wgu_t = to_dev(wgu, cfg.w_dt, device)
    wd_t = to_dev(wd.reshape(-1, cfg.H), cfg.w_dt, device)
    out_t = to_dev(torch.zeros(cfg.G * cfg.RMAX * 32, cfg.H), cfg.out_dt, device)
    prog, io, st = build(device, cfg, x_t, idx_t, counts_t, wgu_t, wd_t, out_t)
    ttnn.generic_op(io, prog)
    ttnn.synchronize_device(device)
    clk = aiclk_mhz(0)

    out = ttnn.to_torch(out_t).float()
    wguq = quantize(wgu, cfg.w_dt).reshape(cfg.nexp, cfg.H, -1)
    wdq = quantize(wd.reshape(-1, cfg.H), cfg.w_dt).reshape(cfg.nexp, cfg.I, cfg.H)
    pcs = []
    for g in range(cfg.G):
        e = max(range(g, cfg.nexp, cfg.G))
        M = Ms[e]
        X = x[idx[e].long()].to(torch.bfloat16).float()
        wgq = wguq[e].reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
        wuq = wguq[e].reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
        h = quantize(torch.nn.functional.silu(X @ wgq) * (X @ wuq), cfg.h_dt)
        ref = h @ wdq[e]
        o = out[g * cfg.RMAX * 32 : g * cfg.RMAX * 32 + M]
        pcs.append(pcc(o, ref[:M]) if M else 1.0)

    NC = cfg.G * cfg.GC
    s = read_stamps(st, NC)
    ref0 = int(s[0, 0])
    rel = ((s - ref0 + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    t0 = int(rel[:, 0].min())
    end_all = int((rel[:, 63] - t0).max())
    # steady state: per group, the end of its first expert to the end of its last
    per_group = []
    for g in range(cfg.G):
        n_g = len(range(g, cfg.nexp, cfg.G))
        rg = rel[g * cfg.GC : (g + 1) * cfg.GC]
        first = int((rg[:, 1] - t0).max())
        last = int((rg[:, n_g] - t0).max())
        per_group.append((first, last, n_g))
    first = max(p[0] for p in per_group)
    last = max(p[1] for p in per_group)
    done_after_first = [e for e in range(cfg.nexp) if e >= cfg.G]
    R = [(m + 31) // 32 for m in Ms]
    ops = lambda es: sum(R[e] * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS) for e in es)
    span = last - first
    util_tile = ops(done_after_first) * 16 / (NC * span)
    util_useful = util_tile * sum(Ms[e] for e in done_after_first) / sum(32 * R[e] for e in done_after_first)
    util_all = ops(range(cfg.nexp)) * 16 / (NC * end_all)
    run_group.last_stamps = (s, t0, rel)
    return dict(
        per_expert=span / len(done_after_first),
        util_tile=util_tile,
        util_useful=util_useful,
        util_all=util_all,
        pcc=min(pcs),
        pcs=pcs,
        aiclk=clk,
        cores=NC,
        unpack=(s[:, 40:44].double().mean(dim=0)).tolist(),
        nc=(s[:, 48:50].double().mean(dim=0)).tolist(),
    )


def report(tag, cfg, Ms, r):
    logger.info(
        f"WSG {tag} M mean {sum(Ms) / len(Ms):.0f} (min {min(Ms)} max {max(Ms)}) nexp {cfg.nexp} groups {cfg.G} "
        f"cores {r['cores']} (+{cfg.NINJ} injectors): {r['per_expert']:.0f} cyc/expert ({r['per_expert'] / r['aiclk']:.2f} "
        f"us), util tile {100 * r['util_tile']:.1f}% useful {100 * r['util_useful']:.1f}% (whole run incl. fill "
        f"{100 * r['util_all']:.1f}%); pcc per group {[round(p, 5) for p in r['pcs']]}"
    )
    logger.info(f"UNPACK totals per core (mean): tilize / x wait / weight wait / h wait {[round(v) for v in r['unpack']]}")
    logger.info(f"NCRISC x multicast: blocked on credits / staging empty {[round(v) for v in r['nc']]}")


@pytest.mark.parametrize("M", [586, 1024, 2048])
def test_ws_group(device, M):
    cfg = GCfg(nexp=32)
    Ms = [M] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"const {M}", cfg, Ms, r)
    assert r["pcc"] > 0.999


def test_ws_group_nogather(device):
    cfg = GCfg(nexp=32, wdefs=("WG_NO_GATHER",))
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report("no gather", cfg, Ms, r)


@pytest.mark.parametrize("BK,NX,LOOK,XRM", [(24, 8, 16, 2), (30, 6, 12, 2), (24, 6, 12, 2)])
def test_ws_group_x(device, BK, NX, LOOK, XRM):
    cfg = GCfg(nexp=32, BK=BK, NX=NX, LOOK=LOOK, XRM=XRM)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"BK {BK} NX {NX} LOOK {LOOK}", cfg, Ms, r)
    assert r["pcc"] > 0.999


@pytest.mark.parametrize("G", [1, 2, 4])
def test_ws_group_count(device, G):
    cfg = GCfg(nexp=8 * G, G=G)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"G {G}", cfg, Ms, r)


def test_ws_group_trace(device):
    cfg = GCfg(nexp=8, G=1, trace0=380)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report("trace", cfg, Ms, r)
    s, t0, rel = run_group.last_stamps
    base = int(rel[:, 64:104][rel[:, 64:104] > -(1 << 30)].min())
    for c in range(40):
        owner = (380 + c) % 20  # (expert 2 starts at stream 380: block l = c + 0 .. owner l % 20, 380 % 190 = 0)
        til = int(rel[owner, 256 + c]) - base if int(s[owner, 256 + c]) else -1
        mc = int(rel[owner, 128 + c]) - base
        push = [int(rel[j, 192 + c]) - base for j in (0, 5, 15)]
        got = [int(rel[j, 64 + c]) - base for j in (0, 5, 15)]
        logger.info(f"TRACE block {380 + c} owner {owner}: tilized {til} mcast {mc} pushed {push} got {got}")
