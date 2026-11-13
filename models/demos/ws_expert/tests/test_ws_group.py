# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert with streamed weights (20-core groups, expert e on group e % G, weights landed by injector cores from
DRAM into 2-slot L1 rings). Every expert has its own weights; each group's last expert is checked against torch."""

import pytest
import torch
from loguru import logger

import ttnn
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
    ns = cfg.P if cfg.wsrc == "l1" else cfg.nexp  # weight sets
    wset = (lambda e: (e // cfg.G) % ns) if cfg.wsrc == "l1" else (lambda e: e)
    wg = torch.randn(ns, cfg.H, cfg.I) / cfg.H**0.5
    wu = torch.randn(ns, cfg.H, cfg.I) / cfg.H**0.5
    wd = torch.randn(ns, cfg.I, cfg.H) / cfg.I**0.5
    wgu = torch.cat([interleave_gu(wg[e], wu[e]) for e in range(ns)], 0)  # [sets * H, 2 I]
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    idx_t = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    counts_t = ttnn.from_torch(counts, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    wgu_t = to_dev(wgu, cfg.w_dt, device)
    wd_t = to_dev(wd.reshape(-1, cfg.H), cfg.w_dt, device)
    out_t = to_dev(torch.zeros(cfg.G * cfg.RMAX * 32, cfg.H), cfg.out_dt, device)
    prog, io, st, _ = build(device, cfg, x_t, idx_t, counts_t, wgu_t, wd_t, out_t)
    ttnn.generic_op(io, prog)
    ttnn.synchronize_device(device)
    clk = aiclk_mhz(0)

    out = ttnn.to_torch(out_t).float()
    wguq = quantize(wgu, cfg.w_dt).reshape(ns, cfg.H, -1)
    wdq = quantize(wd.reshape(-1, cfg.H), cfg.w_dt).reshape(ns, cfg.I, cfg.H)
    pcs = []
    for g in range(cfg.G):
        e = max(range(g, cfg.nexp, cfg.G))
        M = Ms[e]
        X = quantize(x[idx[e].long()].to(torch.bfloat16).float(), cfg.x_dt)
        wgq = wguq[wset(e)].reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
        wuq = wguq[wset(e)].reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
        h = quantize(torch.nn.functional.silu(X @ wgq) * (X @ wuq), cfg.h_dt)
        ref = h @ wdq[wset(e)]
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
        groups=[(p[1] - p[0]) // max(1, p[2] - 1) for p in per_group],
        xw_group=[float(s[g * cfg.GC : (g + 1) * cfg.GC, 41].double().mean()) for g in range(cfg.G)],
        unpack=(s[:, 40:44].double().mean(dim=0)).tolist(),
        nc=(s[:, 48:50].double().mean(dim=0)).tolist(),
        br0=(s[0:10, 50:55].double().mean(dim=0)).tolist(),
        br1=(s[10:20, 50:55].double().mean(dim=0)).tolist(),
        brmax0=(s[0:10, 104:109].double().max(dim=0).values).tolist(),
        brmax1=(s[10:20, 104:109].double().max(dim=0).values).tolist(),
        nc0=(s[0:10, 470:474].double().mean(dim=0)).tolist(),
        nc1=(s[10:20, 470:474].double().mean(dim=0)).tolist(),
        ncmax0=(s[0:10, 474:478].double().max(dim=0).values).tolist(),
        ncmax1=(s[10:20, 474:478].double().max(dim=0).values).tolist(),
    )


def report(tag, cfg, Ms, r):
    logger.info(
        f"WSG {tag} M mean {sum(Ms) / len(Ms):.0f} (min {min(Ms)} max {max(Ms)}) nexp {cfg.nexp} groups {cfg.G} "
        f"cores {r['cores']} (+{cfg.NINJ} injectors): {r['per_expert']:.0f} cyc/expert ({r['per_expert'] / r['aiclk']:.2f} "
        f"us), util tile {100 * r['util_tile']:.1f}% useful {100 * r['util_useful']:.1f}% (whole run incl. fill "
        f"{100 * r['util_all']:.1f}%); pcc per group {[round(p, 5) for p in r['pcs']]}"
    )
    logger.info(
        f"UNPACK totals per core (mean): tilize / x wait / weight wait / h wait {[round(v) for v in r['unpack']]}"
    )
    logger.info(f"NCRISC x multicast: blocked on credits / staging empty {[round(v) for v in r['nc']]}")
    logger.info(f"per group cycles / expert {r['groups']}, x wait {[round(v) for v in r['xw_group']]}")
    logger.info(
        f"BRISC max call qx/gather/hsend/hlead/y: col 0 {[round(v) for v in r['brmax0']]} col 1 {[round(v) for v in r['brmax1']]}"
    )
    logger.info(
        f"NCRISC cycles qw/qa/qb/qh: col 0 {[round(v) for v in r['nc0']]} col 1 {[round(v) for v in r['nc1']]}; max call col 0 {[round(v) for v in r['ncmax0']]} col 1 {[round(v) for v in r['ncmax1']]}"
    )
    logger.info(
        f"BRISC cycles in qx / gather / h send / h leader / y: col 0 {[round(v) for v in r['br0']]} col 1 {[round(v) for v in r['br1']]}"
    )


@pytest.mark.parametrize("M", [64, 256, 586, 1024, 2048])
def test_ws_group(device, M):
    cfg = GCfg(nexp=32)
    Ms = [M] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"const {M}", cfg, Ms, r)
    assert r["pcc"] > 0.999


@pytest.mark.parametrize("weights", [True, False])
def test_ws_group_nogather(device, weights):
    cfg = GCfg(nexp=32, wdefs=("WG_NO_GATHER",), idefs=() if weights else ("WG_INJ_NOWRITE",))
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"no gather weights {int(weights)}", cfg, Ms, r)


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
        owner = (c % 2) * 10 + (c // 2) % 10  # (expert 2 starts at stream 380: block l = c)
        til = int(rel[owner, 256 + c]) - base if int(s[owner, 256 + c]) else -1
        mc = int(rel[owner, 128 + c]) - base
        push = [int(rel[j, 192 + c]) - base for j in (0, 5, 15)]
        got = [int(rel[j, 64 + c]) - base for j in (0, 5, 15)]
        seen = int(rel[owner, 320 + c]) - base if int(s[owner, 320 + c]) else -1
        logger.info(
            f"TRACE block {380 + c} owner {owner}: tilized {til} seen {seen} blocked-by {int(s[owner, 384 + c]) - 1}"
            f"@{int(s[owner, 448 + c])} mcast {mc} pushed {push} got {got}"
        )


@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("xorder", [0, 1, 2])
def test_ws_group_noweights(device, G, xorder):
    """Debug: injectors signal arrival without writing (x / h traffic only; PCC not meaningful)."""
    cfg = GCfg(nexp=8 * G, G=G, idefs=("WG_INJ_NOWRITE",), xorder=xorder)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"noweights G {G} xorder {xorder}", cfg, Ms, r)


@pytest.mark.parametrize("CH,ICB", [(24, 2), (24, 4), (60, 2), (60, 3), (120, 2)])
def test_ws_group_inj(device, CH, ICB):
    cfg = GCfg(nexp=32, CH=CH, ICB=ICB)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"inj CH {CH} ICB {ICB}", cfg, Ms, r)


@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("BK,NX,LOOK", [(12, 15, 30), (24, 15, 30), (12, 30, 60)])
def test_ws_group_bfp8x(device, G, BK, NX, LOOK):
    cfg = GCfg(nexp=8 * G, G=G, x_dt=ttnn.bfloat8_b, BK=BK, NX=NX, LOOK=LOOK)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"bfp8 x G {G} BK {BK} NX {NX} LOOK {LOOK}", cfg, Ms, r)


@pytest.mark.parametrize(
    "G,wdefs,idefs",
    [
        (2, (), ()),
        (3, (), ()),
        (4, (), ("WG_INJ_NOWRITE",)),
        (4, ("WG_NO_GATHER",), ()),
        (4, ("WG_NO_GATHER",), ("WG_INJ_NOWRITE",)),
    ],
)
def test_ws_group_iso(device, G, wdefs, idefs):
    """bfp8 x, BK 24: what costs at G > 1 (debug defines break PCC)."""
    cfg = GCfg(nexp=8 * G, G=G, x_dt=ttnn.bfloat8_b, BK=24, wdefs=wdefs, idefs=idefs)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"iso G {G} {wdefs} {idefs}", cfg, Ms, r)


@pytest.mark.parametrize("G,NINJ", [(4, 10), (4, 20), (3, 20)])
def test_ws_group_ninj(device, G, NINJ):
    cfg = GCfg(nexp=8 * G, G=G, x_dt=ttnn.bfloat8_b, BK=24, NINJ=NINJ)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"ninj G {G} NINJ {NINJ}", cfg, Ms, r)


@pytest.mark.parametrize("dn_lag,NS", [(1, 4), (2, 4), (3, 6)])
def test_ws_group_lag(device, dn_lag, NS):
    cfg = GCfg(nexp=32, G=4, x_dt=ttnn.bfloat8_b, BK=24, NINJ=20, dn_lag=dn_lag, NS=NS)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"lag {dn_lag} NS {NS}", cfg, Ms, r)


@pytest.mark.parametrize(
    "G,idefs", [(4, ("WG_INJ_NOREAD",)), (4, ("WG_INJ_NOREAD", "WG_INJ_NOWRITE")), (3, ("WG_INJ_NOREAD",))]
)
def test_ws_group_noread(device, G, idefs):
    """Injectors send without reading DRAM (fabric-like landing; weights garbage, PCC not meaningful)."""
    cfg = GCfg(nexp=8 * G, G=G, x_dt=ttnn.bfloat8_b, BK=24, NINJ=20, idefs=idefs)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"noread G {G} {idefs}", cfg, Ms, r)


@pytest.mark.parametrize("G", [1, 2, 3, 4])
@pytest.mark.parametrize("wsrc", ["l1", "dram"])
def test_ws_group_fab(device, G, wsrc):
    cfg = GCfg(nexp=8 * G, G=G, x_dt=ttnn.bfloat8_b, BK=24, NINJ=20, wsrc=wsrc)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"wsrc {wsrc} G {G}", cfg, Ms, r)


@pytest.mark.parametrize("BK,NX,LOOK", [(12, 15, 30), (24, 8, 16), (12, 24, 48)])
def test_ws_group_bf16x(device, BK, NX, LOOK):
    cfg = GCfg(nexp=32, x_dt=ttnn.bfloat16, BK=BK, NX=NX, LOOK=LOOK)
    Ms = [586] * cfg.nexp
    r = run_group(device, cfg, Ms)
    report(f"bf16 x BK {BK} NX {NX}", cfg, Ms, r)
