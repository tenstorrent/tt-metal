# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert groups with weights landing over the fabric (LoudBox 2 x 4): step 1, one receiving chip, its
neighbours only send weights (read from their DRAM). Checks each group's last expert on the receiving chip."""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, to_dev
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_fab import FCfg, build_fab, slice_layout
from models.demos.ws_expert.ws_group import GCfg


def router_cfg(payload):
    c = ttnn._ttnn.fabric.FabricRouterConfig()
    c.max_packet_payload_size_bytes = payload
    return c


def run_fab(mesh, cfg: GCfg, fc: FCfg, Ms, seed=0):
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
    wgu = torch.cat([interleave_gu(wg[e], wu[e]) for e in range(cfg.nexp)], 0)
    rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    x_t, idx_t, counts_t = rm(x, ttnn.bfloat16), rm(idx, ttnn.uint32), rm(counts, ttnn.uint32)
    ws_t = to_dev(slice_layout(cfg, fc.CHT, wg, wu, wd), cfg.w_dt, mesh)
    out_t = to_dev(torch.zeros(cfg.G * cfg.RMAX * 32, cfg.H), cfg.out_dt, mesh)
    mpd, io, st, sst = build_fab(mesh, cfg, fc, x_t, idx_t, counts_t, ws_t, out_t)
    ttnn.generic_op(io, mpd)
    ttnn.synchronize_device(mesh)
    ri = fc.recv[0] * mesh.shape[1] + fc.recv[1]
    out = ttnn.to_torch(ttnn.get_device_tensors(out_t)[ri]).float()
    wguq = quantize(wgu, cfg.w_dt).reshape(cfg.nexp, cfg.H, -1)
    wdq = quantize(wd.reshape(-1, cfg.H), cfg.w_dt).reshape(cfg.nexp, cfg.I, cfg.H)
    pcs = []
    for g in range(cfg.G):
        e = max(range(g, cfg.nexp, cfg.G))
        M = Ms[e]
        X = quantize(x[idx[e].long()].to(torch.bfloat16).float(), cfg.x_dt)
        wgq = wguq[e].reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
        wuq = wguq[e].reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
        h = quantize(torch.nn.functional.silu(X @ wgq) * (X @ wuq), cfg.h_dt)
        o = out[g * cfg.RMAX * 32 : g * cfg.RMAX * 32 + M]
        pcs.append(pcc(o, (h @ wdq[e])[:M]) if M else 1.0)
    NC = cfg.G * cfg.GC
    s = (ttnn.to_torch(ttnn.get_device_tensors(st)[ri]).to(torch.int64) & 0xFFFFFFFF).reshape(NC, -1)
    rel = ((s - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    t0 = int(rel[:, 0].min())
    first = last = 0
    for g in range(cfg.G):
        n_g = len(range(g, cfg.nexp, cfg.G))
        rg = rel[g * cfg.GC : (g + 1) * cfg.GC]
        first = max(first, int((rg[:, 1] - t0).max()))
        last = max(last, int((rg[:, n_g] - t0).max()))
    done = [e for e in range(cfg.nexp) if e >= cfg.G]
    R = [(m + 31) // 32 for m in Ms]
    ops = sum(R[e] * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS) for e in done)
    span = last - first
    util = ops * 16 / (NC * span)
    useful = util * sum(Ms[e] for e in done) / sum(32 * R[e] for e in done)
    clk = aiclk_mhz(0)
    from models.demos.ws_expert.ws_fab import neighbours

    snd = []
    for nb in neighbours(tuple(mesh.shape), fc.recv):
        v = ttnn.to_torch(ttnn.get_device_tensors(sst)[nb[0] * mesh.shape[1] + nb[1]]).to(torch.int64)
        snd += [v[1 + l, :8].tolist() for l in range(fc.links)]  # (row 0: the relay's)
    wbytes = 3 * cfg.H * cfg.I * 576 / 1024 * len(done)
    return dict(
        per_expert=span / len(done),
        util=util,
        useful=useful,
        pcs=pcs,
        clk=clk,
        wgbs=wbytes / (span / clk * 1e3),
        unpack=(s[:, 40:44].double().mean(dim=0)).tolist(),
        snd=snd,
    )


FAB = {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": router_cfg(15232)}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("G,M", [(1, 586), (4, 586), (4, 2048)])
def test_ws_fab_recv(mesh_device, G, M):
    cfg = GCfg(nexp=8 * G, G=G)
    fc = FCfg()
    Ms = [M] * cfg.nexp
    r = run_fab(mesh_device, cfg, fc, Ms)
    logger.info(
        f"WSF recv {fc.recv} G {G} M {M}: {r['per_expert']:.0f} cyc/expert ({r['per_expert'] / r['clk']:.2f} us), "
        f"util tile {100 * r['util']:.1f}% useful {100 * r['useful']:.1f}%, weights in {r['wgbs']:.0f} GB/s; "
        f"pcc {[round(p, 5) for p in r['pcs']]}"
    )
    logger.info(
        f"UNPACK totals per core (mean): tilize / x wait / weight wait / h wait {[round(v) for v in r['unpack']]}"
    )
    assert min(r["pcs"]) > 0.999


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("RD,SCB", [(1, 4), (4, 8), (8, 12)])
def test_ws_fab_send(mesh_device, RD, SCB):
    """Sender throughput: DRAM reads in flight (G 4, M 586: needs ~160 GB/s of weights into the receiver)."""
    cfg = GCfg(nexp=32, G=4)
    fc = FCfg(RD=RD, SCB=SCB)
    Ms = [586] * cfg.nexp
    r = run_fab(mesh_device, cfg, fc, Ms)
    logger.info(
        f"WSF send RD {RD} SCB {SCB}: {r['per_expert']:.0f} cyc/expert, util tile {100 * r['util']:.1f}%, "
        f"weights in {r['wgbs']:.0f} GB/s; pcc {[round(p, 5) for p in r['pcs']]}"
    )
    logger.info(
        f"senders cb wait / send / credit wait / total | reader reserve / issue / barrier / total (kcyc): {[[v // 1000 for v in x] for x in r['snd']]}"
    )
    assert min(r["pcs"]) > 0.999
