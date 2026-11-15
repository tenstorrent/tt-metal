# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""What a third in-core weight slot costs: the single-chip group program (ws_group, weights landed from injector L1,
"l1" mode; a 1 x 1 mesh device, as the multi-chip runs open it) at M 586, G 4, with the token-row ring (NX blocks x BK tiles, bfp8) and its staging shrunk until a third
207 KB slot fits beside the two weight slots (GCfg.spare_slot reserves it; a variant that does not fit fails to
build). WS_WARM (default 20) warm-up + WS_MEAS (default 5) measured launches a variant; util and x waits from the group
cores' stamps (steady state, as test_ws_group.run_group), PCC on the last launch."""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, read_stamps, to_dev
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_group import GCfg, build

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


def inputs(device, cfg: GCfg, Ms, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(cfg.T + 1, cfg.H)
    x[cfg.T] = 0
    idx = torch.full((cfg.nexp, cfg.RMAX * 32), cfg.T, dtype=torch.int32)
    for e, M in enumerate(Ms):
        idx[e, :M] = torch.randperm(cfg.T)[:M].to(torch.int32)
    counts = torch.tensor(Ms, dtype=torch.int32).reshape(1, -1)
    ns = cfg.P
    wg = torch.randn(ns, cfg.H, cfg.I) / cfg.H**0.5
    wu = torch.randn(ns, cfg.H, cfg.I) / cfg.H**0.5
    wd = torch.randn(ns, cfg.I, cfg.H) / cfg.I**0.5
    wgu = torch.cat([interleave_gu(wg[e], wu[e]) for e in range(ns)], 0)
    rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    dev = dict(
        x_t=rm(x, ttnn.bfloat16),
        idx_t=rm(idx, ttnn.uint32),
        counts_t=rm(counts, ttnn.uint32),
        wgu_t=to_dev(wgu, cfg.w_dt, device),
        wd_t=to_dev(wd.reshape(-1, cfg.H), cfg.w_dt, device),
        out_t=to_dev(torch.zeros(cfg.G * cfg.RMAX * 32, cfg.H), cfg.out_dt, device),
    )
    wguq = quantize(wgu, cfg.w_dt).reshape(ns, cfg.H, -1)
    wdq = quantize(wd.reshape(-1, cfg.H), cfg.w_dt).reshape(ns, cfg.I, cfg.H)
    refs = []
    for g in range(cfg.G):
        e = max(range(g, cfg.nexp, cfg.G))
        st = (e // cfg.G) % ns
        X = quantize(x[idx[e].long()].to(torch.bfloat16).float(), cfg.x_dt)
        wgq = wguq[st].reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
        wuq = wguq[st].reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
        h = quantize(torch.nn.functional.silu(X @ wgq) * (X @ wuq), cfg.h_dt)
        refs.append((e, (h @ wdq[st])[: Ms[e]]))
    return dev, refs


def measure(cfg: GCfg, Ms, st):
    """Steady-state util tile and mean x wait per core (cycles) from the group cores' stamps."""
    NC = cfg.G * cfg.GC
    s = read_stamps(ttnn.get_device_tensors(st)[0], NC)
    rel = ((s - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    t0 = int(rel[:, 0].min())
    firsts, lasts = [], []
    for g in range(cfg.G):
        n_g = len(range(g, cfg.nexp, cfg.G))
        rg = rel[g * cfg.GC : (g + 1) * cfg.GC]
        firsts.append(int((rg[:, 1] - t0).max()))
        lasts.append(int((rg[:, n_g] - t0).max()))
    span = max(lasts) - max(firsts)
    R = [(m + 31) // 32 for m in Ms]
    es = [e for e in range(cfg.nexp) if e >= cfg.G]
    ops = sum(R[e] * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS) for e in es)
    unpack = s[:, 40:44].double().mean(dim=0).tolist()  # tilize / x wait / weight wait / h wait
    return ops * 16 / (NC * span), span / len(es), unpack


def bench(device, cfg: GCfg, Ms, tag, check=True):
    dev, refs = inputs(device, cfg, Ms)
    rows = []
    for i in range(WARM + MEAS):
        args = [dev[k] for k in ("x_t", "idx_t", "counts_t", "wgu_t", "wd_t", "out_t")]
        prog, io, st, ex = build(device, cfg, *args)
        mpd = ttnn.MeshProgramDescriptor()
        mc = ttnn.MeshCoordinate(0, 0)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = prog
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(device)
        if i >= WARM:
            rows.append(measure(cfg, Ms, st))
        del prog, mpd, io, st, ex  # (this launch's L1 tensors go before the next build allocates its own)
    out = ttnn.to_torch(ttnn.get_device_tensors(dev["out_t"])[0]).float()
    pcs = []
    for g, (e, ref) in enumerate(refs):
        o = out[g * cfg.RMAX * 32 : g * cfg.RMAX * 32 + Ms[e]]
        pcs.append(pcc(o, ref))
    clk = aiclk_mhz(0)
    ut = [100 * r[0] for r in rows]
    xw = statistics.median(r[2][1] for r in rows)
    logger.info(
        f"WS3 {tag}: {WARM} warm-up + {MEAS} measured, util tile median {statistics.median(ut):.1f}% (min {min(ut):.1f}, "
        f"max {max(ut):.1f}), {statistics.median(r[1] for r in rows):.0f} cyc/expert at {clk} MHz; waits per core "
        f"tilize / x / weight / h {[round(v) for v in rows[-1][2]]} (x median {xw:.0f}); pcc {min(pcs):.5f}"
    )
    assert not check or min(pcs) > 0.999
    return rows


# (BK, NX, LOOK, spare): the current layout, then token-row rings small enough for a third weight slot
VARIANTS = [
    (24, 15, 30, False),
    (24, 15, 30, True),
    (12, 15, 30, True),
    (12, 20, 40, True),
    (12, 24, 48, True),
    (24, 7, 14, True),
    (20, 9, 18, True),
]


IDS = [f"bk{v[0]}_nx{v[1]}_look{v[2]}_{'s3' if v[3] else 's2'}" for v in VARIANTS]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("BK,NX,LOOK,spare", VARIANTS, ids=IDS)
def test_ws_slot3(mesh_device, BK, NX, LOOK, spare):
    device = mesh_device
    cfg = GCfg(nexp=32, G=4, x_dt=ttnn.bfloat8_b, BK=BK, NX=NX, LOOK=LOOK, NINJ=20, wsrc="l1", P=2, spare_slot=spare)
    Ms = [586] * cfg.nexp
    bench(device, cfg, Ms, f"BK {BK} NX {NX} LOOK {LOOK} {'3 slots' if spare else '2 slots'}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("spare", [False, True], ids=["s2", "s3"])
def test_ws_slot3_noread(mesh_device, spare):
    """Injectors land without reading weights (smaller injector buffers; weights garbage, no PCC): fit check."""
    cfg = GCfg(nexp=32, G=4, x_dt=ttnn.bfloat8_b, BK=12, NX=15, LOOK=30, NINJ=20, wsrc="dram", spare_slot=spare,
               idefs=("WG_INJ_NOREAD",))
    bench(mesh_device, cfg, [586] * cfg.nexp, f"noread BK 12 NX 15 {'3 slots' if spare else '2 slots'}", check=False)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("y_noc1", [False, True], ids=["y_noc0", "y_noc1"])
def test_ws_y_noc(mesh_device, y_noc1):
    device = mesh_device
    """The down outputs to DRAM from the BRISC (NOC0) or the NCRISC (NOC1): same speed expected."""
    cfg = GCfg(nexp=32, G=4, x_dt=ttnn.bfloat8_b, BK=24, NINJ=20, wsrc="l1", P=2, y_noc1=y_noc1)
    Ms = [586] * cfg.nexp
    bench(device, cfg, Ms, f"down outputs on NOC{int(y_noc1)}")
