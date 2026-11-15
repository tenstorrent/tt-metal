# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Store-and-forward weight landing (ws_sf.py): every chip runs the 4 groups on its own tokens, each expert's slices
owned by chips spread round the cycles; the weights travel one fabric hop at a time around Hamiltonian cycles. test_sf_schedule checks the
ring streams on the host; the device tests make the inputs once and launch WS_WARM (default 20) warm-up + WS_MEAS
(default 5) measured times (every launch builds the program again: fresh credit / arrival words). The steady-state time
per expert comes from the group cores' expert-done stamps; PCC (each group's last expert) is checked on the last
launch, and the forwarders' timers of the last launch are logged."""

import os
import statistics
from collections import Counter

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, quantize, to_dev
from models.demos.ws_expert.tests.test_ws_fab_bench import FAB, TORUS, chip_pcc, chip_time, make_inputs
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_fab import slice_layout
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_sf import (
    INJ,
    TERM,
    RCfg,
    build_sf,
    conveyor,
    flow,
    local_items,
    make_tables,
    owner_table,
    rings,
    torus_cycles,
)

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


def check_schedule(shape, rc: RCfg, cfg: GCfg, seqs, items=None):
    """The ring orders (ws_sf.conveyor): each chip's outgoing stream on a ring is exactly the next chip's incoming one,
    only owners inject, every chip receives every slice it does not own exactly once and its local lander lands the
    ones it owns. Returns the incoming packets per chip and ring (load balance)."""
    R, Cn = shape
    NCHIP = R * Cn
    own = owner_table(shape, rc, cfg)
    got = [Counter() for _ in range(NCHIP)]
    load = {}
    for f, (lane, d, cyc, _) in enumerate(rings(shape, rc)):
        fl = flow(cyc, d, Cn)
        for i, chip in enumerate(fl):
            nx = fl[(i + 1) % len(fl)]
            out = [t[1:] for t in seqs[(chip, f)] if t[0] != TERM]
            inc = [t[1:] for t in seqs[(nx, f)] if t[0] != INJ]
            assert out == inc, f"ring {f}: stream {chip} -> {nx} differs"
            for kind, e, c, u in seqs[(chip, f)]:
                assert kind != INJ or own[e][c % cfg.GC] == chip
                if kind != INJ and u == 0:
                    got[chip][(e, c)] += 1
            load[(chip, f)] = sum(k != INJ for k, *_ in seqs[(chip, f)])
    for me in range(NCHIP):
        want = {(e, (e % cfg.G) * cfg.GC + j) for e in range(cfg.nexp) for j in range(cfg.GC) if own[e][j] != me}
        assert set(got[me]) == want and all(v == 1 for v in got[me].values()), f"chip {me} receives a wrong slice set"
        if items is not None:
            owned = [(e, (e % cfg.G) * cfg.GC + j) for e in range(cfg.nexp) for j in range(cfg.GC) if own[e][j] == me]
            assert sorted((it >> 8, it & 0xFF) for it in items[me]) == owned, f"chip {me}: local lander slices"
    return load


@pytest.mark.parametrize("shape,nexp", [((2, 4), 32), ((8, 4), 128)])
def test_sf_schedule(shape, nexp):
    cfg = GCfg(nexp=nexp, G=4)
    rc = RCfg() if shape[0] == 2 else RCfg(cycles=torus_cycles(*shape))
    seqs, steps, last = conveyor(shape, rc, cfg)
    load = check_schedule(shape, rc, cfg, seqs, local_items(shape, rc, cfg, last))
    out_max = max(sum(k != TERM for k, *_ in sq) for sq in seqs.values())
    logger.info(
        f"WSS schedule {shape} {nexp} experts: incoming packets a chip and ring min {min(load.values())} max "
        f"{max(load.values())}; conveyor model {steps} steps for at most {out_max} packets a link "
        f"({100 * out_max / steps:.1f}% busy)"
    )
    assert max(load.values()) - min(load.values()) <= max(load.values()) // 4


def make_inputs_spread(mesh, cfg: GCfg, rc: RCfg, Ms, seed=0):
    """Inputs for a whole-mesh run with owners per slice (ws_sf.owner_table): each chip's slices tensor holds only the
    slices it owns, slice j of expert e at local index (e // N) GC + j (WS_OWN_SLICES), weights drawn in bf16;
    references for each group's last expert."""
    N = mesh.get_num_devices()
    own = owner_table(tuple(mesh.shape), rc, cfg)
    torch.manual_seed(seed)
    x = torch.randn(cfg.T + 1, cfg.H)
    x[cfg.T] = 0
    idx = torch.full((cfg.nexp, cfg.RMAX * 32), cfg.T, dtype=torch.int32)
    for e, M in enumerate(Ms):
        idx[e, :M] = torch.randperm(cfg.T)[:M].to(torch.int32)
    counts = torch.tensor(Ms, dtype=torch.int32).reshape(1, -1)
    wg = torch.randn(cfg.nexp, cfg.H, cfg.I, dtype=torch.bfloat16) / cfg.H**0.5
    wu = torch.randn(cfg.nexp, cfg.H, cfg.I, dtype=torch.bfloat16) / cfg.H**0.5
    wd = torch.randn(cfg.nexp, cfg.I, cfg.H, dtype=torch.bfloat16) / cfg.I**0.5
    K, dw = cfg.nexp // N, 32 * cfg.DCOLS
    per_chip = []
    for X in range(N):
        # chip X's slices as K whole experts: local expert k's slice j is slice j of the expert in [k N, (k + 1) N)
        # that X owns (gate / up column tile j, down columns [dw j, dw (j + 1)))
        vg, vu, vd = torch.empty(K, cfg.H, cfg.I), torch.empty(K, cfg.H, cfg.I), torch.empty(K, cfg.I, cfg.H)
        for k in range(K):
            for j in range(cfg.GC):
                (e,) = [e for e in range(k * N, (k + 1) * N) if own[e][j] == X]
                vg[k][:, 32 * j : 32 * j + 32] = wg[e][:, 32 * j : 32 * j + 32].float()
                vu[k][:, 32 * j : 32 * j + 32] = wu[e][:, 32 * j : 32 * j + 32].float()
                vd[k][:, dw * j : dw * j + dw] = wd[e][:, dw * j : dw * j + dw].float()
        per_chip.append(slice_layout(cfg, rc.CHT, vg, vu, vd))
        del vg, vu, vd
    rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    dev = dict(
        x_t=rm(x, ttnn.bfloat16),
        idx_t=rm(idx, ttnn.uint32),
        counts_t=rm(counts, ttnn.uint32),
        ws_t=ttnn.from_torch(
            torch.cat(per_chip, 0),
            dtype=cfg.w_dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        ),
        out_t=to_dev(torch.zeros(cfg.G * cfg.RMAX * 32, cfg.H), cfg.out_dt, mesh),
    )
    del per_chip
    refs = []
    for g in range(cfg.G):
        e = max(range(g, cfg.nexp, cfg.G))
        wguq = quantize(interleave_gu(wg[e].float(), wu[e].float()), cfg.w_dt)
        wdq = quantize(wd[e].float(), cfg.w_dt)
        X = quantize(x[idx[e].long()].to(torch.bfloat16).float(), cfg.x_dt)
        wgq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
        wuq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
        h = quantize(torch.nn.functional.silu(X @ wgq) * (X @ wuq), cfg.h_dt)
        refs.append((e, (h @ wdq)[: Ms[e]]))
    return dev, refs


def fwd_report(rc: RCfg, nf, sst_chip, clk):
    """Per forwarder: NCRISC (arrival / downstream credit / owner read / send / drain waits, K cycles), packets out,
    link GB/s over the send span; BRISC (arrival / slot credit / land / DRAM read, K cycles)."""
    s = (ttnn.to_torch(sst_chip).to(torch.int64) & 0xFFFFFFFF).reshape(1 + nf, 32)
    chb = rc.CHT * 576
    out = []
    for f in range(nf):
        w = s[1 + f].tolist()
        span = max(w[12], 1)
        out.append(
            dict(
                nc=[w[i] // 1000 for i in range(6)],
                sent=w[6],
                tin=w[7],
                relays=w[9],
                nbuf=w[10],
                stage=w[11],
                gbs=w[6] * chb / (span / clk * 1e3),
                br=[w[16 + i] // 1000 for i in range(5)],
                landed=w[21],
            )
        )
    return out


def run(mesh, cfg: GCfg, rc: RCfg, Ms, dev, refs, tag, warm=WARM, meas=MEAS):
    chips = list(range(mesh.get_num_devices()))
    tabs = make_tables(mesh, cfg, rc)
    check_schedule(tuple(mesh.shape), rc, cfg, tabs["seqs"], tabs["items"])
    clk = aiclk_mhz(0)
    nf = len(rings(tuple(mesh.shape), rc))
    rows, fw = [], None
    for i in range(warm + meas):
        logger.debug(f"WSS {tag} launch {i + 1} of {warm + meas}")
        mpd, io, st, sst = build_sf(
            mesh, cfg, rc, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"], tabs
        )
        ttnn.synchronize_device(mesh)  # (every chip's fresh credit words are written before any chip starts)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        if i >= warm:
            sts = ttnn.get_device_tensors(st)
            row = {ci: chip_time(cfg, Ms, sts[ci]) for ci in chips}
            rows.append(row)
            del sts
            logger.info(
                f"WSS {tag} launch {i - warm + 1}/{meas}: util tile per chip "
                f"{[round(100 * row[ci][1], 1) for ci in chips]}"
            )
        if i == warm + meas - 1:
            outs = ttnn.get_device_tensors(dev["out_t"])
            pcs = {ci: chip_pcc(cfg, Ms, refs, outs[ci]) for ci in chips}
            ssts = ttnn.get_device_tensors(sst)
            fw = {ci: fwd_report(rc, nf, ssts[ci], clk) for ci in chips}
            del ssts, outs
        del mpd, io, st, sst
    we = 3 * cfg.H * cfg.I * 576 / 1024  # bfp4 bytes of one expert
    stages = {f["stage"] for ci in chips for f in fw[ci]}
    assert len(stages) == 1, f"staging CB address differs between forwarders: {stages}"
    for ci in chips:
        cyc = statistics.median(r[ci][0] for r in rows)
        ut = [100 * r[ci][1] for r in rows]
        logger.info(
            f"WSS {tag} chip {ci}: median {cyc:.0f} cyc/expert ({cyc / clk:.2f} us), util tile median "
            f"{statistics.median(ut):.1f}% (min {min(ut):.1f}, max {max(ut):.1f}), weights in "
            f"{we / (cyc / clk * 1e3):.0f} GB/s; waits (K) tilize/x/weight/h {rows[-1][ci][2]}; pcc {pcs[ci]:.5f}"
        )
        for f, r in enumerate(fw[ci]):
            logger.info(
                f"WSS {tag} chip {ci} ring {f // 2}{'+-'[f % 2]}: out {r['sent']} in {r['tin']} pkts, "
                f"{r['gbs']:.1f} GB/s over the send span; NCRISC K arr/dcred/own/send/drain/total {r['nc']}, "
                f"relays {r['relays']}, fabric slots {r['nbuf']}; BRISC K arr/slot/land/dram/total {r['br']}"
            )
    allu = [100 * r[ci][1] for r in rows for ci in chips]
    logger.info(
        f"WSS {tag} SUMMARY: {warm} warm-up + {meas} measured launches, aiclk {clk} MHz, util tile over chips and "
        f"launches median {statistics.median(allu):.1f}% (min {min(allu):.1f}, max {max(allu):.1f}), "
        f"min pcc {min(pcs.values()):.5f}"
    )
    return rows, pcs, fw


def rcfg_from_env(**kw):
    """RCfg with overrides from WS_SF (e.g. "S=12,CRB=4")."""
    for item in filter(None, os.environ.get("WS_SF", "").split(",")):
        k, v = item.split("=")
        kw[k] = int(v)
    return RCfg(**kw)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("M", [586, 2048])
def test_ws_sf_tray(mesh_device, M):
    """Stage A: one 2 x 4 tray, the row-snake cycle (0,0)..(0,3),(1,3)..(1,0), both directions x 2 links."""
    cfg, rc = GCfg(nexp=32, G=4), rcfg_from_env()
    Ms = [M] * cfg.nexp
    dev, refs = make_inputs(mesh_device, cfg, rc.CHT, Ms)
    warm = int(os.environ.get("WS_SF_WARM", WARM))
    rows, pcs, fw = run(mesh_device, cfg, rc, Ms, dev, refs, f"tray M {M}", warm=warm)
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
@pytest.mark.parametrize("nexp,T,M", [(128, 8192, 586), (128, 8192, 2048)], ids=["e128m586", "e128m2048"])
def test_ws_sf_glx(mesh_device, nexp, T, M):
    """Stage B: the whole Galaxy (8 x 4, FABRIC_2D_TORUS_XY), two edge-disjoint Hamiltonian cycles x 2 directions x 2
    links = 8 directed rings, one per incoming link of every chip; each chip holds only the slices it owns."""
    cfg = GCfg(nexp=nexp, G=4, T=T)
    rc = rcfg_from_env(cycles=torus_cycles(*tuple(mesh_device.shape)), own_slices=True)
    Ms = [M] * cfg.nexp
    dev, refs = make_inputs_spread(mesh_device, cfg, rc, Ms)
    warm = int(os.environ.get("WS_SF_WARM", WARM))
    rows, pcs, fw = run(mesh_device, cfg, rc, Ms, dev, refs, f"glx {nexp} experts M {M}", warm=warm)
    assert min(pcs.values()) > 0.999
