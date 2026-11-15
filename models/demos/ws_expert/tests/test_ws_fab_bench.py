# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Timed runs of the fabric programs in test_ws_fab.py (step 1: one receiving chip; step 2: every chip runs experts),
with warm-up launches. The inputs are made once; every launch builds the program again (fresh credit / arrival words),
WS_WARM launches (default 20) run first and WS_MEAS (default 5) are measured. The steady-state time per expert comes
from the group cores' own expert-done stamps, as in test_ws_fab.py; PCC is checked on the last launch."""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, to_dev
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_fab import ACfg, FCfg, build_all, build_fab, slice_layout
from models.demos.ws_expert.ws_group import GCfg

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


def router_cfg(payload):
    c = ttnn._ttnn.fabric.FabricRouterConfig()
    c.max_packet_payload_size_bytes = payload
    return c


FAB = {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": router_cfg(15232)}


def make_inputs(mesh, cfg, CHT, Ms, seed=0):
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
    rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    dev = dict(
        x_t=rm(x, ttnn.bfloat16),
        idx_t=rm(idx, ttnn.uint32),
        counts_t=rm(counts, ttnn.uint32),
        ws_t=to_dev(slice_layout(cfg, CHT, wg, wu, wd), cfg.w_dt, mesh),
        out_t=to_dev(torch.zeros(cfg.G * cfg.RMAX * 32, cfg.H), cfg.out_dt, mesh),
    )
    wgu = torch.cat([interleave_gu(wg[e], wu[e]) for e in range(cfg.nexp)], 0)
    wguq = quantize(wgu, cfg.w_dt).reshape(cfg.nexp, cfg.H, -1)
    wdq = quantize(wd.reshape(-1, cfg.H), cfg.w_dt).reshape(cfg.nexp, cfg.I, cfg.H)
    refs = []
    for g in range(cfg.G):  # each group's last expert
        e = max(range(g, cfg.nexp, cfg.G))
        X = quantize(x[idx[e].long()].to(torch.bfloat16).float(), cfg.x_dt)
        wgq = wguq[e].reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
        wuq = wguq[e].reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
        h = quantize(torch.nn.functional.silu(X @ wgq) * (X @ wuq), cfg.h_dt)
        refs.append((e, (h @ wdq[e])[: Ms[e]]))
    return dev, refs


def chip_time(cfg, Ms, st_chip):
    """(cycles per expert in the steady state, tile utilization, mean UNPACK waits in K cycles) of one chip."""
    NC = cfg.G * cfg.GC
    R = [(m + 31) // 32 for m in Ms]
    done = [e for e in range(cfg.nexp) if e >= cfg.G]
    ops = sum(R[e] * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS) for e in done)
    s = (ttnn.to_torch(st_chip).to(torch.int64) & 0xFFFFFFFF).reshape(NC, -1)
    rel = ((s - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    t0 = int(rel[:, 0].min())
    first = last = 0
    for g in range(cfg.G):
        n_g = len(range(g, cfg.nexp, cfg.G))
        rg = rel[g * cfg.GC : (g + 1) * cfg.GC]
        first = max(first, int((rg[:, 1] - t0).max()))
        last = max(last, int((rg[:, n_g] - t0).max()))
    span = last - first
    return span / len(done), ops * 16 / (NC * span), (s[:, 40:44].double().mean(dim=0) / 1000).round().tolist()


def chip_pcc(cfg, Ms, refs, out_chip):
    out = ttnn.to_torch(out_chip).float()
    return min(pcc(out[g * cfg.RMAX * 32 : g * cfg.RMAX * 32 + Ms[e]], ref) for g, (e, ref) in enumerate(refs))


def launches(mesh, build_fn, chips, cfg, Ms, dev, refs, tag):
    """WARM + MEAS launches; per measured launch and chip: (cycles per expert, util). PCC of the last launch."""
    meas = []
    for i in range(WARM + MEAS):
        mpd, io, st, sst = build_fn()
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        if i >= WARM:
            sts = ttnn.get_device_tensors(st)
            row = {ci: chip_time(cfg, Ms, sts[ci]) for ci in chips}
            meas.append(row)
            del sts  # (so this launch's tensors are freed before the next build)
            logger.info(
                f"WSB {tag} launch {i - WARM + 1}/{MEAS}: util tile per chip "
                f"{[round(100 * row[ci][1], 1) for ci in chips]}"
            )
        if i == WARM + MEAS - 1:
            outs = ttnn.get_device_tensors(dev["out_t"])
            pcs = {ci: chip_pcc(cfg, Ms, refs, outs[ci]) for ci in chips}
        del mpd, io, st, sst
    return meas, pcs


def summary(tag, cfg, Ms, chips, meas, pcs, clk):
    we = 3 * cfg.H * cfg.I * 576 / 1024  # bfp4 bytes of one expert
    for ci in chips:
        cyc = statistics.median(m[ci][0] for m in meas)
        ut = [100 * m[ci][1] for m in meas]
        logger.info(
            f"WSB {tag} chip {ci}: median {cyc:.0f} cyc/expert ({cyc / clk:.2f} us), util tile median "
            f"{statistics.median(ut):.1f}% (min {min(ut):.1f}, max {max(ut):.1f}), weights in "
            f"{we / (cyc / clk * 1e3):.0f} GB/s; waits (K) tilize/x/weight/h {meas[-1][ci][2]}; pcc {pcs[ci]:.5f}"
        )
    allu = [100 * m[ci][1] for m in meas for ci in chips]
    logger.info(
        f"WSB {tag} SUMMARY: {WARM} warm-up + {MEAS} measured launches, aiclk {clk} MHz, util tile over chips and "
        f"launches median {statistics.median(allu):.1f}% (min {min(allu):.1f}, max {max(allu):.1f}), "
        f"min pcc {min(pcs.values()):.5f}"
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("M", [586, 2048])
def test_ws_fab_all_bench(mesh_device, M):
    """Step 2: all 8 chips run the 4 groups; expert e owned by chip e % 8 (test_ws_fab_all, timed)."""
    cfg, ac = GCfg(nexp=32, G=4), ACfg()
    Ms = [M] * cfg.nexp
    dev, refs = make_inputs(mesh_device, cfg, ac.CHT, Ms)
    build_fn = lambda: build_all(
        mesh_device, cfg, ac, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"]
    )
    chips = list(range(mesh_device.get_num_devices()))
    meas, pcs = launches(mesh_device, build_fn, chips, cfg, Ms, dev, refs, f"all M {M}")
    summary(f"all M {M}", cfg, Ms, chips, meas, pcs, aiclk_mhz(0))
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("G,M", [(4, 586), (4, 2048)])
def test_ws_fab_recv_bench(mesh_device, G, M):
    """Step 1: one receiving chip (0, 1), its 3 neighbours send over 6 links (test_ws_fab_recv, timed)."""
    cfg, fc = GCfg(nexp=8 * G, G=G), FCfg()
    Ms = [M] * cfg.nexp
    dev, refs = make_inputs(mesh_device, cfg, fc.CHT, Ms)
    build_fn = lambda: build_fab(
        mesh_device, cfg, fc, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"]
    )
    ri = fc.recv[0] * mesh_device.shape[1] + fc.recv[1]
    meas, pcs = launches(mesh_device, build_fn, [ri], cfg, Ms, dev, refs, f"recv G {G} M {M}")
    summary(f"recv G {G} M {M}", cfg, Ms, [ri], meas, pcs, aiclk_mhz(0))
    assert min(pcs.values()) > 0.999


TORUS = {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": router_cfg(15232)}


def make_inputs_owned(mesh, cfg, CHT, Ms, nchip, seed=0):
    """Inputs for a whole-mesh run: each chip's slices tensor holds only the experts it owns (e % nchip == chip, in
    expert order; the reader's WS_OWN_SLICES), weights drawn in bf16; references for each group's last expert."""
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
    per_chip = [slice_layout(cfg, CHT, wg[o::nchip].float(), wu[o::nchip].float(), wd[o::nchip].float()) for o in range(nchip)]
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


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
@pytest.mark.parametrize("nexp,T,M", [(128, 8192, 586), (640, 31250, 586)], ids=["smoke128", "layer640"])
def test_ws_fab_glx_bench(mesh_device, nexp, T, M):
    """Step 2 on the whole Galaxy (8 x 4, FABRIC_2D_TORUS_XY): every chip runs the 4 groups on its own tokens, expert
    e owned by chip e % 32 and multicast over the torus. layer640 is a real layer at 1M tokens: 640 experts, 31,250
    tokens a chip, M rows an expert on every chip."""
    nchip = mesh_device.get_num_devices()
    cfg, ac = GCfg(nexp=nexp, G=4, T=T), ACfg(torus=True, own_slices=True)
    Ms = [M] * cfg.nexp
    dev, refs = make_inputs_owned(mesh_device, cfg, ac.CHT, Ms, nchip)
    build_fn = lambda: build_all(
        mesh_device, cfg, ac, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"]
    )
    chips = list(range(nchip))
    tag = f"glx {nexp} experts T {T} M {M}"
    meas, pcs = launches(mesh_device, build_fn, chips, cfg, Ms, dev, refs, tag)
    summary(tag, cfg, Ms, chips, meas, pcs, aiclk_mhz(0))
    assert min(pcs.values()) > 0.999
