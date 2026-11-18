# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Lane design with 5 x 5 K-split groups (ws_lane5.py) on the 2 x 4 tray: the row-snake cycle x 2 links = 2 lanes,
G 2, 32 experts (16 a lane), T 8192, every expert's output written and checked. WS_WARM / WS_MEAS launches (as
test_ws_lane); WS_LANE / WS_GCFG / WS_L5 override LCfg / GCfg / L5Cfg fields ("k=v;k=v")."""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, quantize, to_dev
from models.demos.ws_expert.tests.test_ws_fab_bench import FAB, TORUS
from models.demos.ws_expert.tests.test_ws_lane import chip_pcc, chip_time, lcfg_from_env
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane import make_tables, tables_tensor
from models.demos.ws_expert.ws_lane5 import L5Cfg, build_lane5, slice_layout5
from models.demos.ws_expert.ws_sf import torus_cycles

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))
LANE_DEFAULT = "CHT=20;slots=d0,d1,d0d1,any;keep=2;sched_c=7;wpoll=1"


def env_kv(name):
    out = {}
    for item in filter(None, os.environ.get(name, "").split(";")):
        k, v = item.split("=")
        out[k] = v
    return out


def configs(nexp, G, T):
    kw = dict(h_dt=ttnn.bfloat8_b, XRM=2, NS=5, dn_lag=3)
    for k, v in env_kv("WS_GCFG").items():
        kw[k] = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[v] if k.endswith("_dt") else int(v)
    cfg = GCfg(nexp=nexp, G=G, T=T, MMAX=T, GC=25, **kw)
    os.environ["WS_LANE"] = ";".join(filter(None, [LANE_DEFAULT, os.environ.get("WS_LANE_EXTRA", "")]))
    lc = lcfg_from_env()
    l5 = L5Cfg(**{k: int(v) for k, v in env_kv("WS_L5").items()})
    return cfg, lc, l5


def make_inputs5(mesh, cfg, lc, T, Ms, out_rows, seed=0):
    N = mesh.get_num_devices()
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
    per_chip = [slice_layout5(cfg, lc.CHT, wg[o].float(), wu[o].float(), wd[o].float()) for o in T["own"]]
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
        out_t=to_dev(torch.zeros(cfg.G * out_rows * 32, cfg.H), cfg.out_dt, mesh),
    )
    cache = {}

    def ref(e):
        if e not in cache:
            wguq = quantize(interleave_gu(wg[e].float(), wu[e].float()), cfg.w_dt)
            wdq = quantize(wd[e].float(), cfg.w_dt)
            Xq = quantize(x[idx[e].long()].to(torch.bfloat16).float(), cfg.x_dt)
            wgq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
            wuq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
            h = quantize(torch.nn.functional.silu(Xq @ wgq) * (Xq @ wuq), cfg.h_dt)
            cache[e] = (h @ wdq)[: Ms[e]]
        return cache[e]

    refs = {}
    for X in range(N):
        for g in range(cfg.G):
            lst, row = [], g * out_rows
            for e in T["meta"][(X, g)]["order"]:
                if Ms[e]:
                    lst.append((row, e, ref(e)))
                row += (Ms[e] + 31) // 32
            refs[(X, g)] = lst
    return dev, refs


def run5(mesh, cfg, lc, l5, Ms, tag, warm=WARM, meas=MEAS):
    shape = tuple(mesh.shape)
    N = mesh.get_num_devices()
    T = make_tables(shape, cfg, lc)
    R = [(m + 31) // 32 for m in Ms]
    out_rows = max(1, max(sum(R[e] for e in T["meta"][(X, g)]["order"]) for X in range(N) for g in range(cfg.G)))
    dev, refs = make_inputs5(mesh, cfg, lc, T, Ms, out_rows)
    tab_t = tables_tensor(mesh, T["tabs"])
    clk = aiclk_mhz(0)
    logger.info(f"WL5 {tag}: {T['E']} experts a lane, {lc}, {l5}, GCfg NS {cfg.NS} dn_lag {cfg.dn_lag} h {cfg.h_dt}")
    chips = list(range(N))
    rows, pcs = [], None
    for i in range(warm + meas):
        mpd, io, st = build_lane5(
            mesh, cfg, lc, l5, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"], tab_t, T, out_rows
        )
        ttnn.synchronize_device(mesh)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh)
        if i >= warm:
            sts = ttnn.get_device_tensors(st)
            row = {ci: chip_time(cfg, Ms, T, ci, sts[ci], x0=512) for ci in chips}
            rows.append(row)
            logger.info(
                f"WL5 {tag} launch {i - warm + 1}/{meas}: util tile per chip {[round(100 * row[ci][1], 1) for ci in chips]}"
                f", end-to-end us per chip {[round(row[ci][3] / clk, 1) for ci in chips]}"
            )
            if i == warm + meas - 1:
                s = (ttnn.to_torch(sts[0]).to(torch.int64) & 0xFFFFFFFF).reshape(cfg.G * cfg.GC, -1).double()
                own = [k for k in range(cfg.G * cfg.GC) if k % 25 % 5 < 4]
                r4 = [k for k in range(cfg.G * cfg.GC) if k % 25 % 5 == 4]
                for name, ks in (("owners", own), ("row 4", r4)):
                    logger.info(
                        f"WL5 chip 0 {name}: MATH (K) gate/up / fold / down / tilize "
                        f"{(s[ks, 45:49].mean(0) / 1e3).round().tolist()}; waits (K) tilize / x / weights / h / "
                        f"landing {(s[ks, 40:45].mean(0) / 1e3).round().tolist()}"
                    )
                logger.info(
                    f"WL5 chip 0 BRISC busy (K) qx / gather / partials / credits / h out / own "
                    f"{(s[:, 50:56].mean(0) / 1e3).round().tolist()}; NCRISC busy (K) weights / x ring / landing / "
                    f"partials up / h ring / y {(s[:, 470:476].mean(0) / 1e3).round().tolist()}; packets forwarded "
                    f"{s[:, 58].mean():.0f}, own slices {s[:, 56].mean():.1f} a core"
                )
            if l5.why and i == warm + meas - 1:
                for ci in chips:
                    s = (ttnn.to_torch(sts[ci]).to(torch.int64) & 0xFFFFFFFF).reshape(cfg.G * cfg.GC, -1).double()
                    for g in range(cfg.G):
                        q = s[g * cfg.GC : (g + 1) * cfg.GC]
                        logger.info(
                            f"WL5 why chip {ci} group {g}: idle us none / arrival / credit / mux / refill "
                            f"{(q[:, 477:482].mean(0) / clk).round().tolist()}, busy us {(q[:, 470].mean() / clk):.0f}, "
                            f"loop us {(q[:, 484].mean() / clk):.0f}, first / last forward us "
                            f"{(q[:, 482].mean() / clk):.0f} / {(q[:, 483].mean() / clk):.0f}"
                        )
            del sts
        if i == warm + meas - 1:
            outs = ttnn.get_device_tensors(dev["out_t"])
            pcs = {ci: chip_pcc(refs, ci, cfg.G, outs[ci]) for ci in chips}
            del outs
        del mpd, io, st
    allu = [100 * r[ci][1] for r in rows for ci in chips]
    e2e = statistics.median(max(r[ci][3] for ci in chips) for r in rows) / clk
    wend = [round(statistics.median(max(r[ci][4][g] for ci in chips) for r in rows) / clk, 1) for g in range(cfg.G)]
    logger.info(
        f"WL5 {tag} SUMMARY: {warm} warm-up + {meas} measured launches, aiclk {clk} MHz, util tile over chips and "
        f"launches median {statistics.median(allu):.1f}% (min {min(allu):.1f}, max {max(allu):.1f}), min pcc "
        f"{min(pcs.values()):.5f}, end to end (slowest chip) median {e2e:.1f} us, NCRISC end per group {wend} us"
    )
    return rows, pcs


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("M", [586, 2048, 0, 5], ids=["m586", "m2048", "mzero", "mfive"])
def test_ws_lane5_tray(mesh_device, M):
    cfg, lc, l5 = configs(32, 2, 8192)
    Ms = [M] * cfg.nexp
    rows, pcs = run5(mesh_device, cfg, lc, l5, Ms, f"tray M {M}")
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
def test_ws_lane5_tray_dynm(mesh_device):
    """Uneven M per expert (zeros, 1-row, large), every expert checked."""
    import random

    cfg, lc, l5 = configs(32, 2, 8192)
    rnd = random.Random(3)
    Ms = [rnd.choice([0, 1, 5, 31, 33, 64, 300, 586, 1100, 2048, 4000]) for _ in range(cfg.nexp)]
    rows, pcs = run5(mesh_device, cfg, lc, l5, Ms, "tray dynamic M", warm=2, meas=1)
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
@pytest.mark.parametrize("nexp,T,M", [(128, 8192, 586), (640, 31250, 586)], ids=["e128m586", "e640m586"])
def test_ws_lane5_glx(mesh_device, nexp, T, M):
    """The whole Galaxy (8 x 4, FABRIC_2D_TORUS_XY): two edge-disjoint Hamiltonian cycles x 2 links = 4 lanes, G 4
    groups of 5 x 5 (12 x 10 grid). Not run yet."""
    cfg, lc, l5 = configs(nexp, 4, T)
    lc.cycles = torus_cycles(*tuple(mesh_device.shape))
    Ms = [M] * cfg.nexp
    rows, pcs = run5(mesh_device, cfg, lc, l5, Ms, f"glx {nexp} experts M {M}")
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
def test_ws_lane5_tray_e640(mesh_device):
    """The full expert count on the tray: 640 experts (320 a lane; WS_NEXP overrides), T 8192, M 586, every expert
    checked."""
    cfg, lc, l5 = configs(int(os.environ.get("WS_NEXP", "640")), 2, 8192)
    Ms = [586] * cfg.nexp
    rows, pcs = run5(mesh_device, cfg, lc, l5, Ms, "tray 640 experts M 586", warm=2, meas=3)
    assert min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("M", [586, 0], ids=["m586", "mzero"])
def test_ws_lane5_tray_oneway(mesh_device, M):
    """One-way lanes (LCfg.oneway) on the 8-chip snake cycle as a Galaxy column's ring: 2 links x 2 directions = 4
    lanes, G 4 (the 12 x 10 grid), 160 experts (40 a lane, 5 a chip a lane: a Galaxy EP-4 column's load), credits on
    header-only channels of the partner lane's mux; hub h multicast (hmc 2), muxes on NOC1. WS_ONEWAY_SLOTS: the slot
    types (default d0,d0,any,any); WS_NEXP overrides 160. M 0: weights only."""
    cfg, lc, l5 = configs(int(os.environ.get("WS_NEXP", "160")), 4, 8192)
    lc.oneway = True
    lc.slots = os.environ.get("WS_ONEWAY_SLOTS", "d0,d0,any,any")
    if not os.environ.get("WS_L5"):
        l5 = L5Cfg(hmc=2)
    Ms = [M] * cfg.nexp
    rows, pcs = run5(mesh_device, cfg, lc, l5, Ms, f"tray one-way G 4 M {M}", warm=3, meas=3)
    assert M == 0 or min(pcs.values()) > 0.999


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
def test_ws_lane5_tray_oneway_link0(mesh_device):
    """test_ws_lane5_tray_oneway's per-lane load (40 experts a lane) on link 0 only: 2 one-way lanes, G 2, both muxes
    in column 0 - half the muxes a chip, so at most 2 mux flows share a NoC link of the ethernet row. M 0."""
    cfg, lc, l5 = configs(int(os.environ.get("WS_NEXP", "80")), 2, 8192)
    lc.oneway = True
    lc.NL = 1
    lc.slots = os.environ.get("WS_ONEWAY_SLOTS", "d0,d0,any,any")
    if not os.environ.get("WS_L5"):
        l5 = L5Cfg(hmc=2)
    Ms = [0] * cfg.nexp
    run5(mesh_device, cfg, lc, l5, Ms, "tray one-way link 0 G 2 M 0", warm=3, meas=3)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FAB], indirect=True)
@pytest.mark.parametrize("M", [586, 0], ids=["m586", "mzero"])
def test_ws_lane5_tray_bidir160(mesh_device, M):
    """The bidirectional lanes' twin of test_ws_lane5_tray_oneway: the same 8-chip snake, hmc 2 with muxes on NOC1,
    160 experts on G 2 (80 a lane: the same 35 experts a link direction a mux)."""
    cfg, lc, l5 = configs(int(os.environ.get("WS_NEXP", "160")), 2, 8192)
    if not os.environ.get("WS_L5"):
        l5 = L5Cfg(hmc=2)
    Ms = [M] * cfg.nexp
    rows, pcs = run5(mesh_device, cfg, lc, l5, Ms, f"tray bidirectional G 2 M {M}", warm=3, meas=3)
    assert M == 0 or min(pcs.values()) > 0.999
