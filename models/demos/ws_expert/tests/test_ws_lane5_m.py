# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""test_ws_lane5_glx at any load: WSW_NEXP experts (default 640), WSW_T tokens a chip (31250), WSW_M rows an expert;
WS_L5 selects the L5Cfg (hmc=1: h by the group hub's multicast); WSW_PCC_CHIPS (e.g. "0,31") limits the PCC readback
to those chips (default every chip). test_ws_lane5_glx_soak: launches for WS_SOAK_SECS, PCC on chips 0 and the last
every WS_SOAK_CHECK launches.
make_inputs5 is replaced by a copy that keeps each chip's weight slices in bf16 and allocates the output on the device
(no host zeros), and its references multiply only each expert's M rows (the original multiplies all RMAX x 32
index rows and keeps the full product alive through a slice: ~480 MB an expert, ~300 GB at 640); WL5MEM lines log this process's RSS and the node's MemAvailable after each input step and launch build."""

import os
import statistics
import time

import pytest
import torch

import ttnn
import models.demos.ws_expert.tests.test_ws_lane5 as L5
from models.demos.ws_expert.bench_util import quantize, to_dev
from models.demos.ws_expert.tests.test_ws_fab_bench import TORUS
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_lane5 import slice_layout5
from models.demos.ws_expert.ws_sf import torus_cycles
from loguru import logger


def mem(tag):
    """This process's RSS and the node's MemAvailable, GB."""
    rss = avail = 0
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) / 2**20
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) / 2**20
    logger.info(f"WL5MEM {tag}: rss {rss:.1f} GB, node MemAvailable {avail:.1f} GB")


def make_inputs5_lean(mesh, cfg, lc, T, Ms, out_rows, seed=0):
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
    mem("weights drawn")
    per_chip = [
        slice_layout5(cfg, lc.CHT, wg[o].float(), wu[o].float(), wd[o].float()).to(torch.bfloat16) for o in T["own"]
    ]
    mem("per-chip slices (bf16)")
    ws_host = torch.cat(per_chip, 0)
    del per_chip
    mem("slices concatenated")
    rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    dev = dict(
        x_t=rm(x, ttnn.bfloat16),
        idx_t=rm(idx, ttnn.uint32),
        counts_t=rm(counts, ttnn.uint32),
        ws_t=ttnn.from_torch(
            ws_host,
            dtype=cfg.w_dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        ),
        out_t=ttnn.allocate_tensor_on_device(
            ttnn.Shape([cfg.G * out_rows * 32, cfg.H]), cfg.out_dt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        ),
    )
    del ws_host
    mem(f"device inputs (out_rows {out_rows})")
    cache = {}

    def ref(e):
        if e not in cache:
            wguq = quantize(interleave_gu(wg[e].float(), wu[e].float()), cfg.w_dt)
            wdq = quantize(wd[e].float(), cfg.w_dt)
            Xq = quantize(x[idx[e, : Ms[e]].long()].to(torch.bfloat16).float(), cfg.x_dt)  # (the M routed rows only)
            wgq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
            wuq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
            h = quantize(torch.nn.functional.silu(Xq @ wgq) * (Xq @ wuq), cfg.h_dt)
            cache[e] = h @ wdq
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
    mem("references")
    return dev, refs


_chip_pcc = L5.chip_pcc


def chip_pcc_some(refs, ci, G, out_chip):
    chips = os.environ.get("WSW_PCC_CHIPS", "")
    if chips and ci not in [int(c) for c in chips.split(",")]:
        return 1.0
    return _chip_pcc(refs, ci, G, out_chip)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
def test_ws_lane5_glx_m(mesh_device):
    M = int(os.environ["WSW_M"])
    nexp, T = int(os.environ.get("WSW_NEXP", "640")), int(os.environ.get("WSW_T", "31250"))
    cfg, lc, l5 = L5.configs(nexp, 4, T)
    lc.cycles = torus_cycles(*tuple(mesh_device.shape))
    L5.make_inputs5 = make_inputs5_lean
    L5.chip_pcc = chip_pcc_some
    build0, n = L5.build_lane5, [0]

    def build_logged(*a, **k):
        out = build0(*a, **k)
        n[0] += 1
        mem(f"launch {n[0]} built")
        return out

    L5.build_lane5 = build_logged
    mem("start")
    L5.run5(mesh_device, cfg, lc, l5, [M] * cfg.nexp, f"glx {nexp} experts M {M}")


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
def test_ws_lane5_glx_soak(mesh_device):
    M = int(os.environ.get("WSW_M", "586"))
    nexp, T_tok = int(os.environ.get("WSW_NEXP", "640")), int(os.environ.get("WSW_T", "31250"))
    secs, every = float(os.environ.get("WS_SOAK_SECS", "900")), int(os.environ.get("WS_SOAK_CHECK", "500"))
    cfg, lc, l5 = L5.configs(nexp, 4, T_tok)
    lc.cycles = torus_cycles(*tuple(mesh_device.shape))
    N = mesh_device.get_num_devices()
    Ms = [M] * cfg.nexp
    T = L5.make_tables(tuple(mesh_device.shape), cfg, lc)
    R = [(m + 31) // 32 for m in Ms]
    out_rows = max(1, max(sum(R[e] for e in T["meta"][(X, g)]["order"]) for X in range(N) for g in range(cfg.G)))
    dev, refs = make_inputs5_lean(mesh_device, cfg, lc, T, Ms, out_rows)
    tab_t = L5.tables_tensor(mesh_device, T["tabs"])
    clk = L5.aiclk_mhz(0)
    logger.info(f"WL5S soak: {secs:.0f} s, {T['E']} experts a lane, M {M}, {lc}, {l5}, aiclk {clk} MHz")
    t0, n, checks, e2es, worst = time.time(), 0, 0, [], 1.0
    while time.time() - t0 < secs:
        mpd, io, st = L5.build_lane5(
            mesh_device, cfg, lc, l5, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"], tab_t, T,
            out_rows,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
        n += 1
        if n % every == 0 or n == 1:
            sts = ttnn.get_device_tensors(st)
            e2e = max(L5.chip_time(cfg, Ms, T, ci, sts[ci])[3] for ci in (0, N - 1)) / clk
            outs = ttnn.get_device_tensors(dev["out_t"])
            pcs = {ci: _chip_pcc(refs, ci, cfg.G, outs[ci]) for ci in (0, N - 1)}
            del sts, outs
            checks += 1
            e2es.append(e2e)
            worst = min(worst, *pcs.values())
            mem("soak check")
            logger.info(
                f"WL5S launch {n} at {time.time() - t0:.0f} s: e2e (chips 0, {N - 1}) {e2e:.1f} us, pcc {pcs}, "
                f"aiclk {L5.aiclk_mhz(0)} MHz"
            )
            assert min(pcs.values()) > 0.999, f"soak launch {n}: pcc {pcs}"
        del mpd, io, st
    logger.info(
        f"WL5S SUMMARY: {n} launches in {time.time() - t0:.0f} s, {checks} checks (chips 0, {N - 1}, every expert), "
        f"min pcc {worst:.5f}, e2e at the checks median {statistics.median(e2es):.1f} us"
    )
