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


def oaff_mask():
    """L5Cfg.oaff from WS_L5 (the output bank affinity's group mask; 0 off)."""
    for item in filter(None, os.environ.get("WS_L5", "").split(";")):
        k, v = item.split("=")
        if k == "oaff":
            return int(v)
    return 0


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
    assert not oaff_mask() or mesh.dram_grid_size().x == 8, "output bank affinity assumes 8 DRAM banks (4 a side)"
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
            ttnn.Shape([(2 if oaff_mask() else 1) * cfg.G * out_rows * 32, cfg.H]),
            cfg.out_dt,
            ttnn.TILE_LAYOUT,
            mesh,
            ttnn.DRAM_MEMORY_CONFIG,
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


def pcc_chip(refs, ci, G, out_chip):
    """test_ws_lane.chip_pcc, or with the output bank affinity (WS_L5 oaff) the min PCC over the chip's expert rows
    after un-permuting the tiles (group g's logical tile q at page 8 (q / 4) + q % 4 + 4 ((oaff >> g) & 1))."""
    mask = oaff_mask()
    if not mask:
        return _chip_pcc(refs, ci, G, out_chip)
    import torch
    from models.demos.ws_expert.bench_util import pcc

    phys = ttnn.to_torch(out_chip).float()
    H = phys.shape[-1]
    HT, rows2 = H // 32, phys.shape[0] // 32
    ormax = rows2 // (2 * G)
    tiles = phys.reshape(rows2, 32, HT, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)
    del phys
    q = torch.arange(G * ormax * HT)
    h = (mask >> (q // HT // ormax)) & 1
    lt = tiles[8 * (q // 4) + q % 4 + 4 * h].reshape(G * ormax, HT, 32, 32).permute(0, 2, 1, 3).reshape(-1, H)
    del tiles
    vals = [pcc(lt[row * 32 : row * 32 + r.shape[0]], r) for g in range(G) for row, e, r in refs[(ci, g)]]
    return min(vals) if vals else 1.0


def chip_pcc_some(refs, ci, G, out_chip):
    chips = os.environ.get("WSW_PCC_CHIPS", "")
    if chips and ci not in [int(c) for c in chips.split(",")]:
        return 1.0
    return pcc_chip(refs, ci, G, out_chip)


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


def aiclks():
    """Every chip's AICLK from sysfs (MHz)."""
    import glob

    out = []
    for f in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*/tt_aiclk")):
        try:
            out.append(int(open(f).read().strip()))
        except (OSError, ValueError):
            pass
    return out


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
def test_ws_lane5_glx_soak(mesh_device):
    """WS_SOAK_WARM launches (default 20), then launches for WS_SOAK_SECS; every WS_SOAK_CHECK launches and after the
    last one: PCC on chips 0 and the last (every expert), the launch's end-to-end time (slowest of the two chips) and
    every chip's AICLK."""
    M = int(os.environ.get("WSW_M", "586"))
    nexp, T_tok = int(os.environ.get("WSW_NEXP", "640")), int(os.environ.get("WSW_T", "31250"))
    secs, every = float(os.environ.get("WS_SOAK_SECS", "900")), int(os.environ.get("WS_SOAK_CHECK", "500"))
    warm = int(os.environ.get("WS_SOAK_WARM", "20"))
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
    logger.info(f"WL5S soak: {warm} warm-up, then {secs:.0f} s, {T['E']} experts a lane, M {M}, {l5}, aiclk {clk} MHz")

    def launch():
        mpd, io, st = L5.build_lane5(
            mesh_device, cfg, lc, l5, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], dev["out_t"], tab_t, T,
            out_rows,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
        return mpd, io, st

    rows = []

    def check(n, t, st):
        sts = ttnn.get_device_tensors(st)
        e2e = max(L5.chip_time(cfg, Ms, T, ci, sts[ci])[3] for ci in (0, N - 1)) / clk
        outs = ttnn.get_device_tensors(dev["out_t"])
        pcs = {ci: pcc_chip(refs, ci, cfg.G, outs[ci]) for ci in (0, N - 1)}
        del sts, outs
        a = aiclks()
        rows.append((n, t, e2e, min(pcs.values()), statistics.median(a) if a else -1, min(a) if a else -1))
        logger.info(
            f"WL5S check launch {n} at {t:.0f} s: e2e (chips 0, {N - 1}) {e2e:.1f} us, pcc "
            f"{ {k: round(v, 5) for k, v in pcs.items()} }, AICLK {len(a)} chips median "
            f"{statistics.median(a) if a else -1:.0f} min {min(a) if a else -1} MHz"
        )
        assert min(pcs.values()) > 0.999, f"soak launch {n}: pcc {pcs}"

    for _ in range(warm):
        mpd, io, st = launch()
        del mpd, io, st
    t0, n = time.time(), 0
    st = None
    while time.time() - t0 < secs:
        st = None  # (the previous launch's stamps freed before the next build: no L1 tensor outlives its launch)
        mpd, io, st = launch()
        n += 1
        if n % every == 0:
            check(n, time.time() - t0, st)
        del mpd, io
    if not rows or rows[-1][0] != n:
        check(n, time.time() - t0, st)
    e2es = [r[2] for r in rows]
    logger.info(
        f"WL5S SUMMARY: {warm} warm-up + {n} launches in {time.time() - t0:.0f} s, {len(rows)} checks (chips 0, "
        f"{N - 1}, every expert), min pcc {min(r[3] for r in rows):.5f}, e2e at the checks first {e2es[0]:.1f} / last "
        f"{e2es[-1]:.1f} / median {statistics.median(e2es):.1f} / max {max(e2es):.1f} us, AICLK at the checks median "
        f"{statistics.median(r[4] for r in rows):.0f} min {min(r[5] for r in rows)} MHz"
    )

def group_report(cfg, st_chip, tag):
    """Per group of one chip (stamps of the last launch, K cycles): the latest compute end and NCRISC loop end from the
    chip's earliest loop start, and the cores' mean waits (tilize / x / weights / h / landing), MATH (gate/up / fold /
    down / tilize), BRISC busy (x / gather / partials / credits / multicast / -) and NCRISC busy (weights / x ring /
    landing / partials up / h ring / y)."""
    import torch

    NC = cfg.G * cfg.GC
    s = (ttnn.to_torch(st_chip).to(torch.int64) & 0xFFFFFFFF).reshape(NC, -1)
    rel = ((s - int(s[0, 0]) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    t0 = int(min(rel[:, 0].min(), rel[:, 485].min(), rel[:, 496].min()))
    for g in range(cfg.G):
        r = slice(g * cfg.GC, (g + 1) * cfg.GC)
        k = lambda a, b: (s[r, a:b].double().mean(0) / 1000).round().tolist()
        logger.info(
            f"WL5G {tag} group {g}: compute end {(int(rel[r, 63].max()) - t0) / 1000:.0f} K, NCRISC end "
            f"{(int((rel[r, 485] + s[r, 484]).max()) - t0) / 1000:.0f} K; waits {k(40, 45)}; MATH {k(45, 49)}; "
            f"BRISC {k(50, 56)}; NCRISC {k(470, 476)}"
        )


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
def test_ws_lane5_glx_g(mesh_device):
    """test_ws_lane5_glx_m with a per-group report of chips 0 and the last after the last launch."""
    M = int(os.environ["WSW_M"])
    nexp, T_tok = int(os.environ.get("WSW_NEXP", "640")), int(os.environ.get("WSW_T", "31250"))
    cfg, lc, l5 = L5.configs(nexp, 4, T_tok)
    lc.cycles = torus_cycles(*tuple(mesh_device.shape))
    L5.make_inputs5 = make_inputs5_lean
    L5.chip_pcc = chip_pcc_some
    build0, n, last = L5.build_lane5, [0], {}
    warm, meas = int(os.environ.get("WS_WARM", "20")), int(os.environ.get("WS_MEAS", "5"))

    def build_kept(*a, **k):
        out = build0(*a, **k)
        n[0] += 1
        if n[0] == warm + meas:
            last["st"] = out[2]
        return out

    L5.build_lane5 = build_kept
    L5.run5(mesh_device, cfg, lc, l5, [M] * cfg.nexp, f"glx {nexp} experts M {M}")
    sts = ttnn.get_device_tensors(last["st"])
    N = mesh_device.get_num_devices()
    for ci in (0, N - 1):
        group_report(cfg, sts[ci], f"chip {ci}")


def yc_report(cfg, l5, st_chip, yst_chip, tag=None):
    """WS_L5 yc: the collectors' latest end from the chip's earliest group-core loop start (cycles); with tag, a line a
    collector (K cycles): row tiles, total, waits for the 25 arrivals, for the destinations, the row writes (issue / sent
    waits), the slot credits, the end's fence (the BRISC's when split)."""
    import torch

    NC, ny = cfg.G * cfg.GC, cfg.G * l5.yc
    s = (ttnn.to_torch(st_chip).to(torch.int64) & 0xFFFFFFFF).reshape(NC, -1)
    y = (ttnn.to_torch(yst_chip).to(torch.int64) & 0xFFFFFFFF).reshape(ny, -1)
    s00 = int(s[0, 0])
    rel = lambda v: ((v - s00 + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    t0 = int(min(rel(s[:, 0]).min(), rel(s[:, 485]).min(), rel(s[:, 496]).min()))
    if tag:
        k = lambda v: round(int(v) / 1000)
        for i in range(ny):
            r = y[i]
            logger.info(
                f"WL5YC {tag} group {i // l5.yc} collector {i % l5.yc}: tiles {int(r[6])}, end {k(rel(r[7]) - t0)} K, "
                f"total {k(r[1])} K; waits arrivals {k(r[2])} K, destinations {k(r[3])} K; row writes {k(r[4])} K "
                f"(issue {k(r[9])}, sent {k(r[10])}; BRISC go {k(r[16])}, issue {k(r[17])}, sent {k(r[18])}); credits "
                f"{k(r[5])} K; fence {k(r[8])} K (BRISC {k(r[19])})"
            )
    return int(rel(y[:, 7]).max()) - t0


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TORUS], indirect=True)
def test_ws_lane5_glx_yrm(mesh_device):
    """The 5 x 5 lane with row-major result rows (WS_L5 with yrm=1) into a [T 12 + 16, H] bf16 row-major buffer, through
    per-chip destination lists: every (expert, token) its own row, WSW_DEST "spread" (rows in random order over the
    whole buffer, every bank, as the layer's t S + k; the buffer holds max(T 12, the rows routed) rows) or "side" (a group's rows only in the DRAM banks of its side:
    channels 4-7 at NoC x 9 for groups 2 / 3, 0-3 at x 0 for groups 0 / 1). PCC on WSW_PCC_CHIPS (default 0 and the
    last) over every expert's rows. WS_L5 yc: through the row collectors; the time ends with the collectors' last end
    (each fenced its row writes) when that is later than the compute's, and a line a collector of chip 0 is logged."""
    import torch
    from models.demos.ws_expert.bench_util import pcc

    M = int(os.environ["WSW_M"])
    nexp, T_tok = int(os.environ.get("WSW_NEXP", "640")), int(os.environ.get("WSW_T", "31250"))
    mode = os.environ.get("WSW_DEST", "spread")
    cfg, lc, l5 = L5.configs(nexp, 4, T_tok)
    assert l5.yrm and l5.hmc == 2, "WS_L5 needs hmc=2;yrm=1"
    lc.cycles = torus_cycles(*tuple(mesh_device.shape))
    N, G = mesh_device.get_num_devices(), cfg.G
    assert mesh_device.dram_grid_size().x == 8
    Ms = [M] * cfg.nexp
    T = L5.make_tables(tuple(mesh_device.shape), cfg, lc)
    R = [(m + 31) // 32 for m in Ms]
    out_rows = max(1, max(sum(R[e] for e in T["meta"][(X, g)]["order"]) for X in range(N) for g in range(G)))
    dev, refs = make_inputs5_lean(mesh_device, cfg, lc, T, Ms, out_rows)
    rows = -(-max(T_tok * 12, sum(Ms)) // 8) * 8  # (every (expert, token) its own row; whole bank stripes)
    gen = torch.Generator().manual_seed(7)
    dests = torch.zeros(N, cfg.nexp, cfg.RMAX * 32, dtype=torch.int32)
    for X in range(N):
        grp = {e: g for g in range(G) for e in T["meta"][(X, g)]["order"]}
        if mode == "side":
            pools = []
            for h in (0, 1):  # rows r with (r % 8 >= 4) == h, in random order
                r = torch.arange(rows)
                r = r[(r % 8 >= 4) == bool(h)]
                pools.append(r[torch.randperm(len(r), generator=gen)])
            cur = [0, 0]
            for e in range(cfg.nexp):
                h = 1 if grp[e] >= 2 else 0
                dests[X, e, : Ms[e]] = pools[h][cur[h] : cur[h] + Ms[e]].to(torch.int32)
                cur[h] += Ms[e]
                assert cur[h] <= len(pools[h])
        else:
            perm = torch.randperm(rows, generator=gen)
            off = 0
            for e in range(cfg.nexp):
                dests[X, e, : Ms[e]] = perm[off : off + Ms[e]].to(torch.int32)
                off += Ms[e]
    dest_t = ttnn.from_torch(
        dests.reshape(N * cfg.nexp, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    res_t = ttnn.zeros(
        [rows + 16, cfg.H], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dev["out_t"] = res_t
    tab_t = L5.tables_tensor(mesh_device, T["tabs"])
    clk = L5.aiclk_mhz(0)
    warm, meas = int(os.environ.get("WS_WARM", "20")), int(os.environ.get("WS_MEAS", "5"))
    e2e, yc_lat = [], []
    for i in range(warm + meas):
        mpd, io, st = L5.build_lane5(
            mesh_device, cfg, lc, l5, dev["x_t"], dev["idx_t"], dev["counts_t"], dev["ws_t"], res_t, tab_t, T, out_rows,
            dest_t=dest_t, fence_row=rows,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(mesh_device)
        if i >= warm:
            sts = ttnn.get_device_tensors(st)
            ends = [L5.chip_time(cfg, Ms, T, ci, sts[ci])[3] for ci in range(N)]
            if l5.yc:
                ysts = ttnn.get_device_tensors(io[-1])
                yends = [yc_report(cfg, l5, sts[ci], ysts[ci]) for ci in range(N)]
                yc_lat.append(max(b - a for a, b in zip(ends, yends)) / clk)
                ends = [max(a, b) for a, b in zip(ends, yends)]
                if i == warm + meas - 1:
                    yc_report(cfg, l5, sts[0], ysts[0], f"yrm {mode} chip 0")
                del ysts
            e2e.append(max(ends) / clk)
            if i == warm + meas - 1:
                group_report(cfg, sts[0], f"yrm {mode} chip 0")
            del sts
        del mpd, io, st
    chips = [int(c) for c in os.environ.get("WSW_PCC_CHIPS", f"0,{N - 1}").split(",")]
    if os.environ.get("WSW_NOPCC", "0") == "1":  # (timing-only variants whose rows are not the outputs' layout)
        chips = []
    outs = ttnn.get_device_tensors(res_t)
    pcs = {}
    for ci in chips:
        out = ttnn.to_torch(outs[ci]).float()
        vals = [pcc(out[dests[ci, e, : r.shape[0]].long()], r) for g in range(G) for _, e, r in refs[(ci, g)]]
        pcs[ci] = min(vals)
        del out
    logger.info(
        f"WL5Y SUMMARY dest {mode}: {warm} warm-up + {meas} measured launches, aiclk {clk} MHz, end to end (slowest "
        f"chip) median {statistics.median(e2e):.1f} us (samples {[round(v, 1) for v in e2e]}), pcc {pcs}"
        + (f"; collectors yc {l5.yc} sink {l5.yc_sink}: their end minus the compute's (latest over chips) median {statistics.median(yc_lat):.1f} us" if l5.yc else "")
    )
    assert not pcs or min(pcs.values()) > 0.999
