# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert with x streamed from DRAM: row-major bf16 x [T + 1, H] (row T zero), the expert's rows gathered through
a row index table, tilized on the cluster cores and multicast down each column. Weights resident. Reports per-expert
steady state, utilization (tile / useful) and PCC vs torch."""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, read_stamps, to_dev
from models.demos.ws_expert.ws_cluster import plan
from models.demos.ws_expert.ws_expert import interleave_gu
from models.demos.ws_expert.ws_x import XCfg, build


def run_x(device, cfg: XCfg, M, seed=0):
    p = plan(cfg)
    torch.manual_seed(seed)
    x = torch.randn(cfg.T + 1, cfg.H)
    x[cfg.T] = 0
    rows = torch.randperm(cfg.T)[:M]
    idx = torch.full((cfg.R * 32,), cfg.T, dtype=torch.int32)
    idx[:M] = rows.to(torch.int32)
    wg = torch.randn(cfg.H, cfg.I) / cfg.H**0.5
    wu = torch.randn(cfg.H, cfg.I) / cfg.H**0.5
    wd = torch.randn(cfg.I, cfg.H) / cfg.I**0.5
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    idx_t = ttnn.from_torch(idx.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    wgu = interleave_gu(wg, wu)
    w_t = to_dev(wgu, cfg.w_dt, device)
    wd_t = to_dev(wd, cfg.w_dt, device)
    out_t = to_dev(torch.zeros(cfg.R * 32, cfg.H), cfg.out_dt, device)
    prog, io, st, sched = build(device, cfg, p, x_t, idx_t, w_t, wd_t, out_t)
    ttnn.generic_op(io, prog)
    ttnn.synchronize_device(device)
    clk = aiclk_mhz(0)

    X = x[idx.long()].to(torch.bfloat16).float()
    if cfg.x_dt != ttnn.bfloat16:
        X = quantize(X, cfg.x_dt)
    wguq = quantize(wgu, cfg.w_dt)
    wgq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
    wuq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
    wdq = quantize(wd, cfg.w_dt)
    h = quantize(torch.nn.functional.silu(X @ wgq) * (X @ wuq), cfg.h_dt)
    ref = h @ wdq
    out = ttnn.to_torch(out_t).float()
    pc = pcc(out[:M], ref[:M])
    if not pc > 0.999:
        bad = (out - ref).abs().reshape(cfg.R, 32, cfg.GC, cfg.DCOLS * 32).nan_to_num(1e9).amax(dim=(1, 3))
        refm = ref.abs().reshape(cfg.R, 32, cfg.GC, cfg.DCOLS * 32).amax(dim=(1, 3))
        logger.warning(f"bad (row tile, core j): {(bad > 0.05 * refm).nonzero().tolist()[:40]}")

    n = cfg.nexp
    NC = cfg.NCL * cfg.GC
    s = read_stamps(st, NC)
    t0 = int(s[:, 0].min())
    ends = ((s[:, 1 : 2 + n] - t0) & 0xFFFFFFFF).double()
    e_max = ends.max(dim=0).values
    first = 1 if n > 2 else 0
    steady = float((e_max[n - 1] - e_max[first]) / (n - 1 - first))
    tile_ops = cfg.R * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS)
    util_tile = tile_ops * 16 / (NC * steady)
    return dict(
        steady=steady,
        us=steady / clk,
        util_tile=util_tile,
        util_useful=util_tile * M / (cfg.R * 32),
        first=float(e_max[0]),
        pcc=pc,
        aiclk=clk,
        cores=NC,
        NS=p.NS,
        nblocks=[len(sched[k][0]) for k in sched],
        XST=build.last_xst,
        unpack=(s[:, 40:43].double().mean(dim=0) / (n - 2)).tolist(),
        unpack_max=(s[:, 40:43].double().max(dim=0).values / (n - 2)).tolist(),
    )


def report(tag, cfg, M, r):
    logger.info(
        f"WSX {tag} M {M} R {cfg.R} nexp {cfg.nexp} cores {r['cores']}: {r['steady']:.0f} cyc/expert "
        f"({r['us']:.2f} us), util tile {100 * r['util_tile']:.1f}% useful {100 * r['util_useful']:.1f}%, "
        f"first expert {r['first']:.0f} cyc; pcc {r['pcc']:.5f}; NS {r['NS']} XST {r['XST']}; x blocks/cluster {r['nblocks']}; "
        f"aiclk {r['aiclk']}"
    )


@pytest.mark.parametrize("M", [586])
def test_ws_x(device, M):
    R = (M + 31) // 32
    cfg = XCfg(R=R, nexp=16)
    r = run_x(device, cfg, M)
    report("base", cfg, M, r)
    logger.info(f"per expert per core (UNPACK): tilize / x wait / down mean {[round(v) for v in r['unpack']]} max {[round(v) for v in r['unpack_max']]}")
    assert r["pcc"] > 0.999


@pytest.mark.parametrize("BK,NX,x_dt", [(12, 20, "bf16"), (12, 15, "bf16"), (8, 24, "bf16"), (8, 20, "bf16"), (12, 20, "bfp8")])
def test_ws_x_variants(device, BK, NX, x_dt):
    cfg = XCfg(R=19, nexp=16, BK=BK, NX=NX, x_dt={"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[x_dt])
    r = run_x(device, cfg, 586)
    kb = NX * BK * {"bf16": 2048, "bfp8": 1088}[x_dt] // 1024
    report(f"BK {BK} NX {NX} x {x_dt} ring {kb} KB", cfg, 586, r)
    logger.info(f"per expert per core (UNPACK): tilize / x wait / down mean {[round(v) for v in r['unpack']]} max {[round(v) for v in r['unpack_max']]}")


def test_ws_x_nogather(device):
    cfg = XCfg(R=19, nexp=16, wdefs=("WX_NO_GATHER",))
    r = run_x(device, cfg, 586)
    report("no gather", cfg, 586, r)
    logger.info(f"per expert per core (UNPACK): tilize / x wait / down mean {[round(v) for v in r['unpack']]} max {[round(v) for v in r['unpack_max']]}")
