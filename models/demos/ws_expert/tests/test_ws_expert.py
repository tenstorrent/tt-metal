# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert stage 1: gate/up -> SwiGLU -> down on one chip, x and weights resident in L1, NEXP experts back to back.

Reports per-expert steady-state time (from the down cores' per-expert completion stamps), utilization over the cores
used (tile FLOPs = padded rows, and useful FLOPs at the real M), the first expert's latency, and PCC vs torch.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, read_stamps, to_dev
from models.demos.ws_expert.ws_expert import WsCfg, build, effective_x_tiles, interleave_gu, plan


def run_ws(device, cfg: WsCfg, M=None, seed=0):
    M = M or cfg.R * 32
    p = plan(cfg)
    torch.manual_seed(seed)
    stores = torch.randn(cfg.GR, p.XT, 32, 32)
    wg = torch.randn(cfg.H, cfg.I) / cfg.H**0.5
    wu = torch.randn(cfg.H, cfg.I) / cfg.H**0.5
    wd = torch.randn(cfg.I, cfg.H) / cfg.I**0.5
    x_t = to_dev(stores.reshape(-1, 32), cfg.x_dt, device)
    wgu = interleave_gu(wg, wu)
    w_t = to_dev(wgu, cfg.w_dt, device)
    wd_t = to_dev(wd, cfg.w_dt, device)
    out_t = to_dev(torch.zeros(cfg.R * 32, cfg.H), cfg.out_dt, device)
    prog, io, st = build(device, cfg, p, x_t, w_t, wd_t, out_t)
    ttnn.generic_op(io, prog)
    ttnn.synchronize_device(device)
    clk = aiclk_mhz(0)

    # ---- correctness (last expert)
    X = quantize(effective_x_tiles(p, quantize(stores.reshape(-1, 32), cfg.x_dt).reshape(stores.shape)), ttnn.bfloat16)
    wguq = quantize(wgu, cfg.w_dt)
    wgq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
    wuq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
    wdq = quantize(wd, cfg.w_dt)
    h = torch.nn.functional.silu(X @ wgq) * (X @ wuq)
    h = quantize(h, cfg.h_dt)
    ref = h @ wdq
    out = ttnn.to_torch(out_t).float()
    pc = pcc(out[:M], ref[:M])
    if pc < 0.999:
        err = (out - ref).abs().reshape(cfg.R, 32, cfg.ND, cfg.CW * 32).amax(dim=(1, 3))
        refm = ref.abs().reshape(cfg.R, 32, cfg.ND, cfg.CW * 32).amax(dim=(1, 3))
        bad = (err > 0.05 * refm).nonzero().tolist()
        logger.warning(f"bad (row tile, down core): {bad[:60]} ({len(bad)} of {cfg.R * cfg.ND})")
        logger.warning(f"down order {p.order}")

    # ---- timing
    G = cfg.GR * cfg.GC
    s = read_stamps(st, G + cfg.ND)
    t0 = int(s[:, 0].min())
    dn = s[G:, 1 : 1 + cfg.nexp]
    gu = s[:G, 1 : 1 + cfg.nexp]
    if cfg.debug_no_send:
        dn = gu
    ends = ((dn - t0) & 0xFFFFFFFF).max(dim=0).values.double()
    gu_ends = ((gu - t0) & 0xFFFFFFFF).max(dim=0).values.double()
    n = cfg.nexp
    first = 1 if n > 2 else 0
    steady = float((ends[n - 1] - ends[first]) / (n - 1 - first))
    gu_steady = float((gu_ends[n - 1] - gu_ends[first]) / (n - 1 - first))
    cores = G + cfg.ND
    tile_ops = cfg.R * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS)
    util_tile = tile_ops * 16 / (cores * steady)
    util_useful = util_tile * M / (cfg.R * 32)
    res = dict(
        steady_cyc=steady,
        steady_us=steady / clk if clk > 0 else steady / 1350,
        gu_steady_cyc=gu_steady,
        first_cyc=float(ends[0]),
        util_tile=util_tile,
        util_useful=util_useful,
        util_110=util_useful * cores / 110,
        cores=cores,
        pcc=pc,
        aiclk=clk,
        XT=p.XT,
    )
    res["gu_per_core_e1"] = ((gu[:, 2] - gu[:, 1]) & 0xFFFFFFFF).tolist() if n > 2 else []
    res["dn_per_core_e1"] = ((s[G:, 3] - s[G:, 2]) & 0xFFFFFFFF).tolist() if n > 2 else []
    res["gu_prof"] = s[:G, 32:36].double().mean(dim=0).tolist()
    res["dn_prof"] = s[G:, 32:34].double().mean(dim=0).tolist()
    res["lead_prof"] = s[G, 32:34].tolist()
    res["start_skew"] = int(((s[:, 0] - t0) & 0xFFFFFFFF).max())
    return res


def report(tag, cfg, M, r):
    logger.info(
        f"WSEXP {tag} M {M} R {cfg.R} nexp {cfg.nexp} cores {r['cores']} ({cfg.GR}x{cfg.GC}+{cfg.ND}): "
        f"{r['steady_cyc']:.0f} cyc/expert ({r['steady_us']:.2f} us), gate/up {r['gu_steady_cyc']:.0f}; "
        f"util tile {100 * r['util_tile']:.1f}% useful {100 * r['util_useful']:.1f}% (as 110 grid "
        f"{100 * r['util_110']:.1f}%); first expert {r['first_cyc']:.0f} cyc; pcc {r['pcc']:.5f}; aiclk {r['aiclk']}"
    )


@pytest.mark.parametrize("M", [586])
def test_ws_expert(device, M):
    R = (M + 31) // 32
    cfg = WsCfg(R=R, nexp=16)
    r = run_ws(device, cfg, M)
    report("base", cfg, M, r)
    logger.info(f"gate/up e1 per core: {r['gu_per_core_e1'][:16]} ... start skew {r['start_skew']}")
    logger.info(f"down e1 per core: {r['dn_per_core_e1'][:12]}")
    logger.info(f"gate/up writer waits (mean cyc total): h {r['gu_prof'][0]:.0f} cons {r['gu_prof'][1]:.0f} barrier {r['gu_prof'][2]:.0f} partial {r['gu_prof'][3]:.0f}")
    logger.info(f"down reader waits (mean): reserve {r['dn_prof'][0]:.0f} flags {r['dn_prof'][1]:.0f}; leader {r['lead_prof']}")
    assert r["pcc"] > 0.99


def test_ws_expert_gu_only(device):
    cfg = WsCfg(R=19, nexp=16, debug_no_send=True)
    r = run_ws(device, cfg, 586)
    report("gu_only", cfg, 586, r)
    logger.info(f"gate/up e1 per core: {r['gu_per_core_e1'][:16]} ... start skew {r['start_skew']}")


@pytest.mark.parametrize("NS", [4, 16])
def test_ws_expert_ns(device, NS):
    cfg = WsCfg(R=19, nexp=8, NS=NS)
    r = run_ws(device, cfg, 586)
    report(f"NS{NS}", cfg, 586, r)
    logger.info(f"gate/up writer waits (mean cyc total): h {r['gu_prof'][0]:.0f} cons {r['gu_prof'][1]:.0f} barrier {r['gu_prof'][2]:.0f} partial {r['gu_prof'][3]:.0f}")
