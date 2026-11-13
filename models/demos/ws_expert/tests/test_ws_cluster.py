# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert column-cluster scheme: every core runs gate/up and down; x and weights resident; NEXP experts back to
back. Reports per-expert steady state, utilization over the cores used (tile = padded rows, useful = real M), the
first expert's time and PCC vs torch."""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, read_stamps, to_dev
from models.demos.ws_expert.ws_cluster import ClCfg, build, effective_x, plan
from models.demos.ws_expert.ws_expert import interleave_gu


def run_cl(device, cfg: ClCfg, M, seed=0):
    p = plan(cfg)
    torch.manual_seed(seed)
    stores = torch.randn(cfg.NCL, p.XT, 32, 32)
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

    X = quantize(effective_x(p, quantize(stores.reshape(-1, 32), cfg.x_dt).reshape(stores.shape)), ttnn.bfloat16)
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
        badl = (bad > 0.05 * refm).nonzero().tolist()
        logger.warning(f"bad (row tile, core j): {badl[:40]} ({len(badl)}); rows owned per cluster {p.owned}")
    h_exact = torch.nn.functional.silu(X @ wgq) * (X @ wuq)
    rel = float((out[:M] - ref[:M]).norm() / ref[:M].norm())
    rel_h_only = float((quantize(h_exact, cfg.h_dt) @ wdq - h_exact @ wdq)[:M].norm() / (h_exact @ wdq)[:M].norm())

    n = cfg.nexp
    NC = cfg.NCL * cfg.GC
    s = read_stamps(st, NC)
    t0 = int(s[:, 0].min())
    ends = ((s[:, 1 : 2 + n] - t0) & 0xFFFFFFFF).double()
    e_max = ends.max(dim=0).values
    first = 1 if n > 2 else 0
    steady = float((e_max[n - 1] - e_max[first]) / (n - 1 - first))
    per_cl = ends.reshape(cfg.NCL, cfg.GC, -1).max(dim=1).values
    cl_steady = ((per_cl[:, n - 1] - per_cl[:, first]) / (n - 1 - first)).tolist()
    tile_ops = cfg.R * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS)
    util_tile = tile_ops * 16 / (NC * steady)
    return dict(
        steady=steady,
        us=steady / clk,
        util_tile=util_tile,
        util_useful=util_tile * M / (cfg.R * 32),
        first=float(e_max[0]),
        total=float(e_max[n]),
        pcc=pc,
        rel=rel,
        rel_h_rounding=rel_h_only,
        aiclk=clk,
        cores=NC,
        cost=p.cost,
        owned=[len(o) for o in p.owned],
        XT=p.XT,
        cl_steady=cl_steady,
        math_split=(s[:, 40:43].double().mean(dim=0) / (n - 2)).tolist(),
        unpack_split=(s[:, 43:46].double().mean(dim=0) / (n - 2)).tolist(),
    )


def report(tag, cfg, M, r):
    logger.info(
        f"WSCL {tag} M {M} R {cfg.R} nexp {cfg.nexp} cores {r['cores']} ({cfg.NCL}x{cfg.GC}): {r['steady']:.0f} "
        f"cyc/expert ({r['us']:.2f} us), util tile {100 * r['util_tile']:.1f}% useful {100 * r['util_useful']:.1f}%, "
        f"first {r['first']:.0f} cyc, all {r['total']:.0f}; pcc {r['pcc']:.5f} rel err {r['rel']:.2e} "
        f"(bf16 h rounding alone {r['rel_h_rounding']:.2e}); aiclk {r['aiclk']}; owned {r['owned']}"
    )


@pytest.mark.parametrize("order,dn_lag", [("act_hidden", 3), ("eager", 1)])
@pytest.mark.parametrize("glu", [2, 3])
@pytest.mark.parametrize("M", [586])
def test_ws_cluster(device, M, glu, order, dn_lag):
    R = (M + 31) // 32
    cfg = ClCfg(R=R, nexp=16, glu=glu, order=order, dn_lag=dn_lag)
    r = run_cl(device, cfg, M)
    report(f"base glu {glu} {order} lag {dn_lag} NS {plan(cfg).NS}", cfg, M, r)
    logger.info(f"per cluster steady {[round(v) for v in r['cl_steady']]} cost {r['cost']}")
    logger.info(f"per expert per core: MATH gu/dn/pre {[round(v) for v in r['math_split']]} UNPACK {[round(v) for v in r['unpack_split']]}")
    assert r["pcc"] > 0.999


def test_ws_cluster_nosync(device):
    cfg = ClCfg(R=19, nexp=16, nosync=True)
    r = run_cl(device, cfg, 586)
    report("nosync", cfg, 586, r)
    logger.info(f"per cluster steady {[round(v) for v in r['cl_steady']]} cost {r['cost']}")


@pytest.mark.parametrize("glu", [2])
@pytest.mark.parametrize("R", [2])
def test_ws_cluster_single(device, R, glu):
    """One cluster, no h exchange: the compute kernel alone (each core: R rows of gate/up + down)."""
    cfg = ClCfg(R=R, nexp=16, NCL=1, nosync=True, glu=glu)
    r = run_cl(device, cfg, 32 * R)
    report(f"single glu {glu}", cfg, 32 * R, r)
    logger.info(f"per expert per core: MATH gu/dn/pre {[round(v) for v in r['math_split']]} UNPACK {[round(v) for v in r['unpack_split']]}")


@pytest.mark.parametrize("cdefs", [("WC_NO_ACT",)])
def test_ws_cluster_ceiling(device, cdefs):
    cfg = ClCfg(R=19, nexp=16, cdefs=cdefs)
    r = run_cl(device, cfg, 586)
    report(f"ceiling {cdefs}", cfg, 586, r)
    logger.info(f"per expert per core: MATH gu/dn/pre {[round(v) for v in r['math_split']]} UNPACK {[round(v) for v in r['unpack_split']]}")


@pytest.mark.parametrize(
    "M,dn_lag,group,h_dt",
    [
        (586, 1, 3, "bf16"),
        (586, 2, 3, "bf16"),
        (586, 3, 3, "bf16"),
        (586, 3, 3, "bfp8"),
        (128, 3, 3, "bf16"),
        (256, 3, 3, "bf16"),
        (384, 3, 3, "bf16"),
        (768, 3, 3, "bf16"),
        (1024, 3, 3, "bf16"),
        (1024, 1, 2, "bf16"),
        (2048, 1, 2, "bfp8"),
    ],
)
def test_ws_cluster_sweep(device, M, dn_lag, group, h_dt):
    R = (M + 31) // 32
    cfg = ClCfg(R=R, nexp=12, dn_lag=dn_lag, group=group, h_dt={"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[h_dt])
    r = run_cl(device, cfg, M)
    p = plan(cfg)
    report(f"sweep lag {dn_lag} group {group} h {h_dt} NS {p.NS} XT {p.XT}", cfg, M, r)
