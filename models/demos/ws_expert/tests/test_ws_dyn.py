# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert with dynamic M: a stream of experts with different row counts, every schedule computed on device.
Checks the last expert against torch; utilization over experts 1 .. n-1 (tile ops of those experts / time)."""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize, read_stamps, to_dev
from models.demos.ws_expert.ws_dyn import DCfg, build, check_plans, xst_blocks
from models.demos.ws_expert.ws_expert import interleave_gu


def run_dyn(device, cfg: DCfg, Ms, seed=0):
    assert len(Ms) == cfg.nexp
    for M in set(Ms):
        check_plans(cfg, M)
    torch.manual_seed(seed)
    x = torch.randn(cfg.T + 1, cfg.H)
    x[cfg.T] = 0
    idx = torch.full((cfg.nexp, cfg.RMAX * 32), cfg.T, dtype=torch.int32)
    for e, M in enumerate(Ms):
        idx[e, :M] = torch.randperm(cfg.T)[:M].to(torch.int32)
    counts = torch.tensor(Ms, dtype=torch.int32).reshape(1, -1)
    wg = torch.randn(cfg.H, cfg.I) / cfg.H**0.5
    wu = torch.randn(cfg.H, cfg.I) / cfg.H**0.5
    wd = torch.randn(cfg.I, cfg.H) / cfg.I**0.5
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    idx_t = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    counts_t = ttnn.from_torch(counts, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    wgu = interleave_gu(wg, wu)
    w_t = to_dev(wgu, cfg.w_dt, device)
    wd_t = to_dev(wd, cfg.w_dt, device)
    out_t = to_dev(torch.zeros(cfg.RMAX * 32, cfg.H), cfg.out_dt, device)
    XST = xst_blocks(cfg)
    prog, io, st = build(device, cfg, x_t, idx_t, counts_t, w_t, wd_t, out_t, XST)
    ttnn.generic_op(io, prog)
    ttnn.synchronize_device(device)
    clk = aiclk_mhz(0)

    M = Ms[-1]
    X = x[idx[-1].long()].to(torch.bfloat16).float()
    wguq = quantize(wgu, cfg.w_dt)
    wgq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 0].reshape(cfg.H, cfg.I)
    wuq = wguq.reshape(cfg.H, -1, 2, 32)[:, :, 1].reshape(cfg.H, cfg.I)
    wdq = quantize(wd, cfg.w_dt)
    h = quantize(torch.nn.functional.silu(X @ wgq) * (X @ wuq), cfg.h_dt)
    ref = h @ wdq
    out = ttnn.to_torch(out_t).float()
    pc = pcc(out[:M], ref[:M])

    n = cfg.nexp
    NC = cfg.NCL * cfg.GC
    s = read_stamps(st, NC)
    t0 = int(s[:, 0].min())
    ends = ((s[:, 1 : 2 + n] - t0) & 0xFFFFFFFF).double()
    e_max = ends.max(dim=0).values
    R = [(m + 31) // 32 for m in Ms]
    tile_ops = [r * (2 * cfg.PAIRS * cfg.KT + cfg.HT * cfg.PAIRS) for r in R]
    span = float(e_max[n - 1] - e_max[0])
    util_tile = sum(tile_ops[1:]) * 16 / (NC * span)
    util_useful = util_tile * sum(Ms[1:]) / sum(32 * r for r in R[1:])
    return dict(
        span=span,
        per_expert=span / (n - 1),
        util_tile=util_tile,
        util_useful=util_useful,
        pcc=pc,
        aiclk=clk,
        XST=XST,
        cores=NC,
        unpack=(s[:, 40:43].double().mean(dim=0) / (n - 2)).tolist(),
        unpack_max=(s[:, 40:43].double().max(dim=0).values / (n - 2)).tolist(),
        ends=[round(float(v)) for v in e_max[: n + 1]],
    )


def report(tag, cfg, Ms, r):
    logger.info(
        f"WSD {tag} M mean {sum(Ms) / len(Ms):.0f} (min {min(Ms)} max {max(Ms)}, last {Ms[-1]}) nexp {cfg.nexp} "
        f"cores {r['cores']}: {r['per_expert']:.0f} cyc/expert avg ({r['per_expert'] / r['aiclk']:.2f} us), util "
        f"tile {100 * r['util_tile']:.1f}% useful {100 * r['util_useful']:.1f}%; pcc {r['pcc']:.5f}; XST {r['XST']}"
    )


def poisson_ms(n, mean, seed):
    g = torch.Generator().manual_seed(seed)
    return [max(1, int(v)) for v in torch.poisson(torch.full((n,), float(mean)), generator=g).tolist()]


@pytest.mark.parametrize("case", ["const586", "poisson586", "mixed"])
def test_ws_dyn(device, case):
    cfg = DCfg(nexp=16)
    if case == "const586":
        Ms = [586] * 16
    elif case == "poisson586":
        Ms = poisson_ms(16, 586, 1)
    else:
        Ms = [586, 40, 1000, 300, 777, 5, 586, 128, 900, 64, 512, 33, 700, 256, 999, 431]
    r = run_dyn(device, cfg, Ms)
    report(case, cfg, Ms, r)
    logger.info(f"UNPACK per expert per core: tilize / x wait / down mean {[round(v) for v in r['unpack']]} max {[round(v) for v in r['unpack_max']]}")
    logger.info(f"expert ends {r['ends']}")
    assert r["pcc"] > 0.999
