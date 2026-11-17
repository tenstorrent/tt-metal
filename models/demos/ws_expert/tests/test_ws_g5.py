# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""5 x 5 K-split groups on one chip (ws_g5): the test_ws_group harness (injector-landed weights from P preloaded sets,
x gathered from DRAM, each group's last expert checked against torch). On an 11 x 10 grid only one 5-wide NoC-contiguous
column block exists, so G <= 2 (columns 0-4, rows 0-4 / 5-9)."""

import pytest
from loguru import logger

from models.demos.ws_expert.tests.test_ws_group import run_group
from models.demos.ws_expert.ws_g5 import G5Cfg, build


def report(tag, cfg, Ms, r):
    s = run_group.last_stamps[0]
    n = r["cores"]
    own = [i for i in range(n) if i % 25 % 5 < 4]
    r4 = [i for i in range(n) if i % 25 % 5 == 4]
    logger.info(
        f"WG5 {tag} M {Ms[0]} nexp {cfg.nexp} G {cfg.G}: {r['per_expert']:.0f} cyc/expert "
        f"({r['per_expert'] / r['aiclk']:.2f} us), util tile {100 * r['util_tile']:.1f}% (whole run "
        f"{100 * r['util_all']:.1f}%), pcc {[round(p, 5) for p in r['pcs']]}"
    )
    for name, rows in (("owners", own), ("row 4", r4)):
        m = s[rows, 45:49].double().mean(dim=0).tolist()
        w = s[rows, 40:45].double().mean(dim=0).tolist()
        logger.info(
            f"WG5 {name}: MATH (K) gate/up / fold / down / tilize {[round(v / 1e3) for v in m]}; waits (K) tilize / "
            f"x / weights / h / landing {[round(v / 1e3) for v in w]}"
        )
    br = s[:n, 50:56].double().mean(dim=0).tolist()
    nc = s[:n, 470:477].double().mean(dim=0).tolist()
    logger.info(f"WG5 BRISC busy (K) qx / gather / partials / credits / h out / h lead {[round(v / 1e3) for v in br]}")
    logger.info(
        f"WG5 NCRISC busy (K) weights / x ring / landing / partials up / h ring / h lead / y {[round(v / 1e3) for v in nc]}"
    )


@pytest.mark.parametrize("M", [5, 64, 586, 2048])
@pytest.mark.parametrize("G", [1, 2])
def test_ws_g5(device, G, M):
    cfg = G5Cfg(nexp=16 * G, G=G)
    Ms = [M] * cfg.nexp
    r = run_group(device, cfg, Ms, builder=build)
    report("base", cfg, Ms, r)
    assert r["pcc"] > 0.999


SWEEP = [
    dict(TL=9),
    dict(TL=9, D=4, FL=2),
    dict(TL=9, dn_lag=3, D=4, FL=2),
    dict(TL=12, D=4, FL=2),
    dict(TL=12, dn_lag=3, D=4, FL=2, NXR=10),
    dict(TL=9, dn_lag=4, NS=8, D=4, FL=2),
]


@pytest.mark.parametrize("kw", SWEEP, ids=lambda d: ",".join(f"{k}{v}" for k, v in d.items()) or "default")
@pytest.mark.parametrize("M", [586, 2048])
def test_ws_g5_sweep(device, M, kw):
    cfg = G5Cfg(nexp=64, G=2, **kw)
    Ms = [M] * cfg.nexp
    r = run_group(device, cfg, Ms, builder=build)
    report(f"sweep {kw}", cfg, Ms, r)
    assert r["pcc"] > 0.999


@pytest.mark.parametrize("M", [586, 2048])
def test_ws_ks_ref(device, M):
    """The 10 x 2 K-split prototype (ws_ks, its best single-chip config) at the same G 2 / 64 experts."""
    from models.demos.ws_expert.ws_ks import KCfg
    from models.demos.ws_expert.ws_ks import build as ks_build

    cfg = KCfg(nexp=64, G=2, NXR=8, TL=3, dn_lag=2, fold2=False, D=3)
    Ms = [M] * cfg.nexp
    r = run_group(device, cfg, Ms, builder=ks_build)
    logger.info(
        f"WG5 ks ref 10x2 M {M} G 2: {r['per_expert']:.0f} cyc/expert, util tile {100 * r['util_tile']:.1f}%, "
        f"pcc {[round(p, 5) for p in r['pcs']]}"
    )
    assert r["pcc"] > 0.999
