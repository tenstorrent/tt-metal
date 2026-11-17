# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""K-split gate/up groups on one chip (ws_ks): same harness as test_ws_group (injector-landed weights, x gathered from
DRAM, each group's last expert checked against torch), against the 1 x 20 groups."""

import pytest
from loguru import logger

import ttnn
from models.demos.ws_expert.tests.test_ws_group import run_group
from models.demos.ws_expert.ws_ks import KCfg, build


def report(tag, cfg, Ms, r):
    s = run_group.last_stamps[0]
    n = r["cores"]
    w = s[:n, 40:45].double().mean(dim=0).tolist()
    br = s[:n, 50:56].double().mean(dim=0).tolist()
    nc = s[:n, 470:478].double().mean(dim=0).tolist()
    logger.info(
        f"WSK {tag} M {Ms[0]} nexp {cfg.nexp} G {cfg.G}: {r['per_expert']:.0f} cyc/expert, util tile "
        f"{100 * r['util_tile']:.1f}% (whole run {100 * r['util_all']:.1f}%), pcc {[round(p, 5) for p in r['pcs']]}"
    )
    m = s[:n, 45:49].double().mean(dim=0).tolist()
    logger.info(f"WSK MATH time a core (K cyc) gate/up / fold / down / tilize {[round(v / 1e3) for v in m]}")
    logger.info(f"WSK waits a core (K cyc) tilize / x / weights / h / landing {[round(v / 1e3) for v in w]}")
    logger.info(f"WSK BRISC busy (K) qx / gather / partials / credits / h out / h lead {[round(v / 1e3) for v in br]}")
    logger.info(
        f"WSK NCRISC busy (K) weights / qa / x ring / landing / partials up / h ring / h lead / y "
        f"{[round(v / 1e3) for v in nc]}"
    )


@pytest.mark.parametrize("xalt", [True, False], ids=["xalt", "xhalf"])
@pytest.mark.parametrize("M", [64, 586, 2048])
@pytest.mark.parametrize("G", [1, 4])
def test_ws_ks(device, G, M, xalt):
    cfg = KCfg(nexp=8 * G, G=G, xalt=xalt)
    Ms = [M] * cfg.nexp
    r = run_group(device, cfg, Ms, builder=build)
    report(f"xalt {int(xalt)}", cfg, Ms, r)
    assert r["pcc"] > 0.999


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(NXR=8),
        dict(NXR=8, TL=3),
        dict(D=4, FL=2),
        dict(fold2=False, D=3),
        dict(dn_lag=2),
        dict(NS=6),
        dict(NXR=8, TL=3, dn_lag=2, fold2=False, D=3),
        dict(NXR=8, TL=3, dn_lag=2, h_dt=ttnn.bfloat8_b),
        dict(spare_slot=True, h_dt=ttnn.bfloat8_b, XRM=2, fold2=False, D=3, NXR=8, TL=3, dn_lag=2),
        dict(fold2=False),
        dict(NXR=8, TL=3, dn_lag=2),
    ],
    ids=lambda d: ",".join(f"{k}{'bfp8' if v is ttnn.bfloat8_b else v}" for k, v in d.items()) or "default",
)
@pytest.mark.parametrize("M", [586, 2048])
def test_ws_ks_sweep(device, M, kw):
    cfg = KCfg(nexp=32, G=4, **kw)
    Ms = [M] * cfg.nexp
    r = run_group(device, cfg, Ms, builder=build)
    report(f"sweep {kw}", cfg, Ms, r)
    assert r["pcc"] > 0.999


LANE_LIKE = dict(spare_slot=True, h_dt=ttnn.bfloat8_b, XRM=2, fold2=False, D=3, NXR=8, TL=3, dn_lag=2)


@pytest.mark.parametrize("seed", [0, 1])
def test_ws_ks_dynm(device, seed):
    """Uneven M per expert, including 0 and 1-row experts (each group's last expert checked)."""
    import random

    rnd = random.Random(seed)
    cfg = KCfg(nexp=32, G=4, **LANE_LIKE)
    Ms = [rnd.choice([0, 1, 5, 31, 33, 64, 300, 586, 1100, 2048]) for _ in range(cfg.nexp)]
    Ms[-4:] = [rnd.choice([1, 33, 586, 2048]) for _ in range(4)]  # (the checked last experts: non-empty)
    r = run_group(device, cfg, Ms, builder=build)
    report(f"dynm seed {seed}", cfg, Ms, r)
    assert r["pcc"] > 0.999
