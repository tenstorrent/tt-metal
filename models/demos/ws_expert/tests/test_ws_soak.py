# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Soaks that isolate the group-multicast NoC wedge: launches until WS_SOAK_SECS (default 180) or WS_SOAK_MAX, util of
every 50th launch logged, PCC on the last. test_group_soak: the single-chip group program (ws_group, weights landed by
on-chip injectors from their L1 with NoC atomic arrival increments; no fabric) at the load of the 640-expert torus
case: 160 experts a group, T 31250, M 586, G 4, the token-row ring of the lane design (BK 20, NX 9, LOOK 18)."""

import os
import statistics
import time

import pytest
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import pcc
from models.demos.ws_expert.tests.test_ws_slot3 import inputs, measure
from models.demos.ws_expert.ws_group import GCfg, build


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_group_soak(mesh_device):
    device = mesh_device
    T = 31250
    cfg = GCfg(nexp=640, G=4, T=T, MMAX=2048, x_dt=ttnn.bfloat8_b, BK=20, NX=9, LOOK=18, NINJ=20, wsrc="l1", P=2)
    Ms = [586] * cfg.nexp
    dev, refs = inputs(device, cfg, Ms)
    secs, mx = float(os.environ.get("WS_SOAK_SECS", "180")), int(os.environ.get("WS_SOAK_MAX", "100000"))
    logger.info(f"WSG soak: single chip, {cfg.nexp // cfg.G} experts a group, G {cfg.G}, {secs:.0f} s or {mx} launches")
    t0, i, utils = time.time(), 0, []
    while i < mx and time.time() - t0 < secs:
        args = [dev[k] for k in ("x_t", "idx_t", "counts_t", "wgu_t", "wd_t", "out_t")]
        prog, io, st, ex = build(device, cfg, *args)
        mpd = ttnn.MeshProgramDescriptor()
        mc = ttnn.MeshCoordinate(0, 0)
        mpd[ttnn.MeshCoordinateRange(mc, mc)] = prog
        ttnn.generic_op(io, mpd)
        ttnn.synchronize_device(device)
        i += 1
        if i % 50 == 0:
            u = 100 * measure(cfg, Ms, st)[0]
            utils.append(u)
            logger.info(f"WSG soak launch {i}, {time.time() - t0:.0f} s: util tile {u:.1f}%")
        del prog, mpd, io, st, ex
    out = ttnn.to_torch(ttnn.get_device_tensors(dev["out_t"])[0]).float()
    pcs = [pcc(out[g * cfg.RMAX * 32 : g * cfg.RMAX * 32 + Ms[e]], ref) for g, (e, ref) in enumerate(refs)]
    logger.info(
        f"WSG soak SUMMARY: {i} launches in {time.time() - t0:.0f} s without a hang, util tile median "
        f"{statistics.median(utils) if utils else 0:.1f}%, pcc {min(pcs):.5f}"
    )
    assert min(pcs) > 0.999
