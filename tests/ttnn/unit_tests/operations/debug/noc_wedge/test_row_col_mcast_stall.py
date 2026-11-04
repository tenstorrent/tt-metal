# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Blackhole NoC slowdown (not a hang): a row multicast and a column multicast crossing it on the SAME NoC stall each
other for tens of microseconds; on different NoCs they do not. Traffic only.

Pattern: core (0, 4) multicasts 8 KB pieces (unlinked) along row 4 (x 1..last); core (5, 0) multicasts down column 5
(y 1..last); they cross at (5, 4). Cases: each alone, both on NOC0, both on NOC1, row on NOC0 + column on NOC1. The
test reports wall time per launch (host timed, many launches, each launch moves ~33 MB a sender) and the slowdown of
"both" against the slower of the two alone; expect >= ~1.5x on one NoC and ~1x split over two NoCs.

History: measured on bh-lb-09 (12x10, 2026-10-17, K3 block lin2d work): ~52 GB/s a sender down a column, ~44 along a
row; a row and a column multicast on one NoC stalled each other for tens of us (crossing at every core of the W
stream) - separate them in time or by NoC.
"""

import dataclasses

import pytest
from loguru import logger

import ttnn

from tests.ttnn.unit_tests.operations.debug.noc_wedge import noc_wedge_common as nw

ROW_SENDER, ROW = (0, 4), 4
COL_SENDER, COL = (5, 0), 5
PIECE, ITERS = 8192, 4000


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    row_noc: int = -1  # -1: off
    col_noc: int = -1


CASES = [
    Case("row_noc0", row_noc=0),
    Case("col_noc0", col_noc=0),
    Case("both_noc0", row_noc=0, col_noc=0),
    Case("row_noc1", row_noc=1),
    Case("col_noc1", col_noc=1),
    Case("both_noc1", row_noc=1, col_noc=1),
    Case("row_noc0_col_noc1", row_noc=0, col_noc=1),
]
_times = {}


def build(mesh, s, cfg):
    grid = mesh.compute_with_storage_grid_size()
    plan = nw.TrafficPlan(grid)
    if cfg.row_noc >= 0:
        lo, hi = (1, ROW), (grid.x - 1, ROW)
        ct, rt = nw.mcast(
            s, ITERS, nw.mcast_corners(mesh, lo, hi, cfg.row_noc), grid.x - 1, PIECE, 1, linked=False, sem_mcast=False
        )
        plan.add("row_mcast", nw.TRAFFIC_KERNEL, ct, [(ROW_SENDER, rt)], cfg.row_noc)
    if cfg.col_noc >= 0:
        lo, hi = (COL, 1), (COL, grid.y - 1)
        ct, rt = nw.mcast(
            s, ITERS, nw.mcast_corners(mesh, lo, hi, cfg.col_noc), grid.y - 1, PIECE, 1, linked=False, sem_mcast=False
        )
        plan.add("col_mcast", nw.TRAFFIC_KERNEL, ct, [(COL_SENDER, rt)], cfg.col_noc)
    return plan


@pytest.mark.parametrize("cfg", CASES, ids=[c.name for c in CASES])
def test_row_col_mcast_stall(mesh_device, cfg, request):
    launches = nw.launches_option(request, 200)
    s = nw.alloc_scratch(mesh_device)
    plan = build(mesh_device, s, cfg)
    nw.log_config(f"row_col_mcast {cfg.name}", cfg, plan)
    desc = nw.replicated(mesh_device, plan.program())
    per_launch = nw.run_launches(
        mesh_device,
        s.tensors,
        desc,
        launches,
        nw.sync_option(request, 50),
        cfg.name,
        exit_on_hang=nw.exit_on_hang_option(request),
    )
    _times[cfg.name] = per_launch
    gbs = ITERS * PIECE / per_launch / 1e9
    logger.info(f"NOC_WEDGE row_col_mcast {cfg.name}: {per_launch * 1e6:.1f} us/launch, {gbs:.1f} GB/s a sender")
    for both, row, col in (
        ("both_noc0", "row_noc0", "col_noc0"),
        ("both_noc1", "row_noc1", "col_noc1"),
        ("row_noc0_col_noc1", "row_noc0", "col_noc1"),
    ):
        if cfg.name == both and row in _times and col in _times:
            alone = max(_times[row], _times[col])
            logger.info(f"NOC_WEDGE row_col_mcast {both}: {_times[both] / alone:.2f}x the slower sender alone")
