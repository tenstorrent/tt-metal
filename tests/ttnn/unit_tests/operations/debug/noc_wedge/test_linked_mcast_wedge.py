# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Blackhole NoC hang: a linked multicast chain left open while the same NIU issues a transaction from another
command buffer (a DRAM read, or an atomic increment). Traffic only, one RISC.

Pattern: one core (logical (1, 4), RISC owning `noc`) multicasts chains of `chain` x 8704 B pieces into the 4 x 10
rectangle (2..5, 0..9), every piece but the chain's last linked (the path stays reserved between them); after piece
`mid_at` it issues one transaction from another command buffer of the same NoC:
  dram_read   a 2 KB DRAM read (read command buffer) inside the open chain   - the hazard
  atomic      an atomic increment (atomic command buffer) inside the chain     - the hazard, other command buffer
  read_after  the same DRAM read issued only after the chain closed           - control
  none        no other transaction                                            - control
`linked` False makes every piece unlinked (control). Optional DRAM readers on the other NoC can load the chip.

What a hang looks like: the multicasting RISC stuck in noc_cmd_buf_ready / its write or read barrier on the first
launches; no other core involved.

History: the flat routed expert's x relay (2026-10-17, V4.1 E 24) hung whenever its x DRAM reads moved to the NoC of
its linked multicast (MIMO_FL_XRD_NOC=0), including in dynamic-NoC mode (reads from another RISC's command buffer
on the same NoC); with every piece unlinked (MIMO_FL_XMC_UNLINKED=1) it ran. (Its dedicated-NoC form - the relay's
two RISCs both on NOC0 - is NOT a hardware issue: both RISCs then share command buffers 0-3; never do that.) The
MiMo noc_hang_probe had the atomic variant (linked_atomic_in_chain); its result is not recorded. Expected here: the
dram_read / atomic cases hang on the first launches; the controls run clean.
"""

import dataclasses

import pytest

import ttnn

from tests.ttnn.unit_tests.operations.debug.noc_wedge import noc_wedge_common as nw

MCASTER, RECT = (1, 4), ((2, 0), (5, 9))


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    expect: str
    mid: str = "dram_read"
    linked: bool = True
    noc: int = 0
    launches: int = 1000
    iters: int = 2000
    chain: int = 4
    mid_at: int = 1
    load_other_noc: bool = False  # DRAM readers on every other core, on the other NoC


CASES = [
    Case("linked_read_in_chain", "hang"),
    Case("linked_atomic_in_chain", "hang", mid="atomic"),
    Case("linked_read_in_chain_noc1", "hang", noc=1),
    Case("linked_read_after_chain", "clean", mid="read_after"),
    Case("unlinked_read_in_chain", "clean", linked=False),
    Case("linked_no_mid", "clean", mid="none"),
    Case("linked_read_in_chain_loaded", "hang", load_other_noc=True),
]


def build(mesh, s, cfg):
    grid = mesh.compute_with_storage_grid_size()
    plan = nw.TrafficPlan(grid)
    (x0, y0), (x1, y1) = RECT
    rect_hi = (x1, min(y1, grid.y - 1))
    ndests = (x1 - x0 + 1) * (rect_hi[1] - y0 + 1)
    ct, rt = nw.mcast(
        s,
        cfg.iters,
        nw.mcast_corners(mesh, RECT[0], rect_hi, cfg.noc),
        ndests,
        8704,
        cfg.chain,
        linked=cfg.linked,
        sem_mcast=False,
        mid=cfg.mid,
        mid_at=cfg.mid_at,
        atomic_xy=nw.packed_xy(mesh, ttnn.CoreCoord(*RECT[0])),
    )
    plan.add("mcast", nw.TRAFFIC_KERNEL, ct, [(MCASTER, rt)], cfg.noc)
    if cfg.load_other_noc:
        other = 1 - cfg.noc
        per = []
        for i, c in enumerate((x, y) for y in range(grid.y) for x in range(grid.x)):
            ct, rt = nw.dram_read(s, cfg.iters * 4, 8192, i, in_flight=4)
            per.append((c, rt))
        plan.add("dram_reads", nw.TRAFFIC_KERNEL, ct, per, other)
    return plan


@pytest.mark.timeout(86400)
@pytest.mark.parametrize("cfg", CASES, ids=[c.name for c in CASES])
def test_linked_mcast_wedge(mesh_device, cfg, request):
    launches = nw.launches_option(request, cfg.launches)
    s = nw.alloc_scratch(mesh_device)
    plan = build(mesh_device, s, cfg)
    nw.log_config(f"linked_mcast {cfg.name} (expect {cfg.expect})", cfg, plan)
    desc = nw.replicated(mesh_device, plan.program())
    nw.run_launches(
        mesh_device,
        s.tensors,
        desc,
        launches,
        nw.sync_option(request),
        f"linked_mcast {cfg.name}",
        exit_on_hang=nw.exit_on_hang_option(request),
    )
