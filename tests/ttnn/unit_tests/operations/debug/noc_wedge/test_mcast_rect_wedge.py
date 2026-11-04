# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Blackhole NoC wedge: a multicast into a rectangle of cores + unicast writes into (or out of) that rectangle +
Tensix <-> DRAM traffic, all on ONE NoC. Traffic only.

Pattern (geometry of the GLM-5.3-Flash flat routed expert plan, logical worker coordinates; the left half is the same
on 11x10 P150 and 12x10 Galaxy grids, right-hand cells off the grid are dropped and logged):
  mcast      relay cores (1,4) and (8,3) multicast chains of 4 x 8704 B pieces (linked inside a chain) into their
             rectangles (2..5, 0..9) = 40 cores and (9..11, 0..7), each chain followed by a 4 B semaphore-set multicast
             (the x relay of the expert's gate/up cores)
  in_writes  8 cells next to the DRAM columns ((0|7, 0|2|5|8)) write 9216 B blocks round-robin to 4 rectangle cores
             each, an atomic increment after every block, paced to ~28 GB/s (riscv_wait 450 cycles a block) - the
             DRAM-core (DRISC) weight senders of the op, emulated from the nearest worker cells
  out_writes every rectangle core writes 2048 B to one of 7 "down chain head" cores outside it, an atomic increment to
             its relay every 4th write (the gate/up cores' h writes in the MiMo variant of the wedge)
  dram       the 32 "down" cores read 8 KB DRAM pages over the banks (8 in flight), or write them (dram="writes")
Every class has its own NoC; a core runs at most one role per NoC (NOC0 on BRISC, NOC1 on NCRISC), so no RISC pair
ever shares a NoC (no dynamic-NoC mode anywhere). Nothing waits on anything but its own barriers: a dispatch timeout
is a NoC that stopped making progress.

What a hang looks like (as in the op): a relay core's multicast RISC spinning in noc_cmd_buf_ready (or its write
barrier), its NoC's command buffer 0 still active; any chip, at a random launch.

History / rates (op and earlier probes, bh-lb-09, 12x10 Galaxy-like firmware, 2x4 LoudBox, all 8 chips):
  - MiMo-V2 flat expert (2026-10-02): relay multicast + gate/up h writes + DRAM weight reads on NOC0 wedged ~1 in 1e4
    .. 2e5 launches; moving the h writes to NOC1 fixed it (moving any one of the three classes off the NoC stopped
    it). This is the mimo_* cases.
  - GLM-5.3-Flash flat expert, DRISC-only (2026-10-18/19): relay multicast + DRISC weight writes into the rectangle +
    down-weight reads or y writes, all on NOC0, wedged the left relay (1,4) about 1 in 450K op calls (amplified ~20 /
    M). The traffic-only probe (paced) wedged within 1000-4500 launches of 1500 chains on 8 chips; dropping the DRISC
    writes or the DRAM traffic, moving the DRISC writes to NOC1, the multicast to NOC1, or replacing the multicast by
    unicast chains: 2 x 20K launches clean. Unlinked or posted multicasts, other write VCs: still wedged. This is the
    glm_* cases. (That probe ran some cores' two RISCs on one NoC in dynamic-NoC mode - the in-writer cells' DRAM
    reads; here those cells only write, so a few DRAM readers fewer.)

Cases: `expect` is what the earlier probes saw; launches are per case (--noc-wedge-launches overrides). Run each with
scripts/run_safe_pytest.sh, one case at a time, and stop at the first hang (see the run plan in the commit message).
"""

import dataclasses

import pytest

import ttnn

from tests.ttnn.unit_tests.operations.debug.noc_wedge import noc_wedge_common as nw

LEFT_RELAY, LEFT_RECT = (1, 4), ((2, 0), (5, 9))
RIGHT_RELAY, RIGHT_RECT = (8, 3), ((9, 0), (11, 7))
# DRAM-core weight senders on NOC0 (slots 8-15 of the GLM-Flash DRISC plan), from the worker cell nearest each
# sender's DRAM tile -> their 4 receivers
IN_WRITERS = {
    (0, 0): [(2, 0), (3, 0), (2, 2), (3, 2)],
    (0, 8): [(2, 1), (3, 1), (4, 1), (2, 3)],
    (0, 2): [(2, 6), (3, 6), (2, 7), (2, 8)],
    (0, 5): [(2, 4), (3, 4), (2, 5), (4, 4)],
    (7, 0): [(9, 0), (9, 1), (10, 2), (3, 3)],
    (7, 8): [(10, 0), (10, 1), (11, 2), (3, 5)],
    (7, 2): [(9, 7), (9, 2), (3, 7), (2, 9)],
    (7, 5): [(9, 3), (10, 3), (9, 4), (4, 5)],
}
DOWN = [
    (6, 0), (7, 0), (8, 0), (0, 1), (1, 1), (6, 1), (0, 2), (1, 2), (6, 2), (7, 2), (8, 2), (6, 3), (7, 3), (0, 4),
    (6, 4), (0, 5), (6, 5), (7, 5), (8, 5), (0, 6), (1, 6), (6, 6), (6, 7), (7, 7), (8, 7), (0, 8), (1, 8), (6, 8),
    (7, 8), (8, 8), (9, 8), (10, 8),
]  # fmt: skip
READERS = [
    (0, 9), (0, 0), (0, 7), (0, 3), (7, 9), (7, 1), (7, 6), (7, 4), (1, 9), (1, 0), (1, 7), (1, 3), (8, 9), (8, 1),
    (8, 6), (8, 4),
]  # fmt: skip
CHAIN_HEADS = [(7, 0), (0, 2), (6, 2), (6, 6), (7, 7), (0, 8), (10, 8)]


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    expect: str  # what the earlier traffic-only probe / op saw: "hang" or "clean"
    launches: int = 20000
    iters: int = 1500  # multicast chains per launch; the other classes scale with it
    mc_noc: int = 0
    mc_mode: str = "mcast"  # "mcast" | "unicast_chain" (x down each rectangle column as unicast writes, no multicast)
    mc_linked: bool = True
    chain: int = 4
    piece: int = 8704
    right_rect: bool = True
    in_writes: bool = True
    in_noc: int = 0
    in_pace: int = 450  # riscv_wait cycles after each 9216 B block (~28 GB/s a sender); 0 = unpaced
    in_vc: int = 3
    out_writes: bool = False
    out_noc: int = 0
    dram: str = "reads"  # "reads" | "writes" | "none"
    dram_noc: int = 0
    dram_cores: str = "down"  # "down" | "down+readers"


CASES = [
    # GLM-5.3-Flash flat expert (DRISC-only plan): the three NOC0 classes - wedged within 1-5K launches
    Case("glm_noc0", "hang"),
    Case("glm_noc0_dram_writes", "hang", dram="writes"),
    Case("glm_noc0_unlinked", "hang", mc_linked=False),
    # one class removed or moved: clean in 2 x 20K launches
    Case("glm_no_in_writes", "clean", in_writes=False),
    Case("glm_no_dram", "clean", dram="none"),
    Case("glm_in_writes_noc1", "clean", in_noc=1),
    Case("glm_mcast_noc1", "clean", mc_noc=1),
    Case("glm_unicast_chain", "clean", mc_mode="unicast_chain"),
    # MiMo-V2 variant: the rectangle's own writes OUT of it (+ atomics to the relay) instead of writes into it
    Case("mimo_noc0", "hang", in_writes=False, out_writes=True, dram_cores="down+readers", launches=50000),
    Case("mimo_out_noc1", "clean", in_writes=False, out_writes=True, out_noc=1, dram_cores="down+readers"),
]


def _rect_cores(rect, grid):
    (x0, y0), (x1, y1) = rect
    return [(x, y) for x in range(x0, min(x1, grid.x - 1) + 1) for y in range(y0, min(y1, grid.y - 1) + 1)]


def build(mesh, s, cfg):
    grid = mesh.compute_with_storage_grid_size()
    plan = nw.TrafficPlan(grid)
    C = ttnn.CoreCoord
    xy = lambda c: nw.packed_xy(mesh, C(*c))  # noqa: E731
    it = cfg.iters
    relays = [(LEFT_RELAY, LEFT_RECT)] + ([(RIGHT_RELAY, RIGHT_RECT)] if cfg.right_rect else [])
    relays = [(r, ((x0, y0), (min(x1, grid.x - 1), min(y1, grid.y - 1)))) for r, ((x0, y0), (x1, y1)) in relays]

    # the multicast (or its unicast-chain control) first: it is the class under study
    for relay, rect in relays:
        cores = _rect_cores(rect, grid)
        if cfg.mc_mode == "mcast":
            ct, rt = nw.mcast(
                s,
                it,
                nw.mcast_corners(mesh, rect[0], rect[1], cfg.mc_noc),
                len(cores),
                cfg.piece,
                cfg.chain,
                linked=cfg.mc_linked,
            )
            plan.add("mcast", nw.TRAFFIC_KERNEL, ct, [(relay, rt)], cfg.mc_noc)
        else:
            (x0, y0), (x1, y1) = rect
            heads = [xy((x, y0)) for x in range(x0, x1 + 1)]
            ct, rt = nw.unicast(s, it * cfg.chain, heads, cfg.piece, atomic_every=1)
            plan.add("x_chain_relay", nw.TRAFFIC_KERNEL, ct, [(relay, rt)], cfg.mc_noc)
            fwd = []
            for x, y in cores:
                if y < y1:
                    ct, rt = nw.unicast(s, it * cfg.chain, [xy((x, y + 1))], cfg.piece, atomic_every=1)
                    fwd.append(((x, y), rt))
            plan.add("x_chain_fwd", nw.TRAFFIC_KERNEL, ct, fwd, cfg.mc_noc)
    if cfg.in_writes:
        for cell, recv in IN_WRITERS.items():
            targets = [xy(c) for c in recv if plan.in_grid(c)]
            ct, rt = nw.unicast(s, it * 6, targets, 9216, atomic_every=1, pace=cfg.in_pace, vc=cfg.in_vc)
            plan.add("in_writes", nw.TRAFFIC_KERNEL, ct, [(cell, rt)], cfg.in_noc)
    if cfg.out_writes:
        for relay, rect in relays:
            per = []
            for i, c in enumerate(_rect_cores(rect, grid)):
                heads = [h for h in CHAIN_HEADS if plan.in_grid(h)]
                ct, rt = nw.unicast(s, it * 4, [xy(heads[i % len(heads)])], 2048, atomic_every=4, atomic_xy=xy(relay))
                per.append((c, rt))
            plan.add("out_writes", nw.TRAFFIC_KERNEL, ct, per, cfg.out_noc)
    if cfg.dram != "none":
        cores = list(DOWN) + (list(READERS) if cfg.dram_cores == "down+readers" else [])
        per = []
        for i, c in enumerate(dict.fromkeys(cores)):
            if cfg.dram == "reads":
                ct, rt = nw.dram_read(s, it * 4, 8192, i, in_flight=8)
            else:
                ct, rt = nw.dram_write(s, it * 2 // 8, 8, 8192, i, mode="acked", index=i)
            per.append((c, rt))
        plan.add(f"dram_{cfg.dram}", nw.TRAFFIC_KERNEL, ct, per, cfg.dram_noc)
    return plan


@pytest.mark.timeout(86400)
@pytest.mark.parametrize("cfg", CASES, ids=[c.name for c in CASES])
def test_mcast_rect_wedge(mesh_device, cfg, request):
    launches = nw.launches_option(request, cfg.launches)
    s = nw.alloc_scratch(mesh_device)
    plan = build(mesh_device, s, cfg)
    nw.log_config(f"mcast_rect {cfg.name} (expect {cfg.expect})", cfg, plan)
    desc = nw.replicated(mesh_device, plan.program())
    nw.run_launches(
        mesh_device,
        s.tensors,
        desc,
        launches,
        nw.sync_option(request),
        f"mcast_rect {cfg.name}",
        exit_on_hang=nw.exit_on_hang_option(request),
    )
