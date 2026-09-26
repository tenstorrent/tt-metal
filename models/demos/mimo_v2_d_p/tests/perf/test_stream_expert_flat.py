# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flat spatially pipelined streamed routed expert (SwiGLU), one Blackhole chip.

    y = (silu(x @ Wg) * (x @ Wu)) @ Wd        x [M, H], Wg / Wu [H, I], Wd [I, H], M a multiple of 128

  * 16 bank readers stream the gate/up weights (the original streamed-matmul recipe: BRISC reads its bank region on
    NOC0, NCRISC forwards on NOC1), each to 4 gate/up cores.
  * 64 gate/up cores, one gate/up tile-column pair each (N split), all in the reader-free columns (2-5 x rows 0-9 and
    8-10 x rows 0-7), weights resident per expert in a 2-expert ring; every 128-row sub-block reuses them.
  * 2 x relays, one per rectangle of gate/up cores, read x from DRAM themselves (xdl_selfread.cpp, NOC0) and multicast
    it into their rectangle as linked bursts on NOC1 (se8_xmc.cpp): linked multicast into a rectangle that holds no
    reader runs ~50 GB/s of unique x to every core (a relay that is also fed by the readers, unlinked multicasts or
    rectangles over the readers all measured far lower).
  * 28 down cores (the rest), 8 output tile columns each, own down weights from DRAM (se6_dw.cpp), h from the gate/up
    cores through 7 chains (se6_drecv.cpp), y to DRAM.
Weights bfp4 / bfp8 (MIMO_FL_WDTYPE), x and h bfp8, y bf16, LoFi. Tag ``streamfl_H{H}_M{M}_E{E}_w{dtype}``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mimo_v2_d_p.tests.perf.test_dram_read_fwd import noc_hops
from models.demos.mimo_v2_d_p.tests.perf.test_stream_expert_spatial import _bank_sharded
from models.demos.mimo_v2_d_p.tests.perf.test_stream_matmul import BF8_TILE, _crs

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


KDIR = "models/demos/mimo_v2_d_p/tests/perf/kernels/stream_mm"
MS = _env_list("MIMO_FL_M", "128,256,512", int)
EXPERTS = int(os.environ.get("MIMO_FL_EXPERTS", "4"))
ITERS = int(os.environ.get("MIMO_FL_ITERS", "3"))
H = int(os.environ.get("MIMO_FL_H", "7168"))
I = 2048
X_SLOTS = int(
    os.environ.get(
        "MIMO_FL_X_SLOTS",
        "24" if int(os.environ.get("MIMO_FL_E2E", "0")) or int(os.environ.get("MIMO_FL_DYN", "0")) else "16",
    )
)  # e2e: deeper
RELAY_CB = int(os.environ.get("MIMO_FL_RELAY_CB", "16"))  # x blocks buffered on a relay
D_CHAINS = int(os.environ.get("MIMO_FL_DOWN_CHAINS", "7"))
H_PIECES = int(os.environ.get("MIMO_FL_H_PIECES", "32"))
HBUF = int(os.environ.get("MIMO_FL_HBUF", "3"))
DRING = float(os.environ.get("MIMO_FL_DRING", "1.5"))
READ_BATCH = int(os.environ.get("MIMO_FL_READ_BATCH", "1"))
X_BATCH = int(os.environ.get("MIMO_FL_X_BATCH", "4"))  # x blocks per relay read barrier
STATS_PATH = Path(os.environ.get("MIMO_SE_STATS", "generated/mimo_stream_expert/cases.jsonl"))
KBLK, MT_MAX, ND, R = 8, 4, 28, 4  # R gate/up cores per reader; row tiles per sub-block up to MT_MAX
W_DTYPES = {"bf8": (ttnn.bfloat8_b, BF8_TILE), "bf4": (ttnn.bfloat4_b, 576)}
WDTYPES = _env_list("MIMO_FL_WDTYPE", "bf4")
H_TILE = BF8_TILE
DN_NOC = int(os.environ.get("MIMO_FL_DN_NOC", "1"))  # NoC of the down cores' h chain / y writes (weights: the other)
FWD_DEPTH = int(os.environ.get("MIMO_FL_FWD_DEPTH", "0"))  # > 0: pipelined forwarder (se10_fwd.cpp), chunks in flight
RD_SLOTS = int(os.environ.get("MIMO_FL_RD_SLOTS", "0"))  # reader CB slots (0: 2 x READ_BATCH)
DYN = int(
    os.environ.get("MIMO_FL_DYN", "0")
)  # dynamic token counts read on device (implies E2E); MIMO_FL_COUNTS per expert
E2E = (
    int(os.environ.get("MIMO_FL_E2E", "0")) or DYN
)  # row-major bf16 dispatch buffer in, bfp8 tile buffer out (model format)
RM_CHUNKS = int(os.environ.get("MIMO_FL_RM_CHUNKS", "4"))  # e2e relay: row-major chunks (32 rows x 2 KB) buffered
XRD_BATCH = int(os.environ.get("MIMO_FL_XRD_BATCH", "2"))  # e2e relay: chunks per read barrier
SB_SLOTS = int(os.environ.get("MIMO_FL_SB_SLOTS", "3"))  # e2e relay: tilized super-blocks buffered
RDOWN_ENV = os.environ.get(
    "MIMO_FL_RDOWN", "auto"
)  # one reader per down chain also computes down columns (chain tail); auto: M >= 256
RDOWN_PCD_ENV = os.environ.get("MIMO_FL_RDOWN_PCD")  # down columns of each such reader (default 4, 6 with 4 relays)
X2_ENV = os.environ.get("MIMO_FL_X2", "auto")
XCOL_ENV = os.environ.get(
    "MIMO_FL_XCOL", "auto"
)  # e2e: n relays in column 1, each tilizing 1/n of x for both rectangles  # two x relays per rectangle (auto: e2e with reader-down)
FWD_DIR = int(os.environ.get("MIMO_FL_FWD_DIR", "0"))  # per-reader forwarding NoC by direction (else all NOC1)
XNOC = int(os.environ.get("MIMO_FL_XNOC", "0"))  # NoC of the x relays' multicasts (their DRAM reads use the other)


def _chains(cores, n, phys, noc):
    """Greedy nearest (by `noc` hops) path through `cores`, cut into n chains."""
    left = list(cores)
    path = [min(left, key=lambda c: (c[1].y, c[1].x) if noc == 0 else (-c[1].y, -c[1].x))]
    left.remove(path[0])
    while left:
        nxt = min(left, key=lambda c: noc_hops(phys(path[-1][1]), phys(c[1]), noc))
        path.append(nxt)
        left.remove(nxt)
    L = len(path) // n
    return [path[i * L : (i + 1) * L] if i < n - 1 else path[i * L :] for i in range(n)]


def _layout(device, x2=False, xcol=0):
    grid = device.compute_with_storage_grid_size()
    phys = lambda c: device.worker_core_from_logical_core(c)
    opt = list(device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))
    readers = opt + [ttnn.CoreCoord(c.x + 1, c.y) for c in opt]
    rcols = sorted({c.x for c in readers})
    assert rcols == [0, 1, 6, 7] and (grid.x, grid.y) == (11, 10), (rcols, grid)
    rects = [(2, 5, 0, 9), (8, 10, 0, 7)]  # gate/up rectangles (x0, x1, y0, y1): no reader inside
    gu = [ttnn.CoreCoord(x, y) for x0, x1, y0, y1 in rects for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    taken = {(c.x, c.y) for c in readers + gu}
    # relays just east / south of their rectangle (NOC1 multicasts run -x / -y)
    if xcol:  # all relays west of both rectangles (NOC0 multicasts run +x): each feeds both
        assert XNOC == 0
        relays = [ttnn.CoreCoord(1, y) for y in (4, 5, 3, 6, 2, 7, 1, 8) if (1, y) not in taken][:xcol]
        assert len(relays) == xcol
    elif XNOC == 1:
        relays = [next(ttnn.CoreCoord(6, y) for y in (5, 3, 2, 7, 0, 8) if (6, y) not in taken), ttnn.CoreCoord(9, 8)]
    else:  # NOC0 multicasts run +x / +y: relays just west of their rectangle
        relays = [
            next(ttnn.CoreCoord(1, y) for y in (4, 5, 3, 6, 2, 7) if (1, y) not in taken),
            next(ttnn.CoreCoord(7, y) for y in (3, 4, 2, 5, 1, 6) if (7, y) not in taken),
        ]
    taken |= {(c.x, c.y) for c in relays}
    if x2:  # a second relay per rectangle, next to the first (relays k and k + 2 serve rectangle k)
        assert XNOC == 0
        extra = [
            next(ttnn.CoreCoord(1, y) for y in (5, 3, 6, 2, 7, 1, 8) if (1, y) not in taken),
            next(ttnn.CoreCoord(7, y) for y in (4, 2, 5, 1, 6, 0, 7) if (7, y) not in taken),
        ]
        relays += extra
        taken |= {(c.x, c.y) for c in extra}
    down = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    return grid, phys, readers, gu, rects, relays, down


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("wdtype", WDTYPES)
@pytest.mark.parametrize("m", MS, ids=lambda m: f"M{m}")
def test_stream_expert_flat(device, m, wdtype):
    w_dtype, w_tile = W_DTYPES[wdtype]
    assert m % 32 == 0
    # rows per sub-block: M < 128 -> one sub-block of M rows; else 4 row tiles (MIMO_FL_MT overrides), the last
    # sub-block zero-padded when M is not a multiple of it (FLOPs / bytes below count the real M only)
    MT = int(os.environ["MIMO_FL_MT"]) if os.environ.get("MIMO_FL_MT") else min(MT_MAX, m // 32)
    m_pad = -(-m // (MT * 32)) * MT * 32
    RDOWN = m >= 256 if RDOWN_ENV == "auto" else bool(int(RDOWN_ENV))  # compute-bound M: readers help with down
    banks = device.dram_grid_size().x
    # e2e x relays: below reader-down M one relay west of both rectangles (several collapse: overlapping multicasts),
    # with reader-down two per rectangle (X2), taking turns by super-block
    XCOL = (1 if E2E and not RDOWN else 0) if XCOL_ENV == "auto" else int(XCOL_ENV)
    X2 = not XCOL and (E2E and RDOWN if X2_ENV == "auto" else bool(int(X2_ENV)))  # two x relays per rectangle
    grid, phys, readers, gu, rects, relays, down = _layout(device, X2, XCOL)
    ND = len(down)
    nrl = len(relays)
    if os.environ.get("MIMO_FL_SHOW"):
        logger.info(f"readers logical->phys {[((c.x, c.y), (phys(c).x, phys(c).y)) for c in readers]}")
        logger.info(f"gu phys x {sorted({phys(c).x for c in gu})} relays {[(phys(c).x, phys(c).y) for c in relays]}")
        for nm, lst in (("readers", readers), ("gu", gu), ("relays", relays), ("down", down)):
            logger.info(f"{nm}: " + " ".join(f"{c.x},{c.y}->{phys(c).x}-{phys(c).y}" for c in lst))
    n_rd, ngu = len(readers), len(gu)
    Ht, It, E = H // 32, I // 32, EXPERTS
    S = m_pad // (MT * 32)
    V = E * S
    nk_gu = Ht // KBLK
    slot = KBLK * 2
    ring_g = 2 * nk_gu
    n_rdn = D_CHAINS if RDOWN else 0  # readers that also compute down columns: one per down chain, as its tail
    pcd_r = (int(RDOWN_PCD_ENV) if RDOWN_PCD_ENV else 6 if (X2 or XCOL == 4) else 4) if RDOWN else 0
    rem_cols = Ht - n_rdn * pcd_r  # the down cores' columns; uneven when they do not divide: two widths
    base_p, extra = divmod(rem_cols, ND)
    pcds = [base_p + (1 if d < extra else 0) for d in range(ND)]
    col0s = [sum(pcds[:d]) for d in range(ND)]
    assert max(pcds) <= 8 and min(pcds) >= 1, pcds
    kd_of = lambda p_: max(k for k in (8, 4, 2, 1) if k * p_ <= 16 and It % k == 0)

    def dgrp(p_):
        k_ = kd_of(p_)
        return dict(kd=k_, nblk=It // k_, slot=k_ * p_, ring=int(round(DRING * (It // k_))), out=MT * p_)

    dgroups = {p_: [d for d in range(ND) if pcds[d] == p_] for p_ in sorted(set(pcds))}
    pcd = max(pcds)
    kd = kd_of(pcd)
    if RDOWN:
        kd_r = kd_of(pcd_r)
        nblk_r, slot_dr = It // kd_r, kd_r * pcd_r
        ring_dr = int(round(DRING * nblk_r))
        out_tiles_r = MT * pcd_r
    nblk = It // kd
    slot_d = kd * pcd
    ring_d = int(round(DRING * nblk))
    x_blk = MT * KBLK
    x_bytes = x_blk * BF8_TILE
    h_tiles = It * MT  # h of one sub-block: [KT][4 row tiles]
    out_tiles = MT * pcd
    pk = lambda c: (phys(c).x << 16) | phys(c).y

    # gate/up cores per reader: 4 each, nearest by the forwarder's NoC hops, balanced
    rect_of0 = lambda c: next(i for i, (x0, x1, y0, y1) in enumerate(rects) if x0 <= c.x <= x1 and y0 <= c.y <= y1)
    left = list(range(ngu))
    per_reader = {r: [] for r in range(n_rd)}
    fwd_noc = [1] * n_rd
    if FWD_DIR:  # forward the way the NoC runs: NOC0 (+x) to cores east of the reader, NOC1 (-x) to cores west
        in_rect = lambda k: [ci for ci in left if rect_of0(gu[ci]) == k]
        east = [r for r, c in enumerate(readers) if c.x in (0, 1)]  # -> rect 0 over NOC0
        far = sorted([r for r, c in enumerate(readers) if c.x in (6, 7)], key=lambda r: -readers[r].x)
        n2 = len(in_rect(1)) // R
        plan = [(r, 0, 0) for r in east] + [(r, 1, 0) for r in far[:n2]] + [(r, 0, 1) for r in far[n2:]]
        for r, k, noc in plan:
            fwd_noc[r] = noc
        for _ in range(R):
            for r, k, noc in plan:
                cand = in_rect(k)
                best = min(cand, key=lambda ci: noc_hops(phys(readers[r]), phys(gu[ci]), noc))
                per_reader[r].append(best)
                left.remove(best)
        assert not left
    else:
        for _ in range(ngu // n_rd):
            for r, c in enumerate(readers):
                best = min(left, key=lambda ci: noc_hops(phys(c), phys(gu[ci]), 1))
                per_reader[r].append(best)
                left.remove(best)
    order = [ci for r in range(n_rd) for ci in per_reader[r]]  # compute core index = gate/up column pair
    gu = [gu[ci] for ci in order]
    rect_of = lambda c: next(i for i, (x0, x1, y0, y1) in enumerate(rects) if x0 <= c.x <= x1 and y0 <= c.y <= y1)

    # ---- arena (per-role layout, 2 KB aligned) ----
    al = lambda b: (b + 2047) // 2048 * 2048
    X_OFF = al(ring_g * slot * w_tile)
    gu_bytes = X_OFF + al(X_SLOTS * x_bytes)
    rd_slots = RD_SLOTS or 2 * READ_BATCH
    RD_OFF = al(rd_slots * R * slot * w_tile)  # a down-computing reader's in1 ring follows its reader CB
    H_OFF = al(
        max(
            max(dgrp(p_)["ring"] * dgrp(p_)["slot"] for p_ in dgroups) * w_tile,
            RD_OFF + (ring_dr * slot_dr * w_tile if RDOWN else 0),
        )
    )
    O_OFF = H_OFF + al(HBUF * h_tiles * H_TILE)
    D_OFF = O_OFF + al(2 * out_tiles * 2048)
    dn_bytes = D_OFF + 2048
    tok_pad = -(-m // 32) * 32  # e2e: expert e's region starts at row e * tok_pad of the dispatch buffer
    # per-expert token counts (dynamic mode: read on device; the program is built for up to m per expert)
    cnts = (
        [int(c) for c in os.environ["MIMO_FL_COUNTS"].split(",")]
        if DYN and os.environ.get("MIMO_FL_COUNTS")
        else [m] * E
    )
    assert len(cnts) == E and max(cnts) <= m, cnts
    band = [int(v) for v in os.environ.get("MIMO_FL_BAND", "1,1000000000").split(",")]
    cap = E * tok_pad
    nsb = H // 1024  # e2e: super-blocks (32 K tiles) per row
    SB_OFF = al(RM_CHUNKS * 32 * 2048)
    relay_bytes = SB_OFF + al(SB_SLOTS * MT * 32 * BF8_TILE) if E2E else al(RELAY_CB * x_bytes)
    arena_tiles = max(gu_bytes, dn_bytes, relay_bytes, RD_OFF) // 2048
    logger.info(
        f"M {m}: {S} sub-blocks; gu ring {ring_g * slot * w_tile >> 10} KB, x ring {X_SLOTS * x_bytes >> 10} KB; "
        f"down {ND} x {pcd} (kd {kd}); arena {arena_tiles * 2} KB; relays {[(c.x, c.y) for c in relays]}"
    )

    torch.manual_seed(0)
    Wg, Wu, Wd = torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02
    xs = torch.randn(E, m_pad, H)
    xs[:, m:] = 0
    xs = xs.view(V, MT * 32, H)
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)).float()
    x_last = xs[(E - 1) * S :].reshape(m_pad, H)[:m]
    ref = (torch.nn.functional.silu(x_last @ Wg) * (x_last @ Wu)) @ Wd
    ref_q = (torch.nn.functional.silu(x_last @ q(Wg)) * (x_last @ q(Wu))) @ q(Wd)
    tiles = lambda w: w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32).permute(0, 2, 1, 3)
    Wg_t, Wu_t, Wd_t = tiles(Wg), tiles(Wu), tiles(Wd)

    # ---- gate/up weights: per reader region, per expert, per K-block, its 4 cores' [KBLK x (g, u)] blocks ----
    regions = []
    for r in range(n_rd):
        blocks = []
        for c in range(nk_gu):
            ks = slice(c * KBLK, (c + 1) * KBLK)
            for j in range(R):
                ci = r * R + j
                blocks.append(torch.stack([Wg_t[ks, ci], Wu_t[ks, ci]], dim=1).reshape(-1, 32, 32))
        regions.append(torch.cat(blocks).repeat(E, 1, 1))
    w_dev = _bank_sharded(regions, banks, w_dtype, device)
    region_bytes = regions[0].shape[0] * w_tile

    # ---- down weights: per down core region, per expert, per K-block, [kd x pcd] ----
    def dreg(d):
        p_, k_ = pcds[d], kd_of(pcds[d])
        return torch.cat(
            [Wd_t[c * k_ : (c + 1) * k_, col0s[d] : col0s[d] + p_].reshape(-1, 32, 32) for c in range(It // k_)]
        ).repeat(E, 1, 1)

    dregs = [dreg(d) for d in range(ND)]
    n_max = max(r_.shape[0] for r_ in dregs)
    dregs = [torch.cat([r_, torch.zeros(n_max - r_.shape[0], 32, 32)]) for r_ in dregs]  # equal regions
    wd_dev = _bank_sharded(dregs, banks, w_dtype, device)
    wd_region = dregs[0].shape[0] * w_tile
    if RDOWN:  # the down-computing readers' columns follow the down cores'
        rregs = [
            torch.cat(
                [
                    Wd_t[c * kd_r : (c + 1) * kd_r, rem_cols + i * pcd_r : rem_cols + (i + 1) * pcd_r].reshape(
                        -1, 32, 32
                    )
                    for c in range(nblk_r)
                ]
            ).repeat(E, 1, 1)
            for i in range(n_rdn)
        ]
        wr_dev = _bank_sharded(rregs, banks, w_dtype, device)
        wr_region = rregs[0].shape[0] * w_tile
    # ---- x: blocks (v, K-block) of [128 x 256], block k in region k % 16 (bank-spread) ----
    xblocks = [
        xs[v][:, b * KBLK * 32 : (b + 1) * KBLK * 32].reshape(MT, 32, KBLK, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)
        for v in range(V)
        for b in range(nk_gu)
    ]
    nreg = 2 * banks
    assert len(xblocks) % nreg == 0
    xregs = [torch.cat(xblocks[r::nreg]) for r in range(nreg)]
    x_dev = _bank_sharded(xregs, banks, ttnn.bfloat8_b, device)
    x_region = xregs[0].shape[0] * BF8_TILE
    if E2E:  # the model's dispatch buffer: row-major bf16 [cap, H], expert e's m tokens at row e * tok_pad
        xs_e = xs.view(E, m_pad, H)
        disp = torch.zeros(cap, H)
        for e in range(E):
            disp[e * tok_pad : e * tok_pad + cnts[e]] = xs_e[e, : cnts[e]]
        x_dev = ttnn.from_torch(
            disp,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    both = gu + down
    arena_cores = both + relays + readers  # every role keeps its buffers in the one lockstep arena
    arena = ttnn.from_torch(
        torch.zeros(len(arena_cores) * arena_tiles * 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_crs(arena_cores), (arena_tiles * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    base = arena.buffer_address()
    land_addr, x_ring_addr, h_all_addr = base, base + X_OFF, base + H_OFF
    y_dram = (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([cap, H]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        if E2E
        else ttnn.allocate_tensor_on_device(
            ttnn.Shape([V * MT * 32, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
    )
    e2e_rt = (
        [v_ for e in range(E) for v_ in (e * tok_pad // 32, tok_pad // 32)] if E2E else []
    )  # region tile, count tiles
    dyn_args = []
    if DYN:  # the routing's outputs: per global expert token count and region row, local experts at odd global ids
        NG = 2 * E + 2
        gids = [2 * e + 1 for e in range(E)]
        c_host = torch.full((1, NG), 7777, dtype=torch.int32)
        r_host = torch.full((1, NG), 999999, dtype=torch.int32)
        for e in range(E):
            c_host[0, gids[e]] = cnts[e]
            r_host[0, gids[e]] = e * tok_pad
        counts_dev = ttnn.from_torch(
            c_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        regions_dev = ttnn.from_torch(
            r_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dyn_args = [counts_dev.buffer_address(), regions_dev.buffer_address(), 4 * NG, band[0], band[1]] + gids
        e2e_rt = dyn_args  # the down kernels take the se_dyn.hpp args where the static region list went
    words_zero = ttnn.from_torch(torch.zeros(len(relays) * 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    words = ttnn.from_torch(
        torch.zeros(len(relays) * 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_crs(relays), (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    rect_cores = [[ci for ci in range(ngu) if rect_of(gu[ci]) == k] for k in range(len(rects))]
    word_of = {
        ci: words.buffer_address() + 4 * (ci if XCOL else rect_cores[rect_of(gu[ci])].index(ci)) for ci in range(ngu)
    }

    DATA, HARR, GO, DONE, GATH, XARR, SFREE, HFREE, HSFREE, WORD, GATH1, GATH2 = (
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    sems = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=_crs(readers + both + relays), initial_value=0) for i in range(16)
    ]
    d_pred, d_succ, d_heads = {}, {}, []
    rdn = []  # (reader index, chain tail down index)
    for seg in _chains([(d, down[d]) for d in range(ND)], D_CHAINS, phys, DN_NOC):
        d_heads.append(seg[0][0])
        if RDOWN:
            tail = seg[-1][0]
            used = {r for r, _ in rdn}
            r = min(
                (r for r in range(n_rd) if r not in used),
                key=lambda r: noc_hops(phys(down[tail]), phys(readers[r]), DN_NOC),
            )
            rdn.append((r, tail))
        for a, b in zip(seg, seg[1:]):
            d_succ[a[0]], d_pred[b[0]] = b[0], a[0]
    xy_or0 = lambda lst, i: pk(lst[i]) if i is not None else 0
    head_xy = [pk(down[d]) for d in d_heads]

    rd_vals, fw_vals, gu_rt = {}, {}, ttnn.RuntimeArgs()
    for r, c in enumerate(readers):
        rd_vals[(c.x, c.y)] = [w_dev.buffer_address(), r % banks, (r // banks) * region_bytes, 0] + dyn_args
        fw_vals[(c.x, c.y)] = [land_addr] + [pk(gu[r * R + j]) for j in range(R)] + (dyn_args if DYN else [0] * R)
        for j in range(R):
            ci = r * R + j
            g = gu[ci]
            gu_rt[g.x][g.y] = (
                [
                    pk(c),
                    j,
                    ci,
                    len(head_xy),
                    0,
                    pk(down[0]),
                    GATH,
                    x_ring_addr,
                    0,
                    0,
                    0,
                    *(
                        (pk(relays[0]), word_of[ci], h_all_addr, 0)
                        if XCOL
                        else (pk(relays[rect_of(g)]), word_of[ci], h_all_addr, 0)
                    ),
                    *(
                        [pk(relays[k]) if k < XCOL else 0 for k in (1, 2, 3)]
                        if XCOL
                        else [pk(relays[rect_of(g) + 2]) if X2 else 0, 2, 3]
                    ),
                    0,
                    SFREE,
                    WORD,
                ]
                + head_xy
                + dyn_args
            )
    dr_rts = {p_: ttnn.RuntimeArgs() for p_ in dgroups}
    dw_rts = {p_: ttnn.RuntimeArgs() for p_ in dgroups}
    gu_xy = [pk(c) for c in gu]
    for d, dc in enumerate(down):
        tail_succ = {t: pk(readers[r]) for r, t in rdn}
        dr_rt, dw_rt = dr_rts[pcds[d]], dw_rts[pcds[d]]
        dr_rt[dc.x][dc.y] = (
            [
                h_all_addr,
                xy_or0(down, d_pred.get(d)),
                tail_succ.get(d, xy_or0(down, d_succ.get(d))),
                pk(down[0]),
                int(d == 0),
                ngu,
                y_dram.buffer_address(),
                col0s[d],
                base + D_OFF,
                d,
            ]
            + gu_xy
            + e2e_rt
        )
        dw_rt[dc.x][dc.y] = [wd_dev.buffer_address(), d % banks, (d // banks) * wd_region] + dyn_args
    xr_rt, xm_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    vstride = XCOL or (2 if X2 else 1)
    rl_off = lambda idx: idx if XCOL else idx // 2
    rl_sb = lambda idx: len([g for g in range(V * nsb) if g % vstride == rl_off(idx)])  # super-blocks relay idx sends
    for idx, rl in enumerate(relays):
        k = idx % 2  # its rectangle
        x0, x1, y0, y1 = rects[k]
        lo, hi = ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)
        xr_rt[rl.x][rl.y] = (
            [x_dev.buffer_address(), vstride, rl_off(idx)]
            + (dyn_args or [v_ for e in range(E) for v_ in (e * tok_pad, m)])
            if E2E
            else [x_dev.buffer_address(), x_region, 0]
        )
        a, b = (hi, lo) if XNOC == 1 else (lo, hi)
        xm_l = [
            x_ring_addr,
            pk(a),
            pk(b),
            (x1 - x0 + 1) * (y1 - y0 + 1),
            words.buffer_address(),
            len(rect_cores[k]),
        ] + ([rl_sb(idx), XARR if idx < 2 else 3, vstride, idx // 2, 0, 0, 0, 0] if E2E else [])
        if XCOL:  # both rectangles, every gate/up core's freed word
            (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = rects
            xm_l = [
                x_ring_addr,
                pk(ttnn.CoreCoord(ax0, ay0)),
                pk(ttnn.CoreCoord(ax1, ay1)),
                (ax1 - ax0 + 1) * (ay1 - ay0 + 1),
                words.buffer_address(),
                ngu,
                rl_sb(idx),
                (XARR, 3, 2, 1)[idx],
                vstride,
                idx,
                0,
                (bx1 - bx0 + 1) * (by1 - by0 + 1),
                pk(ttnn.CoreCoord(bx0, by0)),
                pk(ttnn.CoreCoord(bx1, by1)),
            ]
        xm_rt[rl.x][rl.y] = xm_l + dyn_args

    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    dyn_def = [("SE_DYN", "1")] if DYN else []
    FWD = max(FWD_DEPTH, 2) if DYN else FWD_DEPTH  # dynamic counts use the pipelined forwarder
    e2e_def = [("SE_E2E", "1")] if E2E else []
    tz_rt = ttnn.RuntimeArgs()
    for rl in relays:
        tz_rt[rl.x][rl.y] = [rl_sb(relays.index(rl))]
    zones = [("SE_ZONES", "1")] if os.environ.get("MIMO_SE_ZONES") else []
    zones += [("SE_WAITZ", os.environ["MIMO_SE_WAITZ"])] if os.environ.get("MIMO_SE_WAITZ") else []
    total_x = V * nk_gu
    gu_crs, dn_crs, rd_crs, rl_crs = _crs(gu), _crs(down), _crs(readers), _crs(relays)
    swap = bool(int(os.environ.get("MIMO_FL_RD_SWAP", "0")))  # readers read on NOC1 and forward on NOC0
    RD_NOC, FW_NOC = (ttnn.NOC.NOC_1, ttnn.NOC.NOC_0) if swap else (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)
    kernels = []
    for fn in (0, 1):
        grp = [c for r, c in enumerate(readers) if fwd_noc[r] == fn]
        if not grp:
            continue
        rd_noc, fw_noc_ = (ttnn.NOC.NOC_1, ttnn.NOC.NOC_0) if fn == 0 else (RD_NOC, FW_NOC)
        g_rd, g_fw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rdn_cores = {(readers[r].x, readers[r].y) for r, _ in rdn}
        plain = [c for c in grp if (c.x, c.y) not in rdn_cores]
        for c in plain:
            g_rd[c.x][c.y] = rd_vals[(c.x, c.y)]
        for c in grp:
            g_fw[c.x][c.y] = fw_vals[(c.x, c.y)]
        mine = [(i, r, t) for i, (r, t) in enumerate(rdn) if fwd_noc[r] == fn]
        if mine:
            assert READ_BATCH == 1
            g_rn = ttnn.RuntimeArgs()
            for i, r, t in mine:
                c = readers[r]
                g_rn[c.x][c.y] = [
                    w_dev.buffer_address(),
                    r % banks,
                    (r // banks) * region_bytes,
                    wr_dev.buffer_address(),
                    i % banks,
                    (i // banks) * wr_region,
                    pk(down[t]),
                    pk(down[0]),
                    base + D_OFF,
                    ND + i,
                    y_dram.buffer_address(),
                    rem_cols + i * pcd_r,
                ] + e2e_rt
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/se9_rdown.cpp",
                    source_type=FP,
                    core_ranges=_crs([readers[r] for _, r, _ in mine]),
                    compile_time_args=[
                        0,
                        w_tile,
                        R * slot,
                        E * nk_gu,
                        1,
                        slot_dr,
                        E * nblk_r,
                        2,
                        h_tiles,
                        H_TILE,
                        H_PIECES,
                        16,
                        out_tiles_r,
                        V,
                        HARR,
                        HSFREE,
                        HBUF,
                        MT,
                        pcd_r,
                        Ht,
                        S,
                        E,
                    ],
                    defines=([("SE_GU_FIRST", "1")] if int(os.environ.get("MIMO_FL_GU_FIRST", "0")) else [])
                    + e2e_def
                    + dyn_def,
                    runtime_args=g_rn,
                    config=dm(ttnn.DataMovementProcessor.RISCV_0, rd_noc),
                )
            )
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/se6_dcompute.cpp",
                    source_type=FP,
                    core_ranges=_crs([readers[r] for _, r, _ in mine]),
                    compile_time_args=[MT, 1, It, kd_r, pcd_r, E, S, slot_dr, ring_dr],
                    runtime_args=[],
                    defines=zones
                    + dyn_def
                    + ([("SE_EARLY_POP", "1")] if int(os.environ.get("MIMO_FL_EARLY_POP", "1")) else []),
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
                )
            )
        if plain:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/se_reader.cpp",
                    source_type=FP,
                    core_ranges=_crs(plain),
                    compile_time_args=[0, w_tile, R * slot, READ_BATCH, nk_gu, 0, 1, E, R * slot, 0],
                    defines=dyn_def,
                    runtime_args=g_rd,
                    config=dm(ttnn.DataMovementProcessor.RISCV_0, rd_noc),
                )
            )
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se_forward.cpp",
                source_type=FP,
                core_ranges=_crs(grp),
                compile_time_args=[
                    0,
                    R,
                    w_tile,
                    R * slot,
                    slot,
                    ring_g,
                    0,
                    DATA,
                    KBLK,
                    nk_gu,
                    0,
                    1,
                    E,
                    READ_BATCH,
                    0,
                    1,
                    slot,
                    0,
                ],
                runtime_args=g_fw,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, fw_noc_),
            )
            if not FWD
            else ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se10_fwd.cpp",
                source_type=FP,
                core_ranges=_crs(grp),
                compile_time_args=[
                    0,
                    R,
                    w_tile,
                    R * slot,
                    slot,
                    ring_g,
                    0,
                    DATA,
                    nk_gu if DYN else E * nk_gu,
                    rd_slots,
                    FWD,
                    E,
                ],
                defines=dyn_def,
                runtime_args=g_fw,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, fw_noc_),
            ),
        ]
    kernels += (
        []
        + (
            [
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/se11_xrd.cpp",
                    source_type=FP,
                    core_ranges=rl_crs,
                    compile_time_args=[0, H * 2, E, MT, nsb, S, XRD_BATCH],
                    runtime_args=xr_rt,
                    defines=zones + dyn_def,
                    config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0 if XNOC == 1 else ttnn.NOC.NOC_1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/se11_tz.cpp",
                    source_type=FP,
                    core_ranges=rl_crs,
                    compile_time_args=[0, 1, MT],
                    runtime_args=tz_rt,
                    defines=zones + dyn_def,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/se11_xmc.cpp",
                    source_type=FP,
                    core_ranges=rl_crs,
                    compile_time_args=[1, MT, BF8_TILE, X_SLOTS, XARR, WORD, KBLK, E, nsb],
                    runtime_args=xm_rt,
                    defines=zones + dyn_def,
                    config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1 if XNOC == 1 else ttnn.NOC.NOC_0),
                ),
            ]
            if E2E
            else [
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/xdl_selfread.cpp",
                    source_type=FP,
                    core_ranges=rl_crs,
                    compile_time_args=[0, x_bytes, total_x, nreg, banks, X_BATCH, x_blk, 1],
                    runtime_args=xr_rt,
                    config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0 if XNOC == 1 else ttnn.NOC.NOC_1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/se8_xmc.cpp",
                    source_type=FP,
                    core_ranges=rl_crs,
                    compile_time_args=[0, x_blk, x_bytes, total_x, X_SLOTS, XARR, RELAY_CB, WORD],
                    runtime_args=xm_rt,
                    defines=zones + ([("XMC_SAFE", "1")] if os.environ.get("MIMO_FL_XMC_SAFE") else []),
                    config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1 if XNOC == 1 else ttnn.NOC.NOC_0),
                ),
            ]
        )
        + [
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se5_recv.cpp",
                source_type=FP,
                core_ranges=gu_crs,
                compile_time_args=[
                    0,
                    x_blk,
                    1,
                    slot,
                    E * nk_gu,
                    V,
                    ring_g,
                    16,
                    1,
                    3,
                    2,
                    MT,
                    H_TILE,
                    ngu,
                    DATA,
                    HARR,
                    GO,
                    DONE,
                    KBLK,
                    HARR,
                    SFREE,
                    HBUF,
                    XARR,
                    HFREE,
                    X_SLOTS,
                    x_bytes,
                    S,
                    1,
                    HSFREE,
                    H_PIECES,
                    nk_gu,
                    GATH1,
                    GATH2,
                    1,
                    1,
                    E,
                ],
                defines=zones
                + dyn_def
                + [("SE_GU_ONLY", "1"), ("SE_X_RELAY", "1"), ("SE_NO_PARTNER", "1")]
                + ([("SE_X_RELAY2", str(32 // KBLK)), ("SE_X_NRELAY", str(XCOL or 2))] if (X2 or XCOL) else []),
                runtime_args=gu_rt,
                config=dm(
                    ttnn.DataMovementProcessor.RISCV_0,
                    ttnn.NOC.NOC_1 if int(os.environ.get("MIMO_FL_GU_NOC", "1")) else ttnn.NOC.NOC_0,
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se3_compute.cpp",
                source_type=FP,
                core_ranges=gu_crs,
                compile_time_args=[KBLK, MT, nk_gu, 0, 1, E, S, slot, 1, 1, 1, 0, ring_g],
                runtime_args=[],
                defines=zones + dyn_def + [("SE_GU_ONLY", "1")],
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
            ),
        ]
    )
    for p_, ds in dgroups.items():
        g_ = dgrp(p_)
        g_crs = _crs([down[d] for d in ds])
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se6_drecv.cpp",
                source_type=FP,
                core_ranges=g_crs,
                compile_time_args=[
                    2,
                    h_tiles,
                    H_TILE,
                    H_PIECES,
                    16,
                    g_["out"],
                    V,
                    S,
                    ngu,
                    ND + n_rdn,
                    HARR,
                    HSFREE,
                    GATH,
                    DONE,
                    GO,
                    HBUF,
                    MT,
                    p_,
                    Ht,
                    GATH,
                    GATH1,
                    GATH2,
                    E,
                ],
                defines=e2e_def + dyn_def,
                runtime_args=dr_rts[p_],
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1 if DN_NOC else ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se6_dw.cpp",
                source_type=FP,
                core_ranges=g_crs,
                compile_time_args=[
                    1,
                    g_["slot"],
                    w_tile,
                    E * g_["nblk"],
                    int(os.environ.get("MIMO_FL_DW_BATCH", "2")),
                    int(wdtype == "bf8"),
                    E,
                ],
                runtime_args=dw_rts[p_],
                defines=dyn_def
                + ([("SE_DW_DELAY", os.environ["MIMO_FL_DW_DELAY"])] if os.environ.get("MIMO_FL_DW_DELAY") else []),
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0 if DN_NOC else ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/se6_dcompute.cpp",
                source_type=FP,
                core_ranges=g_crs,
                compile_time_args=[MT, 1, It, g_["kd"], p_, E, S, g_["slot"], g_["ring"]],
                runtime_args=[],
                defines=zones
                + dyn_def
                + ([("SE_EARLY_POP", "1")] if int(os.environ.get("MIMO_FL_EARLY_POP", "1")) else []),
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
            ),
        ]
    OUT_FMT = (ttnn.bfloat8_b, BF8_TILE) if E2E else (ttnn.bfloat16, 2048)
    fmt = lambda i, d_, page: [ttnn.CBFormatDescriptor(buffer_index=i, data_format=d_, page_size=page)]

    def arena_cb(idx, off, size, crs, d_, page):
        cb = ttnn.cb_descriptor_from_sharded_tensor(idx, arena, address_offset=off, total_size=size, core_ranges=crs)
        cb.format_descriptors = fmt(idx, d_, page)
        return cb

    cbs = (
        [
            arena_cb(0, 0, rd_slots * R * slot * w_tile, rd_crs, w_dtype, w_tile),
        ]
        + (
            [
                arena_cb(0, 0, RM_CHUNKS * 32 * 2048, rl_crs, ttnn.bfloat16, 2048),
                arena_cb(1, SB_OFF, SB_SLOTS * MT * 32 * BF8_TILE, rl_crs, ttnn.bfloat8_b, BF8_TILE),
            ]
            if E2E
            else [arena_cb(0, 0, RELAY_CB * x_bytes, rl_crs, ttnn.bfloat8_b, BF8_TILE)]
        )
        + [
            arena_cb(1, 0, ring_g * slot * w_tile, gu_crs, w_dtype, w_tile),
            arena_cb(0, X_OFF, X_SLOTS * x_bytes, gu_crs, ttnn.bfloat8_b, BF8_TILE),
            ttnn.CBDescriptor(
                total_size=HBUF * MT * H_TILE, core_ranges=gu_crs, format_descriptors=fmt(3, ttnn.bfloat8_b, H_TILE)
            ),
            ttnn.CBDescriptor(total_size=2048, core_ranges=gu_crs, format_descriptors=fmt(16, ttnn.bfloat16, 2048)),
            arena_cb(2, H_OFF, HBUF * h_tiles * H_TILE, dn_crs, ttnn.bfloat8_b, H_TILE),
        ]
        + [
            cb_
            for p_, ds in dgroups.items()
            for cb_ in (
                arena_cb(
                    1, 0, dgrp(p_)["ring"] * dgrp(p_)["slot"] * w_tile, _crs([down[d] for d in ds]), w_dtype, w_tile
                ),
                arena_cb(16, O_OFF, 2 * dgrp(p_)["out"] * OUT_FMT[1], _crs([down[d] for d in ds]), *OUT_FMT),
            )
        ]
        + []
        + (
            []
            if not RDOWN
            else [
                arena_cb(1, RD_OFF, ring_dr * slot_dr * w_tile, _crs([readers[r] for r, _ in rdn]), w_dtype, w_tile),
                arena_cb(2, H_OFF, HBUF * h_tiles * H_TILE, _crs([readers[r] for r, _ in rdn]), ttnn.bfloat8_b, H_TILE),
                arena_cb(16, O_OFF, 2 * out_tiles_r * OUT_FMT[1], _crs([readers[r] for r, _ in rdn]), *OUT_FMT),
            ]
        )
    )
    if (
        DYN
    ):  # CB 6: the counts page a data-movement kernel hands its compute; CB 7: DM scratch (BRISC low / NCRISC high half)
        all_crs = _crs(arena_cores)
        cbs += [
            ttnn.CBDescriptor(total_size=64, core_ranges=all_crs, format_descriptors=fmt(6, ttnn.uint32, 64)),
            ttnn.CBDescriptor(total_size=4096, core_ranges=all_crs, format_descriptors=fmt(7, ttnn.uint32, 4096)),
        ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    tag = (
        f"streamfl_H{H}_M{m}_E{E}_w{wdtype}"
        + (f"_mt{MT}" if os.environ.get("MIMO_FL_MT") else "")
        + ("_e2e" if E2E else "")
    )
    tok_act = sum(c for c in cnts if c and band[0] <= c <= band[1]) if DYN else E * m  # tokens this program processes
    if DYN:
        tag += "_dyn" + "-".join(map(str, cnts)) + (f"_b{band[0]}-{band[1]}" if os.environ.get("MIMO_FL_BAND") else "")
    w_bytes = E * 3 * H * I * w_tile / 1024
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "M": m,
                    "E": E,
                    "wdtype": wdtype,
                    "weight_bytes": w_bytes,
                    "flops": 6 * tok_act * H * I,
                    "x_bytes": tok_act * H * 1.0625,
                    "y_bytes": tok_act * H * 2,
                }
            )
            + "\n"
        )
    for it in range(1 + ITERS):
        ttnn.copy_host_to_device_tensor(words_zero, words)  # relay freed words restart every launch
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op([w_dev, wd_dev, x_dev, arena, y_dram, words] + ([wr_dev] if RDOWN else []), program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
        if it == 0:
            if E2E:  # every expert's rows at its region of the model-shaped output
                yh = ttnn.to_torch(y_dram).float()
                xs_e = xs.view(E, m_pad, H)
                act = [e for e in range(E) if cnts[e] and band[0] <= cnts[e] <= band[1]]  # experts this program serves
                qg, qu, qd = q(Wg), q(Wu), q(Wd)
                refs_q = {
                    e: (torch.nn.functional.silu(xs_e[e, : cnts[e]] @ qg) * (xs_e[e, : cnts[e]] @ qu)) @ qd for e in act
                }
                pccs = [comp_pcc(refs_q[e], yh[e * tok_pad : e * tok_pad + cnts[e]], 0.99) for e in act]
                logger.info(f"e2e per-expert PCC {[round(float(p_[1]), 5) for p_ in pccs]}")
                assert all(p_[0] for p_ in pccs)
                if DYN:
                    logger.info(f"dyn: counts {cnts} band {band} -> active {act}")
                    ok = True
                    pcc_q = min(float(p_[1]) for p_ in pccs) if pccs else 1.0
                got = yh[(E - 1) * tok_pad : (E - 1) * tok_pad + m]
            else:
                got = ttnn.to_torch(y_dram).float()[(E - 1) * S * MT * 32 :][:m]
            if not DYN:
                ok, pcc_q = comp_pcc(ref_q, got, 0.99)
            if not ok and os.environ.get("MIMO_FL_DEBUG"):
                yall = ttnn.to_torch(y_dram).float().view(V, MT * 32, H)
                ar = (
                    ttnn.to_torch(arena).view(torch.int16).reshape(len(arena_cores), arena_tiles * 32 * 32)
                )  # untilized per 2 KB
                xr = ar[:ngu, X_OFF // 2 : (X_OFF + X_SLOTS * x_bytes) // 2]
                same = [int((xr[i] == xr[0]).all()) for i in range(ngu)]

                def raw(core):  # re-tilize the untilized bf16 view back to the L1 bytes
                    t = ar[core].view(-1, 32, 32)
                    f = t.view(-1, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(-1, 1024)
                    return f.contiguous().view(torch.uint8).reshape(-1)

                from ttnn._ttnn import bfp_utils

                order = ttnn.corerange_to_cores(_crs(arena_cores), None, True)
                shard_of = {(c.x, c.y): i for i, c in enumerate(order)}

                def xblock(t):
                    v, b = t // nk_gu, t % nk_gu
                    return (
                        xs[v][:, b * KBLK * 32 : (b + 1) * KBLK * 32]
                        .reshape(MT, 32, KBLK, 32)
                        .permute(0, 2, 1, 3)
                        .reshape(-1, 32, 32)
                    )

                def packed(t):
                    return torch.cat(
                        [
                            torch.from_numpy(
                                bfp_utils.pack_bfp8(tl.reshape(-1).float().contiguous().numpy(), True)
                            ).view(torch.uint8)
                            for tl in xblock(t)
                        ]
                    )

                if os.environ.get("MIMO_FL_DUMP"):
                    torch.save(
                        {
                            "ring": [raw(shard_of[(c.x, c.y)])[X_OFF : X_OFF + X_SLOTS * x_bytes].clone() for c in gu],
                            "rect": [rect_of(c) for c in gu],
                            "xs": xs,
                            "nk_gu": nk_gu,
                        },
                        os.environ["MIMO_FL_DUMP"],
                    )
                exp = {t: packed(t) for t in range(total_x - X_SLOTS, total_x)}
                logger.info(f"packed block bytes {exp[total_x - 1].numel()} vs x_bytes {x_bytes}")
                for ci in (0, 1, 40, 63):
                    rb = raw(shard_of[(gu[ci].x, gu[ci].y)])[X_OFF : X_OFF + X_SLOTS * x_bytes].view(X_SLOTS, x_bytes)
                    res = []
                    for k in range(X_SLOTS):
                        t = next(t for t in exp if t % X_SLOTS == k)
                        eq = (rb[k] == exp[t]).float().mean().item()
                        tiles_ok = [
                            (rb[k].view(x_blk, -1)[i] == exp[t].view(x_blk, -1)[i]).all().item() for i in range(x_blk)
                        ]
                        res.append((k, t, round(eq, 3), sum(tiles_ok)))
                    logger.info(
                        f"gu {ci} {(gu[ci].x, gu[ci].y)} rect {rect_of(gu[ci])}: (slot, block, byte-eq, tiles-eq) {res}"
                    )
                for i in range(0, ngu, 5):
                    logger.info(
                        f"core {i} {(gu[i].x, gu[i].y)} slot nz {[round(float((xr[i].view(X_SLOTS, -1)[k] != 0).float().mean()), 2) for k in range(X_SLOTS)]}"
                    )
                logger.info(
                    f"x ring equal to core 0: {sum(same)}/{ngu} {same}; nonzero frac {(xr != 0).float().mean():.3f}"
                )
                hr = ar[ngu:, H_OFF // 2 : (H_OFF + HBUF * h_tiles * H_TILE) // 2]
                logger.info(
                    f"h_all equal across down: {[int((hr[i] == hr[0]).all()) for i in range(ND)]}; nonzero {(hr != 0).float().mean():.3f}"
                )
                for v in range(V):
                    xv = xs[v]
                    rv = (torch.nn.functional.silu(xv @ q(Wg)) * (xv @ q(Wu))) @ q(Wd)
                    logger.info(
                        f"v{v}: |y| {yall[v].abs().mean():.4g} |ref| {rv.abs().mean():.4g} pcc {comp_pcc(rv, yall[v], 0)[1]}"
                        f" nan {torch.isnan(yall[v]).sum()} cols-pcc {[round(float(comp_pcc(rv[:, d*256:(d+1)*256], yall[v][:, d*256:(d+1)*256], 0)[1]), 3) for d in range(0, ND, 7)]}"
                    )
            _, pcc = comp_pcc(ref, got, 0.0)
            logger.info(f"{tag}: PCC {pcc_q} vs quantized-weight reference, {pcc} vs fp32")
            assert ok, pcc_q
    logger.info(f"ran {tag}")
