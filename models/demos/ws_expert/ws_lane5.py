# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Lane design (ws_lane.py) with 5 x 5 K-split groups (ws_g5.py): the lanes' static schedule, slot tables, muxes and
credits unchanged; each group is 5 columns x 5 rows (core j = 5 c + r, rows 0-3 own a gate/up pair each and run 4
down columns with 25 K tiles; row 4 runs 8 down columns with 20 K tiles), the weights move in packets of CHT 20 bfp4
tiles (owners 10 gate/up + 4 down packets a slice, row 4 8 + 8), h goes to the group as a unicast fan-out with slot
credits (no multicasts). Kernels: l5_writer.cpp (BRISC), l5_reader.cpp (NCRISC), l5_compute.cpp.
L5Cfg.hmc: h by one multicast a row from the group's hub (core 0, the rectangle's NOC0 multicast start corner): the
owners send their tiles to the hub, whose BRISC (l5h_hub.cpp) multicasts the 20-tile row to the group; every other
transfer of the group's cores (and the muxes) runs on NOC1 from the NCRISC (l5h_reader.cpp), so that inside a group's
rectangle NOC0 carries only its hub's multicasts. hmc 2: the BRISC (l5h_writer.cpp) also keeps the transfers that
stay inside the rectangle on NOC0 (x rows east, partials and landing credits south) and multicasts from the hub; so
nothing enters a group's rectangle on NOC0 from outside.

Layout (L5Cfg): groups in logical columns 1-5 (and 7-11 on a 12 x 10 Galaxy chip; each block NoC contiguous), rows
0-4 / 5-9; each group's two mux cores in the column just west of it (0 / 6: the cores reach them on NOC1). The 11 x 10
tray grid holds G 2. With hmc, L5Cfg.place puts each chip's lanes on the rectangles by the ethernet cores they send on
(place_lanes): a lane's cores reach the same lane's cores on the neighbouring chips wherever those chips put it."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node
from models.demos.ws_expert.ws_g5 import NCOL, NR, OWN_ROWS, core_geom, row_geom
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane import LCfg, lane_cycle, lane_cycles, lanes

fabric_mux = ttnn.experimental.fabric_mux
GC5 = NR * NCOL
NW = 8  # partial tiles a row (4 pairs x gate / up)


@dataclass
class L5Cfg:
    """5 x 5 specifics (GCfg / LCfg hold the rest: G, nexp, T, slots, keep, sched_c, nbuf, NS, dn_lag, h_dt, XRM)."""

    NXR: int = 6  # x ring rows
    D: int = 3  # landing slots
    PD: int = 2  # partial ring rows
    FL: int = 2  # fold lag
    TL: int = 9  # tilize look-ahead (rows; own rows are 1 in 5)
    RD: int = 4  # own-expert chunk reads in flight
    # h by the group hub's multicast: 1 (l5h_reader.cpp / l5h_hub.cpp) NOC0 only for the hub's multicasts; 2
    # (l5h_writer.cpp / l5h_reader.cpp WL5_HMC2) NOC0 also for the transfers that stay inside the group's rectangle
    hmc: int = 0
    mux_noc: int = -1  # the mux kernels' NoC: -1 NOC1 with hmc, else NOC0; 0 / 1 force it (diagnostics)
    mux_mask: int = -1  # >= 0: group g's muxes on NOC1 iff bit g (overrides mux_noc; diagnostics)
    gather0: int = 0  # hmc 2: the x gather from the BRISC on NOC0 (WL5_GATHER0; outside the NoC rule, diagnostics)
    # output bank affinity (WL5_OAFF, a group mask): group g's output tiles only in the DRAM banks at NoC x 9 (bit g
    # set) or x 0 (bit clear), so no output write crosses the chip; the output tensor holds 2x the pages (half unused)
    oaff: int = 0
    # row-major result rows (hmc 2; WL_Y_RM / WL5_Y_RM): every down row to its row of a row-major [rows, H] result buffer,
    # given by the expert's destination list (build_lane5's dest_t), posted, fenced at fence_row
    yrm: int = 0
    remap_row: int = 0  # yrm: DEST's untilize remap only around each down row (WL5_REMAP_ROW), not for the whole kernel
    yt: int = 0  # yrm: plain tiles from the compute, transposed to rows by the BRISC into a staging slot (WL5_YT)
    # the compute's pack-untilize of the down rows (WL_Y_RM on the compute only) regardless of the output path: -1 with
    # yrm (and not yt), 0 never, 1 always (diagnostics: untilize cost with tile writes, row writes without untilize)
    cu: int = -1
    ystate: int = (
        0  # yrm: a row tile's row writes grouped by DRAM bank with the stateful one-packet writes (WL5_YSTATE)
    )
    # yrm: the rows through yc row collectors a group (l5_ycoll.cpp, WL5_YC): spare cores in the muxes' column, which
    # gather each row tile's column slices from the 25 cores and write every row whole; yc_sink: without the row writes
    # (a diagnostic: the cores' traffic and the protocol only)
    yc: int = 0
    yc_sink: int = 0
    yc_sink_mask: int = 0  # yc_sink for the collectors of rectangle r only (bit r; diagnostics)
    # yc: the east half's collectors (DRAM column x 9 east of them) write on NOC0 from the BRISC instead of NOC1 the
    # long way round; outside the NoC rule (the fence's acks return through the rectangles on NOC0), diagnostics
    yc_n0: int = 0
    # the group NCRISCs record what held up the weight forwards (idle loop time by blocker: arrival, downstream slot
    # credit, mux space, slot refill; first / last forward), stamp words 477-483; diagnostics
    why: int = 0
    # lanes onto the group rectangles a chip (place_lanes; hmc only): 0 lane g on rectangle g on every chip; 1 by the
    # chip's ethernet cores, so that the muxes' NOC1 paths to their routers (north, then west along the ethernet row)
    # share as few of the row's links as the chip's ethernet cores allow
    place: int = 1
    # group g: top-left logical core, its muxes' column (west of the group) and rows (direction 0, direction 1). The tray
    # (11 x 10, G 2) uses the first two; a Galaxy chip (12 x 10, logical columns 0-6 and 7-11 NoC contiguous) all four
    origins: tuple = ((1, 0), (1, 5), (7, 0), (7, 5))
    mux_cols: tuple = (0, 0, 6, 6)
    mux_rows: tuple = ((1, 3), (6, 8), (1, 3), (6, 8))


def g5_lane_cores(l5: L5Cfg, g):
    x0, y0 = l5.origins[g]
    return [(x0 + c, y0 + r) for c in range(NCOL) for r in range(NR)]


# Blackhole ethernet channel -> NoC x (the ethernet row, y 1)
BH_ETH_X = (1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10)
NOC_W = 17  # (Blackhole NoC columns: the row wraps)


def eth_x(mesh, mcfg, src_rc, dst_rc, link):
    """NoC x of the ethernet core the fabric uses from chip src_rc towards dst_rc on link (the routing plane the mux
    kernel's runtime args name: after the regions to clear, the connection's ethernet channel)."""
    a = list(
        mcfg.kernel_runtime_args(
            source_node_id=node(mesh, src_rc),
            destination_node_id=node(mesh, dst_rc),
            link_index=link,
            program_descriptor=ttnn.ProgramDescriptor(),
            mux_logical_core=ttnn.CoreCoord(0, 0),
        )
    )
    chan = a[1 + 2 * a[0]]
    assert chan < len(BH_ETH_X), chan
    return BH_ETH_X[chan]


def row_load(flows):
    """(max, total) of the westward NOC1 ethernet-row links the flows (mux x, ethernet x) cross."""
    use = [0] * NOC_W
    for xm, xe in flows:
        x = xm
        while x != xe:
            use[x] += 1
            x = (x - 1) % NOC_W
    return max(use), sum(use)


PLACED = {}  # (the last build's placement: chip -> (lane -> rectangle, lane's ethernet core NoC x a direction))


def place_lanes(mx, ex):
    """A chip's lane -> rectangle assignment: mx[r][d] the NoC x of rectangle r's direction-d mux, ex[g][d] the NoC x
    of the ethernet core lane g's direction d sends on (None: no mux). Every mux writes its packets on NOC1, north to
    the ethernet row, then west along it (wrapping at x 0): the assignment with the fewest flows on the row's busiest
    link, then the fewest link crossings, then the identity."""
    import itertools

    G = len(ex)
    best = None
    for perm in itertools.permutations(range(G)):
        flows = [(mx[perm[g]][d], ex[g][d]) for g in range(G) for d in range(len(ex[g])) if ex[g][d] is not None]
        key = (*row_load(flows), perm != tuple(range(G)))
        if best is None or key < best[0]:
            best = (key, perm)
    return best[1]


def pk_of(cfg, CHT, j):
    """(gate/up packets, packets) of core j's slice."""
    _, _, _, ksl, _, dc, _ = core_geom(cfg, j)
    wgu, wdn = ksl * NW, cfg.PAIRS * dc
    assert wgu % CHT == 0 and wdn % CHT == 0, (j, wgu, wdn, CHT)
    return wgu // CHT, (wgu + wdn) // CHT


def chunk_base(cfg, CHT):
    """Per core j its first chunk within an expert's chunks (PKE chunks an expert, cores in order)."""
    out, acc = [], 0
    for j in range(GC5):
        out.append(acc)
        acc += pk_of(cfg, CHT, j)[1]
    return out, acc


def slice_layout5(cfg, CHT, wg, wu, wd):
    """The chip's own experts' slices in DRAM: expert e, core j: gate/up W_gu[K tiles K0 .. K0 + KSL, interleaved
    columns 8 c .. 8 c + 8] ([KSL][8]) then down columns [DS, DS + DC) ([KTD][DC]); chunk q = e PKE + CH0_j + u of CHT
    tiles, stored as pages (q // 8) CHT 8 + q % 8 + 8 k (one DRAM bank a chunk, one read). [pages * 32, 32] rows."""
    from models.demos.ws_expert.ws_expert import interleave_gu

    E, KT, KTD, HT, P = wg.shape[0], cfg.KT, cfg.PAIRS, cfg.HT, cfg.PAIRS
    tiles = []
    for e in range(E):
        gu = interleave_gu(wg[e], wu[e]).reshape(KT, 32, 2 * P, 32).permute(0, 2, 1, 3)  # [KT, 2P, 32, 32]
        dn = wd[e].reshape(KTD, 32, HT, 32).permute(0, 2, 1, 3)  # [KTD, HT, 32, 32]
        for j in range(GC5):
            c, r, k0, ksl, ds, dc, _ = core_geom(cfg, j)
            tiles.append(gu[k0 : k0 + ksl, NW * c : NW * (c + 1)].reshape(-1, 32, 32))
            tiles.append(dn[:, ds : ds + dc].reshape(-1, 32, 32))
    t = torch.cat(tiles, 0)
    nq = t.shape[0] // CHT
    assert nq * CHT == t.shape[0]
    nq_pad = -(-nq // 8) * 8
    t = torch.cat([t, torch.zeros((nq_pad - nq) * CHT, 32, 32)], 0).reshape(nq_pad, CHT, 32, 32)
    q = torch.arange(nq_pad)
    pages = ((q // 8) * CHT * 8 + q % 8).unsqueeze(1) + 8 * torch.arange(CHT).unsqueeze(0)
    out = torch.empty(nq_pad * CHT, 32, 32)
    out[pages.flatten()] = t.reshape(-1, 32, 32)
    return out.reshape(-1, 32)


def build_lane5(
    mesh,
    cfg: GCfg,
    lc: LCfg,
    l5: L5Cfg,
    x_t,
    idx_t,
    counts_t,
    ws_t,
    out_t,
    tab_t,
    T,
    out_rows,
    dest_t=None,
    fence_row=None,
):
    """generic_op mesh program of the 5 x 5 lane design; returns (mesh program, io tensors, stamps). With l5.yrm, out_t
    is the row-major result buffer [rows, H] and dest_t the destination lists [nexp, RMAX 32] uint32 (out_rows unused);
    fence_row: its spare rows (one a DRAM bank) for the posted row writes' fence."""
    R, Cn = tuple(mesh.shape)
    N, G = R * Cn, cfg.G
    assert cfg.GC == GC5 and cfg.PAIRS == OWN_ROWS * NCOL and cfg.KT == 120 and cfg.HT == 120
    assert lc.out_all, "the 5 x 5 lane writes every expert's rows"
    assert not l5.yrm or (l5.hmc == 2 and dest_t is not None and not l5.oaff), "yrm: hmc 2, dest_t, no oaff"
    assert not l5.yc or (l5.yrm and not l5.yt and not l5.ystate and fence_row is not None), "yc: yrm rows, fenced"
    ln = lanes((R, Cn), lc)
    # (a 2-chip cycle: both directions lead to the same chip over the group's link - one mux a group, WL5_ONEMUX)
    onemux = len(lane_cycles(ln[0])[0]) == 2
    # (one-way lanes: group g's packets on its own mux (direction 0), its credits on header-only channels of its
    # partner g ^ 1's mux - the partner's direction 0 is g's direction 1; one mux a group)
    oneway = lc.oneway
    assert not (oneway and onemux), "one-way lanes need cycles of 3+ chips"
    nmux = 1 if (onemux or oneway) else 2
    groups = [g5_lane_cores(l5, g) for g in range(G)]
    C = [c for gr in groups for c in gr]
    muxes = [(l5.mux_cols[g], l5.mux_rows[g][d]) for g in range(G) for d in range(nmux)]
    assert not set(muxes) & set(C)
    # yc: group g's collectors - the first yc spare cores of its muxes' column within its rows (outside the rectangle;
    # the cores reach them west on NOC1); every collector transfer on NOC1 (the row writes, the destination reads, the
    # slot credits), except with yc_n0 (split: the east half's row writes on NOC0 from the BRISC)
    colls = []
    for g in range(G if l5.yc else 0):
        x0, y0 = l5.origins[g]
        free = [(l5.mux_cols[g], y0 + r) for r in range(NR) if (l5.mux_cols[g], y0 + r) not in muxes]
        assert len(free) >= l5.yc, (g, free)
        colls.append(free[: l5.yc])
    ycores = [xy for cl in colls for xy in cl]
    assert not set(ycores) & set(C)
    KTD, NS, LT, NH = cfg.PAIRS, cfg.NS, 2 * NR, OWN_ROWS * NCOL
    assert NS >= cfg.dn_lag + 1, "h ring below DN_LAG + 1 rows"
    DB = 4
    K, E, TLW = T["K"], T["E"], T["TLW"]
    wtile, htile, xtile, ptile = (TILE_BYTES[d] for d in (cfg.w_dt, cfg.h_dt, cfg.x_dt, ttnn.bfloat16))
    SLOT_MAX = max(row_geom(cfg, r)[1] * NW + KTD * row_geom(cfg, r)[2] for r in range(NR))  # 320 tiles
    U, RM = ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    w16 = lambda n: sharded_l1(mesh, C, [n, 4], U, RM, torch.int32)
    wring = sharded_l1(mesh, C, [K * SLOT_MAX * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    assert wring.buffer_address() % 64 == 0
    warr, cred, freew = w16(K), w16(2 * K), w16(K)
    hring = sharded_l1(mesh, C, [NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    # h flags: a word a (slot, producer) for stage A, a word a (slot, column) for stage B; credits: words 0-4 the
    # column's cores (by row, at the owners), 5-9 the row's cores (by column)
    hflags, hcolf, hcred = w16(NS * NH), w16(NS * NCOL), w16(2 * NR)
    # hmc: the cores' consumed h rows at the hub (a word a core)
    hcons = w16(GC5) if l5.hmc else None
    xring = sharded_l1(mesh, C, [l5.NXR * 25 * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    xflags, xcred = w16(l5.NXR), w16(NCOL)
    land = sharded_l1(mesh, C, [l5.D * LT * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    lflags, lcred = w16(l5.D * NR), w16(NR)
    pring = sharded_l1(mesh, C, [l5.PD * NW * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    scr = w16(16 if l5.hmc else 8)
    # (yrm: the row writes need CB 17 64 B aligned: a sharded tensor of 8 tiles a core - owners 2 x 4, row 4 1 x 8; with
    # yt the same tensor is the BRISC's row-major staging slot, 32 rows x DCOLS 64 B, and CB 17 holds plain tiles)
    YDEP_T = 8
    ysh = sharded_l1(mesh, C, [YDEP_T * 32, 32], cfg.out_dt, ttnn.TILE_LAYOUT) if l5.yrm else None
    counts_l1 = sharded_l1(mesh, C, [1, 32 + max(4, cfg.nexp)], U, RM, torch.int32)
    idx_l1 = sharded_l1(mesh, C, [4, 32], U, RM, torch.int32)
    table_l1 = sharded_l1(mesh, C, [1, TLW + 16], U, RM, torch.int32)
    a64 = lambda a: (a + 63) // 64 * 64
    cnt_base, idx_addr, tab_base = (a64(t.buffer_address()) for t in (counts_l1, idx_l1, table_l1))
    if l5.yc:
        assert cfg.H * 2 == cfg.HT * 64 and cfg.out_dt == ttnn.bfloat16
        SLOTB = 32 * cfg.H * 2
        NSL = min(3, K * SLOT_MAX * wtile // SLOTB)  # (the staging slots fit in the weight ring's range)
        assert NSL >= 2, NSL
        ycw = w16(8)  # (on the group cores: word c the collector c's freed row tiles, word 3 the collectors done)
        ycw_b = a64(ycw.buffer_address())
        yw = sharded_l1(mesh, ycores, [40, 4], U, RM, torch.int32)
        ycnt = sharded_l1(mesh, ycores, [1, 32 + max(4, cfg.nexp)], U, RM, torch.int32)
        ytab = sharded_l1(mesh, ycores, [1, TLW + 16], U, RM, torch.int32)
        yst = stamps_tensor(mesh, ycores)
        yw_b = a64(yw.buffer_address())
    # (a core's stamps: words 0-511 the RISCs', 512 + n the n-th expert's end)
    st = stamps_tensor(mesh, C, words=max(64, -(-(512 + T["E"]) // 128) * 16))
    mcfg = fabric_mux.Config(
        num_full_size_channels=GC5,
        num_header_only_channels=GC5 if oneway else 0,
        num_buffers_per_full_size_channel=lc.nbuf,
        num_buffers_per_header_only_channel=lc.nbuf if oneway else 0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh),
        base_l1_address=ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1),
    )
    ctype = fabric_mux.ChannelType.FULL_SIZE
    hotype = fabric_mux.ChannelType.HEADER_ONLY
    nclients = 2 * GC5 if oneway else GC5  # (a mux's clients: its termination master waits for the rest)
    client_ct = list(fabric_mux.client_compile_time_args(num_clients=nclients, channel_type=ctype, config=mcfg))
    ho_defs = []
    if oneway:
        ho_ct = list(fabric_mux.client_compile_time_args(num_clients=nclients, channel_type=hotype, config=mcfg))
        assert ho_ct[0] == client_ct[0] and ho_ct[2:] == client_ct[2:], (ho_ct, client_ct)
        ho_defs = [("WL5_HO_BUFSZ", str(ho_ct[1]))]
    mesh_id = int(node(mesh, (0, 0)).mesh_id)
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
        ("DN_LAG", str(cfg.dn_lag)),
    ]
    ch0, PKE = chunk_base(cfg, lc.CHT)
    TB = TLW * 4
    own_sets = {own: [xy for gr in groups for j, xy in enumerate(gr) if core_geom(cfg, j)[6] == own] for own in (1, 0)}

    # (lane g's neighbours of chip rc: direction 0's next chip, direction 1's)
    def nbrs(g, rc):
        cyc = [tuple(v) for v in lane_cycle(ln[g], rc)]
        i = cyc.index(rc)
        return cyc[(i + 1) % len(cyc)], cyc[(i - 1) % len(cyc)]

    # lane g's cores on chip rc: rectangle perm[rc][g] (L5Cfg.place); a lane's core j sends to / credits the core j of
    # the same lane on the next / previous chip, wherever that chip put the lane
    chips = [(r, c) for r in range(R) for c in range(Cn)]
    perm = {rc: tuple(range(G)) for rc in chips}
    if l5.place and l5.hmc:  # (l5h_reader.cpp takes the remote cores' coordinates; l5_reader.cpp its own)
        mx = [[noc(mesh, l5.mux_cols[q], l5.mux_rows[q][d])[0] for d in range(nmux)] for q in range(G)]
        for rc in chips:
            ex = [[eth_x(mesh, mcfg, rc, nbrs(g, rc)[d], ln[g][1]) for d in range(nmux)] for g in range(G)]
            perm[rc] = place_lanes(mx, ex)
            PLACED[rc] = (perm[rc], ex)
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            me_rc = (r, c)
            pm = perm[me_rc]
            gl = [groups[pm[g]] for g in range(G)]  # (lane g's cores on this chip)
            mux_at = lambda g, d: (l5.mux_cols[pm[g]], l5.mux_rows[pm[g]][d])
            cll = [colls[pm[g]] for g in range(G)] if l5.yc else []
            kernels, rd_idx = [], {}
            nxt = {}
            hmc_rt = {}
            for g, gr in enumerate(gl):
                link = ln[g][1]
                cyc = [tuple(v) for v in lane_cycle(ln[g], me_rc)]
                i = cyc.index(me_rc)
                nxt[g] = (cyc[(i + 1) % len(cyc)], cyc[(i - 1) % len(cyc)])
                nid = [int(node(mesh, nxt[g][d]).chip_id) for d in range(2)]
                assert max(nid) < 256

                def rxy(d, j, g=g):  # (lane g's core j on the direction-d neighbour, x | y << 8, above the chip id)
                    q = noc(mesh, *groups[perm[nxt[g][d]][g]][j])
                    return (q[0] | q[1] << 8) << 8

                cx = [noc(mesh, *gr[NR * k])[0] for k in range(NCOL)]
                ry = [noc(mesh, *gr[k])[1] for k in range(NR)]
                for own in (1, 0):
                    sub = [(j, xy) for j, xy in enumerate(gr) if core_geom(cfg, j)[6] == own]
                    _, ksl, dc = row_geom(cfg, 0 if own else OWN_ROWS)
                    wr_rt, rd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                    for j, (x, y) in sub:
                        _, _, k0, _, ds, _, _ = core_geom(cfg, j)
                        xy = noc(mesh, x, y)
                        wr_rt[x][y] = (
                            [j, g, E, x_t.buffer_address(), idx_t.buffer_address()]
                            + [
                                counts_t.buffer_address(),
                                st.buffer_address(),
                                k0,
                                tab_t.buffer_address(),
                                ws_t.buffer_address(),
                                ch0[j],
                            ]
                            + cx
                            + ry
                        )
                        rd_rt[x][y] = (
                            [j, g, E, st.buffer_address(), out_t.buffer_address(), ds, mesh_id]
                            + [
                                nid[0] | rxy(0, j),
                                nid[1] | rxy(1, j),
                                xy[0] | xy[1] << 8,
                            ]
                            + cx
                            + ry
                        )
                    cs = crs([xy for _, xy in sub])
                    if l5.hmc:
                        for j, (x, y) in sub:
                            _, _, k0, _, _, _, _ = core_geom(cfg, j)
                            hmc_rt[(x, y)] = [x_t.buffer_address(), idx_t.buffer_address()]
                            hmc_rt[(x, y)] += [counts_t.buffer_address(), k0, tab_t.buffer_address()]
                            hmc_rt[(x, y)] += [ws_t.buffer_address(), ch0[j]]
                            hmc_rt[(x, y)] += [dest_t.buffer_address()] if l5.yrm else []
                            hmc_rt[(x, y)] += (
                                [(lambda q: q[0] | q[1] << 8)(noc(mesh, *xy)) for xy in cll[g]] if l5.yc else []
                            )
                    if l5.hmc == 2:
                        for j, (x, y) in sub:
                            (hx0, hy0), (hx1, hy1) = noc(mesh, *gr[0]), noc(mesh, *gr[-1])
                            assert (hx1 - hx0, hy1 - hy0) == (NCOL - 1, NR - 1), "a group rectangle NoC contiguous"
                            wr_rt[x][y].extend([hx0, hy0, hx1, hy1])
                    if l5.hmc != 1:
                        kernels.append(
                            ttnn.KernelDescriptor(
                                kernel_source=kpath("l5h_writer.cpp" if l5.hmc == 2 else "l5_writer.cpp"),
                                core_ranges=cs,
                                compile_time_args=[
                                    ksl,
                                    dc,
                                    own,
                                    l5.NXR,
                                    xring.buffer_address(),
                                    xflags.buffer_address(),
                                ]
                                + [xcred.buffer_address(), xtile, l5.D, land.buffer_address(), lflags.buffer_address()]
                                + [lcred.buffer_address(), ptile, scr.buffer_address(), KTD, NS, hring.buffer_address()]
                                + [
                                    hflags.buffer_address(),
                                    htile,
                                    cnt_base,
                                    tab_base,
                                    TB,
                                    cfg.RMAX,
                                    cfg.H * 2,
                                    idx_addr,
                                ]
                                + [cfg.nexp, wring.buffer_address(), warr.buffer_address(), freew.buffer_address()]
                                + [lc.CHT, wtile, l5.RD, hcred.buffer_address(), K, PKE]
                                + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                                + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                                + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args())
                                + list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
                                + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args())
                                + ([hcons.buffer_address(), hcolf.buffer_address()] if l5.hmc == 2 else [])
                                + ([ysh.buffer_address()] if l5.yrm and l5.yt else []),
                                defines=([("WL5_GATHER0", "1")] if l5.hmc == 2 and l5.gather0 else [])
                                + ([("WL5_YT", "1")] if l5.yrm and l5.yt else []),
                                runtime_args=wr_rt,
                                config=dm(0),
                            )
                        )
                    rd_idx[(g, own)] = len(kernels)
                    hmc_ct = []
                    if l5.hmc:
                        hmc_ct = [xring.buffer_address(), xtile, TB, cfg.RMAX, cfg.H * 2, idx_addr, cfg.nexp, PKE]
                        hmc_ct += [l5.RD, hcons.buffer_address()]
                        for t in (x_t, idx_t, counts_t, tab_t, ws_t) + ((dest_t,) if l5.yrm else ()):
                            hmc_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=kpath("l5h_reader.cpp" if l5.hmc else "l5_reader.cpp"),
                            core_ranges=cs,
                            compile_time_args=[ksl, dc, own, l5.NXR, xflags.buffer_address(), xcred.buffer_address()]
                            + [l5.D, land.buffer_address(), lflags.buffer_address(), lcred.buffer_address(), ptile]
                            + [scr.buffer_address(), KTD, NS, hflags.buffer_address(), cnt_base, tab_base]
                            + [warr.buffer_address(), out_rows, cfg.HT, wring.buffer_address(), cred.buffer_address()]
                            + [freew.buffer_address(), lc.CHT, wtile, l5.PD, pring.buffer_address()]
                            + [hcred.buffer_address(), K, hcolf.buffer_address(), hring.buffer_address(), htile]
                            + client_ct
                            + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
                            + hmc_ct,
                            runtime_args=rd_rt,
                            defines=FAB_DEFS
                            + ([("WL5_OAFF", str(l5.oaff))] if l5.oaff else [])
                            + ([("WL5_Y_RM", "1")] if l5.yrm else [])
                            + ([("WL5_YSTATE", "1")] if l5.yrm and l5.ystate else [])
                            + ([("WL5_YT", str(ysh.buffer_address()))] if l5.yrm and l5.yt else [])
                            + ([("WL5_Y_FENCE", str(fence_row))] if l5.yrm and fence_row is not None else [])
                            + (
                                [("WL5_YC", str(l5.yc)), ("WL5_YC_NSL", str(NSL)), ("WL5_YC_ARR", str(yw_b))]
                                + [("WL5_YC_CRED", str(ycw_b))]
                                if l5.yc
                                else []
                            )
                            + ([("WL5_HMC2", "1")] if l5.hmc == 2 else [])
                            + ([("WL5_ONEMUX", "1")] if onemux else [])
                            + ([("WL5_WHY", "1")] if l5.why else [])
                            + ho_defs
                            + ([("WL5_GATHER0", "1")] if l5.hmc == 2 and l5.gather0 else []),
                            config=dm(1),
                        )
                    )
            if l5.hmc == 1:  # the hubs' multicasters (core 0 of each group)
                hub_rt = ttnn.RuntimeArgs()
                for g, gr in enumerate(gl):
                    (hx0, hy0), (hx1, hy1) = noc(mesh, *gr[0]), noc(mesh, *gr[-1])
                    assert (hx1 - hx0, hy1 - hy0) == (NCOL - 1, NR - 1), "a group rectangle NoC contiguous"
                    hub_rt[gr[0][0]][gr[0][1]] = [st.buffer_address(), E, hx0, hy0, hx1, hy1, GC5]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("l5h_hub.cpp"),
                        core_ranges=crs([gr[0] for gr in gl]),
                        compile_time_args=[NS, KTD, hring.buffer_address(), hflags.buffer_address()]
                        + [hcons.buffer_address(), hcolf.buffer_address(), htile, scr.buffer_address(), cnt_base]
                        + [tab_base],
                        runtime_args=hub_rt,
                        config=dm(0),
                    )
                )
            if l5.yc:
                y_ct = [NSL, cfg.H * 2, wring.buffer_address(), yw_b, ycw_b, cfg.RMAX, cfg.nexp, TB]
                y_ct += [a64(ycnt.buffer_address()), a64(ytab.buffer_address()), GC5, 0, l5.yc_sink]
                for t in (counts_t, tab_t, dest_t, out_t):
                    y_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
                sunk = {xy for r, cl in enumerate(colls) if l5.yc_sink_mask >> r & 1 for xy in cl}
                y_rt = {k: ttnn.RuntimeArgs() for k in (0, 1)}
                w_rt, split_cores = {k: ttnn.RuntimeArgs() for k in (0, 1)}, []
                for g, cl in enumerate(cll):
                    split = 1 if l5.yc_n0 and mux_at(g, 0)[0] != 0 else 0
                    for ci, (x, y) in enumerate(cl):
                        args = [g, ci, l5.yc, E, counts_t.buffer_address(), tab_t.buffer_address()]
                        args += [
                            dest_t.buffer_address(),
                            out_t.buffer_address(),
                            fence_row,
                            yst.buffer_address(),
                            split,
                        ]
                        args += [(lambda q: q[0] | q[1] << 8)(noc(mesh, *xy)) for xy in gl[g]]
                        y_rt[(x, y) in sunk][x][y] = args
                        if split:
                            w_rt[(x, y) in sunk][x][y] = list(args)
                            split_cores.append((x, y))
                for sink in (0, 1):
                    yset = [xy for xy in ycores if (xy in sunk) == bool(sink)]
                    if not yset:
                        continue
                    ct = y_ct[:12] + [l5.yc_sink or sink] + y_ct[13:]
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=kpath("l5_ycoll.cpp"),
                            core_ranges=crs(yset),
                            compile_time_args=ct,
                            runtime_args=y_rt[sink],
                            config=dm(1),
                        )
                    )
                    wset = [xy for xy in split_cores if (xy in sunk) == bool(sink)]
                    if wset:
                        kernels.append(
                            ttnn.KernelDescriptor(
                                kernel_source=kpath("l5_ycoll.cpp"),
                                core_ranges=crs(wset),
                                compile_time_args=ct,
                                defines=[("YC_WRITER", "1")],
                                runtime_args=w_rt[sink],
                                config=dm(0),
                            )
                        )
            cp_rt = {own: ttnn.RuntimeArgs() for own in (1, 0)}
            for g, gr in enumerate(gl):
                for j, (x, y) in enumerate(gr):
                    cp_rt[core_geom(cfg, j)[6] and 1 or 0][x][y] = [st.buffer_address(), j, g, E]
            cbs = []
            for own in (1, 0):
                cs = crs(own_sets[own])
                _, ksl, dc = row_geom(cfg, 0 if own else OWN_ROWS)
                wgu, wdn = ksl * NW, KTD * dc
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("l5_compute.cpp"),
                        core_ranges=cs,
                        compile_time_args=[ksl, KTD, dc, DB, cnt_base, K, NW, l5.FL, l5.TL, own, tab_base + 64],
                        named_compile_time_args=[("expert_activation", 0)],
                        defines=act_defs
                        + ([("WL_Y_RM", "1")] if (l5.cu == 1 or (l5.cu < 0 and l5.yrm and not l5.yt)) else [])
                        + ([("WL5_REMAP_ROW", "1")] if l5.yrm and l5.remap_row else []),
                        runtime_args=cp_rt[own],
                        config=ccfg,
                    )
                )
                sh = ttnn.cb_descriptor_from_sharded_tensor
                cbs += [
                    sh(0, xring, total_size=l5.NXR * ksl * xtile, core_ranges=cs),
                    sh(1, wring, total_size=K * wgu * wtile, core_ranges=cs),
                    sh(3, wring, address_offset=K * wgu * wtile, total_size=K * wdn * wtile, core_ranges=cs),
                    cb(7, ttnn.bfloat16, cfg.XRM * ksl, cs),
                    cb(8, cfg.x_dt, (l5.TL // NCOL + 3) * ksl, cs),
                    (
                        sh(17, ysh, total_size=YDEP_T * TILE_BYTES[cfg.out_dt], core_ranges=cs)
                        if l5.yrm and not l5.yt
                        else cb(17, cfg.out_dt, (YDEP_T // dc if l5.yt else 2) * dc, cs)
                    ),
                ]
            cset = crs(C)
            cbs += [
                ttnn.cb_descriptor_from_sharded_tensor(2, hring),
                ttnn.cb_descriptor_from_sharded_tensor(4, land),
                cb(16, cfg.h_dt, 2, cset),
                ttnn.cb_descriptor_from_sharded_tensor(18, pring),
            ]
            mux_n1 = lambda g: (
                (l5.mux_mask >> g) & 1 if l5.mux_mask >= 0 else (l5.mux_noc == 1 or (l5.mux_noc < 0 and l5.hmc))
            )
            mux_idx = {}
            for n1 in (0, 1):
                ms = [mux_at(g, d) for g in range(G) for d in range(nmux) if bool(mux_n1(g)) == bool(n1)]
                if not ms:
                    continue
                mrt = ttnn.RuntimeArgs()
                for m in ms:
                    mrt[m[0]][m[1]] = []
                    mux_idx[m] = len(kernels)
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source="tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                        core_ranges=crs(ms),
                        compile_time_args=list(mcfg.kernel_compile_time_args()),
                        runtime_args=mrt,
                        defines=FAB_DEFS,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0,
                            noc=ttnn.NOC.RISCV_1_default if n1 else ttnn.NOC.RISCV_0_default,
                        ),
                        opt_level=fabric_mux.KernelBuildOptLevel.O3,
                    )
                )
            sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(2)]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            for g, gr in enumerate(gl):
                link = ln[g][1]
                for d in range(nmux):
                    m = mux_at(g, d)
                    p.kernels[mux_idx[m]].runtime_args[m[0]][m[1]].extend(
                        mcfg.kernel_runtime_args(
                            source_node_id=node(mesh, me_rc),
                            destination_node_id=node(mesh, nxt[g][d]),
                            link_index=link,
                            program_descriptor=p,
                            mux_logical_core=ttnn.CoreCoord(*m),
                        )
                    )
            # (the full-size clients of every group's direction-0 mux first: one-way lanes' header-only clients on the
            # partner's mux take its termination master and semaphore)
            fs_args, fs_sem = {}, {}
            for g, gr in enumerate(gl):
                m = mux_at(g, 0)
                mux_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*m))
                master_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gr[0]))
                term_sem = None
                for j, (x, y) in enumerate(gr):
                    args = fabric_mux.client_runtime_args(
                        connection_valid=True,
                        is_termination_master=j == 0,
                        channel_type=ctype,
                        mux_virtual_core=mux_vc,
                        client_index=j,
                        client_logical_core=ttnn.CoreCoord(x, y),
                        config=mcfg,
                        program_descriptor=p,
                        termination_master_virtual_core=master_vc,
                        termination_master_semaphore_id=term_sem,
                    )
                    if j == 0:
                        term_sem = args[10]
                    fs_args[(g, j)] = list(args)
                fs_sem[g] = term_sem
            for g, gr in enumerate(gl):
                d0_args = {}
                for d in range(2):
                    if d == 0:
                        for j, (x, y) in enumerate(gr):
                            own = 1 if core_geom(cfg, j)[6] else 0
                            p.kernels[rd_idx[(g, own)]].runtime_args[x][y].extend(fs_args[(g, j)])
                            d0_args[j] = fs_args[(g, j)]
                        continue
                    if oneway:  # (credits: header-only channel j of the partner's mux)
                        pg = g ^ 1
                        assert ln[pg][1] == ln[g][1], "partner lanes on one link"
                        m = mux_at(pg, 0)
                        mux_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*m))
                        master_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gl[pg][0]))
                        for j, (x, y) in enumerate(gr):
                            args = fabric_mux.client_runtime_args(
                                connection_valid=True,
                                is_termination_master=False,
                                channel_type=hotype,
                                mux_virtual_core=mux_vc,
                                client_index=j,
                                client_logical_core=ttnn.CoreCoord(x, y),
                                config=mcfg,
                                program_descriptor=p,
                                termination_master_virtual_core=master_vc,
                                termination_master_semaphore_id=fs_sem[pg],
                            )
                            own = 1 if core_geom(cfg, j)[6] else 0
                            p.kernels[rd_idx[(g, own)]].runtime_args[x][y].extend(args)
                        continue
                    if d >= nmux:  # (onemux: direction 1's RT args are direction 0's, unused)
                        for j, (x, y) in enumerate(gr):
                            own = 1 if core_geom(cfg, j)[6] else 0
                            p.kernels[rd_idx[(g, own)]].runtime_args[x][y].extend(d0_args[j])
                        continue
                    m = mux_at(g, d)
                    mux_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*m))
                    master_vc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gr[0]))
                    term_sem = None
                    for j, (x, y) in enumerate(gr):
                        args = fabric_mux.client_runtime_args(
                            connection_valid=True,
                            is_termination_master=j == 0,
                            channel_type=ctype,
                            mux_virtual_core=mux_vc,
                            client_index=j,
                            client_logical_core=ttnn.CoreCoord(x, y),
                            config=mcfg,
                            program_descriptor=p,
                            termination_master_virtual_core=master_vc,
                            termination_master_semaphore_id=term_sem,
                        )
                        if j == 0:
                            term_sem = args[10]
                        d0_args[j] = list(args)
                        own = 1 if core_geom(cfg, j)[6] else 0
                        p.kernels[rd_idx[(g, own)]].runtime_args[x][y].extend(args)
            if l5.hmc:  # (l5h_reader.cpp's extra RT args, after both directions' mux client args)
                for g, gr in enumerate(gl):
                    for j, (x, y) in enumerate(gr):
                        own = 1 if core_geom(cfg, j)[6] else 0
                        p.kernels[rd_idx[(g, own)]].runtime_args[x][y].extend(hmc_rt[(x, y)])
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    io = [x_t, idx_t, counts_t, ws_t, tab_t, wring, warr, cred, freew, hring, hflags, hcolf, hcred, xring, xflags]
    io += [xcred]
    io += [land, lflags, lcred, pring, scr, counts_l1, idx_l1, table_l1, st, out_t] + ([hcons] if l5.hmc else [])
    io += [ysh, dest_t] if l5.yrm else []
    io += [ycw, yw, ycnt, ytab, yst] if l5.yc else []  # (yc: the collectors' stamps last)
    return mpd, io, st
