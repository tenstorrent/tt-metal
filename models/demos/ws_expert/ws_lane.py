# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ws_expert lane design: every chip runs the G groups on its own tokens; the weights of group g's experts travel lane
g (one Hamiltonian cycle and link, both directions), forwarded by the group cores themselves. Core j of a group sends
its slice of an expert one hop, packet by packet, into the same core of the next chip through the group's mux core of
that direction (fabric_mux: one full-size channel a core), straight into a weight slot. Every group core has 3 weight
slots at the same address on every chip: slot 0 takes experts arriving on direction 0, slot 1 on direction 1, slot 2
any, including the chip's own experts, which the core's BRISC reads from DRAM. An expert's down projection is drained
before the next expert starts, so a slot frees once its expert is computed and forwarded on.

The order of everything comes from a static schedule (ws_lane_sched.lane_sched, the same on all lanes): per chip and
slot the experts it holds in turn, per chip the compute order. The group cores follow it through a table per chip and
group (make_tables, layout in kernels/wl_reader.cpp); a slot's next occupant is only sent (or read) once the slot freed
the previous one: a credit from the receiving core to the sending one, one hop back, through the other direction's
mux. Every wait is on an event earlier in the schedule's model time, so the lanes cannot deadlock (ws_lane_sched).

Core layout: group g in logical columns LANE_PAIRS[g] (contiguous in NoC x), its two mux cores in the column just west
of it (MUX_COL[g], rows MUX_ROWS): the cores reach their muxes on NOC1 (north, then west) without entering another
group's columns, whose linked multicasts run on both NoCs.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane_sched import lane_sched
from models.demos.ws_expert.ws_sf import snake

LANE_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11)]
MUX_COL = [0, 3, 6, 9]
MUX_ROWS = (4, 5)  # direction 0, direction 1
KIND_OWN, KIND_D0, KIND_D1 = 1, 2, 3
fabric_mux = ttnn.experimental.fabric_mux


@dataclass
class LCfg:
    CHT: int = 24  # tiles a packet (24 x 576 B bfp4 = 13.8 KB): 15 packets a slice, the first 10 gate / up
    NL: int = 2  # links a cycle edge (lanes = cycles x links)
    cycles: tuple = None  # Hamiltonian cycles (mesh coordinates in order); default: the row snake
    nbuf: int = 2  # mux buffers a channel
    D: int = 4  # own-expert chunk reads in flight
    keep: int = 1  # schedule: slots an owner leaves free when it injects
    out_all: bool = False  # every expert's down rows to DRAM (WL_OUT_ALL), not only each group's last
    wpoll: int = 2  # the reader serves its weight queues every wpoll turns of its loop (WL_WPOLL)


def lane_cores(g):
    a, b = LANE_PAIRS[g]
    return [(a, y) for y in range(10)] + [(b, y) for y in range(10)]  # core j = c * 10 + y


def mux_core(g, d):
    return (MUX_COL[g], MUX_ROWS[d])


def lanes(shape, lc: LCfg):
    """Lane g: (cycle as a list of mesh coordinates, link index); direction 0 follows the list, 1 goes back."""
    R, Cn = shape
    cycles = [list(c) for c in (lc.cycles or [snake(shape)])]
    out = []
    for cyc in cycles:
        assert sorted(cyc) == sorted((r, c) for r in range(R) for c in range(Cn)), "a cycle visits every chip once"
        for link in range(lc.NL):
            out.append((cyc, link))
    return out


def slot_packets(cfg: GCfg, lc: LCfg):
    WGU, WDN = 2 * cfg.KT, cfg.PAIRS * cfg.DCOLS
    assert WGU % lc.CHT == 0 and WDN % lc.CHT == 0
    return WGU // lc.CHT, (WGU + WDN) // lc.CHT  # (gate / up packets, packets) a slice


def schedule(N, E, cfg: GCfg, lc: LCfg):
    GU, P = slot_packets(cfg, lc)
    return lane_sched(N, E, P=P, GU=GU, keep=lc.keep)


def make_tables(shape, cfg: GCfg, lc: LCfg):
    """Per chip and group the table of kernels/wl_reader.cpp, and per chip its own experts (global ids, in the order
    of their local index in the chip's slices tensor). Lane g holds the experts e = i G + g (lane-local i)."""
    R, Cn = shape
    N, G = R * Cn, cfg.G
    ln = lanes(shape, lc)
    assert len(ln) >= G, f"{len(ln)} lanes for {G} groups"
    assert cfg.nexp % G == 0
    E = cfg.nexp // G
    sch = schedule(N, E, cfg, lc)
    owners, kf = sch["owners"], sch["kf"]
    TLW = -(-(16 + 5 * E) // 16) * 16  # words a table (64 B multiple)
    tabs = torch.zeros(N, G, TLW, dtype=torch.int64)
    own = [[] for _ in range(N)]  # chip -> [(g, i)]
    meta = {}
    for g in range(G):
        cyc = ln[g][0]
        chip_at = [r * Cn + c for r, c in cyc]
        for pos in range(N):
            for i in range(E):
                if owners[i] == pos:
                    own[chip_at[pos]].append((g, i))
    own = [sorted(o) for o in own]
    own_idx = {(X, g, i): k for X in range(N) for k, (g, i) in enumerate(own[X])}

    def arrives(i, pos):
        """(kind, directions it goes on from pos) of lane-local expert i at position pos."""
        o = owners[i]
        if o == pos:
            return KIND_OWN, [d for d, n in ((0, kf[i]), (1, N - 1 - kf[i])) if n > 0]
        h0 = (pos - o) % N  # hops on direction 0
        if 1 <= h0 <= kf[i]:
            return KIND_D0, [0] if h0 < kf[i] else []
        h1 = (o - pos) % N
        assert 1 <= h1 <= N - 1 - kf[i]
        return KIND_D1, [1] if h1 < N - 1 - kf[i] else []

    where = {}  # (pos, i) -> (slot, occupant index)
    for (pos, s), lst in sch["held"].items():
        for k, i in enumerate(lst):
            assert (pos, i) not in where, "an expert is held once a chip"
            where[(pos, i)] = (s, k)
    for g in range(G):
        cyc = ln[g][0]
        chip_at = [r * Cn + c for r, c in cyc]
        for pos in range(N):
            X = chip_at[pos]
            t = tabs[X, g]
            t[0] = E
            off = 16 + E
            for s in range(3):
                lst = sch["held"][(pos, s)]
                t[1 + s] = len(lst)
                t[4 + s] = off
                for k, i in enumerate(lst):
                    kind, dirs = arrives(i, pos)
                    nxt = 0
                    if k + 1 < len(lst):
                        nkind = arrives(lst[k + 1], pos)[0]
                        nxt = 1 if nkind == KIND_OWN else (2 if nkind == KIND_D0 else 3)
                    w0 = (i * G + g) | kind << 16 | sum(1 << d for d in dirs) << 18 | nxt << 20
                    w = [w0, 0, 0, own_idx.get((X, g, i), 0) if kind == KIND_OWN else 0]
                    for d in dirs:
                        y = (pos + 1) % N if d == 0 else (pos - 1) % N
                        s2, k2 = where[(y, i)]
                        assert arrives(i, y)[0] == (KIND_D0 if d == 0 else KIND_D1)
                        w[1 + d] = s2 | k2 << 2
                    t[off + 4 * k : off + 4 * k + 4] = torch.tensor(w)
                off += 4 * len(lst)
            assert off <= TLW
            order = sch["order"][pos]
            assert sorted(order) == list(range(E)), "every expert computed once a chip"
            for n, i in enumerate(order):
                t[16 + n] = (i * G + g) | where[(pos, i)][0] << 16
            meta[(X, g)] = dict(pos=pos, order=[i * G + g for i in order])
    return dict(tabs=tabs, TLW=TLW, own=[[i * G + g for g, i in o] for o in own], meta=meta, sch=sch, E=E)


def tables_tensor(mesh, tabs):
    N, G, TLW = tabs.shape
    return ttnn.from_torch(
        tabs.reshape(N * G, TLW).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def build_lane(mesh, cfg: GCfg, lc: LCfg, x_t, idx_t, counts_t, ws_t, out_t, tab_t, T, out_rows):
    """generic_op mesh program of the lane design; returns (mesh program, io tensors, stamps). T: make_tables();
    out_rows: tile rows of an output block (a group's last expert, or with out_all a group's experts)."""
    R, Cn = tuple(mesh.shape)
    N, G, GC = R * Cn, cfg.G, cfg.GC
    assert GC == 20 and cfg.PAIRS == GC and cfg.DCOLS * GC == cfg.HT
    ln = lanes((R, Cn), lc)
    groups = [lane_cores(g) for g in range(G)]
    C = [c for gr in groups for c in gr]
    muxes = [mux_core(g, d) for g in range(G) for d in range(2)]
    assert not set(muxes) & set(C)
    gx, gy = mesh.compute_with_storage_grid_size().x, mesh.compute_with_storage_grid_size().y
    assert all(x < gx and y < gy for x, y in C + muxes), f"core layout outside the {gx} x {gy} grid"
    cset = crs(C)
    KT, KTD, DC, BK, NX, NS = cfg.KT, cfg.PAIRS, cfg.DCOLS, cfg.BK, cfg.NX, cfg.NS
    DB = 3
    WGU, WDN = KT * 2, KTD * DC
    wtile, htile, xtile = TILE_BYTES[cfg.w_dt], TILE_BYTES[cfg.h_dt], TILE_BYTES[cfg.x_dt]
    E, TLW = T["E"], T["TLW"]
    # L1 tensors (same address on every core and chip); the weight slots first, at the top of L1: the BRISC reads
    # whole chunks from DRAM straight into them (64 B aligned)
    wgu_ring = sharded_l1(mesh, C, [3 * WGU * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    wd_ring = sharded_l1(mesh, C, [3 * WDN * 32, 32], cfg.w_dt, ttnn.TILE_LAYOUT)
    assert wgu_ring.buffer_address() % 64 == 0 and wd_ring.buffer_address() % 64 == 0, "weight slots 64 B aligned"
    w16 = lambda n: sharded_l1(mesh, C, [n, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    warr, cred, freew = w16(3), w16(6), w16(3)
    hring = sharded_l1(mesh, C, [NS * KTD * 32, 32], cfg.h_dt, ttnn.TILE_LAYOUT)
    hflags = w16(NS * 2)
    xring = sharded_l1(mesh, C, [NX * BK * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    xflags = w16(NX)
    xprog = w16(GC)
    # DRAM -> L1 reads need the L1 address congruent to the DRAM one mod 64: 64-aligned landing spots inside these
    counts_l1 = sharded_l1(mesh, C, [1, 32 + max(4, cfg.nexp)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    idx_l1 = sharded_l1(mesh, C, [4, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    table_l1 = sharded_l1(mesh, C, [1, TLW + 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    a64 = lambda a: (a + 63) // 64 * 64
    cnt_base, idx_addr, tab_base = a64(counts_l1.buffer_address()), a64(idx_l1.buffer_address()), a64(
        table_l1.buffer_address()
    )
    st = stamps_tensor(mesh, C)
    mcfg = fabric_mux.Config(
        num_full_size_channels=GC,
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=lc.nbuf,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh),
        base_l1_address=ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1),
    )
    ctype = fabric_mux.ChannelType.FULL_SIZE
    client_ct = list(fabric_mux.client_compile_time_args(num_clients=GC, channel_type=ctype, config=mcfg))
    mesh_id = int(node(mesh, (0, 0)).mesh_id)
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    ccfg.unpack_to_dest_mode = utd
    act_defs = [
        ("EXPERT_ACTIVATION_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
        ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
        ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
    ]
    ydefs = [("WG_XORDER", str(cfg.xorder)), ("WG_Y_NCRISC", "1")] + ([("WL_OUT_ALL", "1")] if lc.out_all else [])
    XST = -(-cfg.LOOK // GC) + 4
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            me_rc, me = (r, c), r * Cn + c
            kernels, rd_idx = [], []
            nxt = {}
            for g, gr in enumerate(groups):
                cyc, link = ln[g]
                i = cyc.index(me_rc)
                nxt[g] = (cyc[(i + 1) % N], cyc[(i - 1) % N])
                nid = [int(node(mesh, nxt[g][d]).chip_id) for d in range(2)]
                rs = noc_rects(mesh, gr)
                assert len(rs) == 1, rs
                r0 = list(rs[0][:4])
                r_noc1 = [r0[2], r0[3], r0[0], r0[1]]
                cx0, ny0 = noc(mesh, *gr[0])
                cx1, _ = noc(mesh, *gr[10])
                cp_rt, rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
                for j, (x, y) in enumerate(gr):
                    assert noc(mesh, x, y) == ((cx0, cx1)[j // 10], ny0 + j % 10)
                    xy = noc(mesh, x, y)
                    cp_rt[x][y] = [st.buffer_address(), j, g, E]
                    wr_rt[x][y] = [j, g, E, cx0, cx1, ny0, out_t.buffer_address(), x_t.buffer_address()] + [
                        idx_t.buffer_address(),
                        counts_t.buffer_address(),
                        st.buffer_address(),
                        tab_t.buffer_address(),
                        ws_t.buffer_address(),
                    ]
                    rd_rt[x][y] = [j, g, E, cx0, cx1, ny0, st.buffer_address(), out_t.buffer_address(), mesh_id] + [
                        nid[0],
                        nid[1],
                        xy[0] | xy[1] << 8,
                    ]
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("wl_writer.cpp"),
                        core_ranges=crs(gr),
                        compile_time_args=[KT, BK, KTD, DC, GC, G, NS, hring.buffer_address(), hflags.buffer_address()]
                        + r0
                        + [htile, cfg.HT, cfg.RMAX, cfg.H * 2, cnt_base, idx_addr, cfg.nexp]
                        + [NX, xring.buffer_address(), xflags.buffer_address(), xprog.buffer_address(), xtile]
                        + [tab_base, TLW * 4, wgu_ring.buffer_address(), wd_ring.buffer_address()]
                        + [warr.buffer_address(), freew.buffer_address(), lc.CHT, wtile, lc.D]
                        + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
                        + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args()),
                        runtime_args=wr_rt,
                        defines=ydefs,
                        config=dm(0),
                    )
                )
                rd_idx.append(len(kernels))
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("wl_reader.cpp"),
                        core_ranges=crs(gr),
                        compile_time_args=[KT, BK, KTD, DC, GC, G, NS, NX, hflags.buffer_address()]
                        + [xring.buffer_address(), xflags.buffer_address(), xprog.buffer_address()]
                        + r_noc1
                        + [xtile, cnt_base, tab_base, hring.buffer_address(), htile, warr.buffer_address()]
                        + [out_rows, cfg.HT, wgu_ring.buffer_address(), wd_ring.buffer_address()]
                        + [cred.buffer_address(), freew.buffer_address(), lc.CHT, wtile]
                        + client_ct
                        + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args()),
                        runtime_args=rd_rt,
                        defines=ydefs + FAB_DEFS + [("WL_WPOLL", str(lc.wpoll))],
                        config=dm(1),
                    )
                )
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kpath("wl_compute.cpp"),
                        core_ranges=crs(gr),
                        compile_time_args=[KT, BK, KTD, DC, DB, NX, cfg.LOOK, cnt_base, G, GC, tab_base + 64],
                        named_compile_time_args=[("expert_activation", 0)],
                        defines=act_defs
                        + [("DN_LAG", str(cfg.dn_lag)), ("WC_GLU", str(cfg.glu))]
                        + ([("WX_X_FMT_DIFFERS", "1")] if cfg.x_dt != cfg.h_dt else []),
                        runtime_args=cp_rt,
                        config=ccfg,
                    )
                )
            mux_idx = len(kernels)
            mrt = ttnn.RuntimeArgs()
            for m in muxes:
                mrt[m[0]][m[1]] = []
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source="tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    core_ranges=crs(muxes),
                    compile_time_args=list(mcfg.kernel_compile_time_args()),
                    runtime_args=mrt,
                    defines=FAB_DEFS,
                    config=dm(0),
                    opt_level=fabric_mux.KernelBuildOptLevel.O3,
                )
            )
            cbs = [
                ttnn.cb_descriptor_from_sharded_tensor(0, xring),
                ttnn.cb_descriptor_from_sharded_tensor(1, wgu_ring),
                ttnn.cb_descriptor_from_sharded_tensor(2, hring),
                ttnn.cb_descriptor_from_sharded_tensor(3, wd_ring),
                cb(7, ttnn.bfloat16, cfg.XRM * BK, cset),
                cb(8, cfg.x_dt, XST * BK, cset),
                cb(16, cfg.h_dt, 2, cset),
                cb(17, cfg.out_dt, 2 * DC, cset),
            ]
            sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(6)]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            for g, gr in enumerate(groups):
                cyc, link = ln[g]
                for d in range(2):
                    m = mux_core(g, d)
                    p.kernels[mux_idx].runtime_args[m[0]][m[1]].extend(
                        mcfg.kernel_runtime_args(
                            source_node_id=node(mesh, me_rc),
                            destination_node_id=node(mesh, nxt[g][d]),
                            link_index=link,
                            program_descriptor=p,
                            mux_logical_core=ttnn.CoreCoord(*m),
                        )
                    )
                for d in range(2):
                    m = mux_core(g, d)
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
                        p.kernels[rd_idx[g]].runtime_args[x][y].extend(args)
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    io = [x_t, idx_t, counts_t, ws_t, tab_t, wgu_ring, wd_ring, hring, hflags, xring, xflags, xprog, counts_l1]
    io += [idx_l1, table_l1, warr, cred, freew, st, out_t]
    return mpd, io, st
