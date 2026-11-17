# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Lane design (ws_lane.py) with 5 x 5 K-split groups (ws_g5.py): the lanes' static schedule, slot tables, muxes and
credits unchanged; each group is 5 columns x 5 rows (core j = 5 c + r, rows 0-3 own a gate/up pair each and run 4
down columns with 25 K tiles; row 4 runs 8 down columns with 20 K tiles), the weights move in packets of CHT 20 bfp4
tiles (owners 10 gate/up + 4 down packets a slice, row 4 8 + 8), h goes to the group as a unicast fan-out with slot
credits (no multicasts). Kernels: l5_writer.cpp (BRISC), l5_reader.cpp (NCRISC), l5_compute.cpp.

Layout (G <= 2 on the 11 x 10 tray grid): group g in logical columns 1-5 (NoC contiguous), rows 5 g .. 5 g + 4; its two
mux cores in column 0 (west of the group: the cores reach them on NOC1)."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert.bench_util import TILE_BYTES, cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, sharded_l1
from models.demos.ws_expert.ws_fab import FAB_DEFS, node
from models.demos.ws_expert.ws_g5 import NCOL, NR, OWN_ROWS, core_geom, row_geom
from models.demos.ws_expert.ws_group import GCfg
from models.demos.ws_expert.ws_lane import LCfg, lanes

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
    origins: tuple = ((1, 0), (1, 5))  # top-left logical core of group g
    mux_col: int = 0
    mux_rows: tuple = ((1, 3), (6, 8))  # (direction 0, direction 1) mux rows of group g


def g5_lane_cores(l5: L5Cfg, g):
    x0, y0 = l5.origins[g]
    return [(x0 + c, y0 + r) for c in range(NCOL) for r in range(NR)]


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


def build_lane5(mesh, cfg: GCfg, lc: LCfg, l5: L5Cfg, x_t, idx_t, counts_t, ws_t, out_t, tab_t, T, out_rows):
    """generic_op mesh program of the 5 x 5 lane design; returns (mesh program, io tensors, stamps)."""
    R, Cn = tuple(mesh.shape)
    N, G = R * Cn, cfg.G
    assert cfg.GC == GC5 and cfg.PAIRS == OWN_ROWS * NCOL and cfg.KT == 120 and cfg.HT == 120
    assert lc.out_all, "the 5 x 5 lane writes every expert's rows"
    ln = lanes((R, Cn), lc)
    groups = [g5_lane_cores(l5, g) for g in range(G)]
    C = [c for gr in groups for c in gr]
    muxes = [(l5.mux_col, l5.mux_rows[g][d]) for g in range(G) for d in range(2)]
    assert not set(muxes) & set(C)
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
    xring = sharded_l1(mesh, C, [l5.NXR * 25 * 32, 32], cfg.x_dt, ttnn.TILE_LAYOUT)
    xflags, xcred = w16(l5.NXR), w16(NCOL)
    land = sharded_l1(mesh, C, [l5.D * LT * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    lflags, lcred = w16(l5.D * NR), w16(NR)
    pring = sharded_l1(mesh, C, [l5.PD * NW * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    scr = w16(8)
    counts_l1 = sharded_l1(mesh, C, [1, 32 + max(4, cfg.nexp)], U, RM, torch.int32)
    idx_l1 = sharded_l1(mesh, C, [4, 32], U, RM, torch.int32)
    table_l1 = sharded_l1(mesh, C, [1, TLW + 16], U, RM, torch.int32)
    a64 = lambda a: (a + 63) // 64 * 64
    cnt_base, idx_addr, tab_base = (a64(t.buffer_address()) for t in (counts_l1, idx_l1, table_l1))
    st = stamps_tensor(mesh, C)
    mcfg = fabric_mux.Config(
        num_full_size_channels=GC5,
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=lc.nbuf,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh),
        base_l1_address=ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1),
    )
    ctype = fabric_mux.ChannelType.FULL_SIZE
    client_ct = list(fabric_mux.client_compile_time_args(num_clients=GC5, channel_type=ctype, config=mcfg))
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
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            me_rc = (r, c)
            kernels, rd_idx = [], {}
            nxt = {}
            for g, gr in enumerate(groups):
                cyc, link = ln[g]
                i = cyc.index(me_rc)
                nxt[g] = (cyc[(i + 1) % N], cyc[(i - 1) % N])
                nid = [int(node(mesh, nxt[g][d]).chip_id) for d in range(2)]
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
                                nid[0],
                                nid[1],
                                xy[0] | xy[1] << 8,
                            ]
                            + cx
                            + ry
                        )
                    cs = crs([xy for _, xy in sub])
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=kpath("l5_writer.cpp"),
                            core_ranges=cs,
                            compile_time_args=[ksl, dc, own, l5.NXR, xring.buffer_address(), xflags.buffer_address()]
                            + [xcred.buffer_address(), xtile, l5.D, land.buffer_address(), lflags.buffer_address()]
                            + [lcred.buffer_address(), ptile, scr.buffer_address(), KTD, NS, hring.buffer_address()]
                            + [hflags.buffer_address(), htile, cnt_base, tab_base, TB, cfg.RMAX, cfg.H * 2, idx_addr]
                            + [cfg.nexp, wring.buffer_address(), warr.buffer_address(), freew.buffer_address()]
                            + [lc.CHT, wtile, l5.RD, hcred.buffer_address(), K, PKE]
                            + list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
                            + list(ttnn.TensorAccessorArgs(idx_t).get_compile_time_args())
                            + list(ttnn.TensorAccessorArgs(counts_t).get_compile_time_args())
                            + list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
                            + list(ttnn.TensorAccessorArgs(ws_t).get_compile_time_args()),
                            runtime_args=wr_rt,
                            config=dm(0),
                        )
                    )
                    rd_idx[(g, own)] = len(kernels)
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=kpath("l5_reader.cpp"),
                            core_ranges=cs,
                            compile_time_args=[ksl, dc, own, l5.NXR, xflags.buffer_address(), xcred.buffer_address()]
                            + [l5.D, land.buffer_address(), lflags.buffer_address(), lcred.buffer_address(), ptile]
                            + [scr.buffer_address(), KTD, NS, hflags.buffer_address(), cnt_base, tab_base]
                            + [warr.buffer_address(), out_rows, cfg.HT, wring.buffer_address(), cred.buffer_address()]
                            + [freew.buffer_address(), lc.CHT, wtile, l5.PD, pring.buffer_address()]
                            + [hcred.buffer_address(), K, hcolf.buffer_address(), hring.buffer_address(), htile]
                            + client_ct
                            + list(ttnn.TensorAccessorArgs(out_t).get_compile_time_args()),
                            runtime_args=rd_rt,
                            defines=FAB_DEFS,
                            config=dm(1),
                        )
                    )
            cp_rt = {own: ttnn.RuntimeArgs() for own in (1, 0)}
            for g, gr in enumerate(groups):
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
                        defines=act_defs,
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
                    cb(17, cfg.out_dt, 2 * dc, cs),
                ]
            cset = crs(C)
            cbs += [
                ttnn.cb_descriptor_from_sharded_tensor(2, hring),
                ttnn.cb_descriptor_from_sharded_tensor(4, land),
                cb(16, cfg.h_dt, 2, cset),
                ttnn.cb_descriptor_from_sharded_tensor(18, pring),
            ]
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
            sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(2)]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            for g, gr in enumerate(groups):
                cyc, link = ln[g]
                for d in range(2):
                    m = (l5.mux_col, l5.mux_rows[g][d])
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
                    m = (l5.mux_col, l5.mux_rows[g][d])
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
                        own = 1 if core_geom(cfg, j)[6] else 0
                        p.kernels[rd_idx[(g, own)]].runtime_args[x][y].extend(args)
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    io = [x_t, idx_t, counts_t, ws_t, tab_t, wring, warr, cred, freew, hring, hflags, hcolf, hcred, xring, xflags]
    io += [xcred]
    io += [land, lflags, lcred, pring, scr, counts_l1, idx_l1, table_l1, st, out_t]
    return mpd, io, st
