# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""EP-4 with weights on Y: the MoE layer's device phases (WORKLOG "EP-4 with weights on Y (LoudBox)", host reference
ep4_ref.py). Mesh R x C, C = 4: a mesh row is an EP group, chip column c holds experts [EPC c, EPC (c + 1)).

- head (A): ws_router in EP-4 mode on the chip's own n tokens - a 128-B topk row a token (uint32: the gate weights'
  fp32 bits in words 0-15, the expert ids in 16 + k, the route word in 28; [TX / 32, 1024], 4-KB pages of 32 tokens) and
  the dispatch route words; then grp, the all-gather of the topk pages along X (cluster_axis 1): [C TX / 32, 1024],
  home c's token t at row c TX + t of the [C TX, 32] view."""

import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.ws_expert import bench_util
from models.demos.ws_expert.ep4_ref import zipf_bias
from models.demos.ws_expert.ws_layer import LayerCfg
from models.demos.ws_expert.ws_router import build_router, ep4_outputs, router_inputs

C_EP = 4  # chips an EP group (a mesh row)


@dataclass
class EP4Cfg(LayerCfg):
    """LayerCfg with T = n, the tokens a chip owns. The dispatch runs in KP pieces of SP tokens a chip (its manifest
    rings hold a call's every row in L1): token t is in piece t // SP."""

    C: int = C_EP
    SP: int = int(os.environ.get("EP4_SP", "1184"))

    EPC = property(lambda s: s.nexp // s.C)
    KP = property(lambda s: -(-s.TX // s.SP))
    TXP = property(lambda s: s.KP * s.SP)  # (route words a chip: tokens >= T route nowhere)


def xrow(lc: EP4Cfg, N, ch, t):
    """xbuf row of chip ch's token t: piece p = t // SP holds rows [p N SP, (p + 1) N SP), chip ch's at ch SP + t % SP."""
    return (t // lc.SP) * N * lc.SP + ch * lc.SP + t % lc.SP


def x_rows(lc: EP4Cfg, N):
    return lc.KP * N * lc.SP


def make_inputs(mesh, lc: EP4Cfg, routing="uniform", seed=0):
    """Host tensors and the head's device inputs: x [N, TX, H] bf16 per chip (rows >= T zero), the router W_r [H, E]
    bf16; zipf routing: a per-expert logit bias (ep4_ref.zipf_bias) as x column 0 = 1 and W_r row 0 = the bias."""
    N = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed)
    H, E = lc.H, lc.nexp
    wr = (torch.randn(H, E, generator=g) / H**0.5).to(torch.bfloat16)
    x = torch.zeros(N, lc.TX, H, dtype=torch.bfloat16)
    x[:, : lc.T] = torch.randn(N, lc.T, H, generator=g).to(torch.bfloat16)
    if routing == "zipf":
        x[:, : lc.T, 0] = 1.0
        wr[0, :] = zipf_bias(E, seed).to(torch.bfloat16)
    else:
        assert routing == "uniform", routing
    shard = ttnn.ShardTensorToMesh(mesh, dim=0)
    x_rm = ttnn.from_torch(
        x.reshape(N * lc.TX, H), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=shard
    )
    wrT, ids_t = router_inputs(mesh, wr)
    return dict(x=x, wr=wr), dict(x_rm=x_rm, wrT=wrT, ids_t=ids_t)


def head_outputs(mesh, lc: EP4Cfg):
    """ws_router.ep4_outputs with the route words in KP pieces, each [1, 1, 1, SP] zeroed (tokens >= TX stay 0: no
    route): the head writes them, the dispatch's calls read them (persistent, so a traced layer can replay)."""
    outs = ep4_outputs(mesh, lc)
    from models.demos.ws_expert.ws_router import ep4_rw

    z = lambda: ttnn.zeros(
        [1, 1, 1, lc.SP * ep4_rw(mesh)],
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    outs["route_pieces"] = [z() for _ in range(lc.KP)]
    outs["route_sp"] = lc.SP
    return outs


def head_program(mesh, lc: EP4Cfg, dev, outs):
    """The head's mesh program (EP-4 mode) into outs (ws_router.ep4_outputs): (mesh program, io, stamps)."""
    with bench_util.self_zeroing():
        return build_router(mesh, lc, dev["x_rm"], dev["wrT"], dev["ids_t"], outs, ep4=lc.EPC)


def grp_tensor(mesh, lc: EP4Cfg):
    """The exchange's output: the group's topk rows [C TX / 32, 1024] uint32 (4-KB pages of 32 tokens)."""
    return ttnn.empty(
        [lc.C * lc.TX // 32, 1024],
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def exchange(tk, grp):
    """grp = the all-gather of every chip's topk rows along its mesh row (X)."""
    return ttnn.all_gather(tk, 0, cluster_axis=1, output_tensor=grp)


def unpack_topk(rows, K):
    """[n, 32] uint32 topk rows (torch int32/int64) -> ids [n, K] int64, w [n, K] fp32, route [n] int64."""
    rows = rows.to(torch.int64) & 0xFFFFFFFF
    w = rows[:, :K].to(torch.int32).view(torch.float32)
    return rows[:, 16 : 16 + K], w, rows[:, 28]


# ------------------------------------------------------------------------------------------------- dispatch (B)
mb = ttnn.experimental.moe_block


def dispatch_resources(mesh, lc: EP4Cfg):
    """The MoE block's resources for sparse_dispatch_2d at S = TX slots a chip (the mesh plan, port cores, links)."""
    dims = mb.MoEModelDims(num_experts=lc.nexp, top_k=lc.K, emb_dim=lc.H, routed_expert_hidden_dim=lc.I)
    return mb.MoEBlockResources(mesh, dims, mb.MoEBlockConfig(chunk_tokens=lc.SP * mesh.get_num_devices()))


def xbuf_tensor(mesh, lc: EP4Cfg, x):
    """xbuf [1, 1, KP N SP, H] bf16 row-major: chip ch's own rows (x [N, TX, H], rows >= T zero) at xrow(ch, t), every
    other row zero; the dispatch writes chip home's token t at xrow(home, t) on the chips its route word names. Built a
    piece at a time (piece p: rows [p N SP, (p + 1) N SP), chip ch's own at ch SP) and joined on device, so the host
    holds one piece of every chip at once."""
    N = mesh.get_num_devices()
    SP = lc.SP
    pieces = []
    for p_ in range(lc.KP):
        buf = torch.zeros(N, N * SP, lc.H, dtype=torch.bfloat16)
        for ch in range(N):
            seg = x[ch, p_ * SP : (p_ + 1) * SP]  # (the last piece may run past TX: those rows stay zero)
            buf[ch, ch * SP : ch * SP + seg.shape[0]] = seg
        pieces.append(
            ttnn.from_torch(
                buf.reshape(N, 1, N * SP, lc.H),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
        )
        del buf
    out = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if len(pieces) > 1:
        for t_ in pieces:
            ttnn.deallocate(t_)
    return ttnn.reshape(out, [1, 1, x_rows(lc, N), lc.H])


def route_pieces(lc: EP4Cfg, outs):
    """The route words of each dispatch piece (the head's outputs)."""
    return outs["route_pieces"]


def dispatch(lc: EP4Cfg, xbuf, routes, res, N):
    """Each own row to the chips of its mesh row its route word names, piece by piece (sparse_dispatch_2d at row
    offset p N SP; each call a barrier: when it ends on a chip, every row sent to that chip has landed)."""
    for p, rp in enumerate(routes):
        mb.sparse_dispatch_2d(xbuf, rp, res, row_offset=p * N * lc.SP)
    return xbuf


# ------------------------------------------------------------------------------------------------- lists (C)
LISTS_EPO = 2  # local experts an owner core
LISTS_CAP = 24576  # records an owner core takes (its experts' pairs + ready records)
O_SLACK = 1.25  # O rows allocated: O_SLACK x 12 n (the busiest zipf chip measured 1.14 x on the host)


def o_rows(lc: EP4Cfg):
    """O's allocated rows (the last one the lists' pad row)."""
    return -(-int(O_SLACK * lc.K * lc.T) // 32) * 32 + 32


def list_width(lc: EP4Cfg):
    """RS: list page entries (every token of the group fits one expert's list)."""
    return lc.C * lc.TX


def lists_outputs(mesh, lc: EP4Cfg):
    """The lists program's outputs (DRAM, row-major uint32): runs [C TX, 32] (a group token's run record), per local
    expert its token list idx / O rows dest / ready list rdy [EPC, RS], counts [1, EPC] (the lane's counts), rcounts
    [1, EPC + 16] (ready counts; word EPC: the O rows used; EPC + 1 + c: the partials this chip's home pass gets from the
    row's chip c), fence [16, 16]."""
    rm, dram = ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
    e = lambda shape: ttnn.empty(shape, dtype=ttnn.uint32, layout=rm, device=mesh, memory_config=dram)
    RS = list_width(lc)
    return dict(
        runs=e([lc.C * lc.TX, 32]),
        idx=e([lc.EPC, RS]),
        dest=e([lc.EPC, RS]),
        rdy=e([lc.EPC, RS]),
        counts=e([1, lc.EPC]),
        rcounts=e([1, lc.EPC + 16]),
        fence=e([16, 16]),
    )


def pos_tensor(mesh, pos):
    """[1, EPC] uint32 stream position key of each local expert (the lane's compute order): pos [EPC] (replicated) or
    [N, EPC] (chip ch's row on chip ch)."""
    pos = torch.as_tensor(pos, dtype=torch.int32)
    if pos.dim() == 1:
        return ttnn.from_torch(
            pos.reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return ttnn.from_torch(
        pos.reshape(-1, pos.shape[-1]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def build_lists(mesh, lc: EP4Cfg, grp, pos_t, outs, cores=None):
    """generic_op mesh program of the receive-side lists (kernels e4_lists_rd.cpp / e4_lists.cpp; one program a
    chip): (mesh program, io, stamps)."""
    from models.demos.ws_expert.bench_util import crs, kpath, stamps_tensor
    from models.demos.ws_expert.ws_end import end_cores, token_ranges
    from models.demos.ws_expert.ws_expert import noc, sharded_l1
    from models.demos.ws_expert.ws_head import dm

    R, Cn = tuple(mesh.shape)
    assert Cn == lc.C
    cores = cores or end_cores(mesh)
    NC, EPC, EPO, CAP = len(cores), lc.EPC, LISTS_EPO, LISTS_CAP
    NOWN = -(-EPC // EPO)
    assert NOWN <= NC
    G = lc.C * lc.TX
    rng = token_ranges(G, NC)
    ntmax = max(n for _, n in rng)
    U = ttnn.uint32
    w16 = lambda words: sharded_l1(mesh, cores, [-(-words // 16), 16], U, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    bufs = dict(
        tok=w16(ntmax * 32),
        srt=w16(2 * (ntmax * (lc.K + 1) + 2 * NOWN)),
        cnt_in=w16(NC),
        off_in=w16(NOWN),
        recbuf=w16(2 * CAP),
        la=w16(CAP + 32 * EPO),
        lb=w16(CAP + 32 * EPO),
        lc=w16(CAP + 32 * EPO),
        cnt_all=w16(2 * EPC + 16),
        aux=w16(3 * 256),
        rstage=w16(32 * 32),
        pos=w16(EPC),
        sb=w16(2 * NC + 48),
    )
    for t in bufs.values():
        assert t.buffer_address() % 64 == 0
    st = stamps_tensor(mesh, cores)
    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    b = lambda k: bufs[k].buffer_address()
    ct = [lc.K, EPC, NOWN, NC, CAP, list_width(lc), lc.TX, EPO]
    ct += [b(k) for k in ("tok", "srt", "cnt_in", "off_in", "recbuf", "la", "lb", "lc", "cnt_all", "aux", "rstage")]
    ct += [b("pos"), b("sb"), lc.T, lc.C, lc.SP, R * Cn * lc.SP]
    for k in ("runs", "idx", "dest", "rdy", "counts", "rcounts", "fence"):
        ct += ta(outs[k])
    rd_ct = [b("tok"), b("pos"), EPC] + ta(grp) + ta(pos_t)
    xy = [noc(mesh, *c)[0] | noc(mesh, *c)[1] << 8 for c in cores]
    cset = crs(cores)
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(5)]
    pad_row = o_rows(lc) - 1
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            ch = r * Cn + c
            rd_rt, rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for me, ((x, y), (a, nt)) in enumerate(zip(cores, rng)):
                rd_rt[x][y] = [a, nt, grp.buffer_address(), pos_t.buffer_address()]
                args = [me, a, nt, r * Cn, c, xrow(lc, R * Cn, ch, lc.T), pad_row]
                args += [outs[k].buffer_address() for k in ("runs", "idx", "dest", "rdy", "counts", "rcounts")]
                rt[x][y] = args + [outs["fence"].buffer_address(), st.buffer_address()] + xy
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=kpath("e4_lists_rd.cpp"),
                    core_ranges=cset,
                    compile_time_args=rd_ct,
                    runtime_args=rd_rt,
                    config=dm(0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("e4_lists.cpp"),
                    core_ranges=cset,
                    compile_time_args=ct,
                    runtime_args=rt,
                    config=dm(1),
                ),
            ]
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=[])
    io = (
        [grp, pos_t, st]
        + list(bufs.values())
        + [outs[k] for k in ("runs", "idx", "dest", "rdy", "counts", "rcounts", "fence")]
    )
    return mpd, io, st


# ------------------------------------------------------------------------------------------------- experts (D + E)
def column_cycles(mesh):
    """The Y rings: one cycle a mesh column, (0, c) .. (R - 1, c)."""
    R, Cn = tuple(mesh.shape)
    return [[(r, c) for r in range(R)] for c in range(Cn)]


def lane_cfgs(mesh, lc: EP4Cfg, G=2, l5kw=None, lkw=None):
    """(GCfg, LCfg, L5Cfg) of the 5 x 5 lane on the Y columns: EPC local experts (lane-local ids), the group's token
    lists (RMAX 32 = C TX entries), row-major O rows through the row collectors."""
    from models.demos.ws_expert.ws_group import GCfg
    from models.demos.ws_expert.ws_lane import LCfg
    from models.demos.ws_expert.ws_lane5 import L5Cfg

    RS = list_width(lc)
    cfg = GCfg(nexp=lc.EPC, G=G, T=RS, MMAX=RS, GC=25, h_dt=ttnn.bfloat8_b, XRM=2, NS=5, dn_lag=3)
    lkw = dict(CHT=20, slots="d0,d1,d0d1,any", keep=2, sched_c=7, wpoll=1, **(lkw or {}))
    lane = LCfg(cycles=column_cycles(mesh), subcycles=True, **lkw)
    l5 = L5Cfg(**{**dict(hmc=2, mux_noc=0, yrm=1, yc=3, yc_n0=1), **(l5kw or {})})
    return cfg, lane, l5


def lane_tables(mesh, cfg, lane):
    """ws_lane.make_tables on the Y columns, plus per chip the stream-position key of each local expert (its index in
    its group's compute order x G + g) for the lists' ready records."""
    from models.demos.ws_expert.ws_lane import make_tables

    R, Cn = tuple(mesh.shape)
    T = make_tables((R, Cn), cfg, lane)
    pos = torch.zeros(R * Cn, cfg.nexp, dtype=torch.int64)
    for X in range(R * Cn):
        for g in range(cfg.G):
            for k, le in enumerate(T["meta"][(X, g)]["order"]):
                pos[X, le] = k * cfg.G + g
    return T, pos


def own_slices(cfg, lane, T, lc: EP4Cfg, wg, wu, wd, Cn):
    """Per chip its own experts' slices (ws_lane5.slice_layout5; T["own"][X]: local ids, global EPC c + id)."""
    from models.demos.ws_expert.ws_lane5 import slice_layout5

    out = []
    for X, own in enumerate(T["own"]):
        ge = [lc.EPC * (X % Cn) + le for le in own]
        out.append(slice_layout5(cfg, lane.CHT, wg[ge].float(), wu[ge].float(), wd[ge].float()).to(torch.bfloat16))
    return out


def o_tensor(mesh, lc: EP4Cfg):
    """O [o_rows + 16, H] bf16 row-major (the last 16 rows the lane's posted-write fence), zero."""
    return ttnn.zeros(
        [o_rows(lc) + 16, lc.H],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def lane_program(mesh, lc: EP4Cfg, cfg, lane, l5, T, tab_t, xbuf, louts, ws_t, O):
    """The experts' mesh program: ws_lane5.build_lane5 over xbuf's rows and the lists, every down row to its O row."""
    from models.demos.ws_expert.ws_lane5 import build_lane5

    x2 = ttnn.reshape(xbuf, [xbuf.shape[-2], lc.H])
    return build_lane5(
        mesh,
        cfg,
        lane,
        l5,
        x2,
        louts["idx"],
        louts["counts"],
        ws_t,
        O,
        tab_t,
        T,
        0,
        dest_t=louts["dest"],
        fence_row=o_rows(lc),
    )


# ------------------------------------------------------------------------------------------------- combine (F, H)
S5 = C_EP + 1  # P slots a token: one a row chip, then the shared expert's
COMB_NBUF = 8  # adder: O rows in flight
COMB_NSL = 1  # adder slots a port
SLOTB = 7744  # a port slot: the row (7680 B), its descriptor, 64 B aligned


def p_tensor(mesh, lc: EP4Cfg):
    """P [TX S5 + 16, H] bf16 row-major: row t S5 + c the partial from the row's chip c, t S5 + C the shared expert's;
    16 spare rows (the shared expert's posted-write fence)."""
    return ttnn.empty(
        [lc.TX * S5 + 16, lc.H],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def gate_core(mesh):
    from models.demos.ws_expert.ws_end import end_cores

    return end_cores(mesh)[0]


def counters_tensor(mesh):
    """The home pass's counters on the gate core: word c (16 B apart) the partials landed from the row's chip c; kept
    across layers (the ports increment it, the home pass's gate clears it)."""
    from models.demos.ws_expert.ws_expert import sharded_l1

    return sharded_l1(mesh, [gate_core(mesh)], [4, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)


def comb_ports(mesh, res):
    """{(r, c): [(dir, link index, (x, y), fabric link, peer)]} the X ports of every chip (resources.port_cores)."""
    R, Cn = tuple(mesh.shape)
    topo = res.topology
    plan = mb.mesh_plan_uniform(R, Cn, topo["wrap_x"], topo["wrap_y"], 2)
    out = {}
    for (r, c, d, li), (x, y, link) in res.port_cores().items():
        if d not in ("X+", "X-"):
            continue
        out.setdefault((r, c), []).append((d, li, (x, y), link, tuple(plan[(r, c)][d]["peer"])))
    return {k: sorted(v) for k, v in out.items()}


def build_combine(mesh, lc: EP4Cfg, res, louts, O, P, cnt_t, fence):
    """generic_op mesh program of the pre-sum and its sends (kernels e4_comb_*.cpp; one program a chip): adders on every
    core but the X ports (each a range of the group's tokens), the ports' fabric sends into the home chips' P.
    Returns (mesh program, io, stamps of the adders, stamps of the ports)."""
    from models.demos.ws_expert.bench_util import cb, crs, kpath, stamps_tensor
    from models.demos.ws_expert.ws_end import end_cores, token_ranges
    from models.demos.ws_expert.ws_expert import noc, sharded_l1
    from models.demos.ws_expert.ws_fab import FAB_DEFS, node
    from models.demos.ws_expert.ws_head import dm

    R, Cn = tuple(mesh.shape)
    ports = comb_ports(mesh, res)
    hops = res.first_hops()
    pcores = sorted({p[2] for v in ports.values() for p in v})
    acores = [c for c in end_cores(mesh) if c not in pcores]
    NA = len(acores)
    G = lc.C * lc.TX
    rng = token_ranges(G, NA)
    ntmax = max(n for _, n in rng)
    U, RM = ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    rb = lc.H * 2
    a16 = lambda cores, words: sharded_l1(mesh, cores, [-(-words // 16), 16], U, RM, torch.int32)
    tab = a16(acores, ntmax * 32 + 16)  # (the run table, then the flag word)
    tab_flag = tab.buffer_address() + ntmax * 128
    cred = a16(acores, 4 * 8)
    scr = a16(acores, 4 * 9)
    rows_l1 = sharded_l1(mesh, acores, [COMB_NBUF * 4 * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    ring = sharded_l1(mesh, pcores, [NA * COMB_NSL * SLOTB // 4 // 16, 16], U, RM, torch.int32)
    pfl = a16(pcores, 4 * NA)
    pend = a16(pcores, 4 * NA)
    for t in (tab, rows_l1, ring, pfl, pend, scr):
        assert t.buffer_address() % 64 == 0
    ast = stamps_tensor(mesh, acores)
    pst = stamps_tensor(mesh, pcores)
    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    xy = lambda c: noc(mesh, *c)[0] | noc(mesh, *c)[1] << 8
    gx = xy(gate_core(mesh))
    aset = crs(acores)
    rd_ct = [COMB_NBUF, tab.buffer_address(), tab_flag, rb, COMB_NBUF * 2048] + ta(louts["runs"]) + ta(O)
    cp_ct = [tab.buffer_address(), tab_flag]
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            pl = ports[(r, c)]
            NP = len(pl)
            # port of (home, link): the fabric's first hop toward home (on a ring it splits the tie itself), link li
            pidx = {}
            for pi, (d, li, _, _, _) in enumerate(pl):
                for h in range(Cn):
                    if h != c and hops[(r, c, r, h)] == d:
                        pidx[(h, li)] = pi
            wr_ct = [lc.TX, rb, tab.buffer_address(), tab_flag, NP, COMB_NSL, SLOTB, ring.buffer_address()]
            wr_ct += [pfl.buffer_address(), pend.buffer_address(), cred.buffer_address(), S5, scr.buffer_address()]
            wr_ct += ta(P) + ta(fence)
            rd_rt, cp_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for me, ((x, y), (a, nt)) in enumerate(zip(acores, rng)):
                rd_rt[x][y] = [louts["runs"].buffer_address(), O.buffer_address(), a, nt, ast.buffer_address()]
                cp_rt[x][y] = [nt, ast.buffer_address()]
                args = [me, a, nt, c, P.buffer_address(), fence.buffer_address(), ast.buffer_address()]
                args += [pidx.get((h, l), 0xFF) for h in range(4) for l in range(2)]
                wr_rt[x][y] = args + [xy(p[2]) for p in pl] + [0] * (8 - NP)
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=kpath("e4_comb_rd.cpp"),
                    core_ranges=aset,
                    compile_time_args=rd_ct,
                    runtime_args=rd_rt,
                    config=dm(0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("e4_comb_cp.cpp"),
                    core_ranges=aset,
                    compile_time_args=cp_ct,
                    runtime_args=cp_rt,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("e4_comb_wr.cpp"),
                    core_ranges=aset,
                    compile_time_args=wr_ct,
                    runtime_args=wr_rt,
                    config=dm(1),
                ),
            ]
            mycores = [p[2] for p in pl]
            pt_ct = [NA, COMB_NSL, SLOTB, ring.buffer_address(), pfl.buffer_address(), pend.buffer_address(), rb, S5]
            pt_ct += [cnt_t.buffer_address(), cred.buffer_address()] + ta(P)
            pt_rt = ttnn.RuntimeArgs()
            chip_ids = [int(node(mesh, (r, h)).chip_id) if h != c else 0xFF for h in range(4)]
            for pi, (d, li, (x, y), link, peer) in enumerate(pl):
                args = [P.buffer_address(), pi, c, gx, int(node(mesh, (r, c)).mesh_id)] + chip_ids
                pt_rt[x][y] = args + [pst.buffer_address()] + [xy(ac) for ac in acores]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("e4_comb_port.cpp"),
                    core_ranges=crs(mycores),
                    compile_time_args=pt_ct,
                    runtime_args=pt_rt,
                    defines=FAB_DEFS,
                    config=dm(1),
                )
            )
            cbs = [
                ttnn.cb_descriptor_from_sharded_tensor(0, rows_l1),
                cb(1, ttnn.bfloat16, COMB_NBUF, aset),
                cb(16, ttnn.bfloat16, 2 * 4, aset),
                cb(17, ttnn.bfloat16, 1, aset),
            ]
            sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=aset, initial_value=0)]
            p = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            for pi, (d, li, (x, y), link, peer) in enumerate(pl):
                p.kernels[3].runtime_args[x][y].extend(
                    ttnn.setup_fabric_connection(
                        mesh_device=mesh,
                        src_fabric_node_id=node(mesh, (r, c)),
                        dst_fabric_node_id=node(mesh, peer),
                        link_idx=link,
                        program_descriptor=p,
                        worker_core=ttnn.CoreCoord(x, y),
                    )
                )
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = p
    io = [louts["runs"], O, P, cnt_t, fence, tab, cred, scr, rows_l1, ring, pfl, pend, ast, pst]
    return mpd, io, ast, pst


# ------------------------------------------------------------------------------------------------- home pass (G)
HOME_NBUF = 4


def build_home(mesh, lc: EP4Cfg, P, tk, rcounts, cnt_t, y_t):
    """generic_op mesh program of the home pass (kernels e4_home_rd.cpp, we_compute.cpp with S5 rows, we_writer.cpp;
    one program a chip): the gate on the ports' counters, then y[t] = the present partials + the shared row."""
    from models.demos.ws_expert.bench_util import cb, crs, kpath, stamps_tensor
    from models.demos.ws_expert.ws_end import end_cores, token_ranges
    from models.demos.ws_expert.ws_expert import noc, sharded_l1
    from models.demos.ws_expert.ws_head import dm

    R, Cn = tuple(mesh.shape)
    cores = end_cores(mesh)
    assert cores[0] == gate_core(mesh)
    NC = len(cores)
    rng = token_ranges(lc.T, NC)
    ntmax = max(n for _, n in rng)
    rb = lc.H * 2
    U, RM = ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    rows_l1 = sharded_l1(mesh, cores, [HOME_NBUF * S5 * 4 * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out_l1 = sharded_l1(mesh, cores, [2 * 4 * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    tks = sharded_l1(mesh, cores, [ntmax * 2, 16], U, RM, torch.int32)
    rc_bytes = (lc.EPC + 16) * 4
    rcs = sharded_l1(mesh, cores, [-(-rc_bytes // 64), 16], U, RM, torch.int32)
    for t in (rows_l1, out_l1, tks, rcs):
        assert t.buffer_address() % 64 == 0
    st = stamps_tensor(mesh, cores)
    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    xy = [noc(mesh, *c)[0] | noc(mesh, *c)[1] << 8 for c in cores]
    cset = crs(cores)
    rd_ct = [S5, lc.C, rb, lc.EPC, lc.K, HOME_NBUF, tks.buffer_address(), cnt_t.buffer_address()]
    rd_ct += [rcs.buffer_address(), rc_bytes] + ta(P) + ta(tk) + ta(rcounts)
    wr_ct = [rb] + ta(y_t)
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(R):
        for c in range(Cn):
            rd_rt, cp_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for me, ((x, y), (t0, n)) in enumerate(zip(cores, rng)):
                args = [P.buffer_address(), tk.buffer_address(), rcounts.buffer_address(), t0, n, c, me, NC]
                rd_rt[x][y] = args + [st.buffer_address()] + xy
                cp_rt[x][y] = [n, st.buffer_address()]
                wr_rt[x][y] = [y_t.buffer_address(), t0, n, st.buffer_address()]
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=kpath("e4_home_rd.cpp"),
                    core_ranges=cset,
                    compile_time_args=rd_ct,
                    runtime_args=rd_rt,
                    config=dm(0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("we_compute.cpp"),
                    core_ranges=cset,
                    compile_time_args=[S5],
                    runtime_args=cp_rt,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=kpath("we_writer.cpp"),
                    core_ranges=cset,
                    compile_time_args=wr_ct,
                    runtime_args=wr_rt,
                    config=dm(1),
                ),
            ]
            cbs = [
                ttnn.cb_descriptor_from_sharded_tensor(0, rows_l1),
                cb(1, ttnn.bfloat16, 2 * S5, cset),
                ttnn.cb_descriptor_from_sharded_tensor(16, out_l1),
            ]
            sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cset, initial_value=0)]
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
    io = [P, tk, rcounts, cnt_t, y_t, rows_l1, out_l1, tks, rcs, st]
    return mpd, io, st


# ------------------------------------------------------------------------------------------------- dispatch beside the shared expert
def port_cores_xy(res):
    """The logical cores of every port sparse_dispatch_2d uses (resources.port_cores, every direction and link)."""
    return sorted({(x, y) for (x, y, _l) in res.port_cores().values()})


def port_subdevices(mesh, res, local_l1=0):
    """A sub-device manager of two: 0 the dispatch's port cores, 1 every other worker core. Loaded, sparse_dispatch_2d
    (sub-device 0) and a program on the other cores (the shared expert, sub-device 1) run at once on one command
    queue: a program waits only for the earlier programs of its own sub-device. Returns (manager, port cores)."""
    g = mesh.compute_with_storage_grid_size()
    pc = port_cores_xy(res)
    pcs = set(pc)
    one = lambda x0, x1, y: ttnn.CoreRange(ttnn.CoreCoord(x0, y), ttnn.CoreCoord(x1, y))
    rest = []
    for y in range(g.y):
        x = 0
        while x < g.x:
            if (x, y) in pcs:
                x += 1
                continue
            x1 = x
            while x1 + 1 < g.x and (x1 + 1, y) not in pcs:
                x1 += 1
            rest.append(one(x, x1, y))
            x = x1 + 1
    ports = ttnn.CoreRangeSet({one(x, x, y) for x, y in pc})
    mgr = mesh.create_sub_device_manager(
        [ttnn.SubDevice([ports]), ttnn.SubDevice([ttnn.CoreRangeSet(set(rest))])], local_l1
    )
    return mgr, pc
