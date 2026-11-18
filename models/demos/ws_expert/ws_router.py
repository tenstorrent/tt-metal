# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The MoE layer's head as one program (kernels wsr2_*.cpp, the exchange wsr_exchange.hpp), every chip on its own
tokens, structured as the MoE block's router (moe_head): the router matmul on the cores' own compute, the top-K on
the SFPU over the logits in DEST, transposed (a token a column), the logits gathered as top-K runs to the core that
owns the token's row tile.

- Router groups: column segments of GS = E / (32 EPC) cores (one NOC1 multicast rectangle each); core c of a group
  holds W_r^T's expert tiles [EPC c, EPC (c + 1)) resident. A group takes a contiguous range of row tiles; core c owns
  its group-local row tiles c, c + GS, ..
- x arrives row-major, in pieces of BK K tiles a row tile; piece p of every row tile is group core p % GS's: it reads
  the 32 row pieces (NOC0), tilizes them and writes the tiles into the relay ring of the group's hub (the column's
  NOC1 multicast start end; NOC0 runs toward it: no wrap), posted, the arrival flag after them through the same
  command buffer. The hub alone multicasts the pieces, in stream order, into every group core's x ring (NOC1).
- Every core, every row tile: logits^T for its EPC expert tiles (HiFi4, fp32 DEST), keys (moe_head router_topk_key:
  the expert id in the low mantissa bits) and the bitonic network over the pair, still in DEST -> its best sort_run
  keys, one tile, to the owner's gather ring.
- The owner: the group's GS run tiles through the network -> each token's top K, exp of their logits -> w = e / sum e
  and the records (t, k, e) -> the L1 exchange of ws_head (counts, offsets, runs, per-expert lists), fenced.
DRAM reads on NOC0, DRAM writes posted on NOC1 (fenced), multicasts only inside a group's rectangle, a chip-wide
barrier before the exchange's cross-group traffic, no DRAM round trip inside the program."""

import torch

import ttnn
from models.demos.ws_expert.bench_util import cb, crs, kpath, stamps_tensor
from models.demos.ws_expert.ws_end import token_ranges
from models.demos.ws_expert.ws_expert import noc, noc_rects, sharded_l1
from models.demos.ws_expert.ws_head import CAP, dm, mesh_program

EPC = 2  # expert tiles a core (the pair the network sorts in DEST)
BK = 12  # K tiles an x piece
NX = 20  # x ring pieces
NR = 6  # hub relay slots (pieces)
NG = 2  # gather ring slots (an owned row tile's GS run tiles each)
LAG = 2  # row tiles between a row tile's router and its owner's merge


def router_groups(mesh, NET):
    """[group: [GS logical cores, a column segment]], GS."""
    g = mesh.compute_with_storage_grid_size()
    GS = NET // EPC
    assert NET % EPC == 0 and 1 <= GS <= g.y, f"{NET} expert tiles over groups of {EPC} a core in {g.y}-core columns"
    return [[(x, s * GS + c) for c in range(GS)] for x in range(g.x) for s in range(g.y // GS)], GS


def router_inputs(mesh, wr):
    """W_r^T [E, H] bf16 TILE and the expert-id tiles [E, 32] uint16 TILE (element (e, j) = e), replicated."""
    E = wr.shape[1]
    dram = ttnn.DRAM_MEMORY_CONFIG
    wrT = ttnn.from_torch(
        wr.t().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dram,
    )
    ids = torch.arange(E, dtype=torch.int32).unsqueeze(1).expand(E, 32).contiguous()
    ids_t = ttnn.from_torch(ids, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram)
    return wrT, ids_t


def logits_debug_tensor(mesh, lc):
    """The logits^T [E, TX] fp32 TILE the program writes under debug_logits."""
    return ttnn.empty(
        [lc.nexp, lc.TX],
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def ep4_rw(mesh):
    """sparse_dispatch_2d's route words a token: 1 up to 16 chips, 2 beyond."""
    return 1 if mesh.get_num_devices() <= 16 else 2


def ep4_outputs(mesh, lc):
    """The head's outputs in EP-4 mode (build_router ep4): the topk rows [TX, 32] uint32 (words 0-15 the gate weights'
    fp32 bits, 16 + k the expert ids, 28 the route word) as [TX / 32, 1024] (4-KB pages of 32 tokens) and the dispatch
    route words [1, 1, 1, TX] uint32 (tokens >= T 0); the WS outputs it does not write as one-page placeholders."""
    rm, dram = ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
    e = lambda shape: ttnn.empty(shape, dtype=ttnn.uint32, layout=rm, device=mesh, memory_config=dram)
    small = e([1, 32])
    return dict(
        w=e([lc.TX // 32, 1024]), route=e([1, 1, 1, lc.TX * ep4_rw(mesh)]), idx_t=small, dest_t=small, counts_t=small, fence_t=small
    )


def build_router(mesh, lc, x_rm, wrT, ids_t, outs, logd=None, side=0, ep4=0):
    """generic_op mesh program of the head; returns (mesh program, io tensors, stamps). x_rm: [TX, H] bf16 row-major
    (rows >= T zero); outs: ws_head.head_outputs; logd: the debug logits^T tensor (written when given); side: the
    routed stream's lane groups G (> 0: each (t, k)'s result row t S + slot in a DRAM bank of the side of its expert's
    group, e % G >= G / 2 the upper half of the banks - WSR_SIDE); ep4: experts a chip column (> 0: EP-4 mode, outs from
    ep4_outputs - each token's topk row and dispatch route word, no per-expert exchange; one program a chip)."""
    K, NE, S, T, H = lc.K, lc.nexp, lc.S, lc.T, lc.H
    NET, KT = NE // 32, H // 32
    NB = KT // BK
    assert KT % BK == 0 and K <= 16
    grp, GS = router_groups(mesh, NET)
    cores = [c for g in grp for c in g]
    NC = len(cores)
    EPO = -(-NE // NC)
    NOWN = -(-NE // EPO)
    RTX = lc.TX // 32
    rng = token_ranges(RTX, len(grp))
    own_max = max(-(-n // GS) for _, n in rng)
    rcsrc = own_max * 32 * K
    cset = crs(cores)
    w16 = lambda words: sharded_l1(mesh, cores, [-(-words // 16), 16], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)
    tl = lambda tiles, dt: sharded_l1(mesh, cores, [tiles * 32, 32], dt, ttnn.TILE_LAYOUT)
    wl1 = tl(EPC * KT, ttnn.bfloat16)  # W_r^T slice (CB 0); after the barrier the exchange's record buffers
    ring = tl(NX * BK, ttnn.bfloat16)  # x ring (CB 1)
    relay = tl(NR * BK, ttnn.bfloat16)  # the hub's relay ring
    xst = tl(2 * BK, ttnn.bfloat16)  # x's row-major pieces (CB 3; DRAM reads: 64 B aligned)
    gring = tl(NG * GS, ttnn.float32)  # gather ring (CB 5)
    idl1 = tl(EPC, ttnn.uint16)  # id tiles (CB 6)
    bufs = dict(
        wst=w16(32 * 32 + 32 * ep4_rw(mesh) if ep4 else 32 * 16),
        rec=w16(rcsrc),
        srt=w16(rcsrc + 4 * NOWN),
        cnt_in=w16(NC),
        off_in=w16(NOWN),
        cnt_all=w16(NE),
        aux=w16(3 * 256),
        xflags=w16(4 * NX),
        xprog=w16(4 * GS),
        gcred=w16(4 * GS),
        garr=w16(4),
        rflags=w16(4 * NR),
        rgrant=w16(4),
        bscr=w16(4),
    )
    recbuf = wl1.buffer_address()
    la = recbuf + CAP * 4
    lb = la + (CAP + 32 * EPO) * 4
    assert lb + (CAP + 32 * EPO) * 4 <= recbuf + EPC * KT * 2048
    st = stamps_tensor(mesh, cores)
    debug = logd is not None
    logd = logd if debug else outs["w"]  # (an accessor's placeholder when not written)
    xy = lambda c: noc(mesh, *c)[0] | noc(mesh, *c)[1] << 8
    all_xy = [xy(c) for c in cores]
    rd_rt, cp_rt, rt_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rt_py = {}
    me = 0
    for g, (R0, n) in zip(grp, rng):
        rs = noc_rects(mesh, g)
        assert len(rs) == 1
        x0, y0, x1, y1 = rs[0][:4]
        rect = [x1, y1, x0, y0]  # (NOC1: the multicast starts at the far corner)
        mxy = [xy(c) for c in g]
        h = max(range(len(g)), key=lambda m: noc(mesh, *g[m])[::-1])  # (the hub: the NOC1 start end)
        assert noc(mesh, *g[h]) == (x1, y1)
        for c, (x, y) in enumerate(g):
            rd_rt[x][y] = [x_rm.buffer_address(), wrT.buffer_address(), ids_t.buffer_address(), R0, n, c] + [
                st.buffer_address(),
                mxy[h],
            ]
            cp_rt[x][y] = [n, c, st.buffer_address()]
            rt_py[(x, y)] = rt_rt[x][y] = (
                [me, R0, n, c, T]
                + [outs[k].buffer_address() for k in ("w", "idx_t", "dest_t", "counts_t", "fence_t")]
                + [st.buffer_address(), logd.buffer_address()]
                + rect
                + [h]
                + mxy
                + all_xy
            )
            me += 1
    ccfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    utd[5] = utd[10] = ttnn.UnpackToDestMode.UnpackToDestFp32
    ccfg.unpack_to_dest_mode = utd
    ta = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    defs = [("WSR_DEBUG_LOGITS", "1")] if debug else []
    defs += [("WSR_EP4", str(ep4)), ("WSR_EP4_SP", str(outs["route_sp"])), ("WSR_EP4_RW", str(ep4_rw(mesh)))] if ep4 else []
    if side:
        banks = mesh.dram_grid_size().x
        assert banks % 2 == 0 and side % 2 == 0
        rdefs = defs + [("WSR_SIDE", str(side)), ("WSR_SIDE_BANKS", str(banks))]
    else:
        rdefs = defs
    exch_ct = (
        [K, NE, S, EPO, NOWN, NC, CAP, lc.RS]
        + [bufs[k].buffer_address() for k in ("wst", "rec", "srt", "cnt_in", "off_in")]
        + [recbuf, la, lb, bufs["cnt_all"].buffer_address(), bufs["aux"].buffer_address()]
        + ta(outs["w"])
        + ta(outs["idx_t"])
        + ta(outs["dest_t"])
        + ta(outs["counts_t"])
        + ta(outs["fence_t"])
    )
    idm = 0xFF if NE <= 256 else 0x1FF if NE <= 512 else 0x3FF

    def route_kd(rt):
        return ttnn.KernelDescriptor(
            kernel_source=kpath("wsr2_route.cpp"),
            core_ranges=cset,
            compile_time_args=exch_ct
            + [BK, NX, KT, GS, NG, ring.buffer_address(), bufs["xflags"].buffer_address()]
            + [bufs[k].buffer_address() for k in ("xprog", "gcred", "garr")]
            + [gring.buffer_address(), idm, RTX, NR, relay.buffer_address()]
            + [bufs[k].buffer_address() for k in ("rflags", "rgrant")]
            + ta(logd)
            + (ta(outs["route_pieces"][0]) if ep4 else []),
            defines=rdefs,
            runtime_args=rt,
            config=dm(1),
        )

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=kpath("wsr2_reader.cpp"),
            core_ranges=cset,
            compile_time_args=[KT, BK, EPC, GS, H * 2, NR, relay.buffer_address()]
            + [bufs[k].buffer_address() for k in ("rflags", "rgrant", "bscr")]
            + ta(x_rm)
            + ta(wrT)
            + ta(ids_t),
            runtime_args=rd_rt,
            config=dm(0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=kpath("wsr2_compute.cpp"),
            core_ranges=cset,
            compile_time_args=[KT, BK, EPC, GS, K, LAG, NET],
            defines=defs,
            runtime_args=cp_rt,
            config=ccfg,
        ),
        route_kd(rt_rt),
    ]
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, wl1),
        ttnn.cb_descriptor_from_sharded_tensor(1, ring),
        cb(2, ttnn.bfloat16, 3 * BK, cset),
        ttnn.cb_descriptor_from_sharded_tensor(3, xst),
        cb(4, ttnn.float32, 2, cset),
        ttnn.cb_descriptor_from_sharded_tensor(5, gring),
        ttnn.cb_descriptor_from_sharded_tensor(6, idl1),
        cb(7, ttnn.float32, 1, cset),
        cb(8, ttnn.float32, 1, cset),
    ]
    if debug:
        cbs += [cb(9, ttnn.float32, 2 * EPC, cset), cb(10, ttnn.float32, EPC, cset)]
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cset, initial_value=0) for s in range(7)]
    if not ep4:
        mpd = mesh_program(mesh, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    else:  # (a program a chip: its row's first chip index r C and its column)
        R, Cn = tuple(mesh.shape)
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(R):
            for c in range(Cn):
                rt = ttnn.RuntimeArgs()
                for x, y in cores:
                    rt[x][y] = rt_py[(x, y)] + [r * Cn, c] + [t.buffer_address() for t in outs["route_pieces"]]
                ks = list(kernels[:2]) + [route_kd(rt)]
                mc = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(mc, mc)] = ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs)
    io = [x_rm, wrT, ids_t, wl1, ring, relay, xst, gring, idl1, st] + list(bufs.values())
    io += ([logd] if debug else []) + [outs[k] for k in ("w", "idx_t", "dest_t", "counts_t", "fence_t")]
    io += list(outs["route_pieces"]) if ep4 else []
    return mpd, io, st


def router_span_us(st, ci, clk):
    """The head program's stamps on chip ci, us from its earliest start: the weights in, the main loop done (router,
    top-K, records), the exchange's B and C done, the end (latest over the cores); and the cores' wait / work counters
    (max and mean): compute waiting for x blocks, for an owned row tile's runs, tilizing, the router matmuls, the keys
    and the pair's network, the owner's merges; BRISC waiting for x reads, for relay slots; NCRISC x multicasts (issue
    to sent), pieces in the relay waiting for credits / stream order (the hub), runs waiting for the owner's credit,
    the weights and records (QA)."""
    s = ttnn.to_torch(ttnn.get_device_tensors(st)[ci]).to(torch.int64) & 0xFFFFFFFF
    s = s.reshape(-1, 8 * 64)
    s0 = int(s[:, 0].min())
    rel = lambda v: ((v - s0 + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    out = {k: int(rel(s[:, i]).max()) / clk for k, i in (("weights", 5), ("loop", 2), ("exchange", 3), ("lists", 4))}
    out["end"] = int(rel(s[:, 63]).max()) / clk
    # per-core cycle counters (us): max and mean over the cores
    for k, i in (
        ("x_wait", 10),
        ("gather_wait", 11),
        ("tilize", 12),
        ("matmul", 13),
        ("keys_sort", 14),
        ("owner_merge", 15),
        ("x_read", 6),
        ("relay_grant", 7),
        ("x_mcast", 20),
        ("x_credit", 21),
        ("run_credit", 22),
        ("ncrisc_qa", 23),
    ):
        v = s[:, i].double() / clk
        out[f"{k}_max"], out[f"{k}_mean"] = float(v.max()), float(v.mean())
    return out
