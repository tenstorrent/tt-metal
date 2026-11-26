# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""W7 step 5 (coordinator 2026-10-25 18:48:11Z; design doc/ttnn/16-prefill-production/pmk/w7/step5_gang_design.md):
program G -- a layer's EP routed MoE on GANGS as ONE program, the form that replaces program B inside the fused layer
program:
  plan    (kernels/pmk/plan_{count,dest,runs}.hpp: the served mc2 plan at the EP geometry -- E = this chip's local
           experts, e = id - lo, lo = E r; PMK_EP_SENTINEL: another chip's pair gets dest 0xFFFFFFFF),
  gang    (kernels/pmk/gang_{reader,writer,compute}.hpp = Stage 14's moe_xffn_gang_*.cpp, a642725faca: unit = (local
           expert, M group of <= MG tile rows) on a gang of G = 8 cores, core s = W slice s of the full-width expert, the
           8 partial y blocks ring-reduce-scattered in L1 (CB_RIN, SEM_RING_VALID / SEM_RING_CREDIT), every pair row
           written once to out_rm by its N chunk's owner),
  combine (kernels/pmk/comb_*.hpp, PMK_RM_SKIP: the sentinel / weight-0 pairs read one zeroed part; PMK_SHARED: + y_s,
           the P13 partial, as the (K + 1)-th term)
over the SAME P gang cores (every core of every gang runs all three: kernels/pmk/g_{brisc,ncrisc,trisc}.cpp), barriers
on semaphores 5..8 as program B's. The gang core set is the caller's (14's ask (c): W7 places the 13 gangs over its
roles) in ring order; EpMoe.gang_ring's rules are checked here (gang_ring).
Standalone: `PmkGang(...)(x_ext, ids_rm, wt, c)` -> y [1, 1, C, H] bf16 TILE (this chip's routed partial, + y_s when
given). parts() gives the fused program's window (tt/pmk_b_window.register_g) the raw pieces."""

from __future__ import annotations

import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as V1

TILE, BF16, FP32, DRAM = V1.TILE, V1.BF16, V1.FP32, V1.DRAM
G = 8  # cores per gang (14's GANG)
GANG_FOLD = (0, 2, 4, 6, 7, 5, 3, 1)  # ring position -> grid offset: every hop <= 2 cores (14's GANG_FOLD)
FID_VALUE = dict(LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4)  # pmk_fid.hpp's template value
PMK_GANG_REV = 2  # kernel revision (defines: a long-lived process rebuilds after a kernel edit)


def gang_sets(gx: int, gy: int) -> list:
    """14's default gangs on a gx x gy grid: one per grid row over columns 0..G-1, one per remaining column over rows
    0..G-1 (13 on the 11 x 10 grid), each in GANG_FOLD ring order"""
    gangs = [[(GANG_FOLD[s], y) for s in range(G)] for y in range(gy)] if gx >= G else []
    gangs += [[(x, GANG_FOLD[s]) for s in range(G)] for x in range(G, gx)] if gy >= G else []
    return gangs


def gang_ring(gangs: list) -> dict:
    """14's ring contract (EpMoe.gang_ring): every gang = G distinct logical cores in RING order, no core in two gangs;
    position s holds W slice s and sends to s + 1 mod G -> {core: (gang id, s, successor, predecessor)}"""
    out = {}
    for g_, gang in enumerate(gangs):
        gang = [tuple(int(v) for v in c_) for c_ in gang]
        assert len(gang) == G and len(set(gang)) == G, ("a gang = G distinct cores", g_, gang)
        for s_, c_ in enumerate(gang):
            assert c_ not in out, ("core in two gangs", c_)
            out[c_] = (g_, s_, gang[(s_ + 1) % G], gang[(s_ - 1) % G])
    assert out, "no gang"
    return out


def gang_blocks(h_t: int) -> tuple:
    """14's gang blocks: MG 4 tile rows per unit, KC 8, NC = H_T / (2 G) (the 2 G N chunks of the ring)"""
    nc = h_t // (2 * G)
    assert nc * 2 * G == h_t and nc % 4 == 0, h_t
    return 4, 8, nc


def _h9_args(h9) -> list:
    """the H9 runtime args: list [(x, y, base)] = every consumer's per-32-row-block counter words (base + 16 block); dict
    (TABLE mode, non-B lever (2)) dict(targets=[(x, y, sync base)], table=[(target index, word)] per row block): the unit
    of block b raises only targets[t].base + 16 w, (t, w) = table[b]"""
    if not h9:
        return [0]
    if isinstance(h9, dict):
        tg, tb = list(h9["targets"]), list(h9["table"])
        return [len(tg)] + [int(v) for xyb in tg for v in xyb] + [int(v) for tw in tb for v in tw]
    return [len(h9)] + [int(v) for xyb in h9 for v in xyb]


class PmkGang(V1.PmkMoe):
    CB_INDEX = dict(V1.PmkMoe.CB_INDEX, RIN=19, CZERO=20)  # moe_common.hpp: PMK_MOE_CBX_RIN / CZERO
    CB_FMT = dict(V1.PmkMoe.CB_FMT, RIN=BF16)

    def __init__(self, mesh, *, e_loc: int, top_k: int, hidden: int, limit: float, gate_up: list, down: list, gangs: list,
                 lo: list, fid: str = "LoFi", ys: bool = False, wt_bf16: bool = False, ids_u16: bool = False,
                 h_split: int = 2, probe: bool = False, cb0: int = 32, cb1: int = 27):  # fmt: skip
        """gate_up / down: this chip's E = e_loc LOCAL experts, FULL width (EP), ND [H, 2 tiles] shards (14's
        XFFN_SHARD_BASES layout); gangs: [[(x, y)] * G per gang, ring order]; lo: per chip (mesh order) the global id of
        its local expert 0 (contiguous EP: E r); ys: the combine adds y_s (PMK_SHARED); wt_bf16: the weights are the P12
        tail's bf16 ROW_MAJOR [1, C K] (PMK_WT_BF16) instead of [1, 1, C, K] fp32 TILE; ids_u16: the ids are the P12
        tail's [1, C K] uint16 ROW_MAJOR (PMK_IDS_U16, token-major) instead of [1, K C] uint32; cb0 / cb1: the CB bases
        (standalone: 32.. 52 and 27, 28 -- the combine's tilize CBs < 32)."""
        self.ring = gang_ring(gangs)
        self.gangs = [[tuple(int(v) for v in c_) for c_ in g_] for g_ in gangs]
        cores = [c_ for g_ in self.gangs for c_ in g_]
        super().__init__(mesh, num_experts=e_loc, top_k=top_k, hidden=hidden, limit=limit, gate_up=gate_up, down=down,
                         cores=cores, cb0=cb0, cb1=cb1, h_split=h_split, fid=fid, probe=probe)  # fmt: skip
        self.w_full = self.w_t  # V1's w_t = the full expert width here (EP)
        assert self.w_full % G == 0, self.w_full
        self.w_slice = self.w_full // G
        self.lo = [int(v) for v in lo]
        assert len(self.lo) == int(mesh.get_num_devices()), (self.lo, mesh.get_num_devices())
        self.ys, self.wt_bf16, self.ids_u16 = bool(ys), bool(wt_bf16), bool(ids_u16)
        assert self.shard_bases(), "G: the experts must be ND [H, 2 tiles] shards (XFFN_SHARD_BASES)"
        assert cb0 + max(self.CB_INDEX.values()) < 64 and cb1 + 1 < 32, (cb0, cb1)

    # ------------------------------------------------------------------------------------------ geometry / budget
    def geometry(self, c: int) -> V1.Geometry:
        """the plan / combine geometry: E = e_loc; the FFN CBs at the gang slice width (W / G)"""
        return V1.Geometry(c=c, h=self.h, w=self.w_slice * TILE, e=self.e, k=self.k, p=self.p, h_split=self.h_split)

    def blocks(self, c: int) -> tuple:
        return gang_blocks(self.h_t)

    def cb_plan(self, c: int) -> dict:
        """V1's plan + RIN (its own allocation: the ring is live during the whole FFN) + CZERO (the RM_SKIP zero part,
        combine-time: aliased into ACT); with y_s the combine's CIN / CW hold K + 1 terms"""
        g = self.geometry(c)
        mg, kc, nc = self.blocks(c)
        ffn = V1.ffn_cb_bytes(g, mg, kc, nc)
        comb = V1.combine_cb_bytes(g)
        part = self.h * 2 // self.h_split
        pt = part // 2048
        kk = self.k + (1 if self.ys else 0)
        comb.update(CIN=2 * kk * pt * 2048, CW=2 * kk * 4096)
        pw = V1.plan_words(g)
        plan_w = max(pw["count"], pw["dest"], pw["runs"]) + (
            4 * pw["chunk"] + 16 if self.ids_u16 else 0
        )  # (pmk_moe2's rule)
        sizes = {**ffn, "RIN": 2 * mg * nc * 2048, **comb, "CZERO": part, "PLAN": plan_w * 4}  # fmt: skip
        blocks = V1.push_blocks(g, mg, kc, nc, kk=kk)
        blocks.update(RIN=nc * 2048)
        # V1's aliases lose COUT (WGU's bfp4 pages / blocks round it past the slack) and CW (likewise in WD's group):
        # +192 KB at the gang blocks. Their FFN-only partners of the SAME size and block here: COUT on RIN (the ring is
        # drained before SEM_FFN: the writer's end-of-body credit drain, 14's contract 3), CW on XT, CZERO on ACT
        aliases = dict(
            WGU=("PLAN",), GUP=("CRM",), XRM=("CIN",), TAB=("CIDX",), ACT=("CZERO",), XT=("CW",), RIN=("COUT",)
        )
        if self.ys:  # K + 1 terms: CIN / CW blocks 36,864 B fit WGU's / WD's 73,728-B blocks, not XRM's / XT's 65,536
            aliases.update(WGU=("PLAN", "CIN"), XRM=(), XT=(), WD=("CW",))
        # CZERO: a RAW buffer (comb_reader reads its base; no fifo use) -> ACT's 2 KB page instead of one 64-KB page: the
        # reclaim window's launch declaration of a fresh index is ONE page (cf3h-k B1: -62 KB static on member / MM-only)
        pages = dict(V1.PAGE, RIN=2048, CZERO=2048)
        groups = V1.alias_groups(sizes, aliases, pages, blocks, list(ffn) + ["RIN"])
        assert not V1.check_pushes(groups, blocks), V1.check_pushes(groups, blocks)
        return dict(groups=groups, total=sum(t for _, t, _ in groups), blocks=blocks, kk=kk, part=part, pt=pt, plan=pw)

    def l1_bytes(self, c: int) -> int:
        return self.cb_plan(c)["total"]

    # ------------------------------------------------------------------------------------------ device tables
    def tables(self, c: int) -> dict:
        new = c not in self.bufs
        b = super().tables(c)
        if new:  # EP: the chip's lo (V1 wrote lo = 0 replicated)
            ttnn.deallocate(b["lo"])
            lo = torch.zeros(len(self.lo), 32, dtype=torch.int32)
            lo[:, 0] = torch.tensor(self.lo, dtype=torch.int32)
            mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0) if len(self.lo) > 1 else None
            b["lo"] = ttnn.from_torch(lo, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                      memory_config=DRAM, mesh_mapper=mapper)  # fmt: skip
        return b

    # ------------------------------------------------------------------------------------------ the program pieces
    def parts(self, x_ext, ids_rm, wt, out_rm, y, c: int, *, y_s=None, wait=None, h9=None) -> tuple:
        """as V1.PmkMoe.parts with the gang FFN: x_ext [C (+ 32), H] bf16 ROW_MAJOR (valid rows read only); ids_rm [1, K C]
        uint32 ROW_MAJOR (global ids, pair p = j C + t); wt [1, 1, C, K] fp32 TILE (or [1, C K] bf16 ROW_MAJOR with
        wt_bf16); out_rm [CAP, H] bf16 ROW_MAJOR; y [1, 1, C, H] bf16 TILE; y_s [C, H] bf16 TILE (with ys)."""
        assert (y_s is not None) == self.ys, ("ys", self.ys, y_s is None)
        g = self.geometry(c)
        b = self.tables(c)
        mg, kc, nc = self.blocks(c)
        plan = self.cb_plan(c)
        P, cores = self.p, self.cores
        acc = lambda t_: list(ttnn.TensorAccessorArgs(t_).get_compile_time_args())  # noqa: E731
        ids, lo, hist = ids_rm.buffer_address(), b["lo"].buffer_address(), b["hist"].buffer_address()
        dest, cnt, start, meta = (b[k_].buffer_address() for k_ in ("dest", "cnt", "start", "meta"))
        stage, disp, tbl = b["stage"].buffer_address(), b["disp"].buffer_address(), self.tbl.buffer_address()
        xa, oa, ya, wa = x_ext.buffer_address(), out_rm.buffer_address(), y.buffer_address(), wt.buffer_address()
        pa = self.bufs["probe"].buffer_address() if self.probe else 0
        hs, part, pt = self.h_split, plan["part"], plan["pt"]
        units = (c // TILE) * hs
        per = -(-units // P)
        cunits = lambda i: (min(i * per, units), min(per, units - min(i * per, units)))  # noqa: E731
        # PMK_COMB_STRIDE=1 (pmk_moe's dealing, G v2b's streamed P15): core i takes units i, i + P, .. (blocks complete mid-
        # combine); register_g's per-rank start offset wraps modulo comb_units
        self.comb_stride = int(os.environ.get("PMK_COMB_STRIDE", "0") or 0) == 1
        self.comb_units = units
        if self.comb_stride:
            cunits = lambda i: (i, len(range(i, units, P)) if i < units else 0)  # noqa: E731
        subw = suba = 4
        # ---- compile-time args per RISC (phase bases as defines)
        ct_count = [g.n, self.e, g.ep, b["chunk"]]
        ct_dest = [g.n, self.e, g.ep, b["chunk"], P, c]
        ct_runs = [self.e, g.ep, b["chunk"], P, c, g.cap, b["epc"], b["rmax"]]
        ct_gang_r = (
            [self.h_t, self.w_slice, kc, nc, mg, self.e, g.ep, P, V1.WTB_BFP4]
            + acc(self.gate_up[0])
            + acc(self.down[0])
        )
        ct_comb_w = [pt, self.h_t, hs]
        ct_w, bases_w = [], {}
        for name, ct_ in (
            ("COUNT", ct_count),
            ("DEST", ct_dest),
            ("RUNS", ct_runs),
            ("GANG", ct_gang_r),
            ("COMB", ct_comb_w),
        ):
            bases_w[name] = len(ct_w)
            ct_w += ct_
        ct_gang_a, ct_comb_r = [self.h_t, nc, kc, mg], [c, self.k, self.h * 2, part, pt, hs]
        ct_a = ct_gang_a + ct_comb_r
        bases_a = dict(GANG=0, COMB=len(ct_gang_a))
        ct_gang_c, ct_comb_c = [self.h_t, self.w_slice, kc, nc, mg, subw, suba], [self.k, pt]
        ct_c = ct_gang_c + ct_comb_c
        bases_c = dict(GANG=0, COMB=len(ct_gang_c))
        # ---- runtime args per core
        flat_noc = [v for xy in self.noc for v in xy]
        noc_of = dict(zip(cores, self.noc))
        rt_w, rt_a, rt_c, rbases_w, rbases_a = {}, {}, {}, {}, {}
        for i, xy in enumerate(cores):
            u0, nu = cunits(i)
            gid, s_, succ, pred = self.ring[xy]
            segs = [("COUNT", [ids, lo, hist, i]), ("DEST", [ids, lo, hist, dest, cnt, start, meta, i, stage]),
                    ("RUNS", [hist, cnt, start, meta, stage, disp, i]),
                    ("GANG", [xa, disp, cnt, start, tbl, i, g.cap, pa, gid, s_]), ("COMB", [ya, u0, nu])]  # fmt: skip
            args = []
            for name, a_ in segs:
                rbases_w[name] = len(args)
                args += a_
            rbases_w["CORES"] = len(args)
            args += flat_noc
            rbases_w["WAIT"] = len(args)
            args += [int(wait[1])] if wait else [0]
            rbases_w["H9"] = len(args)
            args += _h9_args(h9)
            rbases_w["STREAMERS"] = len(args)  # (b_wrap.hpp's hand-over release list: G has no streamers)
            args += [0]
            rt_w[xy] = args
            (sx, sy), (px, py) = noc_of[succ], noc_of[pred]
            a_args = [oa, xa, disp, g.cap, pa, P, i, s_, sx, sy, px, py]
            rbases_a["GANG"] = 0
            rbases_a["COMB"] = len(a_args)
            a_args += [oa, dest, wa, u0, nu]
            rbases_a["CORES"] = len(a_args)
            a_args += flat_noc
            if self.ys:
                rbases_a["YS"] = len(a_args)
                a_args += [y_s.buffer_address()]
            rt_a[xy] = a_args
            rt_c[xy] = [nu]
        # ---- defines
        common = [("PMK_MOE_CB0", str(self.cb0)), ("PMK_MOE_CB1", str(self.cb1)), ("PMK_P", str(P)),
                  ("PMK_MOE_REV", str(V1.PMK_MOE_REV)), ("PMK_GANG_REV", str(PMK_GANG_REV)), ("PLAN_SORTED", "1"),
                  ("XFFN_GANG", str(G)), ("XFFN_GANG_NG", str(len(self.gangs))), ("XFFN_SHARD_BASES", "1"),
                  ("XFFN_X_ON_NOC1", "1"), ("XFFN_VALID_ROWS", "1"), ("PMK_EP_SENTINEL", "1"), ("PMK_RM_SKIP", "1"),
                  ("PMK_CW_PAGES", str(2 * plan["kk"]))]  # fmt: skip
        if self.ys:
            common.append(("PMK_SHARED", "1"))
        if self.wt_bf16:
            common.append(("PMK_WT_BF16", "1"))
        if self.ids_u16:
            common += [("PMK_IDS_U16", "1"), ("PMK_IDS_K", str(self.k))]
        if self.probe:
            common.append(("XFFN_PROBE", "1"))
        d_w = common + [(f"PMK_CT_{k_}", str(v)) for k_, v in bases_w.items()] + \
            [(f"PMK_RT_{k_}", str(v)) for k_, v in rbases_w.items()]  # fmt: skip
        if wait:
            d_w.append(("PMK_MOE_WAIT_SEM", str(int(wait[0]))))
        if self.comb_stride:
            d_w += [("PMK_COMB_STRIDE", str(P)), ("PMK_COMB_UNITS", str(units))]
        if h9:
            d_w.append(("PMK_H9", "1"))
            if isinstance(h9, dict):  # the TABLE mode (P15 streamed behind the combine): one word per row block
                d_w.append(("PMK_H9_TABLE", "1"))
        d_a = common + [(f"PMK_CT_{k_}", str(v)) for k_, v in bases_a.items()] + \
            [(f"PMK_RT_{k_}", str(v)) for k_, v in rbases_a.items()]  # fmt: skip
        if self.comb_stride:  # the combine reader's unit loop strides with the writer's
            d_a += [("PMK_COMB_STRIDE", str(P)), ("PMK_COMB_UNITS", str(units))]
        # the kernel fidelity = the served rm combine's HiFi4 (its bf16 row x fp32 weight multiplies: LoFi there read
        # 0.975 x the served combine, d0 gangw 2026-10-25); the gang's matmuls at self.fid (LoFi = HiFi2 on bfp4: D6)
        d_c = (
            common
            + [(f"PMK_CT_{k_}", str(v)) for k_, v in bases_c.items()]
            + [("PMK_RT_GANG", "0"), ("PMK_RT_COMB", "0"), ("PMK_GANG_FID", str(FID_VALUE[self.fid]))]
        )
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, dst_full_sync_en=False)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[self.cb_index("GUP")] = ttnn.UnpackToDestMode.UnpackToDestFp32  # CB_GUP: the fp32 partials copied exactly
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        raw = dict(
            w=("g_brisc.cpp", ct_w, rt_w, d_w, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            a=("g_ncrisc.cpp", ct_a, rt_a, d_a, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
            c=("g_trisc.cpp", ct_c, rt_c, d_c, cc),
        )
        cbs = self.cb_descriptors(plan)
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.crs, initial_value=0) for i in (5, 6, 7, 8, 12, 13)]
        io = [x_ext, ids_rm, wt, b["lo"], b["hist"], b["dest"], b["cnt"], b["start"], b["meta"], b["stage"], b["disp"],
              self.tbl, out_rm, y] + ([y_s] if self.ys else []) + list(self.gate_up) + list(self.down)  # fmt: skip
        if self.probe:
            io.append(self.bufs["probe"])
        return raw, cbs, sems, io

    def program(self, x_ext, ids_rm, wt, out_rm, y, c: int, **kw):
        raw, cbs, sems, io = self.parts(x_ext, ids_rm, wt, out_rm, y, c, **kw)
        from models.autoports.zai_org_glm_5_3_flash.tt.cb_guard import check

        check(cbs, "pmk_gang")
        kernels = [self.kernel(*raw[r]) for r in ("w", "a", "c")]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs), io

    def __call__(self, x_ext, ids_rm, wt, c: int, *, y_s=None, **kw):
        """standalone: ONE program (plan + gang FFN + combine) -> y [1, 1, C, H] bf16 TILE (a persistent scratch: copy it
        out before the next call)"""
        s = self.scratch(c)
        prog, io = self.program(x_ext, ids_rm, wt, s["out_rm"], s["y"], c, y_s=y_s, **kw)
        ttnn.generic_op(io, prog)
        self.calls += 1
        return s["y"]
