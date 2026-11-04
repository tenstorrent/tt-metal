# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a: the PMK layer's P2 -> P3 in ONE program, the TP2 pair or the TP8 2x4 (FabricStreamC1X: the partner over axis 0
and the row over axis 1, the partner's blocks forwarded along the row) (prefill_megakernel.md §4b H1 -> H2 -> H3).

- P1 stand-in (H1): on every chip the holders of its LOCAL blocks fill their XS (CB 35) from DRAM slice by slice, +1
  their own FS_ARRIVE (15) and the lane's LREADY + k per slice (kernels/fabric_stream/c1_stub_prod.cpp);
- P2 (tt/fabric_stream.FabricStreamC1, the AG): the lanes write every local block into the same XS on the peer's holder,
  a fused FS_ARRIVE +1 per slice;
- H2 -> P3 (tt/pf_stream.PfStreamMM): member G waits FS_ARRIVE >= SLICES on its own XS (local or landed block), hands it
  to the compute as the stationary in0, and the qkv projection streams its weights once over the 64 members.
- The members' NCRISC runs [the H1 stub producer] then [the PfStream reader] as one composite (the PMK pattern).
"""

from __future__ import annotations

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS
from models.autoports.zai_org_glm_5_3_flash.tt import pf_stream as PF


class PfAg:
    def __init__(self, mesh, w, *, members, streamers, lanes, nb: int = 4, nw: int = 32, kb: int = 16, park: int = 4,
                 probe: bool = False):  # fmt: skip
        n = int(mesh.get_num_devices())
        assert n in (2, 8), "the pair (TP2) or the 2x4 (TP8: FabricStreamC1X)"
        self.mesh = mesh
        self.members = [tuple(c) for c in members]
        self.lanes = [tuple(c) for c in lanes]
        assert not set(self.lanes) & (set(self.members) | {tuple(c) for c in streamers})
        self.mm = PF.PfStreamMM(mesh, w, members=members, streamers=streamers, nb=nb, nw=nw, kb=kb, park=park,
                                probe=probe)  # fmt: skip
        self.bl = len(self.members) // n
        cls = FS.FabricStreamC1X if n == 8 else FS.FabricStreamC1
        self.c1 = cls(mesh, self.bl, mm_cores=self.members, lanes=self.lanes)
        self.y = self.mm.y

    def program(self, x, r: int, c: int) -> ttnn.ProgramDescriptor:
        mm, c1 = self.mm, self.c1
        p = c1._pos(r, c)
        pt = mm.parts(mm.y, wait=(FS.SEM_FS_ARRIVE, c1.tpb // c1.slice_t))  # no PFS_UNIT: x̂ arrives by H1 / H2
        defs = list(pt["defines"])
        ck, ccbs, _ = c1.parts(r, c)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        sync = c1.sync.buffer_address()
        prod_ct = [c1.tpb, c1.slice_t, FS.CB_XS, FS.SEM_FS_ARRIVE] + acc(x)
        prod_rt = {}
        for j, m in enumerate(self.members):
            b = j - p * self.bl
            if 0 <= b < self.bl:
                lane = mm._noc(self.lanes[b % len(self.lanes)])
                me = mm._noc(m)
                prod_rt[m] = [
                    1,
                    x.buffer_address(),
                    b,
                    lane[0],
                    lane[1],
                    sync,
                    b // len(self.lanes),
                    me[0],
                    me[1],
                    0,
                    0,
                    0,
                ]
            else:
                prod_rt[m] = [0] * 12
        ct0 = pt["ct"]

        def rt_of(cores, pre=None):
            rr = ttnn.RuntimeArgs()
            for cc in cores:
                rr[cc[0]][cc[1]] = (pre[cc] if pre else []) + pt["rt"][cc]
            return rr

        dyn = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                noc=ttnn.NOC.RISCV_0_default, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)  # fmt: skip
        cc_ = ttnn.ComputeConfigDescriptor(**pt["compute"], math_approx_mode=False)
        kd = lambda src, cores, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=PF._crs(cores),
            compile_time_args=ct, runtime_args=rt, config=cfg, defines=list(defs))  # fmt: skip
        s, m = pt["sources"]["streamer"], pt["sources"]["member"]
        # members' NCRISC: [H1 stub producer (phase 0)] then [PfStream reader (phase 1)]
        member_dm1 = PF.composite_source([c1._srcs["c1_stub_prod.cpp"], m["dm1"]], [0, len(prod_ct)], [0, 12], "dm")
        kernels = list(ck) + [
            kd(s["dm0"], mm.all_streamers, ct0, rt_of(mm.all_streamers), dyn),
            kd(member_dm1, self.members, prod_ct + ct0, rt_of(self.members, prod_rt), ttnn.ReaderConfigDescriptor()),
            kd(m["c"], self.members, ct0, rt_of(self.members), cc_),
            kd(m["dm0"], self.members, ct0, rt_of(self.members), ttnn.WriterConfigDescriptor()),
        ]  # fmt: skip
        cbs = list(ccbs)
        for role, cores in (("streamer", mm.all_streamers), ("member", self.members)):
            for idx, dt, page, pages in pt["cbs"][role]:
                cbs.append(ttnn.CBDescriptor(total_size=page * pages, core_ranges=PF._crs(cores), format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)]))  # fmt: skip
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=PF._crs(self.members), initial_value=0)
                for i in list(pt["sems"]["member"]) + [FS.SEM_FS_ARRIVE]]  # fmt: skip
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        c1.connect(prog, r, c, 0)  # ck[0] = c1_send
        return prog

    def __call__(self, x):
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(int(self.mesh.shape[0])):
            for c in range(int(self.mesh.shape[1])):
                mc = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(mc, mc)] = self.program(x, r, c)
        io = [x, self.mm.weight, self.mm.y, self.mm.sync, self.c1.sync] + ([self.mm.mprobe] if self.mm.probe else [])
        ttnn.generic_op(io, mpd)
        return self.mm.y

    def release(self):
        self.mm.release()
        self.c1.release()


class PfAgN:
    """n x (H1 -> AG -> H2 -> PfStream) in ONE program, the XS (CB 35) REUSED by every instance (the layer's P2 -> P3 and
    P11 -> P12/P13 both land x̂ in XS): PMK H0 XS-FREE on silicon. Instance i: the holders credit their lane (G mod 2)'s
    XSFREE word (instance 0 at once: XS is free at launch; later instances after the compute popped the previous
    stationary block), the lanes wait 32 credits per use before crediting the peer, H2 waits FS_ARRIVE >= 8 (i + 1)
    (cumulative in the launch), the PfStream phases use sync slot i and QBASE = the earlier phases' blocks. The lanes run
    the n C1 phases in one composite per RISC, every phase opening / closing the same fabric connection (fab_at)."""

    def __init__(self, mesh, ws: list, *, members, streamers, lanes, nb: int = 4, nw: int = 32, kb: int = 16,
                 park: int = 4):  # fmt: skip
        assert int(mesh.get_num_devices()) == 2, "the pair (TP2)"
        self.mesh = mesh
        self.members = [tuple(c) for c in members]
        self.lanes = [tuple(c) for c in lanes]
        sync = PF.PfStreamMM.make_sync(mesh, [tuple(c) for c in streamers], slots=len(ws))
        self.mms = [PF.PfStreamMM(mesh, w, members=members, streamers=streamers, nb=nb, nw=nw, kb=kb, park=park,
                                  sync=sync) for w in ws]  # fmt: skip
        self.sync = sync
        self.bl = len(self.members) // 2
        self.c1 = FS.FabricStreamC1(mesh, self.bl, mm_cores=self.members, lanes=self.lanes)

    def program(self, xs: list, r: int, c: int) -> ttnn.ProgramDescriptor:
        c1, mms, n = self.c1, self.mms, len(self.mms)
        p = c1._pos(r, c)
        slices = c1.tpb // c1.slice_t
        nl = len(self.lanes)
        # the lanes: n C1 phases; the fabric args after ALL of them (fab_at = the total)
        c1.parts(r, c, xsfree_n=self.bl, fab_at=0)
        send_len = len(next(iter(c1._rt_last.values()))[0])
        read_len = len(next(iter(c1._rt_last.values()))[1])
        ck, ccbs, _ = c1.parts(r, c, xsfree_n=self.bl, fab_at=n * send_len)
        lane_send = {ln: c1._rt_last[ln][0] * n for ln in self.lanes}  # the same lists per instance (g counts uses)
        lane_read = {ln: c1._rt_last[ln][1] * n for ln in self.lanes}
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        fsync = c1.sync.buffer_address()
        qbase, pts, prod_cts, prod_rts = 0, [], [], []
        for i, (mm, x) in enumerate(zip(mms, xs)):
            pts.append(mm.parts(mm.y, qbase=qbase, slot=i, wait=(FS.SEM_FS_ARRIVE, slices * (i + 1))))
            qbase += mm.blocks
            prod_cts.append([c1.tpb, c1.slice_t, FS.CB_XS, FS.SEM_FS_ARRIVE] + acc(x))
            rt = {}
            for j, m in enumerate(self.members):
                b = j - p * self.bl
                xl = mms[0]._noc(self.lanes[j % nl])  # the XS-FREE lane: G mod 2 (the peer's lane G mod 2 writes here)
                xsf = [1 if i == 0 else 2, xl[0], xl[1]]
                if 0 <= b < self.bl:
                    lane = mms[0]._noc(self.lanes[b % nl])
                    me = mms[0]._noc(m)
                    rt[m] = [1, x.buffer_address(), b, lane[0], lane[1], fsync, b // nl, me[0], me[1]] + xsf
                else:
                    rt[m] = [0, 0, 0, 0, 0, fsync, 0, 0, 0] + xsf
            prod_rts.append(rt)
        m0 = mms[0]
        defs = sorted({d for pt in pts for d in pt["defines"]})
        dyn = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                noc=ttnn.NOC.RISCV_0_default, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)  # fmt: skip
        cc_ = ttnn.ComputeConfigDescriptor(**pts[0]["compute"], math_approx_mode=False)

        def comp(srcs, cts, rts, cores, risc, cfg, startup=""):
            cbases = [sum(len(x_) for x_ in cts[:i]) for i in range(len(cts))]
            lens = [len(next(iter(r_.values()))) for r_ in rts]
            rbases = [sum(lens[:i]) for i in range(len(rts))]
            rr = ttnn.RuntimeArgs()
            for cc in cores:
                rr[cc[0]][cc[1]] = [v for r_ in rts for v in r_[cc]]
            return ttnn.KernelDescriptor(kernel_source=PF.composite_source(srcs, cbases, rbases, risc, startup),
                                         source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                         core_ranges=PF._crs(cores), compile_time_args=[v for x_ in cts for v in x_],
                                         runtime_args=rr, config=cfg, defines=defs)  # fmt: skip

        send_src, read_src = c1._srcs["c1_send.cpp"], c1._srcs["c1_read.cpp"]
        send_ct = [c1.tpb, c1.nb, FS.CB_STAGE, c1.slice_t]
        read_ct = [c1.tpb, c1.nb, FS.CB_STAGE, slices, FS.W_XSADDR]
        lane_k = [
            comp([send_src] * n, [send_ct] * n, [{ln: lane_send[ln][i * send_len:(i + 1) * send_len] for ln in self.lanes}
                                                  for i in range(n)], self.lanes, "dm", ttnn.WriterConfigDescriptor()),
            comp([read_src] * n, [read_ct] * n, [{ln: lane_read[ln][i * read_len:(i + 1) * read_len] for ln in self.lanes}
                                                  for i in range(n)], self.lanes, "dm", ttnn.ReaderConfigDescriptor()),
        ]  # fmt: skip
        del ck
        s_src = [pt["sources"]["streamer"]["dm0"] for pt in pts]
        mem = [pt["sources"]["member"] for pt in pts]
        startup = "    compute_kernel_hw_startup<SrcOrder::Reverse>(%d, %d, %d);" % (PF.CB_X, PF.CB_PARK, PF.CB_OUT)
        dm1_srcs, dm1_cts, dm1_rts = [], [], []
        for i in range(n):
            dm1_srcs += [c1._srcs["c1_stub_prod.cpp"], mem[i]["dm1"]]
            dm1_cts += [prod_cts[i], pts[i]["ct"]]
            dm1_rts += [prod_rts[i], pts[i]["rt"]]
        kernels = lane_k + [
            comp(s_src, [pt["ct"] for pt in pts], [pt["rt"] for pt in pts], m0.all_streamers, "dm", dyn),
            comp(dm1_srcs, dm1_cts, dm1_rts, self.members, "dm", ttnn.ReaderConfigDescriptor()),
            comp([x_["c"] for x_ in mem], [pt["ct"] for pt in pts], [pt["rt"] for pt in pts], self.members, "c", cc_, startup),
            comp([x_["dm0"] for x_ in mem], [pt["ct"] for pt in pts], [pt["rt"] for pt in pts], self.members, "dm",
                 ttnn.WriterConfigDescriptor()),
        ]  # fmt: skip
        cbs = list(ccbs)
        for role, cores in (("streamer", m0.all_streamers), ("member", self.members)):
            seen = {}
            for pt in pts:
                for idx, dt, page, pages in pt["cbs"][role]:
                    prev = seen.get(idx)
                    seen[idx] = (dt, page, max(pages, prev[2] if prev else 0))
            for idx, (dt, page, pages) in seen.items():
                cbs.append(ttnn.CBDescriptor(total_size=page * pages, core_ranges=PF._crs(cores), format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)]))  # fmt: skip
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=PF._crs(self.members), initial_value=0)
                for i in list(pts[0]["sems"]["member"]) + [FS.SEM_FS_ARRIVE]]  # fmt: skip
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        c1.connect(prog, r, c, 0)  # kernels[0] = the lanes' send composite: the fabric args go after all phases
        return prog

    def __call__(self, xs: list):
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(int(self.mesh.shape[0])):
            for c in range(int(self.mesh.shape[1])):
                mc = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(mc, mc)] = self.program(xs, r, c)
        io = list(xs) + [t for mm in self.mms for t in (mm.weight, mm.y)] + [self.sync, self.c1.sync]
        ttnn.generic_op(io, mpd)
        return [mm.y for mm in self.mms]

    def release(self):
        for mm in self.mms:
            mm.release()
        ttnn.deallocate(self.sync)
        self.c1.release()
