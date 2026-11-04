# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a: the PMK layer's P7 -> P8 path on the TP2 pair in ONE program (prefill_megakernel.md §4b H5 -> H6).

- P7 (tt/pf_stream.PfStreamMM): the o_proj projection on the 64 members (row block G each, weights streamed once): every
  chip's partial P [2048, 4096] bf16 (DRAM; chip c's own K slice).
- H5: member G, after its whole row block of P is stored, raises +1 on the word of whoever consumes that block: the RS
  lane that sends it to its owner (peer-owned block b: lane b mod 2, LREADY + b / 2) or the reducer that adds it (own
  block b: reducer b mod 8, OWNRDY + b / 8). (FabricStreamRSR ready=True.)
- P8 (tt/fabric_stream.FabricStreamRSR, reducers + members): the lanes send the peer-owned blocks; 8 reducer cores add
  own + landed in exact fp32 and write y fp32 into the 64 members' POST 21 + FS_REDUCED (H6), under POSTFREE credits.
- The members' BRISC runs [the PfStream output store + the H5 raise] then [the H6 consumer stub] as one composite
  (tt/pf_stream.composite_source: the PMK composite pattern). The unit's stub copies POST to y (DRAM).
"""

from __future__ import annotations

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS
from models.autoports.zai_org_glm_5_3_flash.tt import pf_stream as PF


class PfRs:
    def __init__(self, mesh, w, *, members, streamers, lanes, reducers, nb: int = 4, nw: int = 32, kb: int = 16,
                 park: int = 4, probe: bool = False):  # fmt: skip
        assert int(mesh.get_num_devices()) == 2, "the pair (TP2)"
        self.mesh = mesh
        self.members = [tuple(c) for c in members]
        self.lanes = [tuple(c) for c in lanes]
        self.reducers = [tuple(c) for c in reducers]
        others = set(self.lanes) | set(self.reducers)
        assert not others & set(self.members) and not others & {tuple(c) for c in streamers}
        self.mm = PF.PfStreamMM(mesh, w, members=members, streamers=streamers, nb=nb, nw=nw, kb=kb, park=park,
                                probe=probe)  # fmt: skip
        self.bl = len(self.members) // 2
        self.rs = FS.FabricStreamRSR(mesh, self.bl, lanes=self.lanes, reducers=self.reducers, members=self.members,
                                     hidden=int(w.shape[1]))  # fmt: skip
        self.y = self.rs.y

    def signals(self, p: int) -> dict:
        """member -> (noc x, noc y, L1 address) of the H5 word its row block feeds on chip rank p"""
        sync = self.rs.sync.buffer_address()
        nl, nr = len(self.lanes), len(self.reducers)
        out = {}
        for g, m in enumerate(self.members):
            o, b = divmod(g, self.bl)
            core, word = (self.reducers[b % nr], FS._FS["OWNRDY"] + b // nr) if o == p else \
                (self.lanes[b % nl], FS._FS["LREADY"] + b // nl)  # fmt: skip
            assert word < FS.SYNC_WORDS
            x, y = self.mm._noc(core)
            out[m] = (x, y, sync + word * FS.WORD)
        return out

    def program(self, x, r: int, c: int) -> ttnn.ProgramDescriptor:
        p = c if self.rs.C > 1 else r
        mm = self.mm
        pt = mm.parts(mm.y, x=x, signals=self.signals(p))
        rk, rcbs, rsems, stub = self.rs._build(mm.y, r, c, ready=True, stub_kernel=False)
        ct0, defs = pt["ct"], pt["defines"]
        rt0_len = len(pt["rt"][self.members[0]])

        def rt_of(cores, extra=None):
            rr = ttnn.RuntimeArgs()
            for cc in cores:
                rr[cc[0]][cc[1]] = pt["rt"][cc] + (extra[cc] if extra else [])
            return rr

        dyn = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                noc=ttnn.NOC.RISCV_0_default, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)  # fmt: skip
        cc_ = ttnn.ComputeConfigDescriptor(**pt["compute"], math_approx_mode=False)
        kd = lambda src, cores, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=PF._crs(cores),
            compile_time_args=ct, runtime_args=rt, config=cfg, defines=list(defs))  # fmt: skip
        s, m = pt["sources"]["streamer"], pt["sources"]["member"]
        member_dm0 = PF.composite_source([m["dm0"], stub["src"]], [0, len(ct0)], [0, rt0_len], "dm")
        kernels = list(rk) + [
            kd(s["dm0"], mm.all_streamers, ct0, rt_of(mm.all_streamers), dyn),
            kd(m["dm1"], self.members, ct0, rt_of(self.members), ttnn.ReaderConfigDescriptor()),
            kd(m["c"], self.members, ct0, rt_of(self.members), cc_),
            kd(member_dm0, self.members, ct0 + stub["ct"], rt_of(self.members, stub["rt"]), ttnn.WriterConfigDescriptor()),
        ]  # fmt: skip
        cbs = list(rcbs)
        for role, cores in (("streamer", mm.all_streamers), ("member", self.members)):
            for idx, dt, page, pages in pt["cbs"][role]:
                cbs.append(ttnn.CBDescriptor(total_size=page * pages, core_ranges=PF._crs(cores), format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)]))  # fmt: skip
        sems = list(rsems) + [ttnn.SemaphoreDescriptor(id=i, core_ranges=PF._crs(self.members), initial_value=0)
                              for i in pt["sems"]["member"]]  # fmt: skip
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        self.rs.connect(prog, r, c, 0)  # rk[0] = rs_send
        return prog

    def __call__(self, x):
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(int(self.mesh.shape[0])):
            for c in range(int(self.mesh.shape[1])):
                mc = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(mc, mc)] = self.program(x, r, c)
        io = [x, self.mm.weight, self.mm.y, self.mm.sync, self.rs.r, self.rs.y, self.rs.sync] + \
            ([self.mm.mprobe] if self.mm.probe else [])  # fmt: skip
        ttnn.generic_op(io, mpd)
        return self.rs.y

    def release(self):
        self.mm.release()
        self.rs.release()
