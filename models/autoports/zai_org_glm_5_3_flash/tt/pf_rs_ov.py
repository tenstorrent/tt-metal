# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a: the layer's P7 -> P8 with the RS OVERLAPPED under the projection (TP2 pair, ONE program).

With activation-stationary members every row block of the o_proj partial completes at the end of the phase, so the RS
cannot start before it. Split N into `parts` column slices (each 4096 / parts wide): PfStream phase i computes slice i
on the members (the stationary x stays in XS: later phases reuse it) and raises H5 per row block for RS phase i
(FabricStreamRSR ready=True, its own P_i / r / y_i / sync); the lanes and reducers run RS phase i while the members
compute slice i + 1. parts = 1 is the non-overlapped baseline built the same way (one PfStream, one RS).
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS
from models.autoports.zai_org_glm_5_3_flash.tt import fs_compose as FC
from models.autoports.zai_org_glm_5_3_flash.tt import pf_stream as PF


class PfRsOv:
    def __init__(self, mesh, w: torch.Tensor, *, members, streamers, lanes, reducers, parts: int = 2, nb: int = 4,
                 nw: int = 32, kb: int = 16, park: int = 4):  # fmt: skip
        assert int(mesh.get_num_devices()) == 2, "the pair (TP2)"
        self.mesh, self.parts = mesh, parts
        k, n = int(w.shape[0]), int(w.shape[1])
        assert n % parts == 0
        self.members = [tuple(c) for c in members]
        self.lanes = [tuple(c) for c in lanes]
        self.reducers = [tuple(c) for c in reducers]
        ns = n // parts
        sync = PF.PfStreamMM.make_sync(mesh, [tuple(c) for c in streamers], slots=parts)
        self.mms = [PF.PfStreamMM(mesh, w[:, i * ns:(i + 1) * ns].contiguous(), members=members, streamers=streamers,
                                  nb=nb, nw=nw, kb=kb, park=park, sync=sync) for i in range(parts)]  # fmt: skip
        self.sync = sync
        self.bl = len(self.members) // 2
        self.rsrs = [FS.FabricStreamRSR(mesh, self.bl, lanes=self.lanes, reducers=self.reducers, hidden=ns)
                     for _ in range(parts)]  # fmt: skip

    def signals(self, rsr, p: int) -> dict:
        """member -> the H5 word of RS phase `rsr` its row block feeds (lane LREADY / reducer OWNRDY)"""
        sync = rsr.sync.buffer_address()
        nl, nr = len(self.lanes), len(self.reducers)
        out = {}
        for g, m in enumerate(self.members):
            o, b = divmod(g, self.bl)
            core, word = (self.reducers[b % nr], FS._FS["OWNRDY"] + b // nr) if o == p else \
                (self.lanes[b % nl], FS._FS["LREADY"] + b // nl)  # fmt: skip
            x, y = self.mms[0]._noc(core)
            out[m] = (x, y, sync + word * FS.WORD)
        return out

    def program(self, x, r: int, c: int) -> ttnn.ProgramDescriptor:
        p = c if self.rsrs[0].axis == 1 else r
        kws = [dict(x=x, x_fill=i == 0, signals=self.signals(self.rsrs[i], p)) for i in range(self.parts)]
        pk, pcbs, psems = PF.fused_parts(self.mms, kws)
        rk, rcbs = FC.rsr_n_parts(self.rsrs, [m.y for m in self.mms], r, c, ready=True)
        prog = ttnn.ProgramDescriptor(kernels=list(rk) + list(pk), semaphores=list(psems), cbs=list(rcbs) + list(pcbs))
        self.rsrs[0].connect(prog, r, c, 0)  # rk[0] = the lanes' send composite
        return prog

    def __call__(self, x):
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(int(self.mesh.shape[0])):
            for c in range(int(self.mesh.shape[1])):
                mc = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(mc, mc)] = self.program(x, r, c)
        io = [x, self.sync] + [t for m in self.mms for t in (m.weight, m.y)] + \
            [t for q in self.rsrs for t in (q.r, q.y, q.sync)]  # fmt: skip
        ttnn.generic_op(io, mpd)
        return [q.y for q in self.rsrs]

    def release(self):
        for m in self.mms:
            m.release()
        ttnn.deallocate(self.sync)
        for q in self.rsrs:
            q.release()
