# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a: the PMK P13 shared expert as TWO PfStream phases of one program (coordinator 13:16Z item (b), for the
moe2 piece's SharedP13): gate_up (K = H, N = 2 W_s, the weight's columns = (gate_j, up_j) tile pairs) with the SwiGLU
epilogue SiLU(min(g, L)) * clamp(u, -L, L) packed straight into CB_ACT (the down phase's stationary in0, never in
DRAM), then down (K = W_s, N = H) packed UNTILIZED -> y_s [C, H] bf16 ROW_MAJOR (the combine's row parts). The down
phase has its own PARK CB (its 16-tile blocks differ from gate_up's 64); VALID / QBASE continue across both."""

from __future__ import annotations

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import pf_stream as PF

CB_ACT, CB_PARK_D = 36, 37


def interleave_gate_up(wg: torch.Tensor, wu: torch.Tensor) -> torch.Tensor:
    """[K, W] x 2 -> [K, 2 W] with (gate_j, up_j) tile pairs: columns 64 j .. 64 j + 31 = gate tile j, + 32 .. = up"""
    k, w = wg.shape
    assert w % 32 == 0 and wu.shape == wg.shape
    return torch.stack([wg.reshape(k, w // 32, 32), wu.reshape(k, w // 32, 32)], dim=2).reshape(k, 2 * w)


class SharedP13:
    def __init__(self, mesh, wg, wu, wd, *, members, streamers, limit: float = 10.0, nb: int = 4, kb_gu: int = 16,
                 kb_d: int = 4, park: int = 4):  # fmt: skip
        w_gu = interleave_gate_up(wg.float(), wu.float())
        n_gu, ws = int(w_gu.shape[1]), int(wd.shape[0])
        nw_gu = 32 if (n_gu // 32) % 32 == 0 else 16
        self.sync = PF.PfStreamMM.make_sync(mesh, [tuple(c) for c in streamers], slots=2)
        self.gu = PF.PfStreamMM(mesh, w_gu, members=members, streamers=streamers, nb=nb, nw=nw_gu, kb=kb_gu, park=park,
                                sync=self.sync, epi=1, limit=limit, cb_out=CB_ACT)  # fmt: skip
        self.dn = PF.PfStreamMM(mesh, wd.float(), members=members, streamers=streamers, nb=nb, nw=32, kb=kb_d,
                                park=park, sync=self.sync, epi=2, xsrc=True, cb_x=CB_ACT, cb_park=CB_PARK_D)  # fmt: skip
        assert self.dn.kt == ws // 32
        self.mesh, self.y = mesh, self.dn.y

    def program(self, x) -> ttnn.ProgramDescriptor:
        kernels, cbs, sems = PF.fused_parts([self.gu, self.dn], [dict(x=x), dict(x=x, x_fill=False)])
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    def __call__(self, x):
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(int(self.mesh.shape[0])):
            for c in range(int(self.mesh.shape[1])):
                mc = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(mc, mc)] = self.program(x)
        ttnn.generic_op([x, self.sync, self.gu.weight, self.gu.y, self.dn.weight, self.dn.y], mpd)
        return self.y

    def release(self):
        self.gu.release()
        self.dn.release()
        ttnn.deallocate(self.sync)
