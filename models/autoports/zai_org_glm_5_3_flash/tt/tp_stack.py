# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b (PRODUCTION_PATH.md M5): the stack program on a TP mesh — N dense_kda layers in ONE program per chip with
Stage 14's in-program fabric all-reduce (tt/fabric_ar.py, M2) after every o_proj and every down projection (2N
all-reduces per launch; the lanes' counters are monotonic, so they need no reset between iterations).

Composition (tt/mk_stack.py MkStack + Stage 14's tt/tp8_layer.py hooks, split in two):
  * per layer (MkStack.layer_hook): the captured MkLayer descriptor (stack mode, the chip's TP shard) gets the fabric
    hooks of TPDenseKDALayer._hooked — the readers hand their partial o_proj / down rows to the 2 lane cores
    (reader_a.cpp MK_FAB_*), the owners' mhc_post reads the reduced rows from the lanes (MK_POST_Y_FABRIC, a wait count
    of 2 lanes) — BEFORE the argument table is built, so every layer's table row carries the fabric args too;
  * once (MkStack.finalize): one ProgramDescriptor per chip = the composed stack + the lane kernels / CBs / fabric
    connections of that mesh coordinate, iters = 2N -> ONE MeshProgramDescriptor.
The residual is replicated (every chip reduces in the same order: bitwise-identical replicas, Stage 14 M2)."""

from __future__ import annotations

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import SEM_A
from models.autoports.zai_org_glm_5_3_flash.tt.mk_stack import MkStack
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import _kernel, _rt_dict, _src_is


class TPStack(MkStack):
    """N MkLayer instances (TP shards on `mesh`, one kind) + the fabric phase of `fab_layer` (a TPDenseKDALayer of the
    same mesh: its lanes, FabricPhase and owner placement serve every layer)."""

    FAB_MASK = (1 << 2) | (1 << 4)  # the readers' phases whose output is a partial: o_proj (2), down (4)

    def __init__(self, mesh, mks: list, fab_layer, *, t: int, probe: bool = False, rev: str | None = None, res=None,
                 pool=None, live=None):  # fmt: skip
        self.fl = fab_layer
        super().__init__(mesh, mks, t=t, probe=probe, rev=rev, res=res, pool=pool, live=live)
        assert all(mk.owner_noc == fab_layer.owner_noc for mk in mks), "every layer places its owners identically"

    def layer_hook(self, desc):
        fab, rev = self.fl.fabric, self.fl.rev_defs
        kernels, seen = [], {"reader": 0, "post": 0}
        for kd in desc.kernels:
            if _src_is(kd, "mk/reader_a.cpp"):
                n_rt = len(next(iter(_rt_dict(kd).values())))
                defs = list(kd.defines) + [("MK_FAB_MASK", str(self.FAB_MASK)), ("MK_FAB_BROWS", str(fab.brows)),
                                           ("MK_FAB_RT", str(n_rt))] + rev  # fmt: skip
                kernels.append(_kernel(kd, defines=defs, extra_rt=fab.producer_args()))
                seen["reader"] += 1
            elif _src_is(kd, "mk/owner_dm0.cpp"):
                n_rt = len(next(iter(_rt_dict(kd).values())))
                defs = [d for d in kd.defines if d[0] != "MK_POST_WAIT_COUNT"]
                defs += [("MK_POST_WAIT_COUNT", "2"), ("MK_POST_Y_FABRIC", "1"), ("MK_POST_FAB_RT", str(n_rt))] + rev
                kernels.append(_kernel(kd, defines=defs, extra_rt=fab.consumer_args()))
                seen["post"] += 1
            else:
                kernels.append(_kernel(kd, defines=list(kd.defines) + rev) if rev else kd)
        assert seen == {"reader": 1, "post": 1}, seen
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(desc.semaphores), cbs=list(desc.cbs))

    def reserved_sems(self) -> list:
        return [int(s.id) for s in self.fl.extra_sems]

    def finalize(self, io, desc):
        fab = self.fl.fabric
        self.desc = desc  # the composed chip-0 descriptor (the L1 ledger reads its CBs)
        kernels = list(desc.kernels)
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.fl.R):
            for c in range(self.fl.C):
                lk, lcbs = fab.lane_parts(r, c, 2 * self.n, self.fl.n_readers, self.fl.owner_noc, cons_sem=SEM_A)
                prog = ttnn.ProgramDescriptor(kernels=kernels + lk,
                                              semaphores=list(desc.semaphores) + list(self.fl.extra_sems),
                                              cbs=list(desc.cbs) + lcbs)  # fmt: skip
                fab.add_connections(prog, r, c, send_kernel_index=len(kernels))
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        self.info["fabric_iters"] = 2 * self.n
        return list(io) + fab.io(), mpd


class TPSparseStack(TPStack):
    """N MkSparseLayer instances (sparse_kda TP / EP shards: tt/tp8_sparse.py TPSparseKDALayer) as ONE program per chip:
    per layer the o_proj partial (dense reader phase 2) and the routed-expert partial (xa.cpp) go through the fabric
    lanes (2 all-reduces per layer: iters = 2N), P10 waits for the shared down readers + the 2 lanes; the expert
    weight streamer's e_lo (compile time) is the chip's: one weight-streamer kernel variant per mesh coordinate (the
    per-layer expert address tables are runtime args -> the stack's argument table)."""

    FAB_MASK = 1 << 2  # o_proj; the routed experts hand off through MKX_FAB_* (xa.cpp)

    def layer_hook(self, desc):
        fab, rev, nr = self.fl.fabric, self.fl.rev_defs, self.fl.n_readers
        kernels, seen = [], set()
        for kd in desc.kernels:
            if _src_is(kd, "mk/reader_moe_a.cpp"):
                n_rt = len(next(iter(_rt_dict(kd).values())))
                defs = [d for d in kd.defines if d[0] != "MKX_SIGNAL_SEM"]  # the lanes signal the post-mix owners
                defs += [("MK_FAB_MASK", str(self.FAB_MASK)), ("MK_FAB_BROWS", str(fab.brows)), ("MK_FAB_RT", str(n_rt)),
                         ("MKX_FAB_RT", str(n_rt)), ("MKX_FAB_BROWS", str(fab.brows))] + rev  # fmt: skip
                kernels.append(_kernel(kd, defines=defs, extra_rt=fab.producer_args()))
                seen.add("a")
            elif _src_is(kd, "mk/owner_dm0.cpp"):
                n_rt = len(next(iter(_rt_dict(kd).values())))
                defs = [d for d in kd.defines if d[0] not in ("MK_POST_WAIT_COUNT", "MK_POST2_WAIT_COUNT")]
                defs += [("MK_POST_WAIT_COUNT", "2"), ("MK_POST2_WAIT_COUNT", str(nr + 2)), ("MK_POST_Y_FABRIC", "1"),
                         ("MK_POST_FAB_RT", str(n_rt))] + rev  # fmt: skip
                kernels.append(_kernel(kd, defines=defs, extra_rt=fab.consumer_args()))
                seen.add("o")
            else:
                kernels.append(_kernel(kd, defines=list(kd.defines) + rev) if rev else kd)
        assert seen == {"a", "o"}, seen
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(desc.semaphores), cbs=list(desc.cbs))

    def finalize(self, io, desc):
        from models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse import XW_E_LO_CT

        fab = self.fl.fabric
        self.desc = desc
        kernels = list(desc.kernels)
        wk = [i for i, kd in enumerate(kernels) if _src_is(kd, "mk/reader_moe_w.cpp")]
        assert len(wk) == 1, wk
        wk = wk[0]
        e_lo_idx = int(dict(kernels[wk].defines)["MK_X_CT"]) + XW_E_LO_CT
        assert int(kernels[wk].compile_time_args[e_lo_idx]) == 0, "the captured program is chip 0's (e_lo 0)"
        xsl = bool(getattr(self.fl, "xsl", False))
        if xsl:  # batched expert TP (GLM53_TP_XSL): e_lo stays 0 (all experts local), the per-chip arg is the SLICE
            # index (as tp8_sparse._hooked_sparse; 2026-10-20 review: patching e_lo here ran slice 0 everywhere)
            e_lo_idx = int(dict(kernels[wk].defines)["MK_X_CT"]) + self.fl.mk.experts.xw_xsl_ct
            assert int(kernels[wk].compile_time_args[e_lo_idx]) == 0, "the captured program is slice 0's"
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.fl.R):
            for c in range(self.fl.C):
                rank = r * self.fl.C + c
                ks = list(kernels)
                if rank and not getattr(self.fl, "placement", None):  # placed experts (MKX_LOC): e_lo 0 everywhere
                    kw = desc.kernels[wk]  # a fresh copy (ProgramDescriptor.kernels returns copies)
                    ct = list(kw.compile_time_args)
                    # (sliced XSL: every chip's tensor IS its slice -> 0, as tp8_sparse)
                    ct[e_lo_idx] = (
                        (0 if getattr(self.fl, "xsl_sliced", False) else rank) if xsl else self.fl.e_loc * rank
                    )
                    kw.compile_time_args = ct
                    ks[wk] = kw
                lk, lcbs = fab.lane_parts(r, c, 2 * self.n, self.fl.n_readers, self.fl.owner_noc, cons_sem=SEM_A)
                prog = ttnn.ProgramDescriptor(kernels=ks + lk,
                                              semaphores=list(desc.semaphores) + list(self.fl.extra_sems),
                                              cbs=list(desc.cbs) + lcbs)  # fmt: skip
                fab.add_connections(prog, r, c, send_kernel_index=len(ks))
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        self.info["fabric_iters"] = 2 * self.n
        return list(io) + fab.io(), mpd


def tp_layer_launch(fab_layer, mk, x, hist, active, state, *, t: int):
    """ONE layer as Stage 14's single-layer TP program (MkLayer of `mk`'s shard + fab_layer's fabric phase, 2
    all-reduces): the reference the stack is compared with (N of these launches back to back)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.mk_stack import _Capture

    with _Capture() as cap:
        outs = mk(x, hist, active, t=t, state=state)
    ((io, desc),) = cap.calls
    io2, mpd = fab_layer._hooked(io, desc, fab_mask=TPStack.FAB_MASK, cons_sem=SEM_A, iters=2)
    ttnn.generic_op(io2, mpd)
    return outs
