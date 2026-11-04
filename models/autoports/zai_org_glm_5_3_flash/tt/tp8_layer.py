# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14 (PRODUCTION_PATH.md M3): the TP dense_kda decode layer = Stage 7a's layer megakernel (two programs: the
attention half MkStepB step D and the FFN half MkFFN) running on EVERY chip on its TP shard (tt/tp8.py), with the
in-program fabric all-reduce phase (tt/fabric_ar.py FabricPhase, "cb" mode) after o_proj and after down:

  attention program, per chip: P1 mhc_pre -> P2 qkv + low-rank (shard) -> P3 kda_pre / P4 kda_rec (H_r heads on
      owners 0..H_r-1) -> P5 o_proj (row shard: a PARTIAL [B, 4096]) -> FABRIC (the 16 DS readers hand their packed
      partial rows to the 2 lanes, which all-reduce over the mesh and signal the owners) -> P6 mhc_post (reads the
      reduced rows from the lanes) -> s1;
  FFN program, per chip: P7 mhc_pre -> P8 gate_up (shard) -> swiglu -> P9 down (partial) -> FABRIC -> P10 mhc_post
      -> the layer output (identical on every chip).

7a's classes build each program for the whole mesh (replicated); this module captures their ProgramDescriptor
(ttnn.generic_op is intercepted during the build), adds the fabric hooks (reader_a.cpp MK_FAB_* on the readers,
reader_p.cpp MK_POST_Y_FABRIC + a wait count of 2 lanes on the owners), appends the lane kernels / CBs / fabric
connections per chip, and launches ONE MeshProgramDescriptor (one program per chip; only the lanes' arguments
differ between chips). The weight ring keeps streaming during the collective: the readers' BRISC streams the next
phase's weights independently of their NCRISC, which handed the partial to the lanes.
"""

from __future__ import annotations

import copy
import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import SEM_A, MkStepB, _cores_of
from models.autoports.zai_org_glm_5_3_flash.tt.mk_ffn import SEM_Y, MkFFN
from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import MkLayer
from models.autoports.zai_org_glm_5_3_flash.tt.tp8 import TPShards

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def _on(crs, x, y):
    for r in crs.ranges():
        if r.start.x <= x <= r.end.x and r.start.y <= y <= r.end.y:
            return True
    return False


def _cb_report(desc, x, y):
    """Stage 14c diagnostic: the static CBs (index, bytes, format) and kernels of a program descriptor on core (x, y)"""
    tot = 0
    for cb in desc.cbs:
        if _on(cb.core_ranges, x, y):
            tot += cb.total_size
            fmts = [(f.buffer_index, f.page_size) for f in cb.format_descriptors]
            print(f"CB_REPORT ({x},{y}) cb {fmts} total_size {cb.total_size}", flush=True)
    for kd in desc.kernels:
        if _on(kd.core_ranges, x, y):
            print(f"CB_REPORT ({x},{y}) kernel {str(kd.kernel_source).rsplit('/', 1)[-1]}", flush=True)
    print(f"CB_REPORT ({x},{y}) static CB bytes {tot}", flush=True)


def _same_lanes(a, b) -> bool:
    return [tuple(int(v) for v in x) for x in a] == [tuple(int(v) for v in x) for x in b]


class _Capture:
    """Intercept ttnn.generic_op while a 7a megakernel class builds its program: -> (io, ProgramDescriptor)."""

    def __enter__(self):
        self.orig, self.calls = ttnn.generic_op, []
        ttnn.generic_op = lambda io, desc: self.calls.append((list(io), desc)) or io[-1]
        return self

    def __exit__(self, *exc):
        ttnn.generic_op = self.orig
        return False


def _rt_dict(kd) -> dict:
    return {xy: list(kd.runtime_args[xy[0]][xy[1]]) for xy in sorted(_cores_of(kd.core_ranges))}


def _kernel(kd, *, defines=None, extra_rt=None):
    """KernelDescriptor `kd` (a copy: ProgramDescriptor.kernels returns copies) with replaced defines and extra runtime
    args appended on every core (in place: every other field, e.g. buffer bindings, is kept)."""
    if defines is not None:
        kd.defines = list(defines)
    if extra_rt:
        extra = [int(a) for a in extra_rt]
        for x, y in sorted(_cores_of(kd.core_ranges)):
            kd.runtime_args[x][y].extend(extra)
    return kd


def _src_is(kd, name: str) -> bool:
    return str(kd.kernel_source).replace("\\", "/").endswith(name)


class TPDenseKDALayer:
    """One dense_kda layer, TP over every chip of `mesh`, decode rows t <= 4 (7a's megakernel range)."""

    def __init__(
        self,
        mesh,
        cfg,
        layer: int,
        source,
        ocfg,
        *,
        b: int,
        probe: bool = False,
        program: str = "layer",
        kda_spread: int | None = None,
        rec_spread: int = 1,
        rev: str | None = None,
        vtp: int | None = None,
        weights=None,
        fabric=None,
    ):
        """program "layer": 7a's MkLayer (the whole layer in ONE program per chip: the readers' weight stream runs on
        into the FFN weights while the attention all-reduce is in flight); "halves": MkStepB step D + MkFFN.
        weights: the `.w` of an earlier TPDenseKDALayer of the same (layer, vtp, spreads) — the device weights stay
        resident and only the programs are built (Stage 14 job pipeline rule: a knob change never re-uploads)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedKDA, FusedSwiGLU
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedKDA, OptimizedSwiGLU, _upgrade

        self.mesh, self.cfg, self.layer, self.b = mesh, cfg, layer, b
        self._fabric_share = fabric  # an earlier layer's phase to share (see the fabric phase below)
        # rev: an extra define on every kernel (a long-lived process caches FILE_PATH kernel binaries by path + args:
        # a new rev forces the rebuild of edited kernel files in the same process)
        self.rev_defs = [("S14_REV", str(rev))] if rev else []
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        if weights is not None:
            self.w = weights
            self.shards, self.attn, self.mlp = weights.shards, weights.attn, weights.mlp
            self.attn_args, self.ffn_args = weights.attn_args, weights.ffn_args
            # the per-program knob read by KdaRec / KdaPre at program build
            self.attn.kda_rec_ugroups = int(os.environ.get("GLM53_REC_UGROUPS", "1"))
            assert self.attn.kda_rec_ugroups * rec_spread * self.shards.h <= 64
            self.attn.kda_pre_rsplit = int(os.environ.get("GLM53_KDA_RSPLIT", "1"))  # (read by KdaPre at build)
            assert (
                self.attn.kda_pre_spread * self.shards.h * self.attn.kda_pre_rsplit <= 64
            ), "kda_pre cores of the resident spread x row groups: pass kda_spread for this row split"
            assert (weights.rec_spread, weights.kda_spread) == (rec_spread, kda_spread), "weights of other spreads"
            self._build_programs(mesh, probe, program, b)
            return
        sh = TPShards(mesh, source, layer, cfg, vtp=vtp)  # vtp: TP-vtp-sized shards on fewer chips (timing only)
        self.shards = sh
        ocfg = copy.copy(ocfg)
        for flag in ("fused_kda_pre", "fused_kda_rec"):  # the megakernel builds its own phases
            if hasattr(ocfg, flag):
                setattr(ocfg, flag, False)
        kt = sh.kda_tensors()
        r0 = slice(0, sh.h)
        attn = FusedKDA(
            mesh, **kt,
            dt_bias=sh.host("self_attn.dt_bias").float().reshape(-1)[: sh.w],
            a_log=sh.host("self_attn.A_log").float().reshape(-1)[r0],
            num_heads=sh.h, head_dim=sh.hd, hidden=cfg.hidden_size, lower_bound=cfg.linear_lower_bound,
            eps=cfg.rms_norm_eps,
        )  # fmt: skip
        attn.decay_rate, attn.dt_bias = sh.kda_rate_bias()  # per chip (the constructor replicated rank 0's)
        # kda_pre: spread the chip's few heads over more owners (cores per head's worth of columns, <= 64 owners)
        # GLM53_KDA_RSPLIT = R (Stage 14, ROWS mode): kda_pre row groups over the owners its column cores leave idle
        attn.kda_pre_rsplit = int(os.environ.get("GLM53_KDA_RSPLIT", "1"))
        attn.kda_pre_spread = kda_spread if kda_spread is not None else max(
            s for s in (1, 2, 4) if s * sh.h * attn.kda_pre_rsplit <= 64)  # fmt: skip
        assert attn.kda_pre_spread * sh.h * attn.kda_pre_rsplit <= 64, "kda_pre cores (spread x heads x row groups)"
        # kda_rec: value-column split, rec_spread cores per head (the gated norm's sum of squares exchanged in-program)
        assert rec_spread in (1, 2, 4) and rec_spread * sh.h <= 64, (rec_spread, sh.h)
        attn.kda_rec_spread = rec_spread
        # GLM53_REC_UGROUPS=u: the decode rows split over u copies of the head cores (64 owners: u * rec_spread * h)
        attn.kda_rec_ugroups = int(os.environ.get("GLM53_REC_UGROUPS", "1"))
        assert attn.kda_rec_ugroups * rec_spread * sh.h <= 64, (attn.kda_rec_ugroups, rec_spread, sh.h)
        attn.set_decode_weights()
        from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoderConfig

        attn.fcfg = FusedDecoderConfig()
        _upgrade(attn, OptimizedKDA, ocfg)
        self.attn = attn
        mt = sh.mlp_tensors()
        kd = int(mt["intermediate"]) // 32  # the down shard's K tiles (TP8: 48): the single-chip storage-core override
        c = ocfg.role_value("ds_cores", "mlp_down")  # (32) must divide it, else the DS heuristic picks (TP8: 16)
        if c and kd % c:
            ocfg.ds_cores = ";".join(i for i in ocfg.ds_cores.split(";") if i and not i.startswith("mlp_down:"))
        mlp = FusedSwiGLU(mesh, mt["gate"], mt["up"], mt["down"], limit=cfg.swiglu_limit)
        mlp.fcfg = FusedDecoderConfig()
        _upgrade(mlp, OptimizedSwiGLU, ocfg, "mlp")
        self.mlp = mlp
        self.attn_args, self.ffn_args = sh.mhc_args("attn"), sh.mhc_args("ffn")
        import types

        self.w = types.SimpleNamespace(shards=sh, attn=attn, mlp=mlp, attn_args=self.attn_args, ffn_args=self.ffn_args,
                                       rec_spread=rec_spread, kda_spread=kda_spread)  # fmt: skip
        self._build_programs(mesh, probe, program, b)

    def _build_programs(self, mesh, probe: bool, program: str, b: int):
        """MkLayer (or MkStepB + MkFFN) on the resident weights, the lanes and the fabric phase."""
        attn, mlp, sh = self.attn, self.mlp, self.shards
        self.program = program
        if program == "layer":
            lins = dict(kda_qkv=attn.ds_qkv, kda_lowrank=attn.ds_lowrank, kda_o=attn.ds_o, mlp_gate_up=mlp.ds_gate_up,
                        mlp_down=mlp.ds_down)  # fmt: skip
            self.mk = MkLayer(mesh, self.attn_args, self.ffn_args, attn, lins, intermediate=mlp.intermediate,
                              limit=mlp.limit, probe=probe)  # fmt: skip
            mks = (self.mk,)
        else:
            self.mk_attn = MkStepB(mesh, self.attn_args, attn, attn.ds_qkv, attn.ds_lowrank, ds_o=attn.ds_o,
                                   post=True, probe=probe)  # fmt: skip
            self.mk_ffn = MkFFN(mesh, self.ffn_args, mlp.ds_gate_up, mlp.ds_down, intermediate=mlp.intermediate,
                                limit=mlp.limit)  # fmt: skip
            mks = (self.mk_attn, self.mk_ffn)
        sh.drop()
        # lanes: 2 cores used by none of the programs (readers, owners, root); identical on every chip (one grid)
        grid = mesh.compute_with_storage_grid_size()
        used = set()
        for mk in mks:
            used |= {(int(c.x), int(c.y)) for c, _, _ in mk.readers.assign}
            used |= {(int(c.x), int(c.y)) for c in mk.place["k"]} | {(int(mk.place["root"].x), int(mk.place["root"].y))}
        free = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in used]
        self.lanes = free[:2]
        # Stage 14c: a 1-chip mesh has no all-reduce (the lanes would wait for rounds that never come): the program
        # runs unhooked (7a's own reader -> owner hand-offs), i.e. the per-chip TP program minus the collective
        self.nofab = self.R * self.C == 1
        # fabric: an earlier layer's phase on the same lane cores and rows, shared (StackBackend: one per kind; a stack
        # program runs on its first layer's phase, the dsa layers already share theirs) — every phase keeps a sync
        # buffer whose address range is reserved on EVERY core: one per layer of the 45 pushed the lowest L1 address
        # under native B=32's static CBs (14b 17:15Z clash)
        fabric = self._fabric_share
        # (a width-bucket layer, b > 4, runs on its kind's 32-row phase: whole-tile packets, the t > 4 consumer)
        if (
            not self.nofab
            and fabric is not None
            and (fabric.b == b or (b > 4 and fabric.b == 32))
            and _same_lanes(fabric.lanes, self.lanes)
        ):
            self.fabric = fabric
        else:
            self.fabric = None if self.nofab else FAR.FabricPhase(mesh, b, lanes=self.lanes, mode="cb")
            if self.fabric is not None:
                self.fabric.probe()
        self.owner_noc = mks[0].owner_noc
        assert all(mk.owner_noc == self.owner_noc for mk in mks), "the programs place the owners identically"
        self.n_readers = len(mks[0].readers.assign)  # every reader increments both lanes' OWN_READY per fabric phase
        self.extra_sems = mks[0].rec.semaphores() if hasattr(mks[0], "rec") else []  # kda_rec spread: REC_RMS_SEM
        self.calls = 0
        self.last_captures = None

    # ------------------------------------------------------------------ program surgery
    def _hooked(self, io, desc, fab_mask: int, cons_sem: int, iters: int = 1):
        """The captured 7a program + the fabric phase -> MeshProgramDescriptor (one program per chip). `iters` =
        all-reduces per launch (the phases of fab_mask, in order); consumers = the owners (mhc_post), +1 on cons_sem."""
        fab = self.fabric
        kernels, reader_seen, post_seen = [], 0, 0
        for kd in desc.kernels:
            if _src_is(kd, "mk/reader_a.cpp"):
                n_rt = len(next(iter(_rt_dict(kd).values())))
                defs = list(kd.defines) + [("MK_FAB_MASK", str(fab_mask)), ("MK_FAB_BROWS", str(fab.brows)),
                                           ("MK_FAB_RT", str(n_rt))] + self.rev_defs  # fmt: skip
                kernels.append(_kernel(kd, defines=defs, extra_rt=fab.producer_args()))
                reader_seen += 1
            elif _src_is(kd, "mk/owner_dm0.cpp") or _src_is(kd, "mk/owner_ffn_dm0.cpp"):
                n_rt = len(next(iter(_rt_dict(kd).values())))
                defs = [d for d in kd.defines if d[0] != "MK_POST_WAIT_COUNT"]
                defs += [("MK_POST_WAIT_COUNT", "2"), ("MK_POST_Y_FABRIC", "1"),  # 2 lanes signal each owner
                         ("MK_POST_FAB_RT", str(n_rt))] + self.rev_defs  # fmt: skip
                kernels.append(_kernel(kd, defines=defs, extra_rt=fab.consumer_args()))
                post_seen += 1
            else:
                kernels.append(_kernel(kd, defines=list(kd.defines) + self.rev_defs) if self.rev_defs else kd)
        assert reader_seen == 1 and post_seen >= 1, (reader_seen, post_seen)
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                lk, lcbs = fab.lane_parts(r, c, iters, self.n_readers, self.owner_noc, cons_sem=cons_sem)
                prog = ttnn.ProgramDescriptor(kernels=kernels + lk, semaphores=list(desc.semaphores) + self.extra_sems,
                                              cbs=list(desc.cbs) + lcbs)  # fmt: skip
                fab.add_connections(prog, r, c, send_kernel_index=len(kernels))
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        return list(io) + fab.io(), mpd

    def attention(self, x, hist, active, state, *, t: int):
        """x [1,1,t,16384] fp32 (replicated streams), hist = conv history (per chip), active = the masks, state = S
        (per chip, updated in place) -> s1 [t,1,4,4096] fp32 (identical on every chip) + the step D outputs."""
        with _Capture() as cap:
            outs = self.mk_attn(x, hist, active, t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_STACKED,
                                pc_mode=M.PC_STACKED, state=state)  # fmt: skip
        ((io, desc),) = cap.calls
        io2, mpd = self._hooked(io, desc, fab_mask=1 << 2, cons_sem=SEM_A)  # o_proj = the readers' phase 2
        ttnn.generic_op(io2, mpd)
        return outs

    def ffn(self, s1, *, t: int):
        """s1 [t,1,4,4096] fp32 -> (MhcPreOutputs, gu, m, y, the layer output [1,1,t,16384] fp32 DRAM)."""
        with _Capture() as cap:
            outs = self.mk_ffn(s1, t=t)
        ((io, desc),) = cap.calls
        io2, mpd = self._hooked(io, desc, fab_mask=1 << 1, cons_sem=SEM_Y)  # down = the readers' phase 1
        ttnn.generic_op(io2, mpd)
        return outs

    def _unhooked(self, mk, x, hist, active, state, *, t: int):
        """Stage 14c 1-chip mesh: the captured megakernel program with the rev define on every kernel (a long-lived
        process caches FILE_PATH binaries by path + args) and the kda_rec spread semaphore, no fabric phase."""
        with _Capture() as cap:
            outs = mk(x, hist, active, t=t, state=state)
        ((io, desc),) = cap.calls
        kernels = [_kernel(kd, defines=list(kd.defines) + self.rev_defs) for kd in desc.kernels]
        if os.environ.get("GLM53_CB_REPORT"):  # Stage 14c diagnostic: "x,y" -> the static CBs / kernels on that core
            _cb_report(desc, *map(int, os.environ["GLM53_CB_REPORT"].split(",")))
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=list(desc.semaphores) + self.extra_sems,
                                                   cbs=list(desc.cbs)))  # fmt: skip
        return outs

    def run_layer(self, x, hist, active, state, *, t: int):
        """MkLayer: the whole layer in ONE program per chip, 2 all-reduces (o_proj = the readers' phase 2, down =
        phase 4) -> MkLayer's outputs dict (out = the layer output, identical on every chip)."""
        if self.nofab:  # 1-chip mesh (Stage 14c): MkLayer as built, no fabric phase
            return self._unhooked(self.mk, x, hist, active, state, t=t)
        with _Capture() as cap:
            outs = self.mk(x, hist, active, t=t, state=state)
        ((io, desc),) = cap.calls
        io2, mpd = self._hooked(io, desc, fab_mask=(1 << 2) | (1 << 4), cons_sem=SEM_A, iters=2)
        ttnn.generic_op(io2, mpd)
        return outs

    def __call__(self, x, hist, active, state, *, t: int):
        """-> (s1, the layer output, details)."""
        self.calls += 1
        if self.program == "layer":
            o = self.run_layer(x, hist, active, state, t=t)
            return o["s1"], o["out"], o, None
        a = self.attention(x, hist, active, state, t=t)
        s1 = a[-1]
        f = self.ffn(s1, t=t)
        return s1, f[-1], a, f

    # ------------------------------------------------------------------ host helpers
    def masks(self, active: torch.Tensor):
        """the decoder's masks (mask fp32, mask bf16, lower_bound * mask fp32) [1,1,B,1], replicated."""
        m = active.float().reshape(1, 1, -1, 1)
        mk = lambda v, dt: ttnn.from_torch(
            v,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )  # fmt: skip  # noqa: E731
        return mk(m, ttnn.float32), mk(m, ttnn.bfloat16), mk(m * self.attn.lower_bound, ttnn.float32)

    def replicated(self, t: torch.Tensor, dtype=ttnn.float32, memory_config=DRAM):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=memory_config,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip
