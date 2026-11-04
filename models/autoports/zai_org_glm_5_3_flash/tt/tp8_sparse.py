# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14 (PRODUCTION_PATH.md §2, the sparse_kda kind): one sparse_kda layer (KDA attention + 288-expert MoE FFN), TP
over every chip of the mesh, as 7a's MkSparseLayer (the whole layer in ONE program) on every chip + the in-program
fabric all-reduce (tt/fabric_ar.py) after o_proj and after the routed experts:

  attention: the KDA head shard (tt/tp8.py, as the dense layer: 8 heads per chip at TP8), o_proj row shard -> partial;
  router + router tail: replicated (every chip routes every row identically);
  routed experts: EP — chip r runs only experts [E_loc r, E_loc (r + 1)) (E_loc = 288 / TP) of each row's top-k
      (ReaderExperts' e_lo / num_experts; a chip whose experts no row selects writes zeros) -> the routed PARTIAL y;
  shared expert: replicated here (interim: y2 is complete on every chip, so only y needs the all-reduce; the TP8
      column / row shard of the shared expert is the next step);
  P10 mhc_post: y = the all-reduced routed sum (from the lanes), y2 local.

One captured program -> per chip: the expert weight streamer's compile-time e_lo (xw.cpp CT 12, at MK_X_CT + 12 in
the composite reader kernel) and a per-chip expert address table (a sharded mesh tensor at one address). The expert
weights themselves stay the decoder's replicated tensors (one layer fits: ~4.3 GB / chip; the 45-layer image needs
the EP-sharded weights, shard_contract.md)."""

from __future__ import annotations

import copy
import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import SEM_A
from models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse import MkSparseLayer
from models.autoports.zai_org_glm_5_3_flash.tt.tp8 import TPShards
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import TPDenseKDALayer, _Capture, _kernel, _rt_dict, _src_is

REC_RMS_SEM_SPARSE = 14  # MkSparseLayer: 13 = SEM_LOGITS (the readers also signal it on the owners)
XW_E_LO_CT = 12  # mk_experts.py ReaderExperts.parts_raw w_ct index of e_lo


def build_moe(mesh, cfg, source, layer: int, ocfg):
    """The layer's MoE (router + router tail, shared expert, the routed experts) replicated on `mesh`, exactly as the
    decoder builds it (fused_decoder.FusedDecoderLayer / OptimizedDecoder) without the decoder's single-device parts."""
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import (
        FusedMoE,
        FusedRouter,
        FusedSwiGLU,
        load_fused_experts,
    )
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
        OptimizedMoE,
        OptimizedRouter,
        OptimizedSwiGLU,
        _check_expert_weights,
        _upgrade,
    )

    dev = lambda name: source.device(layer, name, mesh)  # noqa: E731
    host = lambda name: source.host(layer, name)  # noqa: E731
    shared = FusedSwiGLU(mesh, dev("mlp.shared_experts.gate_proj.weight"), dev("mlp.shared_experts.up_proj.weight"),
                         dev("mlp.shared_experts.down_proj.weight"), limit=cfg.swiglu_limit)  # fmt: skip
    shared.fcfg = ocfg
    router = FusedRouter(mesh, dev("mlp.gate.weight"), host("mlp.gate.e_score_correction_bias"),
                         num_experts=cfg.n_routed_experts, top_k=cfg.num_experts_per_tok, fcfg=ocfg)  # fmt: skip
    gate_up, down = load_fused_experts(source, layer, cfg.n_routed_experts, mesh, ocfg.fused_experts_cache_dir)
    moe = FusedMoE(mesh, router, gate_up, down, shared, intermediate=cfg.moe_intermediate_size, hidden=cfg.hidden_size,
                   limit=cfg.swiglu_limit, scaling=cfg.routed_scaling_factor, cfg=ocfg)  # fmt: skip
    _check_expert_weights(source, layer, moe.gate_up, moe.down)
    _upgrade(moe.shared, OptimizedSwiGLU, ocfg, "shared")
    _upgrade(moe.router, OptimizedRouter, ocfg)
    moe.__class__ = OptimizedMoE
    return moe


def _same_lanes(a, b) -> bool:
    return [tuple(int(v) for v in x) for x in a] == [tuple(int(v) for v in x) for x in b]


class _LocalExperts:
    """The decoder's OptimizedMoE seen with only experts [lo, lo + n) — or the expert ids `ids` in slot order (a
    load-aware placement) — (MkSparseLayer / ReaderExperts read gate_up / down lists; everything else delegates)."""

    def __init__(self, mlp, lo: int, n: int, ids=None):
        self._m = mlp
        ids = list(range(lo, lo + n)) if ids is None else list(ids)
        self.gate_up, self.down = [mlp.gate_up[e] for e in ids], [mlp.down[e] for e in ids]

    def __getattr__(self, k):
        return getattr(self._m, k)


class TPSparseKDALayer(TPDenseKDALayer):
    """One sparse_kda layer TP / EP over every chip of `mesh`, decode rows t <= 4 (7a's megakernel range)."""

    def __init__(self, mesh, cfg, layer: int, source, ocfg, *, b: int, moe, kda_spread: int | None = None,
                 rec_spread: int = 1, rev: str | None = None, vtp: int | None = None, probe: bool = False,
                 weights=None, fabric=None):  # fmt: skip
        """moe: the layer's OptimizedMoE on `mesh` (replicated: router, router tail, shared expert, all experts).
        weights: the `.w` of an earlier TPSparseKDALayer of the same (layer, vtp, spreads): resident attention weights,
        only the programs are built (Stage 14 job pipeline rule)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoderConfig, FusedKDA
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedKDA, _upgrade

        self.mesh, self.cfg, self.layer, self.b = mesh, cfg, layer, b
        self._fabric_share = fabric  # an earlier layer's phase to share (see the fabric phase below)
        self.rev_defs = [("S14_REV", str(rev))] if rev else []
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        self.ocfg = copy.copy(
            ocfg
        )  # (the TP-sharded shared expert's DSLinear config, also on the resident-weights path)
        if weights is not None:
            assert (weights.rec_spread, weights.kda_spread) == (rec_spread, kda_spread), "weights of other spreads"
            self.w = weights
            sh, attn = weights.shards, weights.attn
            self.shards, self.attn = sh, attn
            attn.kda_rec_ugroups = int(os.environ.get("GLM53_REC_UGROUPS", "1"))
            assert attn.kda_rec_ugroups * rec_spread * sh.h <= 64
            attn.kda_pre_rsplit = int(os.environ.get("GLM53_KDA_RSPLIT", "1"))  # (read by KdaPre at build)
            assert (
                attn.kda_pre_spread * sh.h * attn.kda_pre_rsplit <= 64
            ), "kda_pre cores of the resident spread x row groups: pass kda_spread for this row split"
            self._build_sparse(mesh, moe, b, probe=probe)
            return
        sh = TPShards(mesh, source, layer, cfg, vtp=vtp)
        self.shards = sh
        ocfg = copy.copy(ocfg)
        for flag in ("fused_kda_pre", "fused_kda_rec"):
            if hasattr(ocfg, flag):
                setattr(ocfg, flag, False)
        kt = sh.kda_tensors()
        attn = FusedKDA(
            mesh, **kt,
            dt_bias=sh.host("self_attn.dt_bias").float().reshape(-1)[: sh.w],
            a_log=sh.host("self_attn.A_log").float().reshape(-1)[slice(0, sh.h)],
            num_heads=sh.h, head_dim=sh.hd, hidden=cfg.hidden_size, lower_bound=cfg.linear_lower_bound,
            eps=cfg.rms_norm_eps,
        )  # fmt: skip
        attn.decay_rate, attn.dt_bias = sh.kda_rate_bias()
        # GLM53_KDA_RSPLIT = R (Stage 14, ROWS mode): kda_pre row groups over the owners its column cores leave idle
        attn.kda_pre_rsplit = int(os.environ.get("GLM53_KDA_RSPLIT", "1"))
        attn.kda_pre_spread = kda_spread if kda_spread is not None else max(
            s for s in (1, 2, 4) if s * sh.h * attn.kda_pre_rsplit <= 64)  # fmt: skip
        assert attn.kda_pre_spread * sh.h * attn.kda_pre_rsplit <= 64, "kda_pre cores (spread x heads x row groups)"
        assert rec_spread in (1, 2, 4) and rec_spread * sh.h <= 64, (rec_spread, sh.h)
        attn.kda_rec_spread = rec_spread
        # GLM53_REC_UGROUPS=u: the decode rows split over u copies of the head cores (64 owners: u * rec_spread * h)
        attn.kda_rec_ugroups = int(os.environ.get("GLM53_REC_UGROUPS", "1"))
        assert attn.kda_rec_ugroups * rec_spread * sh.h <= 64, (attn.kda_rec_ugroups, rec_spread, sh.h)
        attn.kda_rec_rms_sem = REC_RMS_SEM_SPARSE
        attn.set_decode_weights()
        attn.fcfg = FusedDecoderConfig()
        _upgrade(attn, OptimizedKDA, ocfg)
        self.attn = attn
        import types

        self.w = types.SimpleNamespace(shards=sh, attn=attn, attn_args=sh.mhc_args("attn"),
                                       ffn_args=sh.mhc_args("ffn"), rec_spread=rec_spread, kda_spread=kda_spread)  # fmt: skip
        self._build_sparse(mesh, moe, b, probe=probe)

    def _placement(self, tp: int):
        """GLM53_EP_PLACEMENT: '1' (Stage 7a's shipped ep_placement.json) or a path -> per chip r (ranks 0 .. ndev - 1)
        its expert ids, ascending; None = contiguous EP."""
        v = os.environ.get("GLM53_EP_PLACEMENT")
        if not v or v == "0":
            return None
        import json

        path = v if v != "1" else os.path.join(os.environ.get("GLM53_WS", "/home/alex/mpi-shfs/tenstorrent/glm53-flash-bringup"),
                                                "doc/ttnn/07a-fused-streaming-layer/ep_placement.json")  # fmt: skip
        sets = json.load(open(path))["layers"][str(self.layer)]
        assert len(sets) == 8 and 8 % tp == 0, (len(sets), tp)
        g = 8 // tp
        out = [sorted(e for s_ in sets[r * g : (r + 1) * g] for e in s_) for r in range(tp)]
        assert all(len(x) == self.e_loc for x in out) and sorted(sum(out, [])) == list(range(self.e_all)), "placement"
        return out[: self.shards.ndev]

    def _build_sparse(self, mesh, moe, b: int, probe: bool = False):
        """MkSparseLayer on the resident attention weights + the layer's MoE, the expert table, lanes, fabric phase."""
        sh, attn = self.shards, self.attn
        self.mlp = moe
        self.attn_args, self.ffn_args = self.w.attn_args, self.w.ffn_args
        self.program = "layer"
        # EP: E_loc experts per chip (TP-vtp-sized shards on fewer chips: experts of ranks 0 .. ndev - 1 only)
        self.e_all = int(moe.num_experts)
        assert self.e_all % sh.tp == 0, (self.e_all, sh.tp)
        self.e_loc = self.e_all // sh.tp
        lins = dict(kda_qkv=attn.ds_qkv, kda_lowrank=attn.ds_lowrank, kda_o=attn.ds_o)
        # M5 lever 2 (GLM53_EP_PLACEMENT=1 or a path): Stage 7a's load-aware EP8 placement (doc/ttnn/07a-fused-streaming-
        # layer/ep_placement.json: per sparse layer 8 x 36 expert ids, equal expected touched experts per chip) instead
        # of contiguous ranges; chip r of a TP < 8 mesh holds the union of placed chips [r 8 / TP, (r + 1) 8 / TP) (TP2:
        # the numerics check); 7a's MKX_LOC: the expert table row gains loc[e] and e_lo stays 0 on every chip
        self.placement = self._placement(sh.tp)
        # 7a GLM53_TP_XSL=1: batched expert TP (mk_experts.py MKX_XSL) instead of EP: every chip holds all the experts
        # (the replicated MoE) and streams slice `rank` of every selected expert (1/TP of each: no busiest chip); the
        # program is captured for slice 0 and the slice is patched per chip like EP's e_lo
        self.xsl = os.environ.get("GLM53_TP_XSL") == "1"
        assert not (self.xsl and self.placement), "GLM53_TP_XSL and GLM53_EP_PLACEMENT are exclusive"
        view = _LocalExperts(moe, 0, self.e_all if self.xsl else self.e_loc,
                             ids=self.placement[0] if self.placement else None)  # fmt: skip
        # GLM53_TP_XSL_SLICED=1 (with GLM53_TP_XSL=1): the production layout — per expert one mesh tensor, chip k = slice
        # k (fused_decoder.load_xsl_experts; a 1-chip virtual TP holds slice 0), read whole by the kernel (XSL_C = 0)
        self.xsl_sliced = self.xsl and os.environ.get("GLM53_TP_XSL_SLICED") == "1"
        if self.xsl_sliced:
            if getattr(self.w, "xsl_experts", None) is None or self.w.xsl_experts[0] != sh.tp:
                from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import load_xsl_experts

                gs, ds = load_xsl_experts(sh.source, self.layer, self.e_all, mesh, self.ocfg.fused_experts_cache_dir,
                                          tp=sh.tp)  # fmt: skip
                self.w.xsl_experts = (sh.tp, gs, ds)
            view.gate_up, view.down = list(self.w.xsl_experts[1]), list(self.w.xsl_experts[2])
        # Stage 14 GLM53_TP_SHARED=1: the shared expert TP-sharded (intermediate slice per chip: its down output is a
        # partial sum) and folded into the routed partial (7a's fold_y2: shared down seeds the experts' sums), so the
        # ONE all-reduce after the experts sums routed (EP) + shared (TP); ~27 MB -> 3.4 MB of shared weights per chip
        self.tp_shared = os.environ.get("GLM53_TP_SHARED") == "1"
        if self.tp_shared:
            if getattr(self.w, "shared_tp", None) is None:
                from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedSwiGLU
                from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedSwiGLU, _upgrade

                st = sh.shared_tensors()
                shared = FusedSwiGLU(mesh, st["gate"], st["up"], st["down"], limit=moe.shared.limit)
                shared.fcfg = getattr(moe.shared, "fcfg", None) or self.ocfg
                _upgrade(shared, OptimizedSwiGLU, self.ocfg, "shared")
                self.w.shared_tp = shared
            view.shared = self.w.shared_tp
        self.mk = MkSparseLayer(mesh, self.attn_args, self.ffn_args, attn, lins, view, e_lo=0, num_experts=self.e_all,
                                fold_y2=True if self.tp_shared else None, probe=probe,
                                local_experts=self.placement[0] if self.placement else None,
                                xsl=0 if self.xsl else None, xsl_tp=sh.tp, xsl_sliced=self.xsl_sliced)  # fmt: skip
        # the per-chip expert address table: chip r = experts [E_loc r, E_loc (r + 1)) — or its placed ids + loc[e]
        # (slot, E_loc = not on this chip) — at the same buffer address everywhere
        ex = self.mk.experts
        rows = []
        for r in range(sh.ndev):
            ids = self.placement[r] if self.placement else list(range(self.e_loc * r, self.e_loc * (r + 1)))
            if self.xsl:  # every expert on every chip
                ids = list(range(self.e_all))
            src_gu, src_dn = (view.gate_up, view.down) if self.xsl_sliced else (moe.gate_up, moe.down)
            row = [src_gu[e].buffer_address() for e in ids] + [src_dn[e].buffer_address() for e in ids]
            if self.placement:
                loc = [self.e_loc] * self.e_all
                for slot, e in enumerate(ids):
                    loc[e] = slot
                row += loc
            rows.append(row)
        tab = torch.tensor(rows, dtype=torch.int64).to(torch.int32).reshape(sh.ndev, 1, 1, -1)
        ttnn.deallocate(ex.table)
        ex.table = ttnn.from_torch(tab, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))  # fmt: skip
        sh.drop()
        # lanes: the first 2 free cores (MkSparseLayer takes its router-tail cores from the end of the same list)
        grid = mesh.compute_with_storage_grid_size()
        used = {(int(c.x), int(c.y)) for c, _, _ in self.mk.readers.assign}
        used |= {(int(c.x), int(c.y)) for c in self.mk.place["k"]} | {(int(self.mk.place["root"].x),
                                                                       int(self.mk.place["root"].y))}  # fmt: skip
        free = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in used]
        self.lanes = free[:2]
        assert not set(self.lanes) & set(self.mk.tail_free[:4]), "lanes vs router-tail cores"
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
        self.owner_noc = self.mk.owner_noc
        self.n_readers = len(self.mk.readers.assign)
        self.extra_sems = self.mk.rec.semaphores()
        self.calls = 0
        self.last_captures = None

    # ------------------------------------------------------------------ program surgery
    def _hooked_sparse(self, io, desc):
        """The captured MkSparseLayer program -> one program per chip: fabric hooks on o_proj (dense reader phase 2)
        and on the routed experts (xa.cpp), the lanes' 2 all-reduces, per-chip e_lo, the post-mix waits."""
        fab, nr = self.fabric, self.n_readers
        base, wkern, wdesc, seen = [], None, None, set()
        for i, kd in enumerate(desc.kernels):  # (ProgramDescriptor.kernels returns copies)
            src = str(kd.kernel_source)
            if _src_is(kd, "mk/reader_moe_a.cpp"):
                n_rt = len(next(iter(_rt_dict(kd).values())))
                defs = [d for d in kd.defines if d[0] != "MKX_SIGNAL_SEM"]  # the lanes signal the post-mix owners
                defs += [("MK_FAB_MASK", str(1 << 2)), ("MK_FAB_BROWS", str(fab.brows)), ("MK_FAB_RT", str(n_rt)),
                         ("MKX_FAB_RT", str(n_rt)), ("MKX_FAB_BROWS", str(fab.brows))] + self.rev_defs  # fmt: skip
                base.append(_kernel(kd, defines=defs, extra_rt=fab.producer_args()))
                seen.add("a")
            elif _src_is(kd, "mk/reader_moe_w.cpp"):
                wkern, wdesc = len(base), i
                base.append(_kernel(kd, defines=list(kd.defines) + self.rev_defs))
                seen.add("w")
            elif _src_is(kd, "mk/owner_dm0.cpp"):
                n_rt = len(next(iter(_rt_dict(kd).values())))
                defs = [d for d in kd.defines if d[0] not in ("MK_POST_WAIT_COUNT", "MK_POST2_WAIT_COUNT")]
                defs += [("MK_POST_WAIT_COUNT", "2"),  # P6: the 2 lanes (o_proj all-reduce)
                         ("MK_POST2_WAIT_COUNT", str(nr + 2)),  # P10: the shared down readers + the 2 lanes (routed)
                         ("MK_POST_Y_FABRIC", "1"), ("MK_POST_FAB_RT", str(n_rt))] + self.rev_defs  # fmt: skip
                base.append(_kernel(kd, defines=defs, extra_rt=fab.consumer_args()))
                seen.add("o")
            else:
                base.append(_kernel(kd, defines=list(kd.defines) + self.rev_defs) if self.rev_defs else kd)
        assert seen == {"a", "w", "o"}, (seen, [str(k.kernel_source) for k in desc.kernels])
        xct = dict(desc.kernels[wkern].defines).get("MK_X_CT") if wkern is not None else None
        assert xct is not None, "reader_moe_w.cpp carries MK_X_CT"
        e_lo_idx = int(xct) + XW_E_LO_CT
        assert int(base[wkern].compile_time_args[e_lo_idx]) == 0, "the captured program is chip 0's (e_lo 0)"
        if self.xsl:  # batched expert TP: e_lo stays 0 (all experts local), the slice index is the per-chip arg
            e_lo_idx = int(xct) + self.mk.experts.xw_xsl_ct
            assert int(base[wkern].compile_time_args[e_lo_idx]) == 0, "the captured program is slice 0's"
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                rank = r * self.C + c
                kernels = list(base)
                if rank and not self.placement:  # a fresh copy of the weight streamer with this chip's e_lo (placed: 0;
                    # XSL: its slice)
                    kw = _kernel(desc.kernels[wdesc], defines=list(base[wkern].defines))
                    ct = list(kw.compile_time_args)
                    ct[e_lo_idx] = (0 if self.xsl_sliced else rank) if self.xsl else self.e_loc * rank
                    kw.compile_time_args = ct
                    kernels[wkern] = kw
                lk, lcbs = fab.lane_parts(r, c, 2, nr, self.owner_noc, cons_sem=SEM_A)
                prog = ttnn.ProgramDescriptor(kernels=kernels + lk, semaphores=list(desc.semaphores) + self.extra_sems,
                                              cbs=list(desc.cbs) + lcbs)  # fmt: skip
                fab.add_connections(prog, r, c, send_kernel_index=len(kernels))
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        return list(io) + fab.io(), mpd

    def run_layer(self, x, hist, active, state, *, t: int):
        if self.nofab:  # 1-chip mesh (Stage 14c): MkSparseLayer as built (this chip's experts), no fabric phase
            return self._unhooked(self.mk, x, hist, active, state, t=t)
        with _Capture() as cap:
            outs = self.mk(x, hist, active, t=t, state=state)
        ((io, desc),) = cap.calls
        io2, mpd = self._hooked_sparse(io, desc)
        ttnn.generic_op(io2, mpd)
        return outs
