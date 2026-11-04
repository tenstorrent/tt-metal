# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel (M4): the WHOLE sparse_kda decode layer as ONE program — MkLayer's attention sublayer (P1..P6)
followed by MkMoE's FFN sublayer (router, router tail, routed experts, shared expert, mhc_post):

  P1  mhc_pre (attention)        owners + root            x -> hn, x_out, coef
  P2  qkv + low-rank             readers (wait hn)        -> qkv, lr
  P3  kda_pre / P4 kda_rec       owners                   -> on, recurrent state
  P5  o_proj                     readers (wait on)        -> a
  P6  mhc_post (attention)       owners (wait a)          -> s1 (stacked, L1)
  P7  mhc_pre (FFN)              owners + root            s1 -> hn2, coef2
  R   router (DS, fp32 logits)   readers (wait hn2)       -> logits                    -- SEM_LOGITS --> tail cores
  RT  router tail                t tail cores (row k)     -> ids, scores               -- SEM_TAIL --> readers
  S1  shared gate_up             readers                  -> gu                        -- SEM_GU --> owners
  P8  swiglu (shared)            owners (wait gu)         -> m                         -- SEM_M --> readers
  X   routed experts (mkx)       readers (wait tail)      -> y (routed; this chip's partial under EP; no signal of its own)
  S2  shared down                readers (wait m)         -> y2   -- SEM_A --> owners (on each reader after X: covers y)
  P10 mhc_post (FFN)             owners (wait shared down) y + y2 + C^T s1 -> the layer output [1,1,T,16384] fp32 DRAM

Kernels: the owners run MkLayer's composites unchanged (kernels/mk/owner_*.cpp: P3, P4, P6, P7, P8 = the shared expert's
swiglu, P10 = mhc_post with y2), the root MkLayer's root composites, the readers MkMoE's composites (kernels/mk/
reader_moe_*.cpp: dense phases qkv, low-rank, o_proj, router, shared gate_up | shared down with the routed experts
between the last two: MK_SPLIT = 5). The router tail runs on its own t cores (the owners have no free CB index: kda_pre
32..49, kda_rec 50..63) with the standalone router_tail kernels and the same hand-off protocol: the readers signal
every phase's consumers = owners + tail cores (a semaphore id not declared on a core is a slot nobody there reads:
the program's semaphore region is sized by its largest id, per core type).
Semaphores — owners: 0..3 (mhc_pre), QKV, KDA, A (both mhc_post), S1, GU; readers: HN, ON, M, the experts' 10..14,
TAIL; tail cores: LOGITS. Every hand-off is waited for and reset before its next use can be signalled.
TP / EP (Stage 14): e_lo / num_experts select this chip's experts (y = this chip's routed partial); the shared expert's
DSLinears may be column / row shards (y2 = a partial); P10 then needs the all-reduce of y + y2 (one fabric phase after
S2 and X, before P10). Tail cores are taken from the END of the free-core list (Stage 14's lanes take the first two)."""

from __future__ import annotations

import os

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as _mpost
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as _mhc
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import (
    KDA_CB_BASE,
    POST_CB_BASE,
    REC_CB_BASE,
    REC_CB_X0,
    SEM_A,
    SEM_HN,
    SEM_KDA,
    SEM_ON,
    SEM_QKV,
    _cores_of,
    _role,
    _rt_args,
    rect_placement,
)
from models.autoports.zai_org_glm_5_3_flash.tt.mk_experts import SEM_LIST, SEM_MR1, ReaderExperts
from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import SEM_GU, SEM_M, SEM_S1, SWI_CB0, f32_bits, residual_modes
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import CB_RING0, MMPhase, ReaderPhases

BF16, FP32 = ttnn.bfloat16, ttnn.float32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
TILE = 32
SEM_LOGITS = 13  # readers -> tail cores (an id no owner declares: the owners also receive this signal)
SEM_TAIL = 15  # tail writers -> readers (the experts' wait)
SEM_Y2 = 6  # fold_y2: readers -> readers, shared down's y2 tiles landed (an id no reader otherwise uses)
# 7a GLM53_MK_SPARSE_DEFER_INV=1 (the dense layer's GLM53_MK_DEFER_INV, for the sparse layer): root -> readers, an
# mhc_pre instance's 1/rms(h) landed in its invh tensor; SEM_QKV's id (declared on the owners only) is free on the readers
SEM_INVH_S = 5
SEM_XTP = SEM_A  # expert TP (xtp = c): senders -> the bank's leader reader (SEM_A is declared on the owners only)
RTAIL_CB0 = 32  # the tail cores' CBs 32..36 (their own cores)
X_CB0 = 48  # the routed experts' reader CBs 48..58 (the dense reader phases: 31, 38..47)
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
assert SEM_MR1 < SEM_TAIL and SEM_LIST > SEM_M, "reader semaphore ids"
assert SEM_XTP not in (SEM_HN, SEM_ON, SEM_M, SEM_TAIL, SEM_Y2) and not SEM_LIST <= SEM_XTP <= SEM_MR1, "expert TP id"


class MkSparseLayer:
    """The sparse_kda layer (attention + MoE FFN sublayers) in ONE program; see the module docstring."""

    PROBE_ROWS = (
        256  # MkLayer's probe slot layout (mk_layer.decode_probe); MkStack: layer l -> rows [256 l, 256 (l + 1))
    )

    def __init__(self, device, a_args: dict, f_args: dict, kda, lins: dict, mlp, *, e_lo: int = 0,
                 num_experts: int | None = None, expert_ring_slots: int | None = None, probe: bool = False,
                 fold_y2: bool | None = None, xtp: int | None = None,
                 local_experts: list | None = None, replica_home: list | None = None,
                 chip: int = 0, xsl: int | None = None, xsl_tp: int = 8, xsl_sliced: bool = False):  # fmt: skip
        """lins: the attention DSLinears {kda_qkv, kda_lowrank, kda_o}; mlp: the layer's OptimizedMoE (router DSLinear +
        RouterTail bias, the shared expert's DSLinears, the routed experts' weight lists). fold_y2 (default
        GLM53_MK_FOLD_Y2=1): shared down runs BEFORE the routed experts and seeds their running sums, so y = routed +
        shared in ONE partial (one all-reduce when the shared expert is TP-sharded and the experts are EP); P10 then
        reads y only (its own wait count MK_POST2_WAIT_COUNT = shared down + experts)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre
        from models.autoports.zai_org_glm_5_3_flash.tt.kda_rec import KdaRec

        self.device = device
        self.fold_y2 = (os.environ.get("GLM53_MK_FOLD_Y2") == "1") if fold_y2 is None else bool(fold_y2)
        router, shared = mlp.router, mlp.shared
        env_kb = lambda k: int(os.environ[k]) if os.environ.get(k) else None  # read per construction
        kb_big, kb_small = env_kb("GLM53_MK_RING_KB_QKV"), env_kb("GLM53_MK_RING_KB_O")
        kq, ko = int(os.environ.get("GLM53_MK_KB_QKV", "2")), int(os.environ.get("GLM53_MK_KB_O", "8"))
        if expert_ring_slots is None:
            expert_ring_slots = int(os.environ.get("GLM53_MKX_RING", "4"))
        on_count = (
            int(kda.h) * int(getattr(kda, "kda_rec_spread", 1)) * int(getattr(kda, "kda_rec_ugroups", 1))
        )  # kda_rec head cores (TP shard: kda.h; user groups)
        alias_ = os.environ.get("GLM53_MK_QKV_ALIAS") == "1"  # (qkv's own ring under the alias, see below)
        # 7a GLM53_MK_QKV_R0=1 (Stage 14's ask, reader range per phase): low-rank on reader 0 of each bank alone, qkv on
        # readers 1 .. R-1 (both start at hn; low-rank no longer queues behind qkv on the same readers)
        r0q = 1 if os.environ.get("GLM53_MK_QKV_R0") == "1" else 0
        # Stage 14's fidelity knobs, wired here as in MkLayer (7a 2026-10-20: the sparse layer ignored them):
        # GLM53_MK_DS_FIDELITY = the bf8 weight phases (qkv, o_proj, shared gate_up / down), GLM53_MK_LR_FIDELITY = the
        # low-rank phase; unset = the DSLinears' own (HiFi2 / HiFi4). Numerics-changing: m3_tp marks such rows EXPLORE.
        fid = lambda k: getattr(ttnn.MathFidelity, os.environ[k]) if os.environ.get(k) else None  # noqa: E731
        dsf, lrf = fid("GLM53_MK_DS_FIDELITY"), fid("GLM53_MK_LR_FIDELITY")
        phases = [
            MMPhase(
                lins["kda_qkv"],
                kb=kq,
                wait=(SEM_HN, 64),
                signal=SEM_QKV,
                ring_kb=kb_big,
                ring_tag=int(alias_),
                r0=r0q,
                math_fidelity=dsf,
            ),
            # the low-rank ring at 2 slots (a short phase; 16 KB of reader L1 for fold_y2's y2 tiles at T = 4)
            MMPhase(lins["kda_lowrank"], kb=8, signal=SEM_QKV, ring_kb=1, rp_max=1 if r0q else None, math_fidelity=lrf),
            MMPhase(lins["kda_o"], kb=ko, wait=(SEM_ON, on_count), signal=SEM_A, ring_kb=kb_small, math_fidelity=dsf),
            # reader L1 (the experts' CBs take ~0.5 MB): the router at K block GLM53_MK_KB_ROUTER (4: a small ring), the
            # shared gate_up at the o_proj / down K block (GLM53_MK_KB_SGU, default ko: its blocks then share their ring)
            # Stage 14 knob GLM53_MK_ROUTER_FIDELITY (HiFi2 / HiFi3): the K-bound router phase's math fidelity (default:
            # the DSLinear's) — numerics judged by the layer_check and its router near-tie rows
            MMPhase(
                self._router_ds(router),
                kb=int(os.environ.get("GLM53_MK_KB_ROUTER", "4")),
                wait=(SEM_HN, 64),
                signal=SEM_LOGITS,
                k_split=self.router_k_split,
                math_fidelity=getattr(ttnn.MathFidelity, os.environ["GLM53_MK_ROUTER_FIDELITY"])
                if os.environ.get("GLM53_MK_ROUTER_FIDELITY")
                else None,
            ),
            MMPhase(
                shared.ds_gate_up,
                kb=int(os.environ.get("GLM53_MK_KB_SGU", str(ko))),
                signal=SEM_GU,
                ring_kb=kb_big,
                math_fidelity=dsf,
            ),
            MMPhase(shared.ds_down, kb=ko, wait=(SEM_M, 64), signal=SEM_A, ring_kb=kb_small, math_fidelity=dsf),
        ]
        # 3 readers per bank (GLM53_MK_READERS overrides): the routed experts need >= 3 (ReaderExperts)
        self.readers = ReaderPhases(device, phases, readers_per_bank=int(os.environ.get("GLM53_MK_READERS") or 3))
        rd = self.readers
        if CB_RING0 + len(rd.rings) > X_CB0:
            raise ValueError(f"MkSparseLayer: {len(rd.rings)} dense rings reach the experts' CB {X_CB0}")
        # 7a GLM53_MK_QKV_ALIAS=1: the qkv weight ring (phase 0's, exclusive) shares its L1 with the experts' resident x
        # (CB_X, 128 bf16 tiles = 256 KB, idle until the expert phase): the ring grows from a few slots to the whole
        # region, so most of a reader's qkv weights stream in while mhc_pre runs (the readers' CBs are ~1.45 of 1.57 MB:
        # a deep ring of its own does not fit). The shared CB's size = slots x block (the streamer's slot addressing)
        # and a multiple of CB_X's 2 KB page, >= CB_X; CB_X's zero fill moves after phase 0 (reader_moe_a.cpp hook)
        self.qkv_alias = os.environ.get("GLM53_MK_QKV_ALIAS") == "1"
        if self.qkv_alias:
            from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import TILE_BYTES

            qring = rd.geo[0]["ring"]
            assert sum(g["ring"] == qring for g in rd.geo) == 1, "GLM53_MK_QKV_ALIAS: the qkv ring must be exclusive"
            dt, block = rd.rings[qring]
            bb = block * TILE_BYTES[dt]
            x_bytes = 2048 * (int(mlp.hidden) // TILE)
            s_ = -(-x_bytes // bb)
            while (s_ * bb) % 2048:
                s_ += 1
            rd.ring_slots[qring] = s_
            self.qkv_alias_ring, self.qkv_alias_bytes = qring, s_ * bb
        self.ring_slots = dict(rd.ring_slots)
        self.experts = ReaderExperts(device, mlp.gate_up, mlp.down, top_k=mlp.top_k, scaling=mlp.scaling,
                                     limit=mlp.limit, intermediate=mlp.intermediate, hidden=mlp.hidden,
                                     readers_per_bank=rd.readers, ring_slots=expert_ring_slots, e_lo=e_lo,
                                     num_experts=num_experts if num_experts is not None else mlp.num_experts,
                                     cb0=X_CB0, m_buffers=int(os.environ.get("GLM53_MKX_MBUF", "1")),
                                     xtp=xtp, xtp_sem=SEM_XTP, local_experts=local_experts,
                                     replica_home=replica_home, chip=chip, xsl=xsl, xsl_tp=xsl_tp,
                                     xsl_sliced=xsl_sliced)  # fmt: skip
        ra = [(int(c.x), int(c.y)) for c, _, _ in rd.assign]
        assert ra == [(int(c.x), int(c.y)) for c, _, _ in self.experts.assign], "same reader cores, same order"
        self.place = rect_placement(device, set(ra))
        self.mhc_a = _mhc.MhcPre(device, **a_args, placement=self.place)
        self.mhc_f = _mhc.MhcPre(device, **f_args, placement=self.place)
        self.kda = KdaPre(kda, rows_max=TILE, cores=self.place["k"], cb_base=KDA_CB_BASE)  # ROWS mode up to 32 rows
        # kda_rec depth (GLM53_REC_DEPTH, read per construction): 1 (default) = lean (single-buffered per-item inputs
        # and state); 2 (Stage 14) = inputs and state double-buffered at every t (+148 KB on the owners: clashes with the
        # layer's L1 tensors at t = 32 on one chip); s (7a) = lean inputs + a second state slot at t > 1 (+64 KB): the
        # next item's 64 KB state streams in while one item computes (lean serialises every state read with compute)
        self._rec_mode = os.environ.get("GLM53_REC_DEPTH", "1")
        assert self._rec_mode in ("1", "2", "s"), self._rec_mode
        self.rec = KdaRec(kda, cores=self.place["k"], cb_base=REC_CB_BASE, cb_x0=REC_CB_X0, lean=self._rec_mode != "2")
        self._rec_kw = dict(cores=self.place["k"], cb_base=REC_CB_BASE, cb_x0=REC_CB_X0)
        self._rec_deep = None
        # kda_rec's spread sum exchange (Stage 14 TP) takes an owner semaphore: never the ids the readers also send to
        # the owners here (13 = SEM_LOGITS) or the readers' tail id
        assert self.rec.spread == 1 or self.rec.rms_sem not in (SEM_LOGITS, SEM_TAIL), self.rec.rms_sem
        self.post = _mpost.MhcPost(device, cores=self.place["k"], cb_base=POST_CB_BASE)
        self.owner_noc = [
            (int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c) for c in self.place["k"])
        ]
        grid = device.compute_with_storage_grid_size()
        used = set(ra) | {(int(c.x), int(c.y)) for c in self.place["k"]}
        used.add((int(self.place["root"].x), int(self.place["root"].y)))
        free = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in used]
        if len(free) < 3:
            raise ValueError(f"MkSparseLayer: {len(free)} free cores (tail cores + 2 fabric lanes)")
        # the tail cores: from the end, leaving free[:2] to Stage 14's fabric lanes; at most one per row, several rows
        # per core when t exceeds them (router_tail kernels: row arg = first row | (rows - 1) << 16)
        self.tail_free = free[2:][::-1]
        self.tail_bias = router.tail.bias
        self.shared_i = shared.intermediate
        self.i_tiles = self.shared_i // TILE
        n_own = len(self.place["k"])
        assert self.i_tiles % n_own == 0 or self.i_tiles < n_own, (self.i_tiles, n_own)
        self.nt = max(1, self.i_tiles // n_own)
        self.limit_shared = shared.limit
        self.persist = {}
        self.probe = None
        if probe:  # per-phase wall-clock marks (kernels/mk/probe.hpp; MkLayer's slot layout: mk_layer.decode_probe)
            import torch

            self.probe = ttnn.from_torch(torch.zeros(256, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)  # fmt: skip
        self.park_plan = None
        self.calls = 0

    def persistent_buffers(self, t: int):
        """the hand-off buffers kept across calls (Stage 14b's stack shares one set between its layers)"""
        mk = lambda shape: ttnn.empty(shape, BF16, ttnn.TILE_LAYOUT, self.device, L1)  # noqa: E731
        self._persistent(("on", t), lambda: self._rec_for(t).output(t))
        self._persistent(("m", t), lambda: mk((1, 1, t, self.shared_i)))

    def _router_ds(self, router):
        """The router phase's DSLinear. Stage 14 GLM53_MK_ROUTER_KSPLIT=RS (2 / 4): the K-bound router (4096 x 320 on
        ~24 readers, 128 K tiles each) as ONE phase over RS times the columns — chunk c's weight rows side by side, the
        readers of chunk c read only activation K tiles [c K / RS, (c + 1) K / RS) (MMPhase.k_split) — and the router
        tail sums the RS partial logits (RTAIL_KSPLIT) before the sigmoid."""
        rs = int(os.environ.get("GLM53_MK_ROUTER_KSPLIT", "1"))
        self.router_ks, self.router_k_split = rs, None
        if rs <= 1:
            return router.ds
        import torch

        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear

        w = router.weights[router.num_experts]
        wt = ttnn.to_torch(ttnn.get_device_tensors(w)[0]).float()
        wt = wt.reshape(-1, wt.shape[-1])  # [K, 288]
        # every chunk padded to the router tail's width (COL_TILES = 10 tiles = 320 columns, router_tail_common.hpp):
        # the tail reads chunk c at column tile c * COL_TILES (the padded experts' bias is -1e9 there). The 288-column
        # stride put chunk 1 one tile early (2026-10-20 06:08Z: TP2 PCC 0.996 / 0.911 at L4 / L44)
        n = 10 * TILE
        assert wt.shape[-1] <= n, wt.shape
        wt = torch.nn.functional.pad(wt, (0, n - wt.shape[-1]))
        k = wt.shape[0]
        assert k % (rs * TILE) == 0, (k, rs)
        kc = k // rs
        wcat = torch.cat([wt[c * kc : (c + 1) * kc] for c in range(rs)], dim=1).reshape(1, 1, kc, rs * n)
        n_dev = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        wdev = ttnn.from_torch(wcat.to(torch.bfloat16), dtype=w.dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
                               memory_config=DRAM,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if n_dev > 1 else None)  # fmt: skip
        ds = DSLinear(self.device, wdev, role="router", cfg=router.ds.ocfg, ckc=router.ds.ckc, dtype=FP32)
        self.router_k_split = (n // TILE, kc // TILE)
        return ds

    def _persistent(self, key, make):
        if key not in self.persist:
            self.persist[key] = make()
        return self.persist[key]

    def _rec_for(self, t: int):
        """kda_rec for t rows: self.rec, or at GLM53_REC_DEPTH=s and t > 1 the lean one with a second state slot"""
        if t > 1 and self._rec_mode == "s":
            if self._rec_deep is None:
                from models.autoports.zai_org_glm_5_3_flash.tt.kda_rec import KdaRec

                self._rec_deep = KdaRec(self.kda.kda, lean=True, state_depth=2, **self._rec_kw)
            return self._rec_deep
        return self.rec

    def tail_rows(self, t: int):
        """[(core, first row, rows)]: t rows over at most len(tail_free) cores, consecutive rows per core"""
        n = min(t, len(self.tail_free))
        per = -(-t // n)
        out, r = [], 0
        for xy in self.tail_free[:n]:
            if r >= t:
                break
            out.append((ttnn.CoreCoord(*xy), r, min(per, t - r)))
            r += per
        return out

    def __call__(self, x, hist, active, *, t: int, state, stack: bool = False, live=None):
        """x: the layer input [1,1,t,16384] fp32 (flat); hist = state.C, state = state.S (both updated in place);
        active = the decoder's masks -> dict(out = the layer output [1,1,t,16384] fp32 DRAM, s1, hn, hn2, qkv, a,
        logits, ids, scores, y (routed), y2 (shared), m, gu).
        stack=True (Stage 14b, tt/mk_stack.py): x = the persistent residual res (t <= 4 stacked [t,1,4,4096], t > 4
        flat [1,1,t,16384], fp32), read by P1, the attention residual of P6, overwritten by P10: out = x (see MkLayer).
        """
        rm = residual_modes(t)
        dev, ma, mf, rd = self.device, self.mhc_a, self.mhc_f, self.readers
        mk = lambda shape, dt, mem=L1: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, dev, mem)
        hn, hn2 = mk((1, 1, t, ma.hidden), BF16), mk((1, 1, t, mf.hidden), BF16)
        post_a, comb_a = (mk(sh, FP32) for sh in rm["pc_shapes"])
        post_f, comb_f = (mk(sh, FP32) for sh in rm["pc_shapes"])
        # t > 4: P6 reads x itself; stack: P6 reads the residual res = x
        x_out = mk((t, 1, 4, ma.hidden), FP32) if rm["stacked"] and not stack else None
        coef_a, coef_f = mk((1, 1, 32, 32), FP32), mk((1, 1, 32, 32), FP32)
        sig = (SEM_HN, rd.noc_coords())
        pa = self.probe.buffer_address() if self.probe is not None else None
        # 7a GLM53_MK_SPARSE_DEFER_INV=1 (probe 2026-10-20: GLM53_MK_DEFER_INV only reaches the DENSE layer; here the
        # K cores held hn until round 2's 1/rms(h): 'pre in -> hn packed' ~13 us per instance at B=1): both mhc_pre
        # instances publish hn = h * w before their round 2; the readers scale qkv / low-rank (attention) and router /
        # shared gate_up (FFN) outputs at pack by the root's per-row 1/rms(h) (all linear in hn), and the routed
        # experts' gate | up before SwiGLU (mk_experts MKX_XINV: xc, from the FFN invh tile)
        defer_a = os.environ.get("GLM53_MK_SPARSE_DEFER_INV") == "1"
        invh_a = self._persistent(("invh_a",), lambda: mk((1, 1, 32, 32), FP32)) if defer_a else None
        invh_f = self._persistent(("invh_f",), lambda: mk((1, 1, 32, 32), FP32)) if defer_a else None
        rd.scale = dict(groups=[((0, 1), invh_a), ((3, 4), invh_f)], sem=SEM_INVH_S) if defer_a else None
        roles_a, mhc_cbs, mhc_sems = ma.parts(x, hn, post_a, comb_a, x_out, coef_a, t=t,
                                              x_mode=rm["s1_x"] if stack else _mhc.X_FLAT,
                                              xout_mode=_mhc.XOUT_NONE if stack else rm["xout"], pc_mode=rm["pc"],
                                              signal=sig,
                                              mk_probe=(pa, 0) if pa is not None else None, coef_plane=True,
                                              defer=(invh_a, SEM_INVH_S, rd.noc_coords()) if defer_a else None)  # fmt: skip
        s1 = self.post.output(t, rm["s1_out"])
        roles_f, mhc_cbs_f, _ = mf.parts(s1, hn2, post_f, comb_f, None, coef_f, t=t, x_mode=rm["s1_x"],
                                         xout_mode=_mhc.XOUT_NONE, pc_mode=rm["pc"], signal=sig, coef_plane=True,
                                         defer=(invh_f, SEM_INVH_S, rd.noc_coords()) if defer_a else None)  # fmt: skip
        _same_cbs(mhc_cbs, mhc_cbs_f, "mhc_pre")
        outs = rd.outputs(t)
        qkv, lr, a, logits, gu, y2 = outs
        rec = self._rec_for(t)
        on = self._persistent(("on", t), lambda: rec.output(t))  # zeroed once (whole-tile writes after)
        m = self._persistent(("m", t), lambda: mk((1, 1, t, self.shared_i), BF16))
        ids = mk((1, 1, t, self.experts.top_k), ttnn.uint16)
        scores = mk((1, 1, t, self.experts.e), BF16)
        # GLM53_MKX_RW (default on): the router tail computes the rows' routing weights (mk_moe.py)
        rw = os.environ.get("GLM53_MKX_RW", "1") == "1"
        weights = mk((1, 1, t, self.experts.top_k), BF16) if rw else None
        y = mk((1, 1, t, mf.hidden), BF16)
        nr = len(rd.assign)
        # P3 kda_pre, P4 kda_rec
        kroles, kda_cbs, kda_io, kda_outs = self.kda.parts(qkv, lr, hist, active, rows=t,
                                                           signal=(SEM_KDA, self.owner_noc))  # fmt: skip
        cons = self.kda.consumers()
        count = {h: sum(h in hs for hs in cons.values()) for h in range(len(self.owner_noc))}
        xy = {h: (int(c.x), int(c.y)) for h, c in enumerate(self.place["k"])}
        rroles, rec_cbs, rec_io = rec.parts(*kda_outs[:6], state, on, rows=t,
                                                 wait=(SEM_KDA, {xy[h]: count[h] for h in count}),
                                                 signal=(SEM_ON, rd.noc_coords()))  # fmt: skip
        # P6 / P10 mhc_post (packed; the same CB set): P10 adds the shared expert's y2 to the routed y
        proles_a, post_cbs, post_io_a = self.post.parts(a, x_out if x_out is not None else x, coef_a, s1, t=t,
                                                        x_mode=rm["post_x"], out_mode=rm["s1_out"], wait=(SEM_A, nr),
                                                        packed=rm["stacked"], fused_v2=True)  # fmt: skip
        out = x if stack else mk((1, 1, t, 4 * ma.hidden), FP32, DRAM)
        proles_f, post_cbs_f, post_io_f = self.post.parts(y, s1, coef_f, out, t=t, x_mode=rm["post_x"],
                                                          out_mode=rm["s1_out"] if stack else _mpost.OUT_FLAT,
                                                          y2=None if self.fold_y2 else y2,
                                                          wait=(SEM_A, nr),
                                                          packed=rm["stacked"], fused_v2=True)  # fmt: skip
        _same_cbs(post_cbs, post_cbs_f, "mhc_post")
        for role in list(proles_a.values()) + list(proles_f.values()) if rm["stacked"] else ():
            role[2].append(("MK_POST_PACKED", "1"))
        # P8 swiglu of the shared expert (owner k: m column tiles nt k ..)
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        rtd = lambda fn: {(int(c.x), int(c.y)): [int(v) for v in fn(k)] for k, c in enumerate(self.place["k"])}
        sdef = [("SWI_CB0", str(SWI_CB0))]
        flat_r = [v for xy_ in rd.noc_coords() for v in xy_]
        hifi4 = lambda: ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                     fp32_dest_acc_en=True)  # fmt: skip
        it = self.i_tiles
        swi = dict(
            reader=([self.nt, it] + acc(gu), rtd(lambda k: [gu.buffer_address(), (self.nt * k) % it, nr]),
                    sdef + [("MK_SWI_WAIT_SEM", str(SEM_GU))], None),
            compute=([self.nt, f32_bits(self.limit_shared), f32_bits(-self.limit_shared)], rtd(lambda k: []), sdef,
                     hifi4()),
            writer=([self.nt] + acc(m), rtd(lambda k: [m.buffer_address(), (self.nt * k) % it, nr] + flat_r),
                    sdef + [("MK_SWI_SIGNAL_SEM", str(SEM_M))], None),
        )  # fmt: skip
        swi_cbs = [ttnn.CBDescriptor(total_size=2048 * n, core_ranges=ma.kcores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=SWI_CB0 + i, data_format=BF16, page_size=2048)])
            for i, n in ((0, self.nt), (1, self.nt), (2, 2))]  # fmt: skip

        def p7(name, lead):
            if name == "reader":
                r = _role(roles_f, "stream.cpp", "MHC_K")
            else:
                f = {"writer": "sync.cpp", "compute": "compute.cpp"}[name]
                r = _role(roles_f, f, "MHC_LEADER") if lead else _role(roles_f, f, "MHC_K", "MHC_LEADER")
            return r["ct"], r["rt"], r["defines"], r["config"]

        def combine(mrole, role_name, lead, extra):
            mine = _cores_of(mrole["cores"])
            ct, defs = list(mrole["ct"]), list(mrole["defines"])
            n_rt = max((len(v) for v in mrole["rt"].values()), default=0)
            rt = {core: list(mrole["rt"].get(core, [])) for core in mine}
            rt = {core: v + [0] * (n_rt - len(v)) for core, v in rt.items()}
            phase_list = [(3, kroles[role_name]), (4, rroles[role_name]), (6, proles_a[role_name]),
                          (7, p7(role_name, lead)), (8, swi[role_name]), (10, proles_f[role_name])]  # fmt: skip
            for n, (pct, prt, pdefs, _) in phase_list:
                defs += [d for d in pdefs if d not in defs] + [
                    (f"MK_P{n}_CT", str(len(ct))),
                    (f"MK_P{n}_RT", str(n_rt)),
                ]
                ct += list(pct)
                width = max(len(prt.get(core, [])) for core in mine)
                for core in mine:
                    v = [int(q) for q in prt.get(core, [])]
                    rt[core] += v + [0] * (width - len(v))
                n_rt += width
            seen = {}
            for k_, v_ in defs + extra:
                if seen.setdefault(k_, v_) != v_:
                    raise ValueError(f"MkSparseLayer: define {k_} = {seen[k_]} / {v_} in one {role_name} kernel")
            return ct, rt, defs + extra

        mkk = lambda src, cores, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=_rt_args(rt), defines=defs, config=cfg,
        )  # fmt: skip
        s1def = [("MK_S1_SEM", str(SEM_S1))] + (
            [("MK_PRE_PREFETCH", "1")] if os.environ.get("GLM53_MK_PRE_PREFETCH", "1") == "1" else []
        )
        kernels = []
        ks = _role(roles_a, "stream.cpp", "MHC_K")
        p10 = [("MK_POST2_WAIT_COUNT", str(2 * nr))] if self.fold_y2 else []  # fold_y2: shared down + experts
        ct, rt, defs = combine(ks, "reader", None,
                               s1def + [("MK_SEM_QKV", str(SEM_QKV)), ("MK_QKV_COUNT", str(2 * nr))]
                               # Stage 14 GLM53_KDA_PREFETCH=1: kda_pre / kda_rec request their step-independent inputs
                               # before their hand-off waits (the owner skips its qkv wait; kda_pre waits itself)
                               + ([("KDA_PREFETCH", "1")] if os.environ.get("GLM53_KDA_PREFETCH") == "1" else []) + p10)  # fmt: skip
        kernels.append(mkk("mk/owner_dm0.cpp", ks["cores"], ct, rt, defs, ks["config"]))
        for lead in (True, False):
            sr = _role(roles_a, "sync.cpp", "MHC_LEADER") if lead else _role(roles_a, "sync.cpp", "MHC_K", "MHC_LEADER")
            ct, rt, defs = combine(sr, "writer", lead, s1def)
            if pa is not None:  # the owner NCRISC's probe args after every phase's: slot 96 + k
                kidx = {(int(c.x), int(c.y)): k for k, c in enumerate(self.place["k"])}
                n_all = len(next(iter(rt.values())))
                rt = {core: v + [pa, 96 + kidx[core]] for core, v in rt.items()}
                defs = defs + [("MK_PROBE_ARG", str(n_all)), ("MK_PROBE_CB", "31")]
            kernels.append(mkk("mk/owner_dm1.cpp", sr["cores"], ct, rt, defs, sr["config"]))
            cr = (_role(roles_a, "compute.cpp", "MHC_LEADER") if lead
                  else _role(roles_a, "compute.cpp", "MHC_K", "MHC_LEADER"))  # fmt: skip
            ct, rt, defs = combine(cr, "compute", lead, s1def)
            cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                              fp32_dest_acc_en=True, dst_full_sync_en=True)  # fmt: skip
            modes = list(cr["config"].unpack_to_dest_mode)
            for roles_ in (kroles, rroles, proles_a, proles_f):
                kmodes = list(roles_["compute"][3].unpack_to_dest_mode)
                modes = [q if q != ttnn.UnpackToDestMode.Default else b for q, b in zip(modes, kmodes)]
            cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
            kernels.append(mkk("mk/owner_c.cpp", cr["cores"], ct, rt, defs, cc))
        for src, comp in (("stream.cpp", "root_dm0.cpp"), ("sync.cpp", "root_dm1.cpp"), ("compute.cpp", "root_c.cpp")):
            ra, rf = _role(roles_a, src, "MHC_ROOT"), _role(roles_f, src, "MHC_ROOT")
            core = (int(self.place["root"].x), int(self.place["root"].y))
            va, vf = list(ra["rt"].get(core, [])), list(rf["rt"].get(core, []))
            defs = _merge_defines(ra["defines"], rf["defines"], where="root")
            defs += [("MK_P7_CT", str(len(ra["ct"]))), ("MK_P7_RT", str(len(va)))]
            kernels.append(mkk("mk/" + comp, ra["cores"], list(ra["ct"]) + list(rf["ct"]), {core: va + vf}, defs,
                               ra["config"]))  # fmt: skip
        # the router tail on its own t cores (row k on tail core k), standalone kernels
        tails = self.tail_rows(t)  # [(core, first row, rows)]
        tail_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in tails])
        tail_noc = [(int(w.x), int(w.y)) for w in (dev.worker_core_from_logical_core(c) for c, _, _ in tails)]
        trt = lambda fn: {(int(c.x), int(c.y)): [int(v) for v in fn(r0 | ((n_ - 1) << 16))] for c, r0, n_ in tails}
        # RTAIL_REV: kernel revision in the compile signature (2: a row range per core) — a long-lived process rebuilds
        tdef = [("RTAIL_CB0", str(RTAIL_CB0)), ("RTAIL_REV", "2")]
        if self.router_ks > 1:  # Stage 14: the tail sums the router's K-chunk partial logits
            tdef = tdef + [("RTAIL_KSPLIT", str(self.router_ks))]
        tail_cc = hifi4()
        tmodes = [ttnn.UnpackToDestMode.Default] * 64
        tmodes[RTAIL_CB0 + 0] = tmodes[RTAIL_CB0 + 1] = ttnn.UnpackToDestMode.UnpackToDestFp32
        tail_cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(tmodes)
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        # probes (tail core k): reader -> slot 224 + 2 k, writer -> 225 + 2 k (the kernels flush under RTAIL_PROBE)
        kidx_t = {(int(c.x), int(c.y)): k for k, (c, _, _) in enumerate(tails)}
        tp = lambda off: (lambda xy: [pa, 224 + 2 * kidx_t[xy] + off] if pa is not None else [])  # noqa: E731
        tpd = lambda n_args: ([("MK_PROBE_ARG", str(n_args)), ("MK_PROBE_CB", "31"), ("RTAIL_PROBE", "1")]
                              if pa is not None else [])  # fmt: skip
        r_args = {xy: v + tp(0)(xy) for xy, v in
                  trt(lambda rows: [logits.buffer_address(), self.tail_bias.buffer_address(), rows, nr]).items()}  # fmt: skip
        w_args = {xy: v + tp(1)(xy) for xy, v in
                  trt(lambda rows: [ids.buffer_address(), scores.buffer_address(), rows, nr] + flat_r
                      + ([weights.buffer_address()] if rw else [])).items()}  # fmt: skip
        n_wa = 4 + len(flat_r) + (1 if rw else 0)  # the writer's runtime args before the probe args
        kernels += [
            mkk("router_tail/reader.cpp", tail_crs, acc(logits) + acc(self.tail_bias), r_args,
                tdef + [("MK_RTAIL_WAIT_SEM", str(SEM_LOGITS))] + tpd(4),
                dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            mkk("router_tail/compute.cpp", tail_crs, [], trt(lambda rows: [rows]), tdef, tail_cc),
            mkk("router_tail/writer.cpp", tail_crs,
                acc(ids) + acc(scores) + (acc(weights) + self.experts.rw_ct() if rw else []), w_args,
                tdef + [("MK_RTAIL_SIGNAL_SEM", str(SEM_TAIL))] + tpd(n_wa)
                + ([("RTAIL_WEIGHTS", "1"), ("RTAIL_W_RT", str(4 + len(flat_r)))] if rw else []),
                dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
        ]  # fmt: skip
        tcb = lambda i, dt, page, n=1: ttnn.CBDescriptor(total_size=n * page, core_ranges=tail_crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=RTAIL_CB0 + i, data_format=dt, page_size=page)])  # fmt: skip
        # CB_LOG: router_ks pages of one packed fp32 tile (the reader reserves / the compute pops router_ks pages; one
        # page of 4096 * router_ks deadlocked the tail on device, 2026-10-20 03:37Z watched pass)
        tail_cbs = [
            tcb(0, FP32, 4096, self.router_ks),
            tcb(1, FP32, 4096),
            tcb(2, FP32, 4096),
            tcb(3, BF16, 2048),
            tcb(4, BF16, 64),
        ]
        if pa is not None:  # the tail cores' probe scratch
            tail_cbs.append(ttnn.CBDescriptor(total_size=128, core_ranges=tail_crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=128)]))  # fmt: skip
        # readers: the dense phases + the routed experts between shared gate_up and shared down; every signalling
        # phase signals owners + tail cores (the router's SEM_LOGITS is the tail cores'; the others the owners')
        draw, r_cbs = rd.parts_raw([hn, hn, on, hn2, hn2, m], outs, consumers=self.owner_noc + tail_noc,
                                   mk_probe=(pa, 160) if pa is not None else None)  # fmt: skip
        # no owner signal from the experts: on every reader the routed phase runs before shared down on the same
        # NCRISC (write barrier first), so shared down's SEM_A signal (count nr) also covers that reader's y tiles —
        # P6 and P10 share one owner kernel whose mhc_post wait count is a define: both must wait for nr
        # fold_y2: the experts run after shared down (MK_SPLIT = 6), seed their sums with y2, and signal the owners
        xkw = dict(signal=(SEM_A, self.owner_noc), y2=y2, y2_sem=SEM_Y2) if self.fold_y2 else {}
        xraw, x_cbs, x_sems, x_io = self.experts.parts_raw(hn2, ids, scores, y, t=t, wait=(SEM_TAIL, len(tails)),
                                                           weights=weights, xinv=invh_f, live=live,
                                                           **xkw)  # fmt: skip
        for r, src in (("w", "reader_moe_w.cpp"), ("a", "reader_moe_a.cpp"), ("c", "reader_moe_c.cpp")):
            _, dct, drt, ddefs, dcfg = draw[r]
            _, xct, xrt, xdefs, xcfg = xraw[r]
            n_drt = max(len(v) for v in drt.values())
            assert all(len(v) == n_drt for v in drt.values()), "uniform dense runtime args"
            rt_ = {core: list(drt[core]) + list(xrt[core]) for core in drt}
            defs_ = _merge_defines(ddefs, xdefs, where=f"reader {r}")
            defs_ += [("MK_SPLIT", "6" if self.fold_y2 else "5"), ("MK_X_CT", str(len(dct))), ("MK_X_RT", str(n_drt))]
            if self.qkv_alias and r == "a":
                defs_ += [("MK_QKV_ALIAS", "1")]
            cfg = dcfg
            if r == "c":  # the experts' fidelity is the kernel-wide one (the dense phases pass theirs per phase)
                cfg = ttnn.ComputeConfigDescriptor(math_fidelity=xcfg.math_fidelity, math_approx_mode=False,
                                                   fp32_dest_acc_en=True)  # fmt: skip
                cfg.unpack_to_dest_mode = self.experts.unpack_modes(dcfg.unpack_to_dest_mode)
            kernels.append(rd.kernel(src, list(dct) + list(xct), rt_, defs_, cfg))
        sem = lambda i, crs: ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0)
        sems = mhc_sems + x_sems + [
            sem(SEM_HN, rd.cores), sem(SEM_ON, rd.cores), sem(SEM_M, rd.cores), sem(SEM_TAIL, rd.cores),
            sem(SEM_QKV, ma.kcores), sem(SEM_KDA, ma.kcores), sem(SEM_A, ma.kcores), sem(SEM_S1, ma.kcores),
            sem(SEM_GU, ma.kcores), sem(SEM_LOGITS, tail_crs),
        ] + ([sem(SEM_INVH_S, rd.cores)] if defer_a else [])  # fmt: skip
        io = [x, ma.fn, ma.w, ma.sb, ma.dbg, post_a, comb_a, coef_a] + ([x_out] if x_out is not None else []) + [hn]
        io += [mf.fn, mf.w, mf.sb, mf.dbg, post_f, comb_f, coef_f, hn2, s1, m]
        io += [t_ for t_ in (out, ids, scores, y, self.tail_bias) if all(t_ is not u for u in io)]
        io += [p.ds.weight for p in rd.phases] + list(outs) + ([invh_a, invh_f] if defer_a else [])
        io += [t_ for t_ in kda_io + rec_io + post_io_a + post_io_f + x_io if all(t_ is not u for u in io)]
        if self.qkv_alias:  # one region, two indices: the qkv ring and CB_X (see __init__)
            idx = lambda c: int(c.format_descriptors[0].buffer_index)  # noqa: E731
            ring_cb = next(c for c in r_cbs if idx(c) == self.qkv_alias_ring)
            x_cb = next(c for c in x_cbs if idx(c) == X_CB0)
            assert int(ring_cb.total_size) == self.qkv_alias_bytes and int(x_cb.total_size) <= self.qkv_alias_bytes, (
                int(ring_cb.total_size),
                int(x_cb.total_size),
                self.qkv_alias_bytes,
            )
            merged = ttnn.CBDescriptor(total_size=self.qkv_alias_bytes, core_ranges=ring_cb.core_ranges,
                                       format_descriptors=[ring_cb.format_descriptors[0], x_cb.format_descriptors[0]])  # fmt: skip
            r_cbs = [c for c in r_cbs if c is not ring_cb] + [merged]
            x_cbs = [c for c in x_cbs if c is not x_cb]
        all_cbs = mhc_cbs + kda_cbs + rec_cbs + post_cbs + swi_cbs + r_cbs + x_cbs + tail_cbs
        if pa is not None:  # the probe scratch (CB 31) on the owners + root (the readers' comes with parts_raw)
            io.append(self.probe)
            all_cbs += [ttnn.CBDescriptor(total_size=128, core_ranges=ma.all, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=128)])]  # fmt: skip
        if os.environ.get("GLM53_MK_CB_REPORT"):  # static CB bytes per core group (the reader cores bind the L1 floor)
            r0 = ra0 = next(iter(_cores_of(rd.cores)))
            k0 = (int(self.place["k"][0].x), int(self.place["k"][0].y))
            for name, core in (("reader", r0), ("owner", k0)):
                mine = [(int(c.format_descriptors[0].buffer_index), int(c.total_size)) for c in all_cbs
                        if core in _cores_of(c.core_ranges)]  # fmt: skip
                print(f"MK_CB_REPORT {name} total={sum(b for _, b in mine)} " + " ".join(f"{i}:{b}" for i, b in
                                                                                       sorted(mine)), flush=True)  # fmt: skip
            del ra0
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=all_cbs))
        self.calls += 1
        return dict(out=out, s1=s1, hn=hn, hn2=hn2, qkv=qkv, lr=lr, a=a, logits=logits, ids=ids, scores=scores, y=y,
                    y2=y2, m=m, gu=gu, on=on, kda=kda_outs, x_out=x_out, coef_a=coef_a, coef_f=coef_f)  # fmt: skip


def _merge_defines(*lists, where: str) -> list:
    """concatenate define lists, dropping exact duplicates; a name with two values is an error (a define is kernel-wide:
    -D keeps one value, and a phase silently getting another phase's count is how P10 once hung)"""
    out, seen = [], {}
    for lst in lists:
        for k, v in lst:
            if k in seen:
                if seen[k] != v:
                    raise ValueError(f"MkSparseLayer: define {k} = {seen[k]} / {v} in one {where} kernel")
                continue
            seen[k] = v
            out.append((k, v))
    return out


def _same_cbs(a, b, what):
    key = lambda cbs: sorted((int(c.total_size), tuple(int(f.buffer_index) for f in c.format_descriptors)) for c in cbs)
    if key(a) != key(b):
        raise ValueError(f"MkSparseLayer: the two {what} instances need different CB sets; they share indices")


def decode_tail_probe(probe_host, n_tails: int, *, mhz: float = 1350.0, t0: int | None = None) -> dict:
    """MkSparseLayer's probe rows of the router-tail cores (reader 224 + 2 k: start, logits landed, rows read; writer
    225 + 2 k: start, per row (choice ready, written), signalled) -> us after t0 (default: the earliest mark of the
    whole probe tensor), per tail core."""
    p = probe_host.reshape(-1, 16).long() & 0xFFFFFFFF
    nz = p[p > 0]
    base = int(nz.min()) if t0 is None and nz.numel() else (t0 or 0)
    us = lambda row: [round((int(v) - base) / mhz, 1) for v in p[row] if int(v) > 0]  # noqa: E731
    return {f"tail{k}": dict(reader=us(224 + 2 * k), writer=us(225 + 2 * k)) for k in range(n_tails)}
