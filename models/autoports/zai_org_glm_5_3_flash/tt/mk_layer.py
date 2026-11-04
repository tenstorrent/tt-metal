# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel (M1): the WHOLE dense_kda decode layer as ONE program (doc/ttnn/07a-fused-streaming-layer/
megakernel_plan.md §9). Phases (owners = 64 cores in rectangles, root = 1 core, readers = the DS bank readers):

  P1  mhc_pre (attention)        owners + root            x -> hn, x_out (stacked residual copy), coef
  P2  qkv + low-rank             readers (wait hn)        -> qkv, lr
  P3  kda_pre                    owners (wait qkv + lr)   -> q, k, v, g, beta, gate_out, conv history
  P4  kda_rec                    owners (wait kda_pre)    -> on (o_proj input), recurrent state
  P5  o_proj                     readers (wait on)        -> a
  P6  mhc_post (attention)       owners (wait a)          -> s1 (stacked, L1)
  P7  mhc_pre (FFN)              owners (wait own s1) + root  s1 -> hn2, coef2
  P8  gate_up                    readers (wait hn2)       -> gu
  P8b swiglu                     owners (wait gu)         -> m
  P9  down                       readers (wait m)         -> y
  P10 mhc_post (FFN)             owners (wait y)          -> the layer output [1,1,T,16384] fp32 DRAM

The readers' weight streamer never waits for activations: after o_proj's last block it streams gate_up's first blocks
during P6 / P7 (the cross-sublayer overlap two programs cannot have). P7 / P10 reuse the CB indices of P1 / P6 (rewound
at the boundary, kernels/mk/owner_*.cpp, root_*.cpp); mhc_pre's own semaphores are reset after every wait. The s1
hand-off P6 -> P7 is local (owner k writes and reads the same column tiles 2k, 2k+1): the owner's NCRISC raises its own
SEM_S1 after the P6 write barrier, its BRISC waits for it before P7.
Semaphores: owners + root 0..3 (mhc_pre), readers SEM_HN (both mhc_pre instances), SEM_ON, SEM_M; owners SEM_QKV,
SEM_KDA, SEM_A (both mhc_post instances), SEM_S1, SEM_GU. Every hand-off is waited for and reset before its next use
can be signalled (the phases are causally ordered)."""

from __future__ import annotations

import os
import struct

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
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import TILE_BYTES as TILE_BYTES_
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases

BF16, FP32 = ttnn.bfloat16, ttnn.float32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
_S1_DRAM: dict = {}  # t -> (device, the narrow widths' stacked s1 in DRAM)


def _narrow_s1_dram(device, t: int):
    """a narrow width's stacked s1 [t, 1, 4, 4096] fp32 in DRAM (t > 1; stack_model._narrow_knobs: GLM53_MK_S1_DRAM), ONE
    per (device, t) for every layer -- an intermediate of each layer (P6 -> P7), as the pool's shared L1 ones; allocated
    outside the stack's AllocPool (a request the pool never sees leaves its other decisions as they were)."""
    hit = _S1_DRAM.get(t)
    if hit is None or hit[0] is not device or not hit[1].is_allocated():
        hit = _S1_DRAM[t] = (device, ttnn.allocate_tensor_on_device(ttnn.Shape([t, 1, 4, 4096]), ttnn.float32,
                                                                     ttnn.TILE_LAYOUT, device, DRAM))  # fmt: skip
    return hit[1]


TILE = 32
SEM_M = SEM_A + 1  # swiglu writers -> down readers (64 owners)
SEM_S1 = SEM_M + 1  # owner NCRISC -> its own BRISC: s1 columns written (P6 -> P7)
SEM_GU = SEM_S1 + 1  # gate_up readers -> swiglu owners
SEM_PARK = SEM_GU + 1  # owner: its park region's L1 address, read by the readers (forwarding ring)
SEM_INVH = 14  # Stage 14 (GLM53_MK_DEFER_INV): root -> readers, 1/rms(h) of an mhc_pre instance in its invh tensor
PARK_CB = 29  # the owners' park region (never touched by the owner phases)
SWI_CB0 = 18  # swiglu's CBs on the owners: 18..20 (mhc_pre 0..13, kda_rec scratch 14..17, mhc_post 21..28)
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def residual_modes(t: int, stacked_max: int | None = None) -> dict:
    """The residual layouts of a layer program at t decode rows (1..32): t <= 4 = M1's stacked residual (P1 writes a
    stacked copy of x, s1 stacked, the PACKED mhc_post kernels); t > 4 = flat / stream-major (P6 reads the layer
    input x itself, s1 flat [1,1,t,16384], the per-tile mhc_post kernels). Owner k reads / writes the column tiles
    {2k, 2k+1} of every stream in both, so the s1 hand-off P6 -> P7 stays local to the owner."""
    if not 1 <= t <= TILE:
        raise ValueError(f"layer program: t={t} (1..32 decode rows)")
    # stacked_max (default 4 = PACKED_ROWS_MAX): the largest t on the stacked / packed layout; 1 puts t = 2..4 on the
    # flat / per-tile path. Only MkLayer passes it (GLM53_MK_STACKED_MAX, device-validated for the dense layer);
    # MkSparseLayer / MkMoE keep the default
    smax = _mpost.PACKED_ROWS_MAX if stacked_max is None else int(stacked_max)
    assert 1 <= smax <= _mpost.PACKED_ROWS_MAX, smax
    st = t <= smax
    return dict(
        stacked=st,
        xout=_mhc.XOUT_STACKED if st else _mhc.XOUT_NONE,
        pc=_mhc.PC_STACKED if st else _mhc.PC_STREAM_MAJOR,
        pc_shapes=(((t, 1, 4, 1), (t, 1, 4, 4)) if st else ((1, 4, t, 1), (1, 1, t, 16))),
        s1_x=_mhc.X_STACKED if st else _mhc.X_FLAT,  # mhc_pre (FFN) reads s1
        post_x=_mpost.X_STACKED if st else _mpost.X_FLAT,  # both mhc_post read the residual
        s1_out=_mpost.OUT_STACKED if st else _mpost.OUT_FLAT,
    )


class MkLayer:
    """The dense_kda layer (attention + FFN sublayers) in ONE program; see the module docstring."""

    PROBE_ROWS = 256  # owner BRISC k -> k, root NCRISC 64, owner NCRISC k -> 96 + k, readers -> 160 + 2 i (+1)

    def __init__(self, device, a_args: dict, f_args: dict, kda, lins: dict, *, intermediate: int, limit: float,
                 probe: bool = False):  # fmt: skip
        from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre
        from models.autoports.zai_org_glm_5_3_flash.tt.kda_rec import KdaRec

        self.device = device
        # ring budgets (KB): the qkv / gate_up ring (DRAM-bound phases: prefetch during P1 and P6 + P7) and the o_proj /
        # down ring (compute-bound on their readers: shallow suffices); None = mk_readers' default (GLM53_MK_RING_KB)
        env_kb = lambda k: int(os.environ[k]) if os.environ.get(k) else None
        kb_big, kb_small = env_kb("GLM53_MK_RING_KB_QKV"), env_kb("GLM53_MK_RING_KB_O")
        kq, ko = int(os.environ.get("GLM53_MK_KB_QKV", "2")), int(os.environ.get("GLM53_MK_KB_O", "8"))  # K blocks
        # Stage 14 knob GLM53_MK_DS_FIDELITY (LoFi / HiFi2 / ...): the bf8 weight phases' math fidelity (qkv, o_proj,
        # gate_up, down; default the DSLinears', HiFi2) — at TP8 qkv / gate_up are reader-compute-bound
        dsf = (
            getattr(ttnn.MathFidelity, os.environ["GLM53_MK_DS_FIDELITY"])
            if os.environ.get("GLM53_MK_DS_FIDELITY")
            else None
        )
        phases = [
            MMPhase(lins["kda_qkv"], kb=kq, wait=(SEM_HN, 64), signal=SEM_QKV, ring_kb=kb_big, math_fidelity=dsf),
            # Stage 14 knob GLM53_MK_LR_FIDELITY (HiFi2 / HiFi3 / LoFi): the low-rank phase's math fidelity (default: the
            # DSLinear's, HiFi4 parity) — at TP8 the phase is K-bound per reader (128 K tiles x 1 N tile)
            MMPhase(
                lins["kda_lowrank"],
                kb=8,
                signal=SEM_QKV,
                math_fidelity=getattr(ttnn.MathFidelity, os.environ["GLM53_MK_LR_FIDELITY"])
                if os.environ.get("GLM53_MK_LR_FIDELITY")
                else None,
            ),
            # the o_proj input arrives from the kda_rec head cores: one per head of the layer (TP shard: kda.h)
            MMPhase(
                lins["kda_o"],
                kb=ko,
                wait=(
                    SEM_ON,
                    int(kda.h) * int(getattr(kda, "kda_rec_spread", 1)) * int(getattr(kda, "kda_rec_ugroups", 1)),
                ),  # Stage 14: kda_rec user groups
                signal=SEM_A,
                ring_kb=kb_small,
                math_fidelity=dsf,
            ),
            MMPhase(lins["mlp_gate_up"], kb=kq, wait=(SEM_HN, 64), signal=SEM_GU, ring_kb=kb_big, math_fidelity=dsf),
            MMPhase(lins["mlp_down"], kb=ko, wait=(SEM_M, 64), signal=SEM_A, ring_kb=kb_small, math_fidelity=dsf),
        ]
        self.ring_slots = None  # set after ReaderPhases
        # GLM53_MK_NO_DRAM=1 (timing only, garbage outputs): the weight streamer skips its DRAM reads -> the stack's span
        # with every weight already in L1 = the upper bound of any weight-prefetch scheme (ring_across_layers.md)
        self.readers = ReaderPhases(device, phases, no_dram=os.environ.get("GLM53_MK_NO_DRAM") == "1")
        self.ring_slots = dict(self.readers.ring_slots)
        self.place = rect_placement(device, {(int(c.x), int(c.y)) for c, _, _ in self.readers.assign})
        self.mhc_a = _mhc.MhcPre(device, **a_args, placement=self.place)
        self.mhc_f = _mhc.MhcPre(device, **f_args, placement=self.place)
        # ROWS mode (packed user rows, the conv-history shift in the reader) up to 32 rows
        self.kda = KdaPre(kda, rows_max=TILE, cores=self.place["k"], cb_base=KDA_CB_BASE)
        # kda_rec depth (GLM53_REC_DEPTH, read per construction): 1 (default) = lean (single-buffered per-item inputs
        # and state); 2 (Stage 14) = inputs and state double-buffered at every t (+148 KB on the owners: clashes with the
        # layer's L1 tensors at t = 32 on one chip); s (7a) = lean inputs + a second state slot at t > 1 (+64 KB): the
        # next item's 64 KB state streams in while one item computes (lean serialises every state read with compute)
        self._rec_mode = os.environ.get("GLM53_REC_DEPTH", "1")
        assert self._rec_mode in ("1", "2", "s"), self._rec_mode
        self.rec = KdaRec(kda, cores=self.place["k"], cb_base=REC_CB_BASE, cb_x0=REC_CB_X0, lean=self._rec_mode != "2")
        self._rec_kw = dict(cores=self.place["k"], cb_base=REC_CB_BASE, cb_x0=REC_CB_X0)
        self._rec_deep = None
        self.post = _mpost.MhcPost(device, cores=self.place["k"], cb_base=POST_CB_BASE)
        self.owner_noc = [
            (int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c) for c in self.place["k"])
        ]
        self.i_tiles = intermediate // TILE
        n_own = len(self.place["k"])
        # m column tiles per owner; a TP shard with fewer tiles than owners (TP8: I = 1536, 48 tiles): owner k takes tile
        # k mod I_tiles (the extra owners rewrite identical bytes; every owner still signals the down readers)
        assert self.i_tiles % n_own == 0 or self.i_tiles < n_own, (self.i_tiles, n_own)
        self.nt = max(1, self.i_tiles // n_own)
        self.intermediate, self.limit = intermediate, limit
        self.persist = {}  # buffers kept across calls (allocated on the first, eager call: never inside a capture)
        self.probe = None
        self.park_plan = None
        if probe:  # per-phase wall-clock marks (kernels/mk/probe.hpp), decoded by decode_probe()
            import torch

            self.probe = ttnn.from_torch(torch.zeros(self.PROBE_ROWS, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)  # fmt: skip
        self.calls = 0

    def persistent_buffers(self, t: int):
        """the hand-off buffers kept across calls (Stage 14b's stack shares one set between its layers)"""
        mk = lambda shape: ttnn.empty(shape, BF16, ttnn.TILE_LAYOUT, self.device, L1)  # noqa: E731
        self._persistent(("on", t), lambda: self._rec_for(t).output(t))
        self._persistent(("m", t), lambda: mk((1, 1, t, self.intermediate)))

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

    def __call__(self, x, hist, active, *, t: int, state, stack: bool = False):
        """x: the layer input [1,1,t,16384] fp32 (flat); hist = state.C, state = state.S (both updated in place);
        active = the decoder's masks -> dict(s1, out, hn, hn2, qkv, a, gu, m, y): out = the layer output
        [1,1,t,16384] fp32 DRAM.
        stack=True (Stage 14b, tt/mk_stack.py): x = the persistent residual res — t <= 4: STACKED [t,1,4,4096] fp32,
        t > 4: flat [1,1,t,16384] fp32 — P1 reads it (no residual copy), P6 takes it as the attention residual, P10
        overwrites it with the layer output -> out = res, the next layer's input in place (every owner reads / writes
        only its own column tiles {2k, 2k+1} of every stream, in phase order)."""
        rm = residual_modes(t, stacked_max=int(os.environ.get("GLM53_MK_STACKED_MAX", _mpost.PACKED_ROWS_MAX)))
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
        # Stage 14 glue trim (GLM53_MK_DEFER_INV=1): both mhc_pre instances publish hn = h * w before their round 2;
        # the root's 1/rms(h) goes to invh and the readers scale qkv / low-rank (attention) and gate_up (FFN) outputs
        defer = os.environ.get("GLM53_MK_DEFER_INV") == "1"
        invh_a = self._persistent(("invh_a",), lambda: mk((1, 1, 32, 32), FP32)) if defer else None
        invh_f = self._persistent(("invh_f",), lambda: mk((1, 1, 32, 32), FP32)) if defer else None
        rd.scale = dict(groups=[((0, 1), invh_a), ((3,), invh_f)], sem=SEM_INVH) if defer else None
        roles_a, mhc_cbs, mhc_sems = ma.parts(x, hn, post_a, comb_a, x_out, coef_a, t=t,
                                              x_mode=rm["s1_x"] if stack else _mhc.X_FLAT,
                                              xout_mode=_mhc.XOUT_NONE if stack else rm["xout"], pc_mode=rm["pc"],
                                              signal=sig,
                                              mk_probe=(pa, 0) if pa is not None else None, coef_plane=True,
                                              defer=(invh_a, SEM_INVH, rd.noc_coords()) if defer else None)  # fmt: skip
        # narrow widths (stack_model) at t > 1: t = 1's s1 is the pool's prefix view of the flat full-width s1 (no L1 of
        # its own; taking it out of the pool hands that base to another request -> layer kernels differ, pair mock 201)
        if stack and rm["stacked"] and t > 1 and os.environ.get("GLM53_MK_S1_DRAM") == "1":
            s1 = _narrow_s1_dram(dev, t)
        else:
            s1 = self.post.output(t, rm["s1_out"])
        roles_f, mhc_cbs_f, _ = mf.parts(s1, hn2, post_f, comb_f, None, coef_f, t=t, x_mode=rm["s1_x"],
                                         xout_mode=_mhc.XOUT_NONE, pc_mode=rm["pc"], signal=sig, coef_plane=True,
                                         defer=(invh_f, SEM_INVH, rd.noc_coords()) if defer else None)  # fmt: skip
        self._same_cbs(mhc_cbs, mhc_cbs_f, "mhc_pre")
        outs = rd.outputs(t)
        qkv, lr, a, gu, y = outs
        rec = self._rec_for(t)
        on = self._persistent(("on", t), lambda: rec.output(t))  # zeroed once (whole-tile writes after)
        m = self._persistent(("m", t), lambda: mk((1, 1, t, self.intermediate), BF16))
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
        # P6 / P10 mhc_post (packed at t <= 4; the same CB set)
        proles_a, post_cbs, post_io_a = self.post.parts(a, x_out if x_out is not None else x, coef_a, s1, t=t,
                                                        x_mode=rm["post_x"], out_mode=rm["s1_out"], wait=(SEM_A, nr),
                                                        packed=rm["stacked"], fused_v2=True)  # fmt: skip
        out = x if stack else mk((1, 1, t, 4 * ma.hidden), FP32, DRAM)
        proles_f, post_cbs_f, post_io_f = self.post.parts(y, s1, coef_f, out, t=t, x_mode=rm["post_x"],
                                                          out_mode=rm["s1_out"] if stack else _mpost.OUT_FLAT,
                                                          wait=(SEM_A, nr),
                                                          packed=rm["stacked"], fused_v2=True)  # fmt: skip
        self._same_cbs(post_cbs, post_cbs_f, "mhc_post")
        for role in list(proles_a.values()) + list(proles_f.values()) if rm["stacked"] else ():
            role[2].append(("MK_POST_PACKED", "1"))
        # P8b swiglu (owner k: m column tiles nt k ..)
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        rtd = lambda fn: {(int(c.x), int(c.y)): [int(v) for v in fn(k)] for k, c in enumerate(self.place["k"])}
        sdef = [("SWI_CB0", str(SWI_CB0))]
        flat_r = [v for xy_ in rd.noc_coords() for v in xy_]
        cc_swi = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                              fp32_dest_acc_en=True)  # fmt: skip
        swi = dict(
            reader=([self.nt, self.i_tiles] + acc(gu), rtd(lambda k: [gu.buffer_address(), (self.nt * k) % self.i_tiles, nr]),
                    sdef + [("MK_SWI_WAIT_SEM", str(SEM_GU))], None),
            compute=([self.nt, f32_bits(self.limit), f32_bits(-self.limit)], rtd(lambda k: []), sdef, cc_swi),
            writer=([self.nt] + acc(m), rtd(lambda k: [m.buffer_address(), (self.nt * k) % self.i_tiles, nr] + flat_r),
                    sdef + [("MK_SWI_SIGNAL_SEM", str(SEM_M))], None),
        )  # fmt: skip
        swi_cbs = [ttnn.CBDescriptor(total_size=2048 * n, core_ranges=ma.kcores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=SWI_CB0 + i, data_format=BF16, page_size=2048)])
            for i, n in ((0, self.nt), (1, self.nt), (2, 2))]  # fmt: skip

        def p7(name, lead):
            """the FFN mhc_pre role as a phase tuple (ct, rt {core: args}, defines, config)"""
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
            return ct, rt, defs + extra

        mkk = lambda src, cores, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "mk", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=_rt_args(rt), defines=defs, config=cfg,
        )  # fmt: skip
        # forwarding ring (GLM53_MK_PARK=1): park slots in every owner's L1 between its static CB region and the
        # readers' (the larger, which already bounds every L1 tensor); the readers park ring blocks there when their
        # ring is full and pull them back in stream order (kernels/mk/reader_w.cpp stream_phase_park)
        park_cbs, park_sems, park_defs = [], [], []
        rd.park = None
        if os.environ.get("GLM53_MK_PARK") == "1":
            on_owner = lambda cb: any(
                r.start.x <= self.place["k"][0].x <= r.end.x and r.start.y <= self.place["k"][0].y <= r.end.y
                for r in cb.core_ranges.ranges()
            )
            owner_bytes = sum(int(c.total_size) for c in mhc_cbs + kda_cbs + rec_cbs + post_cbs + swi_cbs
                              if on_owner(c)) + (128 if pa is not None else 0)  # fmt: skip
            ring43 = max(rd.rings, key=lambda cb: rd.rings[cb][1] * TILE_BYTES_[rd.rings[cb][0]])
            block_bytes = rd.rings[ring43][1] * TILE_BYTES_[rd.rings[ring43][0]]
            reader_bytes = rd.l1_bytes() + block_bytes + (128 if pa is not None else 0)
            slots = (reader_bytes - owner_bytes - 16 * 1024) // block_bytes
            slots = min(slots, int(os.environ.get("GLM53_MK_PARK_SLOTS", slots)))
            phases_ = [int(v) for v in os.environ.get("GLM53_MK_PARK_PHASES", "0,3").split(",")]
            mask = sum(1 << i for i in phases_ if rd.geo[i]["ring"] == ring43)
            if slots > 0 and mask:
                rd.park = dict(owners=self.owner_noc, slots=slots, sem=SEM_PARK, mask=mask, block_bytes=block_bytes)
                park_cbs = [ttnn.CBDescriptor(total_size=slots * block_bytes, core_ranges=ma.kcores, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=PARK_CB, data_format=ttnn.uint32, page_size=block_bytes)])]  # fmt: skip
                park_sems = [ttnn.SemaphoreDescriptor(id=SEM_PARK, core_ranges=ma.kcores, initial_value=0)]
                park_defs = [("MK_PARK_SEM", str(SEM_PARK)), ("MK_PARK_CB", str(PARK_CB))]
            self.park_plan = dict(slots=slots, mask=mask, owner_bytes=owner_bytes, reader_bytes=reader_bytes,
                                  block_bytes=block_bytes)  # fmt: skip
        s1def = (
            [("MK_S1_SEM", str(SEM_S1))]
            + ([("MK_PRE_PREFETCH", "1")] if os.environ.get("GLM53_MK_PRE_PREFETCH", "1") == "1" else [])
            + park_defs
        )
        kernels = []
        ks = _role(roles_a, "stream.cpp", "MHC_K")
        ct, rt, defs = combine(ks, "reader", None,
                               s1def + [("MK_SEM_QKV", str(SEM_QKV)), ("MK_QKV_COUNT", str(2 * nr))]
                               # Stage 14 GLM53_KDA_PREFETCH=1: kda_pre / kda_rec request their step-independent inputs
                               # before their hand-off waits (the owner skips its qkv wait; kda_pre waits itself)
                               + ([("KDA_PREFETCH", "1")] if os.environ.get("GLM53_KDA_PREFETCH") == "1" else []))  # fmt: skip
        kernels.append(mkk("owner_dm0.cpp", ks["cores"], ct, rt, defs, ks["config"]))
        for lead in (True, False):
            sr = _role(roles_a, "sync.cpp", "MHC_LEADER") if lead else _role(roles_a, "sync.cpp", "MHC_K", "MHC_LEADER")
            ct, rt, defs = combine(sr, "writer", lead, s1def)
            if pa is not None:  # the owner NCRISC's probe args after every phase's: slot 96 + k
                kidx = {(int(c.x), int(c.y)): k for k, c in enumerate(self.place["k"])}
                n_all = len(next(iter(rt.values())))
                rt = {core: v + [pa, 96 + kidx[core]] for core, v in rt.items()}
                defs = defs + [("MK_PROBE_ARG", str(n_all)), ("MK_PROBE_CB", "31")]
            kernels.append(mkk("owner_dm1.cpp", sr["cores"], ct, rt, defs, sr["config"]))
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
            kernels.append(mkk("owner_c.cpp", cr["cores"], ct, rt, defs, cc))
        # the root: mhc_pre's root roles of both instances, composed (root_*.cpp)
        for src, comp in (("stream.cpp", "root_dm0.cpp"), ("sync.cpp", "root_dm1.cpp"), ("compute.cpp", "root_c.cpp")):
            ra, rf = _role(roles_a, src, "MHC_ROOT"), _role(roles_f, src, "MHC_ROOT")
            core = (int(self.place["root"].x), int(self.place["root"].y))
            va, vf = list(ra["rt"].get(core, [])), list(rf["rt"].get(core, []))
            defs = list(ra["defines"]) + [d for d in rf["defines"] if d not in ra["defines"]]
            defs += [("MK_P7_CT", str(len(ra["ct"]))), ("MK_P7_RT", str(len(va)))]
            kernels.append(mkk(comp, ra["cores"], list(ra["ct"]) + list(rf["ct"]), {core: list(va) + list(vf)}, defs,
                               ra["config"]))  # fmt: skip
        r_kernels, r_cbs = rd.parts([hn, hn, on, hn2, m], outs, consumers=self.owner_noc,
                                    mk_probe=(pa, 160) if pa is not None else None)  # fmt: skip
        kernels += r_kernels
        sems = (
            mhc_sems
            + [
                ttnn.SemaphoreDescriptor(id=SEM_HN, core_ranges=rd.cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_ON, core_ranges=rd.cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_M, core_ranges=rd.cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_QKV, core_ranges=ma.kcores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_KDA, core_ranges=ma.kcores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_A, core_ranges=ma.kcores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_S1, core_ranges=ma.kcores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_GU, core_ranges=ma.kcores, initial_value=0),
            ]
            + park_sems
            + ([ttnn.SemaphoreDescriptor(id=SEM_INVH, core_ranges=rd.cores, initial_value=0)] if defer else [])
        )
        io = [x, ma.fn, ma.w, ma.sb, ma.dbg, post_a, comb_a, coef_a] + ([x_out] if x_out is not None else []) + [hn]
        io += [invh_a, invh_f] if defer else []
        io += [mf.fn, mf.w, mf.sb, mf.dbg, post_f, comb_f, coef_f, hn2, s1, m, out]
        io += [p.ds.weight for p in rd.phases] + list(outs)
        io += [t_ for t_ in kda_io + rec_io + post_io_a + post_io_f if all(t_ is not u for u in io)]
        probe_cbs = []
        if pa is not None:
            io.append(self.probe)
            probe_cbs = [ttnn.CBDescriptor(total_size=128, core_ranges=ma.all, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=128)])]  # fmt: skip
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems,
                                                   cbs=mhc_cbs + kda_cbs + rec_cbs + post_cbs + swi_cbs + r_cbs
                                                   + probe_cbs + park_cbs))  # fmt: skip
        self.calls += 1
        return dict(
            s1=s1,
            out=out,
            hn=hn,
            hn2=hn2,
            qkv=qkv,
            lr=lr,
            a=a,
            gu=gu,
            m=m,
            y=y,
            on=on,
            kda=kda_outs,
            x_out=x_out,
            coef_a=coef_a,
            coef_f=coef_f,
        )

    @staticmethod
    def _same_cbs(a, b, what):
        key = lambda cbs: sorted((int(c.total_size), tuple(int(f.buffer_index) for f in c.format_descriptors))
                                 for c in cbs)  # fmt: skip
        if key(a) != key(b):
            raise ValueError(f"MkLayer: the two {what} instances need different CB sets; they share indices")


def decode_probe(probe_host, assign, *, k_cores=64, mhz=1350.0) -> dict:
    """MkLayer's probe tensor [256, 16] (host) -> marks in us after the earliest mark, median over the cores of a role:
    owner BRISC (mhc_pre stream 3 marks, qkv landed, kda_pre inputs, kda_rec inputs, mhc_post inputs, FFN mhc_pre
    stream 3, FFN mhc_pre streamed, swiglu inputs, FFN mhc_post inputs); owner NCRISC per mhc_pre group role (leaders /
    members: their sync marks, then mhc_pre done, kda_pre written, kda_rec written, mhc_post written, FFN sync marks,
    FFN mhc_pre done, m written, layer output written); root NCRISC; readers per index in bank (the weight streamer:
    start + one mark per phase it streams; the activation mover: start + (after wait, after signal) per phase)."""
    import statistics

    p = probe_host.reshape(-1, 16).long() & 0xFFFFFFFF
    nz = p[p > 0]
    t0 = int(nz.min()) if nz.numel() else 0
    us = lambda v: round((int(v) - t0) / mhz, 1)

    def med(rows):
        cols = [[int(v) for v in p[r] if int(v) > 0] for r in rows]
        n = min((len(c) for c in cols), default=0)
        return [us(statistics.median(c[i] for c in cols)) for i in range(n)]

    out = dict(owner_brisc=med(range(k_cores)), root_ncrisc=med([64]),
               owner_ncrisc_leaders=med([96 + k for k in range(k_cores) if k % 8 == 0]),
               owner_ncrisc_members=med([96 + k for k in range(k_cores) if k % 8]))  # fmt: skip
    for idx in sorted({i for _, _, i in assign}):
        rows = [n for n, (_, _, i) in enumerate(assign) if i == idx]
        out[f"reader{idx}_w"] = med([160 + 2 * n for n in rows])
        out[f"reader{idx}_a"] = med([161 + 2 * n for n in rows])
    return out
