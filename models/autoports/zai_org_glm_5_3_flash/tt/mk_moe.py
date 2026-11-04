# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel (M4): the MoE FFN half of a sparse layer as ONE program (the MkFFN analogue for MoE layers):

    owners (64) + root            readers (3 per DRAM bank)
    P1  mhc_pre (FFN) s1 -> hn2  -- SEM_HN -->  P2  router (DS, fp32 logits)
    P5  router tail, owner k = token row k (k < T)  <-- SEM_LOGITS --
                                                 P3  shared gate_up (DS)  -- SEM_GU -->  P8 swiglu (owners) -- SEM_M --+
    P5 -- SEM_TAIL -->  PX  routed experts (mkx: routing union, expert stream by address, m all-gather) -- SEM_Y --+
                                                 P4  shared down (DS)  <-- SEM_M  ... -- SEM_Y --+
    P10 mhc_post: routed + shared + C^T s1 -> the layer output [1,1,T,16384] fp32 DRAM  <-- SEM_Y (2 x readers) --+

The readers run one composite kernel per RISC (kernels/mk/reader_moe_*.cpp): the dense reader phases (router, shared
gate_up | shared down) with the routed-expert phase between them (MK_SPLIT = 2); the owners run kernels/mk/
owner_moe_*.cpp (P1, P5, P8, P10), the root mhc_pre's root kernels. Expert parallelism: e_lo / num_experts as in
ReaderExperts (the routed output is then this chip's partial sum). t <= 32 rows (mk_layer.residual_modes: the stacked
residual + packed mhc_post at t <= 4, flat above)."""

from __future__ import annotations

import os
import struct

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as _mpost
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as _mhc
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import _cores_of, _role, _rt_args, rect_placement
from models.autoports.zai_org_glm_5_3_flash.tt.mk_experts import ReaderExperts
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases

BF16, FP32 = ttnn.bfloat16, ttnn.float32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
TILE = 32
# semaphores: owners + root 0..3 (mhc_pre); readers HN / M / TAIL (+ ReaderExperts' 10..14); owners LOGITS / GU / Y
SEM_HN, SEM_LOGITS, SEM_GU, SEM_Y, SEM_M, SEM_TAIL = 4, 5, 6, 7, 8, 9
SWI_CB0 = 18  # swiglu's CBs on the owners: 18..20 (mhc_pre 0..13, mhc_post 21..28, router tail 32..36)
POST_CB0 = 21
RTAIL_CB0 = 32
X_CB0 = 48  # the routed experts' reader CBs 48..58 (the dense reader phases: 31, 38..46)
SKIP = 0xFFFFFFFF
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class MkMoE:
    def __init__(self, device, mhc_args: dict, mlp, *, e_lo: int = 0, num_experts: int | None = None,
                 expert_ring_slots: int | None = None, xtp: int | None = None,
                 local_experts: list | None = None, replica_home: list | None = None,
                 chip: int = 0, xsl: int | None = None, xsl_tp: int = 8, xsl_sliced: bool = False):  # fmt: skip
        """mlp: the decoder layer's OptimizedMoE (router DSLinear + its RouterTail bias, the shared expert's DSLinears,
        the routed experts' weight lists); e_lo / num_experts: expert parallelism (this chip's experts are mlp.gate_up /
        mlp.down); xtp = c: expert TP slice c of every expert (mk_experts.ReaderExperts; its semaphore SEM_XTP = 15 is
        free on these readers); xsl = c: batched expert TP slice c (MKX_XSL); local_experts: a load-aware EP placement
        (this chip's expert ids in slot order, mk_experts.ReaderExperts)."""
        self.device = device
        router, shared = mlp.router, mlp.shared
        if expert_ring_slots is None:  # read per construction (a long-lived process keeps the module imported)
            expert_ring_slots = int(os.environ.get("GLM53_MKX_RING", "6"))
        # 3 readers per bank (GLM53_MK_READERS overrides): the routed experts need >= 3 (ReaderExperts)
        self.readers = ReaderPhases(device, [
            MMPhase(router.ds, kb=8, wait=(SEM_HN, 64), signal=SEM_LOGITS),
            MMPhase(shared.ds_gate_up, kb=2, signal=SEM_GU),
            MMPhase(shared.ds_down, kb=8, wait=(SEM_M, 64), signal=SEM_Y),
        ], readers_per_bank=int(os.environ.get("GLM53_MK_READERS") or 3))  # fmt: skip
        self.experts = ReaderExperts(device, mlp.gate_up, mlp.down, top_k=mlp.top_k, scaling=mlp.scaling,
                                     limit=mlp.limit, intermediate=mlp.intermediate, hidden=mlp.hidden,
                                     readers_per_bank=self.readers.readers, ring_slots=expert_ring_slots, e_lo=e_lo,
                                     num_experts=num_experts if num_experts is not None else mlp.num_experts,
                                     cb0=X_CB0, xtp=xtp, local_experts=local_experts,
                                     replica_home=replica_home, chip=chip, xsl=xsl, xsl_tp=xsl_tp,
                                     xsl_sliced=xsl_sliced)  # fmt: skip
        ra = [(int(c.x), int(c.y)) for c, _, _ in self.readers.assign]
        xa = [(int(c.x), int(c.y)) for c, _, _ in self.experts.assign]
        assert ra == xa, "the dense reader phases and the expert phases must run on the same reader cores, same order"
        self.place = rect_placement(device, set(ra))
        self.mhc = _mhc.MhcPre(device, **mhc_args, placement=self.place)
        self.post = _mpost.MhcPost(device, cores=self.place["k"], cb_base=POST_CB0)
        self.owner_noc = [
            (int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c) for c in self.place["k"])
        ]
        self.tail_bias = router.tail.bias  # the RouterTail's packed bias row (fp32, DRAM)
        self.shared_i = shared.intermediate
        self.i_tiles = self.shared_i // TILE
        assert self.i_tiles % len(self.place["k"]) == 0
        self.nt = self.i_tiles // len(self.place["k"])
        self.limit_shared = shared.limit
        self.persist = {}
        self.calls = 0

    def _persistent(self, key, make):
        if key not in self.persist:
            self.persist[key] = make()
        return self.persist[key]

    def __call__(self, s1, *, t: int, out=None, live=None):
        """s1: the residual after the attention sublayer, fp32 — stacked [t,1,4,4096] at t <= 4, flat [1,1,t,16384]
        above (mk_layer.residual_modes; t <= 32) -> dict(out = the layer output [1,1,t,16384] fp32 DRAM (or `out`:
        Stage 14b's persistent residual, written in the residual layout of t — stacked at t <= 4), hn2,
        logits, ids, scores, y (routed), y2 (shared), m (shared intermediate)). live: the stack's position page
        (ReaderExperts.parts_raw MKX_LIVE: idle rows route to no expert)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import residual_modes

        rm = residual_modes(t)
        mhc, dev, rd = self.mhc, self.device, self.readers
        mk = lambda shape, dt, mem=L1: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, dev, mem)
        hn = mk((1, 1, t, mhc.hidden), BF16)
        post, comb = (mk(sh, FP32) for sh in rm["pc_shapes"])
        coef = mk((1, 1, 32, 32), FP32)
        roles, mhc_cbs, mhc_sems = mhc.parts(s1, hn, post, comb, None, coef, t=t, x_mode=rm["s1_x"],
                                             xout_mode=_mhc.XOUT_NONE, pc_mode=rm["pc"],
                                             signal=(SEM_HN, rd.noc_coords()))  # fmt: skip
        logits, gu, y2 = rd.outputs(t)
        # the shared expert's intermediate, persistent per TILE ROW count (not per t): every t <= 32 is the same one tile
        # row, so a server's widths (32, 8, 16, narrow 1..4) share ONE buffer per layer instead of each width's first
        # call allocating another L1 buffer below the programs already compiled (Stage 14b: 11 sparse_dsa layers x ~1 KB
        # per L1 bank per extra width; a later allocation overlaps captured programs' static CBs). Widths never run
        # concurrently and each step writes m before reading it.
        mrows = -(-t // TILE) * TILE
        m = self._persistent(("m", mrows), lambda: mk((1, 1, mrows, self.shared_i), BF16))
        ids = mk((1, 1, t, self.experts.top_k), ttnn.uint16)
        scores = mk((1, 1, t, self.experts.e), BF16)
        # GLM53_MKX_RW (default on): the router tail computes the rows' routing weights (one [t, 8] bf16 row each, the
        # same soft-float formula) instead of every expert reader normalising every row (86 us at T=32, A4 probe)
        rw = os.environ.get("GLM53_MKX_RW", "1") == "1"
        weights = mk((1, 1, t, self.experts.top_k), BF16) if rw else None
        y = mk((1, 1, t, mhc.hidden), BF16)
        out_mode = rm["s1_out"] if out is not None else _mpost.OUT_FLAT
        out = out if out is not None else mk((1, 1, t, 4 * mhc.hidden), FP32, DRAM)
        nr = len(rd.assign)
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        rtd = lambda fn: {(int(c.x), int(c.y)): [int(v) for v in fn(k)] for k, c in enumerate(self.place["k"])}
        flat_r = [v for xy in rd.noc_coords() for v in xy]
        hifi4 = lambda: ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                     fp32_dest_acc_en=True)  # fmt: skip
        # P5 router tail: owner k = token row k
        tdef = [("RTAIL_CB0", str(RTAIL_CB0))]
        tail_cc = hifi4()
        tmodes = [ttnn.UnpackToDestMode.Default] * 64
        tmodes[RTAIL_CB0 + 0] = tmodes[RTAIL_CB0 + 1] = ttnn.UnpackToDestMode.UnpackToDestFp32
        tail_cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(tmodes)
        row = lambda k: k if k < t else SKIP
        tail = dict(
            reader=(acc(logits) + acc(self.tail_bias),
                    rtd(lambda k: [logits.buffer_address(), self.tail_bias.buffer_address(), row(k), nr]),
                    tdef + [("MK_RTAIL_WAIT_SEM", str(SEM_LOGITS))], None),
            compute=([], rtd(lambda k: [row(k)]), tdef, tail_cc),
            writer=(acc(ids) + acc(scores) + (acc(weights) + self.experts.rw_ct() if rw else []),
                    rtd(lambda k: [ids.buffer_address(), scores.buffer_address(), row(k), nr] + flat_r
                        + ([weights.buffer_address()] if rw else [])),
                    tdef + [("MK_RTAIL_SIGNAL_SEM", str(SEM_TAIL))]
                    + ([("RTAIL_WEIGHTS", "1"), ("RTAIL_W_RT", str(4 + len(flat_r)))] if rw else []), None),
        )  # fmt: skip
        tcb = lambda i, dt, page: ttnn.CBDescriptor(total_size=page, core_ranges=mhc.kcores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=RTAIL_CB0 + i, data_format=dt, page_size=page)])  # fmt: skip
        tail_cbs = [tcb(0, FP32, 4096), tcb(1, FP32, 4096), tcb(2, FP32, 4096), tcb(3, BF16, 2048), tcb(4, BF16, 64)]
        # P8 swiglu (shared expert): owner k -> m column tiles nt k ..
        sdef = [("SWI_CB0", str(SWI_CB0))]
        swi = dict(
            reader=([self.nt, self.i_tiles] + acc(gu), rtd(lambda k: [gu.buffer_address(), self.nt * k, nr]),
                    sdef + [("MK_SWI_WAIT_SEM", str(SEM_GU))], None),
            compute=([self.nt, f32_bits(self.limit_shared), f32_bits(-self.limit_shared)], rtd(lambda k: []), sdef,
                     hifi4()),
            writer=([self.nt] + acc(m), rtd(lambda k: [m.buffer_address(), self.nt * k, nr] + flat_r),
                    sdef + [("MK_SWI_SIGNAL_SEM", str(SEM_M))], None),
        )  # fmt: skip
        swi_cbs = [ttnn.CBDescriptor(total_size=2048 * n, core_ranges=mhc.kcores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=SWI_CB0 + i, data_format=BF16, page_size=2048)])
            for i, n in ((0, self.nt), (1, self.nt), (2, 2))]  # fmt: skip
        # P10 mhc_post: routed (y) + shared (y2) + C^T s1; waits for the routed readers and the shared-down readers
        proles, post_cbs, post_io = self.post.parts(y, s1, coef, out, t=t, x_mode=rm["post_x"],
                                                    out_mode=out_mode, y2=y2, wait=(SEM_Y, 2 * nr),
                                                    packed=rm["stacked"])  # fmt: skip
        for role in proles.values() if rm["stacked"] else ():
            role[2].append(("MK_POST_PACKED", "1"))
        phase_roles = ((5, tail), (8, swi), (10, proles))

        def combine(mrole, role_name):
            mine = _cores_of(mrole["cores"])
            ct, defs = list(mrole["ct"]), list(mrole["defines"])
            n_rt = max((len(v) for v in mrole["rt"].values()), default=0)
            rt = {core: list(mrole["rt"].get(core, [])) + [0] * (n_rt - len(mrole["rt"].get(core, [])))
                  for core in mine}  # fmt: skip
            for n, roles_ in phase_roles:
                pct, prt, pdefs, _ = roles_[role_name]
                defs += [d for d in pdefs if d not in defs] + [
                    (f"MK_P{n}_CT", str(len(ct))),
                    (f"MK_P{n}_RT", str(n_rt)),
                ]
                ct += list(pct)
                width = max(len(prt[core]) for core in mine)
                for core in mine:
                    rt[core] += [int(a) for a in prt[core]] + [0] * (width - len(prt[core]))
                n_rt += width
            return ct, rt, defs

        mkk = lambda src, cores, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "mk", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=_rt_args(rt), defines=defs, config=cfg,
        )  # fmt: skip
        kernels = []
        ks = _role(roles, "stream.cpp", "MHC_K")
        ct, rt, defs = combine(ks, "reader")
        kernels.append(mkk("owner_moe_dm0.cpp", ks["cores"], ct, rt, defs, ks["config"]))
        for lead in (True, False):
            sr = _role(roles, "sync.cpp", "MHC_LEADER") if lead else _role(roles, "sync.cpp", "MHC_K", "MHC_LEADER")
            ct, rt, defs = combine(sr, "writer")
            kernels.append(mkk("owner_moe_dm1.cpp", sr["cores"], ct, rt, defs, sr["config"]))
            cr = (_role(roles, "compute.cpp", "MHC_LEADER") if lead
                  else _role(roles, "compute.cpp", "MHC_K", "MHC_LEADER"))  # fmt: skip
            ct, rt, defs = combine(cr, "compute")
            cc = hifi4()
            modes = list(cr["config"].unpack_to_dest_mode)
            for kc in (proles["compute"][3], tail_cc):
                kmodes = list(kc.unpack_to_dest_mode)
                modes = [a if a != ttnn.UnpackToDestMode.Default else b for a, b in zip(modes, kmodes)]
            cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
            kernels.append(mkk("owner_moe_c.cpp", cr["cores"], ct, rt, defs, cc))
        for r in roles:  # the root's three kernels, unchanged
            if ("MHC_ROOT", "1") in r["defines"]:
                kernels.append(ttnn.KernelDescriptor(
                    kernel_source=os.path.join(KDIR, "mhc_pre", r["name"]),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=r["cores"],
                    compile_time_args=r["ct"], runtime_args=_rt_args(r["rt"]), defines=r["defines"],
                    config=r["config"]))  # fmt: skip
        # readers: the dense phases (router, shared gate_up | shared down) + the routed experts between them
        draw, r_cbs = rd.parts_raw([hn, hn, m], [logits, gu, y2], consumers=self.owner_noc)
        xraw, x_cbs, x_sems, x_io = self.experts.parts_raw(hn, ids, scores, y, t=t, wait=(SEM_TAIL, t), weights=weights,
                                                           signal=(SEM_Y, self.owner_noc), live=live)  # fmt: skip
        for r, src in (("w", "reader_moe_w.cpp"), ("a", "reader_moe_a.cpp"), ("c", "reader_moe_c.cpp")):
            _, dct, drt, ddefs, dcfg = draw[r]
            _, xct, xrt, xdefs, xcfg = xraw[r]
            n_drt = max(len(v) for v in drt.values())
            assert all(len(v) == n_drt for v in drt.values()), "uniform dense runtime args"
            rt_ = {core: list(drt[core]) + list(xrt[core]) for core in drt}
            defs_ = list(ddefs) + [d for d in xdefs if d not in ddefs]
            defs_ += [("MK_SPLIT", "2"), ("MK_X_CT", str(len(dct))), ("MK_X_RT", str(n_drt))]
            cfg = dcfg
            if r == "c":  # the experts' fidelity is the kernel-wide one (the dense phases pass theirs per phase)
                cfg = ttnn.ComputeConfigDescriptor(math_fidelity=xcfg.math_fidelity, math_approx_mode=False,
                                                   fp32_dest_acc_en=True)  # fmt: skip
                cfg.unpack_to_dest_mode = self.experts.unpack_modes(dcfg.unpack_to_dest_mode)
            kernels.append(rd.kernel(src, list(dct) + list(xct), rt_, defs_, cfg))
        sems = (
            mhc_sems
            + x_sems
            + [
                ttnn.SemaphoreDescriptor(id=SEM_HN, core_ranges=rd.cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_M, core_ranges=rd.cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_TAIL, core_ranges=rd.cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_LOGITS, core_ranges=mhc.kcores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_GU, core_ranges=mhc.kcores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_Y, core_ranges=mhc.kcores, initial_value=0),
            ]
        )
        io = [s1, mhc.fn, mhc.w, mhc.sb, mhc.dbg, post, comb, coef, hn, logits, gu, y2, m, ids, scores, y, out,
              self.tail_bias] + [p.ds.weight for p in rd.phases]  # fmt: skip
        io += [t_ for t_ in x_io + post_io if all(t_ is not u for u in io)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems,
                                                   cbs=mhc_cbs + tail_cbs + swi_cbs + post_cbs + r_cbs + x_cbs))  # fmt: skip
        self.calls += 1
        return dict(out=out, hn2=hn, logits=logits, ids=ids, scores=scores, y=y, y2=y2, m=m, gu=gu, coef=coef)
