# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel, FFN half of the dense_kda layer as ONE program (M1 builds the layer from this + mk_dense's
step D attention half; doc/ttnn/07a-fused-streaming-layer/megakernel_plan.md):

    P7  mhc_pre (FFN sublayer) on the 64 owners + root: s1 (the attention half's stacked residual) -> hn2
    P8  gate_up on the 16 bank-adjacent DS readers (waits for hn2: SEM_HN2)               -> gu [B, 2I]
    P8b swiglu on the owners: m = SiLU(min(gate, 10)) * clamp(up, +-10) (waits for gu: SEM_GU) -> m [B, I]
    P9  down on the readers (waits for m: SEM_M)                                           -> y [B, 4096]
    P10 mhc_post on the owners (waits for y: SEM_Y): post * y + C^T s1 -> the layer output [1,1,B,16384] fp32 DRAM

Hand-offs are point-to-point semaphores (every producer increments every consumer after its write barrier); no
dispatch between the phases. Owner phases are composed into one kernel per RISC (kernels/mk/owner_ffn_*.cpp,
PH_CT0 / ph_rt0 argument bases); the root runs mhc_pre's root kernels unchanged; the readers run kernels/mk/reader_*.
B <= 4 (mhc_pre's stacked residual layout)."""

from __future__ import annotations

import os
import struct

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as _mpost
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as _mhc
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import _cores_of, _role, _rt_args, rect_placement
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases

BF16, FP32 = ttnn.bfloat16, ttnn.float32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
TILE = 32
# semaphores: mhc_pre's 0..3 on owners + root; readers HN2 / M; owners GU / Y
SEM_HN2, SEM_M, SEM_GU, SEM_Y = 4, 5, 6, 7
SWI_CB0 = 21  # swiglu's CBs on the owners: 21..23 (mhc_pre 0..13, root-only 14..20)
POST_CB0 = 32  # mhc_post's CBs on the owners: 32..39
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class MkFFN:
    """P7 mhc_pre (FFN) -> P8 gate_up -> P8b swiglu -> P9 down -> P10 mhc_post in ONE program."""

    def __init__(self, device, mhc_args: dict, ds_gate_up, ds_down, *, intermediate: int, limit: float):
        self.device = device
        self.readers = ReaderPhases(device, [MMPhase(ds_gate_up, kb=2, wait=(SEM_HN2, 64), signal=SEM_GU),
                                             MMPhase(ds_down, kb=8, wait=(SEM_M, 64), signal=SEM_Y)])  # fmt: skip
        self.place = rect_placement(device, {(int(c.x), int(c.y)) for c, _, _ in self.readers.assign})
        self.mhc = _mhc.MhcPre(device, **mhc_args, placement=self.place)
        self.post = _mpost.MhcPost(device, cores=self.place["k"], cb_base=POST_CB0)
        self.owner_noc = [
            (int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c) for c in self.place["k"])
        ]
        self.i_tiles = intermediate // TILE
        n_own = len(self.place["k"])
        # m column tiles per owner (6 for I = 12288). A TP8 shard (I = 1536: 48 tiles) has fewer tiles than owners:
        # 1 tile per owner and owner k takes tile k mod 48 (owners 48..63 recompute tiles 0..15 and write the SAME
        # bytes: no race on the values, and every owner still signals the down readers, so the hand-off count stays 64)
        assert self.i_tiles % n_own == 0 or self.i_tiles < n_own, (self.i_tiles, n_own)
        self.nt = max(1, self.i_tiles // n_own)
        self.intermediate, self.limit = intermediate, limit
        self.persist = {}
        self.calls = 0

    def _persistent(self, key, make):
        if key not in self.persist:
            self.persist[key] = make()
        return self.persist[key]

    def __call__(self, s1, *, t: int):
        """s1: the stacked residual [t,1,4,4096] fp32 (the attention half's output) -> (MhcPreOutputs of P7, gu, m, y,
        the layer output [1,1,t,16384] fp32 DRAM)."""
        if not 1 <= t <= _mpost.PACKED_ROWS_MAX:
            raise ValueError(f"MkFFN: t={t} (the stacked residual layout serves t <= 4)")
        mhc, dev = self.mhc, self.device
        mk = lambda shape, dt, mem=L1: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, dev, mem)
        hn = mk((1, 1, t, mhc.hidden), BF16)
        post, comb = mk((t, 1, 4, 1), FP32), mk((t, 1, 4, 4), FP32)
        coef = mk((1, 1, 32, 32), FP32)
        roles, mhc_cbs, mhc_sems = mhc.parts(s1, hn, post, comb, None, coef, t=t, x_mode=_mhc.X_STACKED,
                                             xout_mode=_mhc.XOUT_NONE, pc_mode=_mhc.PC_STACKED,
                                             signal=(SEM_HN2, self.readers.noc_coords()))  # fmt: skip
        gu, y = self.readers.outputs(t)
        m = self._persistent(("m", t), lambda: mk((1, 1, t, self.intermediate), BF16))
        r_kernels, r_cbs = self.readers.parts([hn, m], [gu, y], consumers=self.owner_noc)
        out = mk((1, 1, t, 4 * mhc.hidden), FP32, DRAM)
        # P8b swiglu roles (owners k: m column tiles nt k ..)
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        rtd = lambda fn: {(int(c.x), int(c.y)): [int(v) for v in fn(k)] for k, c in enumerate(self.place["k"])}
        sdef = [("SWI_CB0", str(SWI_CB0))]
        flat_r = [v for xy in self.readers.noc_coords() for v in xy]
        nr = len(self.readers.assign)
        swi = dict(
            reader=([self.nt, self.i_tiles] + acc(gu), rtd(lambda k: [gu.buffer_address(), (self.nt * k) % self.i_tiles, nr]),
                    sdef + [("MK_SWI_WAIT_SEM", str(SEM_GU))], None),
            compute=([self.nt, f32_bits(self.limit), f32_bits(-self.limit)], rtd(lambda k: []), sdef,
                     ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                  fp32_dest_acc_en=True)),
            writer=([self.nt] + acc(m), rtd(lambda k: [m.buffer_address(), (self.nt * k) % self.i_tiles, nr] + flat_r),
                    sdef + [("MK_SWI_SIGNAL_SEM", str(SEM_M))], None),
        )  # fmt: skip
        swi_cbs = [ttnn.CBDescriptor(total_size=2048 * n, core_ranges=mhc.kcores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=SWI_CB0 + i, data_format=BF16, page_size=2048)])
            for i, n in ((0, self.nt), (1, self.nt), (2, 2))]  # fmt: skip
        # P10 mhc_post roles: the layer output, flat DRAM
        proles, post_cbs, post_io = self.post.parts(y, s1, coef, out, t=t, x_mode=_mpost.X_STACKED,
                                                    out_mode=_mpost.OUT_FLAT, wait=(SEM_Y, nr),
                                                    packed=t <= _mpost.PACKED_ROWS_MAX)  # fmt: skip
        if self.post.last_packed:
            for role in proles.values():
                role[2].append(("MK_POST_PACKED", "1"))
        phase_roles = ((8, swi), (10, proles))

        mkk = lambda src, cores, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "mk", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=_rt_args(rt), defines=defs, config=cfg,
        )  # fmt: skip
        want_size = os.environ.get("GLM53_MK_OPT", "O3") == "size"

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
                ct += pct
                width = max(len(prt[core]) for core in mine)
                for core in mine:
                    rt[core] += [int(a) for a in prt[core]] + [0] * (width - len(prt[core]))
                n_rt += width
            return ct, rt, defs + ([("MK_OPT_SIZE", "1")] if want_size else [])

        kernels = []
        ks = _role(roles, "stream.cpp", "MHC_K")
        ct, rt, defs = combine(ks, "reader")
        kernels.append(mkk("owner_ffn_dm0.cpp", ks["cores"], ct, rt, defs, ks["config"]))
        for lead in (True, False):
            sr = _role(roles, "sync.cpp", "MHC_LEADER") if lead else _role(roles, "sync.cpp", "MHC_K", "MHC_LEADER")
            ct, rt, defs = combine(sr, "writer")
            kernels.append(mkk("owner_ffn_dm1.cpp", sr["cores"], ct, rt, defs, sr["config"]))
            cr = (_role(roles, "compute.cpp", "MHC_LEADER") if lead
                  else _role(roles, "compute.cpp", "MHC_K", "MHC_LEADER"))  # fmt: skip
            ct, rt, defs = combine(cr, "compute")
            cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                              fp32_dest_acc_en=True)  # fmt: skip
            modes = list(cr["config"].unpack_to_dest_mode)
            kmodes = list(proles["compute"][3].unpack_to_dest_mode)
            modes = [a if a != ttnn.UnpackToDestMode.Default else b for a, b in zip(modes, kmodes)]
            cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
            kernels.append(mkk("owner_ffn_c.cpp", cr["cores"], ct, rt, defs, cc))
        for r in roles:  # the root's three kernels, unchanged
            if ("MHC_ROOT", "1") in r["defines"]:
                kernels.append(ttnn.KernelDescriptor(
                    kernel_source=os.path.join(KDIR, "mhc_pre", r["name"]),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=r["cores"],
                    compile_time_args=r["ct"], runtime_args=_rt_args(r["rt"]), defines=r["defines"],
                    config=r["config"]))  # fmt: skip
        kernels += r_kernels
        sems = mhc_sems + [
            ttnn.SemaphoreDescriptor(id=SEM_HN2, core_ranges=self.readers.cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_M, core_ranges=self.readers.cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_GU, core_ranges=mhc.kcores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_Y, core_ranges=mhc.kcores, initial_value=0),
        ]
        io = [s1, mhc.fn, mhc.w, mhc.sb, mhc.dbg, post, comb, coef, hn]
        io += [p.ds.weight for p in self.readers.phases] + [gu, m, y, out]
        io += [t_ for t_ in post_io if all(t_ is not u for u in io)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems,
                                                   cbs=mhc_cbs + swi_cbs + post_cbs + r_cbs))  # fmt: skip
        self.calls += 1
        return _mhc.MhcPreOutputs(hn=hn, post=post, comb=comb, x_out=None, coef=coef), gu, m, y, out
