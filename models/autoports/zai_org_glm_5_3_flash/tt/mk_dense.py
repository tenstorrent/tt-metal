# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel — the dense_kda decode layer as ONE persistent program (doc/ttnn/07a-fused-streaming-layer/
megakernel_plan.md), built phase chain by phase chain (each step validated against the op chain before the next).

Core roles (one program, 110 cores):
  readers R  — the 16 DRAM bank readers of the DS placement (2 per bank): every weight-matmul phase (qkv + low-rank,
               o_proj, gate_up, down) streams its weights from the bank next to it (BRISC / NoC0) and pulls its
               activation from the producer phase (NCRISC / NoC1); kernels/ds_mm/;
  owners O   — 64 other cores: the glue phases (mhc_pre K cores, kda_pre, recurrence head cores, mhc_post), grouped in
               8 columns of 8 (the first of each column is the mhc_pre leader);
  root       — one more core: mhc_pre's coefficient / Sinkhorn root.
Phase hand-off: the producer cores increment a per-phase semaphore on each consumer core once their output tiles have
landed (write barrier first); the consumer waits for the producer count and resets the semaphore (semaphores are
re-initialised per launch).

Step A (this file, first chain): P1 (mhc_pre of the attention sublayer, the K cores = owners) -> P2 (the qkv matmul on
the readers, its activation = hn pulled from the owners' hn tensor pages). Output: hn and qkv tensors (validation:
== MhcPre then DSLinear(qkv) on the same input, bitwise).
"""

from __future__ import annotations

import os

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as _mhc
from models.autoports.zai_org_glm_5_3_flash.tt.ds_mm import DSMatmul

MM_CB_BASE = 24  # the matmul phases' CB indices (mhc_pre uses 0..20)
SEM_HN = _mhc.SEMAPHORES  # first semaphore id after mhc_pre's


def placement(device, readers: list) -> dict:
    """{"k": 64 owner cores, "root": core} on the logical worker grid minus the reader cores (column-major scan, so
    each column of 8 owners is a mhc_pre group)."""
    grid = device.compute_with_storage_grid_size()
    taken = {(int(c.x), int(c.y)) for c in readers}
    free = [(x, y) for x in range(grid.x) for y in range(grid.y) if (x, y) not in taken]
    if len(free) < 65:
        raise ValueError(f"megakernel placement: {len(free)} free cores < 65")
    root = free[-1]
    owners = free[:64]
    return dict(k=[ttnn.CoreCoord(x, y) for x, y in owners], root=ttnn.CoreCoord(*root))


def rect_placement(device, taken: set) -> dict:
    """{"k": 64 owner cores, "root": core} around the reader cores `taken` ((x, y) logical), in as few rectangles as
    the chip allows: whole reader-free columns first (column-major), then the free cells of the other columns. The
    DS readers' columns depend on the chip's harvesting (chip 0: readers at x = 0, 1, 6, 7 / 11; chip 2: x = 0, 1, 5,
    6), so the placement is derived per device, never hard-coded."""
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    clean = [x for x in range(gx) if all((x, y) not in taken for y in range(gy))]
    cells = [(x, y) for x in clean for y in range(gy)]
    cells += [(x, y) for x in range(gx) if x not in clean for y in range(gy) if (x, y) not in taken]
    if len(cells) < 65:
        raise ValueError(f"megakernel placement: {len(cells)} free cores < 65")
    return dict(k=[ttnn.CoreCoord(x, y) for x, y in cells[:64]], root=ttnn.CoreCoord(*cells[64]))


class MkStepA:
    """P1 mhc_pre (attention sublayer) + P2 qkv matmul in ONE program."""

    PROBE_ROWS = 128  # K cores 0..63, root 64, readers 2i / 2i+1 at 65 + ...

    def __init__(self, device, mhc_args: dict, ds_qkv, *, probe: bool = False, rect: str = "ds_rect"):
        self.device = device
        self.probe = None
        if probe:
            import torch

            self.probe = ttnn.from_torch(torch.zeros(self.PROBE_ROWS, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        fid = getattr(ds_qkv.ckc, "math_fidelity", ttnn.MathFidelity.HiFi2)
        if rect == "ds_rect":
            # bank-adjacent DS readers (the rate) + owners in a few rectangles around them (the dispatch): groups 0..6
            # = columns of 8 at x = 2..5 and 8..10, rows 0..7; group 7 = the 4 x 2 block x = 2..5, rows 8..9; root (8, 8)
            self.mm = DSMatmul(device, ds_qkv, fidelity=fid)
            taken = {(int(c.x), int(c.y)) for c, _, _ in self.mm.assign}
            cols = [(x, y) for x in (2, 3, 4, 5, 8, 9, 10) for y in range(8)] + [
                (x, y) for y in (8, 9) for x in (2, 3, 4, 5)
            ]
            root = (8, 8)
            if taken & (set(cols) | {root}):  # another chip's harvesting: derive the rectangles from its readers
                self.place = rect_placement(device, taken)
            else:
                self.place = dict(k=[ttnn.CoreCoord(x, y) for x, y in cols], root=ttnn.CoreCoord(*root))
            self.mhc = _mhc.MhcPre(device, **mhc_args, placement=self.place)
        elif rect == "rect":
            # rectangles (multicast dispatch of binaries / launch messages): owners = the 8 x 8 block at the origin
            # (mhc_pre's default placement), root (8, 0), readers = the 8 x 2 block below it, reader (bank b, index r)
            # at (b, 8 + r) — away from the banks' own columns, the NoC hops are paid on pipelined reads
            readers = [(ttnn.CoreCoord(b, 8 + r), b, r) for b in range(8) for r in range(ds_qkv.readers)]
            self.mm = DSMatmul(device, ds_qkv, fidelity=fid, readers=readers)
            self.place = dict(k=[ttnn.CoreCoord(k // 8, k % 8) for k in range(64)], root=ttnn.CoreCoord(8, 0))
            self.mhc = _mhc.MhcPre(device, **mhc_args)  # default placement = the same cores, rectangle ranges
        else:
            self.mm = DSMatmul(device, ds_qkv, fidelity=fid)
            self.place = placement(device, [c for c, _, _ in self.mm.assign])
            self.mhc = _mhc.MhcPre(device, **mhc_args, placement=self.place)
        self.calls = 0

    def __call__(self, x, *, t: int, x_mode: int, xout_mode: int, pc_mode: int):
        """x: the layer's streams (as MhcPre) -> (MhcPreOutputs, qkv [1,1,t,N_pad] bf16)."""
        mhc = self.mhc
        dev = self.device
        mk = lambda shape, dt: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, dev, ttnn.L1_MEMORY_CONFIG)
        hn = mk((1, 1, t, mhc.hidden), _mhc.BF16)
        if pc_mode == _mhc.PC_STACKED:
            post, comb = mk((t, 1, 4, 1), _mhc.FP32), mk((t, 1, 4, 4), _mhc.FP32)
        else:
            post, comb = mk((1, 4, t, 1), _mhc.FP32), mk((1, 1, t, 16), _mhc.FP32)
        x_out = None
        if xout_mode == _mhc.XOUT_STACKED:
            x_out = mk((t, 1, 4, mhc.hidden), _mhc.FP32)
        elif xout_mode == _mhc.XOUT_FLAT:
            x_out = mk((1, 4, t, mhc.hidden), _mhc.FP32)
        coef = mk((1, 1, 32, 32), _mhc.FP32)
        pa = self.probe.buffer_address() if self.probe is not None else None
        desc = mhc.descriptor(x, hn, post, comb, x_out, coef, t=t, x_mode=x_mode, xout_mode=xout_mode,
                              pc_mode=pc_mode, signal=(SEM_HN, self.mm.noc_coords()),
                              mk_probe=(pa, 0) if pa is not None else None)  # fmt: skip
        qkv = self.mm.output(hn)
        mm_kernels, mm_cbs = self.mm.parts(hn, qkv, cb_base=MM_CB_BASE, wait=(SEM_HN, 64),
                                           mk_probe=(pa, 65) if pa is not None else None)  # fmt: skip
        if pa is not None:
            every = ttnn.CoreRangeSet(list(mhc.all.ranges()) + list(self.mm.cores.ranges()))
            mm_cbs = mm_cbs + [ttnn.CBDescriptor(total_size=128, core_ranges=every, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=128)])]  # fmt: skip
        desc.kernels = list(desc.kernels) + mm_kernels
        desc.cbs = list(desc.cbs) + mm_cbs
        desc.semaphores = list(desc.semaphores) + [
            ttnn.SemaphoreDescriptor(id=SEM_HN, core_ranges=self.mm.cores, initial_value=0)
        ]
        io = [x, mhc.fn, mhc.w, mhc.sb, mhc.dbg, post, comb, coef] + ([x_out] if x_out is not None else [])
        io += [hn, self.mm.ds.weight, qkv] + ([self.probe] if self.probe is not None else [])
        ttnn.generic_op(io, desc)
        self.calls += 1
        return _mhc.MhcPreOutputs(hn=hn, post=post, comb=comb, x_out=x_out, coef=coef), qkv


def decode_probe(probe_host, *, k_cores=64, readers=16, mhz=1350.0) -> dict:
    """The probe tensor [rows, 16] (host int32) -> per-role phase durations in us, relative to the earliest start:
    owners (K cores: start, inputs in, hn out), root (start, z, coef, round-2, end), readers (wreader: start, first
    block, last block; mover: start, ready, pulled, computed, written)."""
    import torch

    p = probe_host.reshape(-1, 16).long() & 0xFFFFFFFF
    starts = [int(p[r, 0]) for r in range(k_cores + 1 + 2 * readers) if int(p[r, 0])]
    t0 = min(starts)
    us = lambda v: (int(v) - t0) / mhz if int(v) else None
    own = torch.tensor([[us(p[k, i]) or 0.0 for i in range(3)] for k in range(k_cores)])
    root = [us(p[k_cores, i]) for i in range(5)]
    wr = torch.tensor([[us(p[k_cores + 1 + 2 * r, i]) or 0.0 for i in range(3)] for r in range(readers)])
    mv = torch.tensor([[us(p[k_cores + 2 + 2 * r, i]) or 0.0 for i in range(5)] for r in range(readers)])
    q = lambda col: (round(float(col.min()), 1), round(float(col.median()), 1), round(float(col.max()), 1))
    return dict(owner_start=q(own[:, 0]), owner_inputs_in=q(own[:, 1]), owner_hn_out=q(own[:, 2]),
                root=[round(v, 1) if v is not None else None for v in root],
                reader_w_start=q(wr[:, 0]), reader_w_first=q(wr[:, 1]), reader_w_last=q(wr[:, 2]),
                mover_start=q(mv[:, 0]), mover_ready=q(mv[:, 1]), mover_pulled=q(mv[:, 2]),
                mover_computed=q(mv[:, 3]), mover_written=q(mv[:, 4]))  # fmt: skip


# ======================================================================================================== step B
SEM_QKV = SEM_HN + 1  # readers -> owners: qkv and low-rank landed (16 readers x 2 phases)
KDA_CB_BASE = 32  # kda_pre's CB indices on the owners (mhc_pre: 0..20, probe 31)
# step C
SEM_KDA = SEM_QKV + 1  # kda_pre writers -> kda_rec head cores (owners): q / k / v / g / gate_out / beta landed
SEM_ON = SEM_KDA + 1  # kda_rec writers -> readers: the o_proj input landed (64 head cores)
# kda_rec's CB indices on the owners: 50..63 + 14..16 (kda_pre: 32..49; mhc_pre's 14..20 live on its root core only)
REC_CB_BASE, REC_CB_X0 = 50, 14
# step D
SEM_A = SEM_ON + 1  # o_proj readers -> owners: the attention output landed (16 readers)
POST_CB_BASE = 21  # mhc_post's CB indices on the owners: 21..28


def _role(roles: list, name: str, has: str, lacks: str | None = None) -> dict:
    """The mhc_pre role dict whose kernel file is `name` and whose defines contain `has` (and not `lacks`)."""
    for r in roles:
        names = {d[0] for d in r["defines"]}
        if r["name"] == name and has in names and (lacks is None or lacks not in names):
            return r
    raise KeyError((name, has, lacks))


class MkStepB:
    """P1 mhc_pre (attention) -> P2 qkv + low-rank (readers) -> P3 kda_pre (owners) in ONE program; with `ds_o`
    (step C) also -> P4 kda_rec (recurrence + gated norm, owners: head h on owner h) -> P5 o_proj (readers)."""

    PROBE_ROWS = 192  # owner BRISC k -> 0..63, root NCRISC 64, owner NCRISC k -> 96 + k, readers -> 160 + 2 i (+1)

    def __init__(self, device, mhc_args: dict, kda, ds_qkv, ds_lr, ds_o=None, post: bool = False,
                 probe: bool = False):  # fmt: skip
        from models.autoports.zai_org_glm_5_3_flash.tt.kda_pre import KdaPre
        from models.autoports.zai_org_glm_5_3_flash.tt.kda_rec import KdaRec
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases

        self.device = device
        # low-rank K block 8 (not 32): the reader cores' CB region bounds every L1 tensor of the layer (IN0 = 2 x the
        # largest K block)
        phases = [MMPhase(ds_qkv, kb=2, wait=(SEM_HN, 64), signal=SEM_QKV), MMPhase(ds_lr, kb=8, signal=SEM_QKV)]
        if ds_o is not None:
            # the o_proj input arrives from the kda_rec head cores: one per head of the layer (shard)
            phases.append(
                MMPhase(
                    ds_o,
                    kb=8,
                    wait=(SEM_ON, int(kda.h) * int(getattr(kda, "kda_rec_spread", 1))),
                    signal=SEM_A if post else None,
                )
            )
        self.readers = ReaderPhases(device, phases)
        self.place = rect_placement(device, {(int(c.x), int(c.y)) for c, _, _ in self.readers.assign})
        self.mhc = _mhc.MhcPre(device, **mhc_args, placement=self.place)
        self.kda = KdaPre(kda, rows_max=4, cores=self.place["k"], cb_base=KDA_CB_BASE)
        self.rec = (
            KdaRec(kda, cores=self.place["k"], cb_base=REC_CB_BASE, cb_x0=REC_CB_X0, lean=True)
            if ds_o is not None
            else None
        )
        from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as _mpost

        assert not post or ds_o is not None, "step D (post) needs step C (ds_o)"
        self.post = _mpost.MhcPost(device, cores=self.place["k"], cb_base=POST_CB_BASE) if post else None
        self.owner_noc = [
            (int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c) for c in self.place["k"])
        ]
        self.persist = {}  # buffers kept across calls (allocated on the first, eager call: never inside a capture)
        self.probe = None
        if probe:  # per-phase wall-clock marks (kernels/mk/probe.hpp), decoded by decode_probe_b
            import torch

            self.probe = ttnn.from_torch(torch.zeros(self.PROBE_ROWS, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        self.calls = 0

    def _persistent(self, key, make):
        if key not in self.persist:
            self.persist[key] = make()
        return self.persist[key]

    def __call__(self, x, hist, active, *, t: int, x_mode: int, xout_mode: int, pc_mode: int, state=None):
        """x: the layer's streams; hist = state.C; active = the decoder's masks -> (MhcPreOutputs, qkv, lr, kda outputs
        (q, k, v, g, beta, gate_out, history_updated)); step C (state = state.S, updated in place) appends (on, a):
        the gated-normed o (o_proj input) and the attention output o_proj(on) [1,1,t,4096]; step D (post) appends
        s1: the residual after the attention post-mix, stacked [t,1,4,4096] fp32 (needs xout_mode XOUT_STACKED)."""
        import os as _os

        mhc, dev = self.mhc, self.device
        mk = lambda shape, dt: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, dev, ttnn.L1_MEMORY_CONFIG)
        hn = mk((1, 1, t, mhc.hidden), _mhc.BF16)
        if pc_mode == _mhc.PC_STACKED:
            post, comb = mk((t, 1, 4, 1), _mhc.FP32), mk((t, 1, 4, 4), _mhc.FP32)
        else:
            post, comb = mk((1, 4, t, 1), _mhc.FP32), mk((1, 1, t, 16), _mhc.FP32)
        x_out = None
        if xout_mode == _mhc.XOUT_STACKED:
            x_out = mk((t, 1, 4, mhc.hidden), _mhc.FP32)
        elif xout_mode == _mhc.XOUT_FLAT:
            x_out = mk((1, 4, t, mhc.hidden), _mhc.FP32)
        coef = mk((1, 1, 32, 32), _mhc.FP32)
        pa = self.probe.buffer_address() if self.probe is not None else None
        roles, mhc_cbs, mhc_sems = mhc.parts(x, hn, post, comb, x_out, coef, t=t, x_mode=x_mode, xout_mode=xout_mode,
                                             pc_mode=pc_mode, signal=(SEM_HN, self.readers.noc_coords()),
                                             mk_probe=(pa, 0) if pa is not None else None)  # fmt: skip
        rec = self.rec
        outs = self.readers.outputs(t)
        qkv, lr = outs[:2]
        on = self._persistent(("on", t), lambda: rec.output(t)) if rec is not None else None  # zeroed once
        r_kernels, r_cbs = self.readers.parts([hn, hn] + ([on] if rec else []), outs, consumers=self.owner_noc,
                                              mk_probe=(pa, 160) if pa is not None else None)  # fmt: skip
        ksig = (SEM_KDA, self.owner_noc) if rec is not None else None
        kroles, kda_cbs, kda_io, kda_outs = self.kda.parts(qkv, lr, hist, active, rows=t, signal=ksig)
        phase_roles = [kroles]
        rec_cbs, rec_io = [], []
        if rec is not None:
            cons = self.kda.consumers()
            count = {h: sum(h in hs for hs in cons.values()) for h in range(len(self.owner_noc))}
            q_, k_, v_, g_, beta_, go_ = kda_outs[:6]
            xy = {h: (int(c.x), int(c.y)) for h, c in enumerate(self.place["k"])}
            rroles, rec_cbs, rec_io = rec.parts(q_, k_, v_, g_, beta_, go_, state, on, rows=t,
                                                wait=(SEM_KDA, {xy[h]: count[h] for h in count}),
                                                signal=(SEM_ON, self.readers.noc_coords()))  # fmt: skip
            phase_roles.append(rroles)
        post_cbs, post_io, s1 = [], [], None
        if self.post is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as _mpost

            assert xout_mode == _mhc.XOUT_STACKED and x_out is not None, "step D reads the stacked residual"
            s1 = self.post.output(t, _mpost.OUT_STACKED)
            proles, post_cbs, post_io = self.post.parts(
                outs[2],
                x_out,
                coef,
                s1,
                t=t,
                x_mode=_mpost.X_STACKED,
                out_mode=_mpost.OUT_STACKED,
                wait=(SEM_A, len(self.readers.assign)),
                packed=t <= _mpost.PACKED_ROWS_MAX,
            )  # fmt: skip
            if self.post.last_packed:  # owner_*.cpp then include the packed mhc_post kernels (*_p.cpp)
                for role in proles.values():
                    role[2].append(("MK_POST_PACKED", "1"))
            phase_roles.append(proles)

        kdir = _os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "mk")
        mkk = lambda src, cores, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=_os.path.join(kdir, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=_rt_args(rt), defines=defs, config=cfg,
        )  # fmt: skip

        def combine(src, mrole, role_name, extra):
            """mhc_pre's role + the later owner phases' `role_name` parts (P3 kda_pre, P4 kda_rec): CT / RT lists
            concatenated, each later phase's bases as MK_P<n>_CT / MK_P<n>_RT."""
            mine = _cores_of(mrole["cores"])
            ct, defs = list(mrole["ct"]), list(mrole["defines"])
            n_rt = max((len(v) for v in mrole["rt"].values()), default=0)
            rt = {core: list(mrole["rt"].get(core, [])) for core in mine}
            rt = {core: v + [0] * (n_rt - len(v)) for core, v in rt.items()}
            for n, roles_ in zip((3, 4, 6), phase_roles):
                pct, prt, pdefs, _ = roles_[role_name]
                defs += [d for d in pdefs if d not in defs] + [
                    (f"MK_P{n}_CT", str(len(ct))),
                    (f"MK_P{n}_RT", str(n_rt)),
                ]
                ct += pct
                width = max(len(prt[core]) for core in mine)  # uniform: the next phase's base is compile-time
                for core in mine:
                    rt[core] += [int(a) for a in prt[core]] + [0] * (width - len(prt[core]))
                n_rt += width
            # step C: the owner kernels compiled for size (kernels/mk/opt_size.hpp; P1+P3+P4 at -O3 = 103 KB > the 69 KB
            # kernel config buffer)
            # GLM53_MK_OPT=O3 keeps the default -O3: needs a kernel-config ring > 69 KB, i.e. the device opened with
            # a smaller worker_l1_size (the ring = L1 below the allocatable region; tests/s7a/dev_mk_c GLM53_WORKER_L1)
            want_size = os.environ.get("GLM53_MK_OPT", "size") == "size"
            size = [("MK_OPT_SIZE", "1")] if want_size and rec is not None and ("MK_OPT_SIZE", "1") not in defs else []
            return ct, rt, defs + extra + size

        kernels = []
        ks = _role(roles, "stream.cpp", "MHC_K")
        ct, rt, defs = combine("owner_dm0.cpp", ks, "reader",
                               [("MK_SEM_QKV", str(SEM_QKV)), ("MK_QKV_COUNT", str(2 * len(self.readers.assign)))])  # fmt: skip
        kernels.append(mkk("owner_dm0.cpp", ks["cores"], ct, rt, defs, ks["config"]))
        for lead in (True, False):
            sr = _role(roles, "sync.cpp", "MHC_LEADER") if lead else _role(roles, "sync.cpp", "MHC_K", "MHC_LEADER")
            ct, rt, defs = combine("owner_dm1.cpp", sr, "writer", [])
            if pa is not None:  # the owner NCRISC's probe args after every phase's: slot 96 + k
                kidx = {(int(c.x), int(c.y)): k for k, c in enumerate(self.place["k"])}
                n_all = len(next(iter(rt.values())))
                rt = {core: v + [pa, 96 + kidx[core]] for core, v in rt.items()}
                defs = defs + [("MK_PROBE_ARG", str(n_all)), ("MK_PROBE_CB", "31")]
            kernels.append(mkk("owner_dm1.cpp", sr["cores"], ct, rt, defs, sr["config"]))
            cr = (
                _role(roles, "compute.cpp", "MHC_LEADER")
                if lead
                else _role(roles, "compute.cpp", "MHC_K", "MHC_LEADER")
            )
            ct, rt, defs = combine("owner_c.cpp", cr, "compute", [])
            cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                              fp32_dest_acc_en=True, dst_full_sync_en=rec is not None)  # fmt: skip
            modes = list(cr["config"].unpack_to_dest_mode)
            for roles_ in phase_roles:
                kmodes = list(roles_["compute"][3].unpack_to_dest_mode)
                modes = [a if a != ttnn.UnpackToDestMode.Default else b for a, b in zip(modes, kmodes)]
            cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
            kernels.append(mkk("owner_c.cpp", cr["cores"], ct, rt, defs, cc))
        for r in roles:  # the root's three kernels, unchanged
            if ("MHC_ROOT", "1") in r["defines"]:
                kernels.append(ttnn.KernelDescriptor(
                    kernel_source=_os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels",
                                                "mhc_pre", r["name"]),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=r["cores"],
                    compile_time_args=r["ct"], runtime_args=_rt_args(r["rt"]), defines=r["defines"], config=r["config"]))  # fmt: skip
        kernels += r_kernels
        sems = mhc_sems + [ttnn.SemaphoreDescriptor(id=SEM_HN, core_ranges=self.readers.cores, initial_value=0),
                           ttnn.SemaphoreDescriptor(id=SEM_QKV, core_ranges=mhc.kcores, initial_value=0)]  # fmt: skip
        if rec is not None:
            sems += [ttnn.SemaphoreDescriptor(id=SEM_KDA, core_ranges=mhc.kcores, initial_value=0),
                     ttnn.SemaphoreDescriptor(id=SEM_ON, core_ranges=self.readers.cores, initial_value=0)]  # fmt: skip
        if self.post is not None:
            sems.append(ttnn.SemaphoreDescriptor(id=SEM_A, core_ranges=mhc.kcores, initial_value=0))
        io = [x, mhc.fn, mhc.w, mhc.sb, mhc.dbg, post, comb, coef] + ([x_out] if x_out is not None else []) + [hn]
        io += [p.ds.weight for p in self.readers.phases] + list(outs)
        io += [t_ for t_ in kda_io + rec_io + post_io if all(t_ is not u for u in io)]
        probe_cbs = []
        if pa is not None:
            io.append(self.probe)
            probe_cbs = [ttnn.CBDescriptor(total_size=128, core_ranges=mhc.all, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=128)])]  # fmt: skip
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems,
                                                   cbs=mhc_cbs + kda_cbs + rec_cbs + post_cbs + r_cbs + probe_cbs))  # fmt: skip
        self.calls += 1
        res = (_mhc.MhcPreOutputs(hn=hn, post=post, comb=comb, x_out=x_out, coef=coef), qkv, lr, kda_outs)
        return res + ((on, outs[2]) if rec is not None else ()) + ((s1,) if self.post is not None else ())


def _rt_args(rtd: dict):
    rt = ttnn.RuntimeArgs()
    for (x, y), args in rtd.items():
        rt[x][y] = [int(a) for a in args]
    return rt


def _cores_of(crs) -> set:
    out = set()
    for r in crs.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                out.add((x, y))
    return out


def decode_probe_b(probe_host, *, k_cores=64, readers=16, mhz=1350.0) -> dict:
    """MkStepB's probe tensor [192, 16] (host) -> per-phase marks in us after the earliest mark (median over the cores):
    owner BRISC [start, inputs in L1, hn written, qkv + low-rank landed, kda_pre inputs read, kda_rec inputs read,
    mhc_post inputs read]; owner NCRISC: its last 4 marks [mhc_pre done, kda_pre written, kda_rec written, mhc_post
    written]; readers: the weight streamer's [start, phase ends ...], the activation mover's per-phase marks."""
    import statistics

    p = probe_host.reshape(-1, 16).long() & 0xFFFFFFFF
    nz = p[p > 0]
    t0 = int(nz.min()) if nz.numel() else 0
    us = lambda v: round((int(v) - t0) / mhz, 1)

    def med(rows, take=None):
        cols = []
        for r in rows:
            m = [int(v) for v in p[r] if int(v) > 0]
            if take is not None:
                m = m[-take:]
            cols.append(m)
        n = min((len(c) for c in cols), default=0)
        return [us(statistics.median(c[i] for c in cols)) for i in range(n)]

    return dict(
        owner_brisc=med(range(k_cores)),
        owner_ncrisc=med(range(96, 96 + k_cores), take=4),
        root_ncrisc=med([64]),
        reader_w=med(range(160, 160 + 2 * readers, 2)),
        reader_a=med(range(161, 161 + 2 * readers, 2)),
    )
