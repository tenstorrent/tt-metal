# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel (M4): the routed experts of a sparse layer on the DS bank readers — fused_experts' math as
reader phases (kernels/mkx/, mkx_common.hpp), so the expert weights stream like the dense matmuls' (bank-adjacent
readers, a ring, no per-expert dispatch) and the phase can join the layer program.

Placement: R readers per DRAM bank (ttnn's DS placement, mk_readers.reader_assignments); reader (bank b, index i) owns
the expert shards j = b + 8 s (s = i, i + R, ... < 8) of fused_experts' ND-sharded weights (64 shards of [rows, 64] round
robin over the 8 banks): it computes m tile j for every expert of the union and output tiles 2j, 2j + 1. The expert
weight base addresses are a device table (uint32 [2E], built once); the union and the routing weights are computed
from the router tail's ids / scores on every reader (no broadcast). m is all-gathered through a double-buffered L1
tensor (semaphores MW / MR per buffer).

Standalone (this step): __call__(x, ids, scores) -> y [1,1,T,H] bf16 == fused_experts(x, ids, scores) (same rounding
points; the device test compares bitwise / PCC)."""

from __future__ import annotations

import os
import struct

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import reader_assignments

BF16, FP32 = ttnn.bfloat16, ttnn.float32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
TILE = 32
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "mkx")
CB0 = 45  # MKX_CB0: CB_X .. CB_RAW = 45 .. 55 (the dense readers use 31, 39 .. 44)
SEM_LIST, SEM_MW0, SEM_MW1, SEM_MR0, SEM_MR1 = 10, 11, 12, 13, 14
SEM_XTP = (
    15  # expert TP: the senders' partials landed on the leader (MkMoE leaves it free; MkSparseLayer passes its own)
)
XTP = 8  # expert TP: m slices (chips)
XSL_RS_PAGES = 6  # MKX_XSL: CB_RS = 2 groups x 3 owned shards of routing weight tiles
LIST_BYTES = 20480  # [n, T, ids[288]] uint32 + weights bf16 [288][32]
RAW_BYTES = 26624  # ids tile + 9 score tiles + the address table + routing work arrays
MKX_REV = 24  # kernel revision (compile signature: a long-lived process rebuilds after a kernel edit): 2 pipelined
# weight reads, gate_up(e + 1) before down(e); 3 expert parallelism (E_LO / E_LOC), zeros for an empty union; 4
# MKX_INFLIGHT blocks in flight; 5 matmul_block (ct=2); 6 phase-relative args; 7 megakernel hand-offs; 11 expert TP;
# 12 expert TP fp32 running sums; 13 load-aware placement (MKX_LOC); 14 routing weights from the tail (MKX_RW); 15 progressive x (MKX_XPROG);
# 16 routing step: table read + NoC zero fill of the weight rows before the hand-off, O(1) duplicate check, word-wise union
# 17 MKX_XROWS: the activation mover reads only the T decode rows of x (T <= 16; default on, GLM53_MKX_XROWS=0 = off)
# 18 MKX_XROWS_PRE: in the megakernel composites the x zero fill is the activation mover's program-start prologue
# 19 MKX_REP: EP replicas, per-batch balanced assignment of the touched replicated experts (xw)
# 20 MKX_RW_XA: the routing weights' scatter on xa (off the stream-start path; GLM53_MKX_RW_XA=1)
# 21 MKX_XSL: batched expert TP (slice c of every selected expert, the union in groups of 8 = one EP-shaped pass each)
# 22 fold_y2 (MKX_Y2_INIT): y2 added to the finished routed sum in its own pass (batch invariance; Stage 14)
TPT = 8  # tiles per NoC read transaction (8 x 576 B)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class ReaderExperts:
    def __init__(self, device, gate_up: list, down: list, *, top_k: int, scaling: float, limit: float,
                 intermediate: int, hidden: int, readers_per_bank: int = 3, kb: int = 32, ring_slots: int = 12,
                 placement: str = "kda_qkv", math_fidelity=ttnn.MathFidelity.LoFi, e_lo: int = 0,
                 num_experts: int | None = None, cb0: int = CB0, m_buffers: int = 2, xtp: int | None = None,
                 xtp_sem: int = SEM_XTP, local_experts: list | None = None, replica_home: list | None = None,
                 chip: int = 0, xsl: int | None = None, xsl_tp: int = 8, xsl_sliced: bool = False):  # fmt: skip
        self.device = device
        # gathered-m buffers of the down's in0 (CB_M): 2 lets the next expert's gather land before the current down
        # pops; 1 (a megakernel's reader L1 budget) is enough at the compute order gu(pos), dn(pos - 1)
        self.m_buffers = m_buffers
        self.cb0 = cb0  # first of the 11 CB indices (a megakernel shifts them past its other reader phases')
        self.gate_up, self.down = gate_up, down
        # expert parallelism: gate_up / down are this chip's experts e_lo .. e_lo + len - 1 of num_experts (routing
        # width); experts outside the range do not enter this chip's union (its output is the partial sum over them)
        self.e_loc, self.e_lo = len(gate_up), e_lo
        self.e = num_experts if num_experts is not None else len(gate_up)
        assert 0 <= e_lo and e_lo + self.e_loc <= self.e
        self.top_k, self.scaling, self.limit = top_k, scaling, limit
        self.i_tiles, self.h_tiles = intermediate // TILE, hidden // TILE
        self.kt_gu, self.kt_dn = hidden // TILE, intermediate // TILE
        self.gu_cols = 2 * intermediate // TILE
        self.kb, self.ring_slots, self.fid = kb, ring_slots, math_fidelity
        if os.environ.get("GLM53_MKX_FIDELITY"):  # precision checks (e.g. HiFi4: the routing-weight multiply exact)
            self.fid = getattr(ttnn.MathFidelity, os.environ["GLM53_MKX_FIDELITY"])
        # expert TP (megakernel_plan §10e; kernels/mkx/mkx_common.hpp MKX_XTP): xtp = c -> this program computes slice c
        # = m tiles [8c, 8c + 8) of EVERY selected expert: on bank b gate_up shard 8c + b with its K split over the
        # bank's 3 readers (the leader, idx 0, adds the senders' fp32 partials, SwiGLU, one m tile), the 8 m tiles
        # all-gathered, then the down rows of those tiles for the reader's output shards (as EP). On one chip the slice
        # is read in place from the full fused_experts tensors (the TP8 per-chip work, emulated); y = the slice's partial
        # (the 8 slices' sum = the routed output). xtp_sem: a reader semaphore id the composite leaves free.
        self.xtp, self.xtp_sem = xtp, xtp_sem
        if xtp is not None:
            assert 0 <= xtp < XTP and self.e_loc == self.e and readers_per_bank == 3, (
                xtp,
                self.e_loc,
                readers_per_bank,
            )
            self.kb = 8  # K blocks of 8 tiles (the slice's down K = 8 tiles)
        # batched expert TP (kernels/mkx/mkx_common.hpp MKX_XSL): xsl = c -> slice c (m tiles [8c, 8c + 8), down rows
        # [256c, + 256)) of EVERY selected expert, read in place from the full tensors (every chip holds all experts:
        # the m3 harness's replicated MoE; production: per-chip slice tensors = the same bytes as EP's 36 experts). The
        # union is taken 8 experts at a time: group g's shard j = b + 8 s is slice tile b of the group's expert s (bank
        # b, full K, like EP's shard j), so a group is ONE EP-shaped pass (same readers, one 64-tile m gather) and every
        # chip streams 1/8 of every selected expert: B=1 = 1 expert-equivalent on every chip (EP: the busiest chip's
        # 2.5 on average), any B = the mean. m carries the routing weight (x w before bfloat8_b), the down sums the
        # group's experts in one K loop (fp32 running sums); the 8 slices' outputs sum in the all-reduce.
        # xsl_tp: the slices (chips): 8 (TP8, slice width 8 m tiles, groups of 8 experts), 4 (16, 4), 2 (32, 2)
        self.xsl, self.xsl_tp = xsl, xsl_tp
        if xsl is not None:
            assert xtp is None and xsl_tp in (2, 4, 8) and 0 <= xsl < xsl_tp, (xsl, xsl_tp)
            assert self.e_loc == self.e and local_experts is None, (
                xsl,
                xtp,
                self.e_loc,
                self.e,
            )
            self.kb = 8  # one ring block = one expert's slice rows of a down shard (8 tiles x 2)
            self.ring_slots = 4 * self.ring_slots  # the EP ring's bytes in 4x smaller blocks
        # xsl_sliced (production layout): gate_up / down are this chip's SLICE tensors (xsl_slices(): gate_up
        # [H, 2W x 32] fused [g|u], down [W x 32, H], ND-sharded like the full tensors: the same bytes per chip as
        # EP's 36 experts); the kernel reads slice 0 of them (XSL_C = 0, GU_COLS = 2W), no per-chip patch
        self.xsl_sliced = bool(xsl_sliced) and xsl is not None
        if self.xsl_sliced:
            self.gu_cols = 2 * (self.i_tiles // xsl_tp)
        self.m_tiles = self.i_tiles // XTP if xtp is not None else self.i_tiles  # gathered m tiles (the down K)
        # weight stream knobs (env overrides for sweeps): blocks in flight, tiles per NoC transaction
        self.inflight = int(os.environ.get("GLM53_MKX_INFLIGHT", "2"))
        if xsl is not None:  # 4x smaller blocks: more in flight (own knob: a job's EP reference keeps its ring / depth)
            self.inflight = int(os.environ.get("GLM53_MKX_XSL_INFLIGHT", "8"))
        self.tpt = int(os.environ.get("GLM53_MKX_TPT", str(TPT)))
        self.tb = 576  # bfloat4_b tile bytes
        assert self.i_tiles == 64, "one m tile per fused_experts shard (64 shards)"
        self.assign = reader_assignments(device, readers_per_bank, os.environ.get("GLM53_MK_PLACEMENT", "")
                                         or self._placement_name(gate_up[0]))  # fmt: skip
        R = readers_per_bank
        self.owned = {}
        for c, bank, idx in self.assign:
            self.owned[(int(c.x), int(c.y))] = [bank + 8 * s for s in range(idx, 8, R)]
        # the kernels carry at most 3 shard indices per reader (xw / xa: shard[3]); fewer than 3 readers per bank
        # would leave shard[3..] unset -> out-of-range expert pages (Stage 7b's watcher assert, 2026-10-19 22:07Z)
        if max(len(v) for v in self.owned.values()) > 3:
            raise ValueError(f"ReaderExperts: readers_per_bank={R} gives a reader > 3 of its bank's 8 shards (>= 3)")
        from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

        self.cores = merged_ranges([(c.x, c.y) for c, _, _ in self.assign])
        self.noc = [(int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c) for c, _, _ in self.assign)]
        table = [g.buffer_address() for g in gate_up] + [d.buffer_address() for d in down]
        # a load-aware EP placement (scripts/ep_placement.py): this chip's experts are any ids, local_experts[slot] =
        # the global id of gate_up[slot] / down[slot]; the table gains loc[e] (slot, or E_LOC = not here; xw MKX_LOC)
        self.local_experts = list(local_experts) if local_experts is not None else None
        self.replica_home, self.chip = replica_home, chip
        assert replica_home is None or local_experts is not None, "EP replicas need a placement (local_experts)"
        if self.local_experts is not None:
            assert e_lo == 0 and len(self.local_experts) == self.e_loc == len(set(self.local_experts)), "placement"
            assert all(0 <= x < self.e for x in self.local_experts), "placement ids"
            loc = [self.e_loc] * self.e
            for slot, x in enumerate(self.local_experts):
                loc[x] = slot
            table += loc
            # EP replicas (scripts/ep_placement.py --replicas): replica_home[e] = (primary chip, second chip or -1) for
            # every expert; the table gains home words + this chip's index and xw.cpp (MKX_REP) assigns every touched
            # replicated expert to one of its homes per batch (the same greedy on every chip)
            if replica_home is not None:
                assert len(replica_home) == self.e and 0 <= chip < 8, (len(replica_home), chip)
                for x in self.local_experts:
                    p_, s_ = replica_home[x]
                    assert chip in (p_, s_), ("a local expert must have this chip as a home", x, p_, s_, chip)
                table += [(p_ & 0xFF) | ((0xFF if s_ < 0 else s_) << 8) for p_, s_ in replica_home] + [chip]
        self.table = ttnn.from_torch(torch.tensor(table, dtype=torch.int64).to(torch.int32).reshape(1, 1, 1, -1),
                                     dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                     memory_config=DRAM)  # fmt: skip
        self.persist = {}
        self.calls = 0

    @staticmethod
    def _placement_name(_):
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoderConfig

        return OptimizedDecoderConfig().ds_placement

    def _persistent(self, key, make):
        if key not in self.persist:
            self.persist[key] = make()
        return self.persist[key]

    def parts(self, x, ids, scores, y, *, t: int):
        """(kernels, cbs, semaphores, io) of the routed-expert phases on the reader cores (standalone program)."""
        raw, cbs, sems, io = self.parts_raw(x, ids, scores, y, t=t)
        return [self._kd(*raw[r]) for r in ("w", "a", "c")], cbs, sems, io

    def parts_raw(self, x, ids, scores, y, *, t: int, wait: tuple | None = None, signal: tuple | None = None,
                  y2=None, y2_sem: int | None = None, weights=None, xinv=None, live=None):  # fmt: skip
        """({"w" / "a" / "c": (kernel file, compile-time args, {(x, y): runtime args}, defines, config)}, cbs,
        semaphores, io). Megakernel hand-offs: wait = (semaphore id, producer count) before the router tail's ids /
        scores are read; signal = (semaphore id, [consumer NoC (x, y)]) after this reader's output columns landed.
        y2 / y2_sem (MKX_Y2_INIT): another phase's [1,1,t,H] bf16 output (the shared expert's down, written by a dense
        reader phase that runs BEFORE this one on the same readers) seeds the running sums, so y = y2 + the routed sum
        (one partial for one all-reduce under TP / EP); the readers sync on y2_sem before reading it. weights (MKX_RW):
        the router tail's [t, 8] bf16 routing weights (router_tail/writer.cpp RTAIL_WEIGHTS, rw_ct()) replace the
        scores input: the readers only build the union and scatter them. live (MKX_LIVE, Stage 14b): the stack's
        position page (int32 ROW_MAJOR [1, 32], row b < 0 = an idle decode row): idle rows select no expert (no union
        entry, no weight, nothing streamed for them; coordinator 2026-10-22 10:51Z batch invariance)."""
        dev = self.device
        m = self._persistent(("m", self.m_tiles), lambda: ttnn.empty((1, 1, TILE, 2 * self.m_tiles * TILE),
                                                                     ttnn.bfloat8_b, ttnn.TILE_LAYOUT, dev, L1))  # fmt: skip
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        rt = lambda fn: {xy: [int(a) for a in fn(xy)] for xy in self.owned}
        pad3 = lambda v: (list(v) + [0, 0, 0])[:3]
        # GLM53_MKX_XPROG=1: x pushed in 8-tile chunks, gate_up waits per K block (the first expert starts earlier)
        # GLM53_MKX_Y2_FOLD_LAST=1: the previous y2 fold into the last expert's add (not batch invariant; A/B only)
        y2fold = [("MKX_Y2_FOLD_LAST", "1")] if os.environ.get("GLM53_MKX_Y2_FOLD_LAST") == "1" else []
        xprog = [("MKX_XPROG", "1")] if os.environ.get("GLM53_MKX_XPROG") == "1" else []
        # MKX_XROWS (T <= 16, default since 2026-10-20 14:2xZ: routed y bitwise at T=1/8/32, sparse layer T=1 bitwise
        # with / without it): xa reads only the T valid rows of x (every reader pulled all KT_GU full tiles);
        # GLM53_MKX_XROWS=0 = the full-tile reads
        if os.environ.get("GLM53_MKX_XROWS", "1") == "1" and t <= 16:
            xprog = [("MKX_XROWS", str(t))]
        if weights is not None:  # the tail's weights in the scores slot (xw.cpp MKX_RW)
            scores = weights
        rep = [("MKX_REP", "1")] if self.replica_home is not None else []
        # MKX_RW_XA (with the tail's weights; default since 2026-10-20 15:1xZ: watched w_f01..03 bitwise / exact,
        # routing step at T=32 17.1 -> 11.5 us, c4_01 / c4_02): the weights' scatter runs on xa, xw publishes the list
        # right after the union; GLM53_MKX_RW_XA=0 = the scatter on xw
        if weights is not None and os.environ.get("GLM53_MKX_RW_XA", "1") == "1":
            rep = rep + [("MKX_RW_XA", "1")]
        defs = xprog + rep + y2fold + ([("MKX_LOC", "1")] if self.local_experts is not None else []) + (
            [("MKX_RW", "1")] if weights is not None else []) + [
                ("MKX_CB0", str(self.cb0)), ("MKX_SEM_LIST", str(SEM_LIST)), ("MKX_SEM_MW0", str(SEM_MW0)),
                ("MKX_SEM_MW1", str(SEM_MW1)), ("MKX_SEM_MR0", str(SEM_MR0)), ("MKX_SEM_MR1", str(SEM_MR1)),
                ("MKX_TPT", str(self.tpt)), ("MKX_INFLIGHT", str(self.inflight)), ("MKX_REV", str(MKX_REV)),
                ("MKX_MMBLOCK", os.environ.get("GLM53_MKX_MMBLOCK", "1"))]  # fmt: skip
        wdefs, adefs = list(defs), list(defs)
        wtail, atail = [], []
        if wait is not None:
            wdefs.append(("MKX_WAIT_SEM", str(wait[0])))
            wtail = [wait[1]]
        if signal is not None:
            adefs.append(("MKX_SIGNAL_SEM", str(signal[0])))
            atail = [len(signal[1])] + [v for xy in signal[1] for v in xy]
        cdefs, a_ct_y2, y2_rt = list(defs), [], []
        if y2 is not None:
            assert y2.dtype == BF16 and tuple(y2.shape)[-1] == self.h_tiles * TILE, (y2.dtype, tuple(y2.shape))
            adefs += [("MKX_Y2_INIT", "1"), ("MKX_Y2_SEM", str(y2_sem)),
                      ("MKX_Y2_RT", str(7 + 2 * len(self.noc) + len(atail)))]  # fmt: skip
            cdefs.append(("MKX_Y2_INIT", "1"))
            a_ct_y2, y2_rt = acc(y2), [y2.buffer_address()]
        # xinv (MKX_XINV, the sparse layer's deferred FFN 1/rms(h)): its [1,1,32,32] fp32 tile; xa reads it into CB_XINV
        # (accessor after y2's, address at MKX_XINV_RT), xc scales gate | up per row before SwiGLU
        a_ct_xinv, xinv_rt = [], []
        if xinv is not None:
            assert self.xtp is None, "MKX_XINV with expert TP partials (the leader's sum) is not wired"
            a_ct_xinv, xinv_rt = acc(xinv), [xinv.buffer_address()]
            xi_def = [("MKX_XINV", "1"), ("MKX_XINV_RT", str(7 + 2 * len(self.noc) + len(atail) + len(y2_rt)))]
            adefs += xi_def
            cdefs += xi_def[:1]
        vc = {(int(c.x), int(c.y)): bank & 3 for c, bank, _ in self.assign}
        kt_dn = self.m_tiles  # down K rows read per output shard (expert TP: the slice's 8)
        xw_rt, xa_rt, xc_rt = {}, {}, {}
        if self.xsl is not None:  # per expert of a group: the slice's 8 rows (the group's K = 8 x its experts)
            kt_dn = self.i_tiles // self.xsl_tp  # the slice width W (m tiles)
            xdefs = [("MKX_XSL", "1"), ("MKX_XSL_RS_PAGES", str(XSL_RS_PAGES)), ("MKX_XSL_W", str(kt_dn))]
            wdefs += xdefs
            adefs += xdefs
            cdefs += xdefs
            # GLM53_MKX_XSL_SEED=1 (xc.cpp MKX_XSL_SEED): the running sum seeds each group's down accumulation -> a row's
            # routed sum does not depend on the group boundaries (= the other rows' routing): batch invariance vs co-rows
            if os.environ.get("GLM53_MKX_XSL_SEED", "0") == "1":
                cdefs.append(("MKX_XSL_SEED", "1"))
            for c, bank, idx in self.assign:
                xy = (int(c.x), int(c.y))
                xc_rt[xy] = pad3(self.owned[xy])
        if self.xtp is not None:
            wtail = wtail or [0]  # the XTP args sit at a fixed index
            leader = {bank: self.noc[i] for i, (_, bank, idx) in enumerate(self.assign) if idx == 0}
            nblk = self.kt_gu // self.kb
            cut = [0, nblk * 5 // 16 * self.kb, nblk * 10 // 16 * self.kb, self.kt_gu]  # K: 40 / 40 / 48 tiles
            for c, bank, idx in self.assign:
                xy = (int(c.x), int(c.y))
                xw_rt[xy] = [bank, cut[idx], cut[idx + 1]]
                xa_rt[xy] = [idx, leader[bank][0], leader[bank][1], bank]
                xc_rt[xy] = [idx, cut[idx], cut[idx + 1] - cut[idx]]
            xdefs = [("MKX_XTP", "1"), ("MKX_XTP_SEM", str(self.xtp_sem)), ("MKX_XTP_W", str(XTP))]
            wdefs += xdefs + [("MKX_XTP_RTW", str(9 + len(wtail)))]
            adefs += xdefs + [("MKX_XTP_RTA", str(7 + 2 * len(self.noc) + len(atail) + len(y2_rt)))]
            cdefs += xdefs
        e_tiles = (self.e + TILE - 1) // TILE
        w_ct = [self.e, self.top_k, self.kt_gu, kt_dn, self.kb, self.tb, self.gu_cols, self.h_tiles,
                f32_bits(self.scaling), f32_bits(1e-20), e_tiles, self.ring_slots, self.e_lo, self.e_loc]  # fmt: skip
        w_ct += acc(ids) + acc(scores) + acc(self.table) + acc(self.gate_up[0]) + acc(self.down[0])
        if self.xtp is not None:
            self.xw_xtp_ct = len(w_ct)  # the slice's index in xw's compile-time args (Stage 14: patched per chip)
            w_ct.append(self.xtp)
        if self.xsl is not None:
            self.xw_xsl_ct = len(w_ct)  # the slice's index in xw's compile-time args (TP8: patched per chip)
            w_ct.append(0 if self.xsl_sliced else self.xsl)
        live_rt = []
        if live is not None:  # appended LAST (the XTP / XSL ct slot and MKX_XTP_RTW keep their indices)
            assert live.dtype == ttnn.int32 and live.layout == ttnn.ROW_MAJOR_LAYOUT and int(live.shape[-1]) == 32 \
                and t <= 32, (live.dtype, live.layout, tuple(live.shape), t)  # fmt: skip
            n_xw = {len(v) for v in xw_rt.values()} or {0}
            assert len(n_xw) == 1, "uniform xw runtime args"
            wdefs += [("MKX_LIVE", "1"), ("MKX_LIVE_RT", str(9 + len(wtail) + n_xw.pop()))]
            w_ct += acc(live)
            live_rt = [live.buffer_address()]
        flat = [v for xy in self.noc for v in xy]
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True)
        ccfg.unpack_to_dest_mode = self.unpack_modes([ttnn.UnpackToDestMode.Default] * 64)
        raw = dict(
            w=("xw.cpp", w_ct, rt(lambda xy: [ids.buffer_address(), scores.buffer_address(),
                                              self.table.buffer_address(), t, len(self.owned[xy])]
                                   + pad3(self.owned[xy]) + [vc[xy]] + wtail + xw_rt.get(xy, []) + live_rt), wdefs,
               dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            a=("xa.cpp", [self.kt_gu, self.m_tiles, len(self.noc)] + acc(x) + acc(m) + acc(y) + a_ct_y2 + a_ct_xinv,
               rt(lambda xy: [x.buffer_address(), m.buffer_address(), y.buffer_address(), len(self.owned[xy])]
                  + pad3(self.owned[xy]) + flat + atail + y2_rt + xinv_rt + xa_rt.get(xy, [])), adefs,
               dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
            c=("xc.cpp", [self.kt_gu, kt_dn, self.kb, self.m_tiles, f32_bits(self.limit)],
               rt(lambda xy: [len(self.owned[xy])] + xc_rt.get(xy, [])), cdefs, ccfg),
        )  # fmt: skip
        nsh = max(len(v) for v in self.owned.values())
        cb = lambda i, dt, page, pages: ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=self.cb0 + i, data_format=dt, page_size=page)])  # fmt: skip
        # CB_ACC: the running sums, bf16 (fused_experts' rounding) or fp32 under expert TP (xc.cpp)
        fp32_acc = self.xtp is not None or self.xsl is not None
        acc_cb = cb(6, FP32, 4096, 2 * (nsh + 1)) if fp32_acc else cb(6, BF16, 2048, 2 * (nsh + 1))
        cbs = [cb(0, BF16, 2048, self.kt_gu), cb(1, ttnn.bfloat4_b, self.tb, self.ring_slots * self.kb * 2),
               cb(2, ttnn.bfloat8_b, 1088, self.m_buffers * self.m_tiles), cb(3, FP32, 4096, 2), cb(4, ttnn.bfloat8_b, 1088, 2 * nsh),
               cb(5, BF16, 2048, XSL_RS_PAGES if self.xsl is not None else 2), acc_cb, cb(7, BF16, 2048, 2), cb(8, BF16, 2048, 2 * nsh),
               cb(9, ttnn.uint32, LIST_BYTES, 1), cb(10, ttnn.uint32, RAW_BYTES, 1)]  # fmt: skip
        if y2 is not None:
            cbs.append(cb(11, BF16, 2048, 2 * nsh))
        if xinv is not None:  # CB_XINV
            cbs.append(cb(14, FP32, 4096, 1))
        if self.xtp is not None:  # CB_PART (2 item slots x 2 senders x 2 tiles), CB_PS
            cbs += [cb(12, FP32, 4096, 8), cb(13, FP32, 4096, 2)]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.cores, initial_value=0)
                for i in (SEM_LIST, SEM_MW0, SEM_MW1, SEM_MR0, SEM_MR1)
                + ((self.xtp_sem,) if self.xtp is not None else ())]  # fmt: skip
        io = [x, ids, scores, self.table, m, y] + list(self.gate_up) + list(self.down)
        if y2 is not None:
            sems.append(ttnn.SemaphoreDescriptor(id=y2_sem, core_ranges=self.cores, initial_value=0))
            io.append(y2)
        if xinv is not None:
            io.append(xinv)
        if live is not None:
            io.append(live)
        return raw, cbs, sems, io

    @staticmethod
    def xsl_slices(gate_up: list, down: list, tp: int, c: int, device, experts=None):
        """Slice c of TP of each expert's fused_experts tensors as their own device tensors (ReaderExperts xsl_sliced):
        gate_up columns [64 W c, + 64 W) of the fused [g0|u0|g1|u1..] layout (W = 64 / TP m tiles = shards W c ..), down
        rows [32 W c, + 32 W); tile-aligned, so the bfloat4_b blocks re-pack bitwise. experts: the ids to build (others
        reuse the first one's slice: a test only reads the union's); production cuts the tile cache instead."""
        from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import _nd_dram

        w = 64 // tp
        ids = list(range(len(gate_up))) if experts is None else sorted(set(int(e) for e in experts))
        gcol, drow = slice(64 * w * c, 64 * w * (c + 1)), slice(32 * w * c, 32 * w * (c + 1))
        gu_s, dn_s = {}, {}
        for e in ids:
            g = ttnn.to_torch(gate_up[e]).reshape(gate_up[e].shape[-2], gate_up[e].shape[-1])[:, gcol]
            d = ttnn.to_torch(down[e]).reshape(down[e].shape[-2], down[e].shape[-1])[drow, :]
            pack = lambda t: ttnn.from_torch(t.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat4_b,
                                             layout=ttnn.TILE_LAYOUT)  # fmt: skip
            gu_s[e] = ttnn.to_device(pack(g), device, memory_config=_nd_dram(g.shape[0], 64))
            dn_s[e] = ttnn.to_device(pack(d), device, memory_config=_nd_dram(d.shape[0], 64))
        e0 = ids[0]
        return ([gu_s.get(e, gu_s[e0]) for e in range(len(gate_up))],
                [dn_s.get(e, dn_s[e0]) for e in range(len(down))])  # fmt: skip

    def rw_ct(self) -> list:
        """The router tail writer's compile-time args after its weights accessor (RTAIL_WEIGHTS): SCALING, EPS bits —
        the experts' normalisation constants."""
        return [f32_bits(self.scaling), f32_bits(1e-20)]

    def unpack_modes(self, modes: list):
        """The reader compute kernel's unpack-to-DST modes with this phase's entries (expert TP: CB_ACC fp32, exact)
        set: a composite passes its dense phases' modes."""
        modes = list(modes)
        if self.xtp is not None or self.xsl is not None:
            modes[self.cb0 + 6] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if self.xsl is not None:  # CB_MM (fp32 matmul outputs) only copied to DST under MKX_XSL: exact
            modes[self.cb0 + 3] = ttnn.UnpackToDestMode.UnpackToDestFp32
        modes[self.cb0 + 14] = ttnn.UnpackToDestMode.UnpackToDestFp32  # CB_XINV (MKX_XINV; unused otherwise)
        return ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)

    def __call__(self, x, ids, scores, *, t: int):
        y = ttnn.empty((1, 1, t, self.h_tiles * TILE), BF16, ttnn.TILE_LAYOUT, self.device, L1)
        kernels, cbs, sems, io = self.parts(x, ids, scores, y, t=t)
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        self.calls += 1
        return y

    def _kd(self, name, ct, rt: dict, defs, cfg):
        return ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, name),
                                     source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                     compile_time_args=ct, runtime_args=self._rt(rt), config=cfg,
                                     defines=list(defs))  # fmt: skip

    @staticmethod
    def _rt(d):
        r = ttnn.RuntimeArgs()
        for (x, y), args in d.items():
            r[x][y] = [int(a) for a in args]
        return r
