# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a `kda_rec`: the KDA decode recurrence + the sigmoid-gated o RMSNorm as ONE generic_op program — the
megakernel's P4 / P5-norm phase (doc/ttnn/07a-fused-streaming-layer/megakernel_plan.md).

Replaces (OptimizedKDA._recurrence_and_output up to o_proj, B rows): ttnn.experimental.kda.fused_recurrent_kda
(in-place state) and the gated norm (rms_norm over the [B*64, 128] view + sigmoid(gate_out) multiply + typecast at
B = 1; sigmoid_gated_rms_norm at B >= 2).

Math per item (b, h) (64 heads x 128; q / k / g / gate_out rows of [1,1,B,8192] fp32, beta [1,1,B,64] fp32, state
S [B,64,128,128] fp32):
    q, k = l2norm(q), l2norm(k) (eps 1e-6);  q *= 1/sqrt(128)
    S = exp(g)[:, None] * S;  delta = (v - k @ S) * beta;  S += k^T delta;  o = q @ S
    on = o / sqrt(mean(o^2) + eps) * o_norm * sigmoid(gate_out) -> bf16 row b, columns 128 h .. of [1,1,B,8192]
Program: one head per core (head h on core h of the 64-core list — kda_pre's core h, which owns head h's g /
gate_out), items b = 0..B-1 per core; the recurrence arithmetic is ttnn's fused_recurrent_kda compute unchanged (fp32
SFPU, so the state is bitwise the op's), the norm an fp32 SFPU epilogue. NoC: BRISC (NoC0) reads, NCRISC (NoC1) the o
row + state writes. `reference_step` is the arithmetic in torch (fp64).
"""

from __future__ import annotations

import math
import os
import struct

import torch

import ttnn

KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "kda_rec")
FP32, BF16 = ttnn.float32, ttnn.bfloat16
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TILE = 32
N_CORES = 64
RMS_SEM = 13  # program semaphore of the spread gated norm's partial sums (MkLayer uses 0..12; Stage 14 adds it)
# Stage 17 verify (REC_CHAIN): writer -> reader of the same owner core, row b's state landed. An id no owner declares in
# MkLayer / MkSparseLayer (15 = the sparse layer's SEM_TAIL lives on the READERS only); MkStack's MK_X_SEM pick skips
# it (tp_stack reserved_sems = the layer's extra semaphores)
CHAIN_SEM = 15
PSUM_CB = 30  # spread: the partial sum of squares (owners: 29 = park, 31 = probe); CB_PIN = cb_x0 + 3 (the debug tap)
# CB slots: 14 at cb_base + n, 3 at cb_x0 + n (kernels/kda_rec/kda_rec_common.hpp)
CB = dict(Q=0, K=1, V=2, G=3, BETA=4, S0=5, QCOL=6, KCOL=7, KFULL=8, DCOL=9, ST=10, O=11, GO=12, ON=13)
CBX = dict(SCR=0, W=1, OT=2, TAP=3)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def reference_step(q, k, v, g, beta, gate_out, state, o_norm, eps):
    """Host oracle (fp64). q / k / v / g / gate_out [B, 64, 128], beta [B, 64], state [B, 64, 128, 128], o_norm [128]
    -> dict(on [B, 64, 128], state [B, 64, 128, 128], o [B, 64, 128])."""
    d = lambda x: x.double()
    q, k, v, g, beta, go, s = d(q), d(k), d(v), d(g), d(beta), d(gate_out), d(state)
    l2 = lambda x: x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    q, k = l2(q) / math.sqrt(q.shape[-1]), l2(k)
    s = s * torch.exp(g)[..., None]
    delta = (v - torch.einsum("bhk,bhkv->bhv", k, s)) * beta[..., None]
    s = s + k[..., None] * delta[..., None, :]
    o = torch.einsum("bhk,bhkv->bhv", q, s)
    on = o / torch.sqrt((o * o).mean(-1, keepdim=True) + eps) * d(o_norm) * torch.sigmoid(go)
    return dict(on=on, state=s, o=o)


class KdaRec:
    """One KDA layer's fused recurrence + gated norm program (bound to the layer's o_norm; call per decode step)."""

    def __init__(self, kda, *, cores: list | None = None, cb_base: int = 0, cb_x0: int | None = None,
                 opt_size: bool = False, lean: bool = False, state_depth: int | None = None,
                 debug: bool = False):  # fmt: skip
        """cores (megakernel): 64 logical CoreCoords, head h on cores[h] (default: kda_pre's 8 x 8 grid order, h = 8 x
        + y); cb_base / cb_x0 place the 14 + 2 CB indices (kernels: REC_CB0 / REC_CB_X0); opt_size: kernels compiled
        for size (kernels/mk/opt_size.hpp); lean: single-buffered per-item inputs and state (megakernel owners, whose
        CB region bounds every L1 tensor of the layer); state_depth: the per-item state CB's slots (default as the
        inputs; lean + state_depth 2 = the megakernel's T > 1 variant: the reader issues the next item's state first,
        REC_S_FIRST, so the 64 KB state read overlaps compute for +64 KB instead of the double inputs' +148 KB);
        debug: the pre-norm o rows as a tap (`last_tap`, checked against reference_step's "o")."""
        self.device = kda.device
        self.kda = kda
        self.h, self.dk, self.dv = kda.h, kda.d, kda.d
        assert self.h <= N_CORES and self.dk % TILE == 0 and self.dk <= 4 * TILE, (self.h, self.dk)
        self.cb_base = cb_base
        self.cb_x0 = cb_base + len(CB) if cb_x0 is None else cb_x0
        self.depth = 1 if lean else 2
        self.s_depth = self.depth if state_depth is None else int(state_depth)
        assert self.s_depth in (1, 2) and self.s_depth >= self.depth, (self.s_depth, self.depth)
        self.debug = debug
        # Stage 14 TP: kda.kda_rec_spread = cores per head (value-column split, 1 / 2 / 4); core c = head c // spread,
        # column block c % spread; the gated norm's sum of squares is exchanged among a head's cores (REC_RMS_SEM)
        self.spread = int(getattr(kda, "kda_rec_spread", 1))
        assert self.spread in (1, 2, 4) and not (debug and self.spread > 1), self.spread
        # Stage 14: kda.kda_rec_ugroups = u > 1 user groups (GLM53_REC_UGROUPS): the B rows split into u contiguous
        # ranges, group g on cores g * h * spread .. (its own copy of every head's column blocks)
        self.ug = int(getattr(kda, "kda_rec_ugroups", 1))
        self.hs = self.h * self.spread  # the head cores of one user group
        self.n_active = self.hs * self.ug
        self.rms_sem = int(getattr(kda, "kda_rec_rms_sem", RMS_SEM))  # MkSparseLayer: 13 is its SEM_LOGITS -> 14
        # Stage 14c (GLM53_REC_MODE=v2): fused SFPI recurrence (kernels/kda_rec/kda_rec_v2_sfpu.hpp), one compact vector
        # tile per item; one head per core (spread 1; user groups for parallelism), no debug tap
        self.v2 = os.environ.get("GLM53_REC_MODE") == "v2"
        if self.v2:  # v2 has its own reader / CB layout (one state slot per input slot): no 7a state slot
            self.s_depth = self.depth
        assert not self.v2 or (
            self.spread == 1 and not debug and self.dk == 128 and self.dv == 128
        ), "kda_rec v2: spread 1, head dim 128, no debug tap"
        # v2 composes with user groups (each group core runs its own rows' items; device-validated: 14c chip 7 watched
        # 03:31Z, Stage 14 TP2 native B=8 / 32 vs Stage 3 0.999959 / 0.999970, job 3432); GLM53_REC_V2_UG is inert
        # Stage 17 speculative verify (kda.kda_verify_sel = the [1, 32] uint32 selector page, set only while a verify
        # width's layers are built): rows chained through the state (kernels/kda_rec/reader.cpp REC_CHAIN)
        self.verify_sel = getattr(kda, "kda_verify_sel", None)
        if self.verify_sel is not None:
            assert self.ug == 1 and self.s_depth == self.depth and not debug, (
                "REC_CHAIN: one user group, no early state read, no debug tap",
                self.ug,
                self.s_depth,
                self.depth,
            )
        self.last_tap = None
        self._taps = {}
        self.defines = (
            [("REC_CB0", str(cb_base)), ("REC_CB_X0", str(self.cb_x0)), ("REC_DEPTH", str(self.depth))]
            # the reader's revision (a kernel binary is cached per path + args + defines: a changed reader needs a new
            # key in a long-lived process); 2 = read_state() / REC_S_FIRST; 3 = no weight read on a 0-row core
            + [("REC_REV", "3")]
            + ([("REC_S_FIRST", "1")] if self.s_depth > self.depth else [])
            + ([("MK_OPT_SIZE", "1")] if opt_size else [])
            + ([("REC_DEBUG", "1")] if debug else [])
            + ([("REC_HEADS", str(self.h))] if self.h != N_CORES else [])  # Stage 14 TP8 shard: cores h >= H skip
            + ([("REC_SPREAD", str(self.spread)), ("REC_RMS_SEM", str(self.rms_sem))] if self.spread > 1 else [])
            + ([("REC_UG", str(self.ug))] if self.ug > 1 else [])
            + ([("REC_V2", "1")] if self.v2 else [])
            # Stage 14c diagnostics (GLM53_REC_PROBE=1 with v2): cross-lane primitive patterns into state tile (0, 0)
            + ([("REC_V2_PROBE", "1")] if self.v2 and os.environ.get("GLM53_REC_PROBE") == "1" else [])
            # Stage 14 glue trim (GLM53_REC_EARLY_SIGNAL=1): the o_proj input is published before the last state write
            + ([("REC_EARLY_SIGNAL", "1")] if os.environ.get("GLM53_REC_EARLY_SIGNAL") == "1" else [])
            # GLM53_REC_ITEM_MARKS=1 (probe runs): per-item writer marks (norm partials exchanged, state written)
            + ([("REC_ITEM_MARKS", "1")] if os.environ.get("GLM53_REC_ITEM_MARKS") == "1" else [])
            + ([("REC_CHAIN", "1"), ("REC_CHAIN_SEM", str(CHAIN_SEM))] if self.verify_sel is not None else [])
        )
        w = torch.zeros(TILE, self.dv)
        w[0] = ttnn.to_torch(ttnn.get_device_tensors(kda.o_norm)[0]).float().reshape(-1)  # replicated weight
        self.host_o_norm = w[0].clone()
        n_dev = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        self.w = ttnn.from_torch(w, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if n_dev > 1 else None)  # fmt: skip
        if cores is None:
            cores = [ttnn.CoreCoord(i // 8, i % 8) for i in range(N_CORES)]
        from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

        assert len(cores) >= self.n_active, (len(cores), self.n_active)
        self.core_list = list(cores)
        self.cores = merged_ranges([(c.x, c.y) for c in cores])
        self.calls = 0

    @staticmethod
    def _acc(t) -> list:
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    @staticmethod
    def _geometry(t) -> list:
        s, p = list(t.shape), list(t.padded_shape)
        return [s[-1], s[-2], p[-2] // TILE, p[-1] // TILE]

    def _cb(self, index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )  # fmt: skip

    def _rtd(self, fn) -> dict:
        return {(int(c.x), int(c.y)): [int(v) for v in fn(h)] for h, c in enumerate(self.core_list)}

    def output(self, rows: int, memory_config=L1):
        """The o_proj input (the program writes whole tiles, padding rows zero; user groups: rows only, so the
        tensor is zeroed here — create it outside any trace capture)."""
        if self.ug > 1:
            return ttnn.zeros((1, 1, rows, self.h * self.dv), BF16, ttnn.TILE_LAYOUT, self.device, memory_config)
        return ttnn.empty((1, 1, rows, self.h * self.dv), BF16, ttnn.TILE_LAYOUT, self.device, memory_config)

    def _hc(self, c: int) -> int:
        """The kernels' core index of core c: its head core within its user group (cores past n_active: c, >= hs)."""
        return c % self.hs if c < self.n_active else c

    def _items(self, c: int, rows: int) -> int:
        """(b0 << 16) | n: core c's rows (user groups: contiguous ranges of ceil(rows / ug); a group may get none)."""
        if self.ug == 1 or c >= self.n_active:
            return rows
        per = -(-rows // self.ug)
        b0 = (c // self.hs) * per
        return (min(b0, rows) << 16) | max(0, min(rows, b0 + per) - b0)

    def parts(self, q, k, v, g, beta, gate_out, state, on, *, rows: int, wait: dict | None = None,
              signal: tuple | None = None, tap=None):  # fmt: skip
        """(roles {reader / writer / compute: (compile-time args, {core: runtime args}, defines, config)}, cbs, io).
        q / k / v / g / gate_out [1,1,B,8192] fp32, beta [1,1,B,64] fp32 (kda_pre's outputs), state = state.S
        [B,64,128,128] fp32 (updated in place), on = the [1,1,B,8192] bf16 output. Megakernel hand-offs: wait = (sem
        id, {core (x, y): producer count}) before the first read; signal = (sem id, consumer NoC coords) after the
        last write."""
        if not 1 <= rows <= TILE:
            raise ValueError(f"kda_rec: decode rows {rows} (1..32)")
        kt, vt = self.dk // TILE, self.dv // TILE
        tensors = (q, k, v, g, beta, gate_out, state, self.w)
        reader_ct = [self.h, kt, vt]
        for t in (q, k, v, g, beta, gate_out):
            reader_ct += self._geometry(t)
        for t in tensors:
            reader_ct += self._acc(t)
        addrs = [t.buffer_address() for t in tensors]
        wdefs = list(self.defines)
        if wait is not None:
            wdefs.append(("MK_REC_WAIT_SEM", str(wait[0])))
        vsel = self.verify_sel
        assert vsel is None or rows <= 8, ("REC_CHAIN: verify rows <= 8", rows)  # (Stage 17: the t = 8 verify)
        reader = (reader_ct, self._rtd(lambda h: [self._hc(h), self._items(h, rows)] + addrs
                                       + [wait[1][self._xy(h)] if wait else 0]
                                       + ([vsel.buffer_address()] if vsel is not None else [])), wdefs,
                  ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                    noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        if self.debug and tap is None:
            raise ValueError("kda_rec debug: pass the tap tensor ([1,1,B,H*V] fp32, zeroed)")
        writer_ct = [self.h, kt, vt] + self._geometry(on) + self._acc(on) + self._acc(state)
        if self.debug:
            writer_ct += self._acc(tap)
        tap_addr = tap.buffer_address() if self.debug else 0
        sdefs = list(self.defines)
        cons = []
        if signal is not None:
            sdefs.append(("MK_REC_SIGNAL_SEM", str(signal[0])))
            cons = [len(signal[1])] + [c for xy in signal[1] for c in xy]
        writer = (writer_ct, self._rtd(lambda h: [self._hc(h), self._items(h, rows), on.buffer_address(),
                                                  state.buffer_address(), tap_addr] + cons
                                       + self._peers(h)),
                  sdefs,
                  ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                    noc=ttnn.NOC.RISCV_1_default))  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, dst_full_sync_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for n in ("Q", "K", "V", "G", "BETA", "S0", "QCOL", "KCOL", "KFULL", "DCOL", "O", "GO"):
            modes[self.cb_base + CB[n]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        modes[self.cb_x0 + CBX["W"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if self.spread > 1:  # the peers' fp32 partials, exact
            modes[self.cb_x0 + CBX["TAP"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        compute_ct = [kt, vt, 1, f32_bits(1.0 / math.sqrt(self.dk)), f32_bits(self.kda.eps), f32_bits(1.0 / self.dv)]
        compute = (compute_ct, self._rtd(lambda h: [self._items(h, rows), self._hc(h)]), list(self.defines), cc)
        t32, t16 = 4096, 2048
        # spread > 1 at any rows <= 32: the peers' partial sums of squares cycle through 2 CB_PIN item slots (writer.cpp:
        # a peer's partial of item b + 2 is causally after this core's partial of item b + 1) and the arrival count is
        # a >= wait, so the exchange does not depend on the number of items (the former rows <= 4 limit is lifted;
        # tests/s7a/dev_kda_rec_spread.py checks spread / user-group layouts up to 32 rows against spread 1)
        vt = vt // self.spread  # this core's value-column tiles
        kv, dp, sd = kt * vt, self.depth, self.s_depth
        c = lambda n, dt, page, pages: self._cb(self.cb_base + CB[n], dt, page, pages)
        if self.v2:  # kernels/kda_rec/kda_rec_common.hpp: VEC (CB_Q), VEC2 (QCOL), DELTA (KCOL), W 1 page
            cbs = [
                c("Q", FP32, t32, dp), c("S0", FP32, t32, dp * kv), c("QCOL", FP32, t32, 1), c("KCOL", FP32, t32, 1),
                c("ST", FP32, t32, kv), c("ON", BF16, t16, 2 * vt),
                self._cb(self.cb_x0 + CBX["SCR"], FP32, t32, 1), self._cb(self.cb_x0 + CBX["W"], FP32, t32, 1),
                self._cb(self.cb_x0 + CBX["OT"], BF16, t16, vt),
            ]  # fmt: skip
            io = list(tensors) + [on] + ([vsel] if vsel is not None else [])
            return dict(reader=reader, writer=writer, compute=compute), cbs, io
        cbs = [
            c("Q", FP32, t32, dp * kt), c("K", FP32, t32, dp * kt), c("V", FP32, t32, dp * vt), c("G", FP32, t32, dp * kt),
            c("BETA", FP32, t32, dp), c("S0", FP32, t32, sd * kv), c("QCOL", FP32, t32, kt), c("KCOL", FP32, t32, kt),
            c("KFULL", FP32, t32, kt), c("DCOL", FP32, t32, kt), c("ST", FP32, t32, kv), c("O", FP32, t32, vt),
            c("GO", FP32, t32, dp * vt), c("ON", BF16, t16, 2 * vt),
            self._cb(self.cb_x0 + CBX["SCR"], FP32, t32, 1), self._cb(self.cb_x0 + CBX["W"], FP32, t32, vt),
            self._cb(self.cb_x0 + CBX["OT"], BF16, t16, vt),
        ] + ([self._cb(self.cb_x0 + CBX["TAP"], FP32, t32, vt)] if self.debug else [])  # fmt: skip
        if self.spread > 1:  # kernels/kda_rec/kda_rec_common.hpp REC_CB_PSUM / REC_CB_PIN (2 item slots)
            cbs += [
                self._cb(PSUM_CB, FP32, t32, 1),
                self._cb(self.cb_x0 + CBX["TAP"], FP32, t32, 2 * (self.spread - 1)),
            ]
        io = list(tensors) + [on] + ([tap] if self.debug else []) + ([vsel] if vsel is not None else [])
        return dict(reader=reader, writer=writer, compute=compute), cbs, io

    def semaphores(self) -> list:
        """The program semaphores the phase needs (spread > 1: REC_RMS_SEM on the head cores)."""
        chain = ([ttnn.SemaphoreDescriptor(id=CHAIN_SEM, core_ranges=self.cores, initial_value=0)]
                 if self.verify_sel is not None else [])  # fmt: skip
        if self.spread == 1:
            return chain
        assert self.rms_sem != CHAIN_SEM, self.rms_sem
        return [ttnn.SemaphoreDescriptor(id=self.rms_sem, core_ranges=self.cores, initial_value=0)] + chain

    def _peers(self, c: int) -> list:
        """Writer args of core c (spread > 1): the NoC (x, y) of the head's other cores, in column-block order."""
        if self.spread == 1 or c >= self.n_active:
            return []
        g0 = (c // self.hs) * self.hs  # the user group's first core
        head, blk = divmod(c - g0, self.spread)
        out = []
        for p in range(self.spread):
            if p != blk:
                w = self.device.worker_core_from_logical_core(self.core_list[g0 + head * self.spread + p])
                out += [int(w.x), int(w.y)]
        return out

    def _xy(self, h: int) -> tuple:
        c = self.core_list[h]
        return (int(c.x), int(c.y))

    def l1_bytes(self) -> int:
        kt, vt = self.dk // TILE, self.dv // TILE
        dp = self.depth
        if self.v2:
            return 4096 * (dp + dp * kt * vt + 2 + kt * vt + 2) + 2048 * 3 * vt
        return (
            4096 * (3 * dp * kt + dp * vt + dp + self.s_depth * kt * vt + 4 * kt + kt * vt + vt + dp * vt + 1 + vt)
            + 2048 * 3 * vt
        )

    def __call__(self, q, k, v, g, beta, gate_out, state, *, rows: int, out_memory_config=L1, on=None):
        """Standalone: -> on [1,1,B,8192] bf16; state.S updated in place. on: a caller-made output (self.output(rows),
        e.g. made outside a trace capture: with user groups it is zeroed by a host write)."""
        on = self.output(rows, out_memory_config) if on is None else on
        tap = None
        if self.debug:
            # one zeroed tap per width, made on the first (eager) call: ttnn.zeros is a host write, not allowed inside a
            # trace capture; the program writes rows < B only, so the padding rows stay zero across replays
            key = (rows, str(out_memory_config))
            if key not in self._taps:
                self._taps[key] = ttnn.zeros((1, 1, rows, self.h * self.dv), FP32, ttnn.TILE_LAYOUT, self.device,
                                             out_memory_config)  # fmt: skip
            tap = self.last_tap = self._taps[key]
        roles, cbs, io = self.parts(q, k, v, g, beta, gate_out, state, on, rows=rows, tap=tap)
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KERNEL_DIR, f"{name}.cpp"),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                  compile_time_args=ct, runtime_args=_rt_from(rt), defines=defs, config=cfg)
            for name, (ct, rt, defs, cfg) in roles.items()
        ]  # fmt: skip
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=self.semaphores(), cbs=cbs))
        self.calls += 1
        return on


def _rt_from(rtd: dict):
    rt = ttnn.RuntimeArgs()
    for (x, y), args in rtd.items():
        rt[x][y] = args
    return rt
