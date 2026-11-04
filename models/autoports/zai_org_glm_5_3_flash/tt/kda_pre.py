# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a `kda_pre`: the KDA decode glue between the q|k|v / low-rank projections and fused_recurrent_kda as ONE
multi-core generic_op program.

Replaces (OptimizedKDA.decode, B rows): the 3 low-rank slices, the f_b / g_b K = 128 linears, the dt_bias add, the
decay multiply, the gated g multiply, the beta multiply, the conv (hist * taps multiply, fast_reduce_nc, slice,
typecast, addcmul, silu), the conv-history shift (slice, concat, where -> state.C) and the q / k / v slices: ~21 ops,
~200 us at B = 1.

Math per user row t (B <= 32 rows; columns: q|k|v 24576 = 3 x 64 heads x 128, g / gate_out 8192, beta 64):
    conv  = silu(h0 * tap0 + h1 * tap1 + h2 * tap2 + qkv * tap_cur)        (h_j = state.C row j, bf16; taps bf16 values)
    q | k | v = conv split in three 8192-column blocks                     (fp32)
    g     = sigmoid((f_a @ w_fb + dt_bias) * exp(A_log)) * (lower_bound * active)
    gate_out = g_a @ w_gb;   beta = sigmoid(b) * active                     (f_a | g_a | b = the low-rank projection)
    state.C rows of ACTIVE users <- (h1, h2, qkv)                           (ROWS mode; inactive users keep theirs)

Program: 64 cores (8 x 8), core k owns q|k|v column tiles [12k, 12k + 12), g / gate_out tiles [4k, 4k + 4) and (k < 2)
beta tile k; no cross-core traffic. ROWS mode (B <= rows_max): the valid user rows are packed — one packed tile per
row pair u holds rows (2u, 2u+1) of the core's 12 column tiles (packed row 2r + p = column tile c0 + r, user row
2u + p; row pairs keep the 64-B DRAM alignment), so the conv runs on ceil(B/2) tiles instead of 12 and the reader
reads / writes only the users' rows of state.C (the history shift happens in the reader, row-pair writes, mixed
active / inactive pairs assembled in L1). TILES mode (larger B): whole tiles, history shift left to the chain. NoC:
BRISC (NoC0) all reads + the history writes, NCRISC (NoC1) the outputs.
Numerics: the conv products / sums are exact fp32 SFPU arithmetic on bf16 values (today: fp32 FPU multiply,
fast_reduce_nc, addcmul), sigmoid / silu are the library's fp32 SFPU functions, the K = 128 projections the same FPU
matmul (HiFi4, fp32 DST) as today's linears. `reference_step` is the arithmetic in torch.
"""

from __future__ import annotations

import os

import torch

import ttnn

KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "kda_pre")
FP32, BF16 = ttnn.float32, ttnn.bfloat16
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
TILE = 32
GRID = 8
N_CORES = GRID * GRID
CONV_PER_CORE = 12
G_PER_CORE = 4
QKV_TILES = 768
CB = dict(H0=0, H1=1, H2=2, Q=3, T=4, CONV=5, FA=6, GA=7, WFB=8, WGB=9, DT=10, RATE=11, ACT=12, G=13, GO=14, B=15,
          BETA=16, SCR=17)  # fmt: skip


def tile_off(r: int, c: int) -> int:
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)


def packed_taps(taps: torch.Tensor, heads: int = N_CORES, spread: int = 1, pack: int = 2) -> torch.Tensor:
    """taps [4, 3 * heads * 128] (rows: hist 0, 1, 2, current) -> [4 * cores * 32, 32] (cores = heads * spread): tile
    (j, k) row pack r + p = tap j of column tile c k + r, c = 12 / spread conv tiles per core (every one of the unit's
    pack user rows; pack 2 = the pair layout), the other rows zero."""
    cores, conv = heads * spread, CONV_PER_CORE // spread
    out = torch.zeros(4, cores, TILE, TILE)
    t = taps.float().reshape(4, 12 * heads, TILE)
    for k in range(cores):
        for r in range(conv):
            for p in range(pack):
                out[:, k, pack * r + p] = t[:, conv * k + r]
    return out.reshape(4 * cores * TILE, TILE)


def kda_pack(spread: int) -> int:
    """GLM53_KDA_PACK (7a): user rows per ROWS-mode packed tile - 2 (default, the pair layout), an even number with
    pack x conv tiles <= 32, or 'max' (spread 4: 10 -> 4 units at B = 32 instead of 16 pairs)."""
    conv = CONV_PER_CORE // spread
    v = os.environ.get("GLM53_KDA_PACK", "2")
    pack = (TILE // conv) // 2 * 2 if v == "max" else int(v)
    assert pack % 2 == 0 and pack * conv <= TILE, (pack, conv)
    return pack


def reference_step(qkv, lr, hist, taps, w_fb, w_gb, dt_bias, rate, lower_bound, active):
    """Host oracle. qkv [B, 24576] bf16, lr [B, 320] bf16, hist [3, B, 24576] bf16, taps [4, 24576], w_fb / w_gb
    [128, 8192] bf16, dt_bias / rate [8192] fp32, active [B] bool -> dict(q, k, v, g, gate_out, beta, hist_new)."""
    d = lambda x: x.double()
    acc = d(hist[0]) * d(taps[0]) + d(hist[1]) * d(taps[1]) + d(hist[2]) * d(taps[2]) + d(qkv) * d(taps[3])
    conv = acc * torch.sigmoid(acc)
    f_a, g_a, b = lr[:, :128], lr[:, 128:256], lr[:, 256:320]
    m = active.double().reshape(-1, 1)
    g = torch.sigmoid((d(f_a) @ d(w_fb) + d(dt_bias)) * d(rate)) * (lower_bound * m)
    gate_out = d(g_a) @ d(w_gb)
    beta = torch.sigmoid(d(b)) * m
    hist_new = hist.clone()
    act = active.bool()
    hist_new[0, act], hist_new[1, act], hist_new[2, act] = hist[1, act], hist[2, act], qkv[act]
    w = 8192
    return dict(q=conv[:, :w], k=conv[:, w : 2 * w], v=conv[:, 2 * w :], g=g, gate_out=gate_out, beta=beta,
                hist_new=hist_new)  # fmt: skip


def _rt_from(rtd: dict):
    rt = ttnn.RuntimeArgs()
    for (x, y), args in rtd.items():
        rt[x][y] = args
    return rt


class KdaPre:
    """One KDA layer's fused pre-recurrence program (bound to the layer's weights; call per decode step)."""

    def __init__(self, kda, *, rows_max: int = 4, cores: list | None = None, cb_base: int = 0):
        """cores (megakernel): 64 logical CoreCoords, core index kk = position in the list (default: the 8 x 8 grid at
        the origin, kk = 8 x + y); cb_base shifts the CB indices (kernels: KDA_CB0)."""
        self.device = kda.device
        self.kda = kda
        self.rows_max = rows_max
        self.cb_base = cb_base
        # heads of the layer shard (Stage 14 TP8: 8 per chip): core k < heads owns one head's worth of columns, the
        # other cores of the list skip the phase (kernels: KDA_HEADS; 64 = the original single-chip layout)
        self.heads = int(kda.h)
        self.width = self.heads * 128
        # Stage 14 TP: kda.kda_pre_spread = cores per head's worth of columns (1, 2, 4; default 1 = the original)
        self.spread = int(getattr(kda, "kda_pre_spread", 1))
        assert self.spread in (1, 2, 4), self.spread
        self.conv, self.gpc = CONV_PER_CORE // self.spread, G_PER_CORE // self.spread
        # Stage 14 kda.kda_pre_rsplit = R row groups (ROWS mode; GLM53_KDA_RSPLIT): core kk = column core kk % (heads
        # spread), row group kk // (heads spread) — on a TP8 shard 32 column cores leave 32 owners idle; with R = 2
        # they convolve half of the row pairs each (group 0 keeps g / gate_out / beta)
        self.rsplit = int(getattr(kda, "kda_pre_rsplit", 1))
        assert self.rsplit >= 1, self.rsplit
        self.n_cols = self.heads * self.spread
        self.n_active = self.n_cols * self.rsplit
        # the kda_rec layout this program signals (consumers()), fixed at construction like KdaRec's: the TP layers of
        # one model layer share the attention object across widths (stack_model), and a later width's construction
        # sets other kda_rec_spread / kda_rec_ugroups on it before this width's program is first built
        self.rec_spread = int(getattr(kda, "kda_rec_spread", 1))
        self.rec_ug = int(getattr(kda, "kda_rec_ugroups", 1))
        # Stage 17 speculative verify (kda.kda_verify_sel = the [1, 32] uint32 selector page, set only while a verify
        # width's layers are built): the rows are consecutive tokens of ONE user (kernels/kda_pre/reader.cpp KDA_VERIFY)
        self.verify_sel = getattr(kda, "kda_verify_sel", None)
        assert self.verify_sel is None or self.rsplit == 1, "KDA_VERIFY: one row group"
        self.pack = kda_pack(self.spread)  # 7a: ROWS-mode user rows per packed tile (kernels: KDA_PACK)
        assert self.verify_sel is None or self.pack == 2, "KDA_VERIFY: the pair layout (KDA_PACK 2)"
        self.defines = (
            ([("KDA_CB0", str(cb_base))] if cb_base else [])
            + ([("KDA_HEADS", str(self.heads))] if self.heads != N_CORES else [])
            + ([("KDA_SPREAD", str(self.spread))] if self.spread != 1 else [])
            + ([("KDA_RSPLIT", str(self.rsplit))] if self.rsplit != 1 else [])
            + ([("KDA_PACK", str(self.pack))] if self.pack != 2 else [])
            + ([("KDA_PRE_REV", "2")] if self.verify_sel is not None else [])
        )  # fmt: skip  (2: the verify reader's CB_H* pages)
        # per device (a TP8 mesh holds each chip's own shard): [4, 3 * heads * 128] (bf16 values)
        hd = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(kda.taps_hist)]
        cd = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(kda.tap_cur)]
        blocks = [torch.cat([h_.reshape(3, -1), c_.reshape(1, -1)]) for h_, c_ in zip(hd, cd)]
        packed = torch.cat([packed_taps(b_, self.heads, self.spread, self.pack) for b_ in blocks]).to(torch.bfloat16)
        self.taps_packed = ttnn.from_torch(
            packed, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0) if len(blocks) > 1 else None,
        )  # fmt: skip
        self.host_taps = blocks[0]
        cr = lambda x0, y0, x1, y1: ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))
        if cores is None:
            self.core_list = [ttnn.CoreCoord(kk // GRID, kk % GRID) for kk in range(N_CORES)]
            self.cores = ttnn.CoreRangeSet([cr(0, 0, GRID - 1, GRID - 1)])
        else:
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

            assert len(cores) >= self.n_active, (len(cores), self.n_active)
            self.core_list = list(cores)
            self.cores = merged_ranges([(c.x, c.y) for c in cores])
        self.calls = 0

    @staticmethod
    def _acc(t) -> list:
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    def _cb(self, index, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=self.cb_base + index, data_format=dtype,
                                                        page_size=page)],
        )  # fmt: skip

    def _rtd(self, fn) -> dict:
        return {(int(c.x), int(c.y)): [int(v) for v in fn(kk)] for kk, c in enumerate(self.core_list)}

    def _rt(self, fn):
        rt = ttnn.RuntimeArgs()
        for (x, y), args in self._rtd(fn).items():
            rt[x][y] = args
        return rt

    def __call__(self, qkv, lr, hist, active, *, rows: int, out_memory_config=L1):
        roles, cbs, io, outs = self.parts(qkv, lr, hist, active, rows=rows, out_memory_config=out_memory_config)
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KERNEL_DIR, f"{name}.cpp"),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                  compile_time_args=ct, runtime_args=_rt_from(rt), defines=defs, config=cfg)
            for name, (ct, rt, defs, cfg) in roles.items()
        ]  # fmt: skip
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        self.calls += 1
        return outs

    def consumers(self) -> dict:
        """kda_rec head cores reading core kk's outputs: {kk: sorted heads} (conv tiles 12 kk .. -> heads of q / k /
        v, g / gate_out tiles 4 kk .. -> head kk, beta tile kk < 2 -> heads 32 kk ..); kda.kda_rec_spread = s > 1:
        the s cores h s .. h s + s - 1 of each head."""
        out = {}
        h = self.heads
        for kk_ in range(len(self.core_list)):
            if kk_ >= self.n_active:  # a core past the shard's heads: no kda_pre work, signals nobody
                out[kk_] = []
                continue
            kk, rg = kk_ % self.n_cols, kk_ // self.n_cols  # column core, row group (KDA_RSPLIT)
            heads = {(c % (4 * h)) // 4 for c in range(self.conv * kk, self.conv * (kk + 1))}  # q / k / v tiles
            if rg == 0:  # row group 0 also writes g / gate_out / beta
                heads |= {g // 4 for g in range(self.gpc * kk, self.gpc * (kk + 1))}  # g / gate_out tiles
                if kk < (h + 31) // 32:
                    heads |= set(range(32 * kk, min(h, 32 * kk + 32)))  # beta
            rs = self.rec_spread  # kda_rec core c = head c // rs (Stage 14 TP)
            ug = self.rec_ug  # kda_rec user groups: every group's copy of the head
            out[kk_] = sorted(u * h * rs + hh * rs + j for u in range(ug) for hh in heads for j in range(rs))
        return out

    def parts(self, qkv, lr, hist, active, *, rows: int, out_memory_config=L1, signal: tuple | None = None):
        """(roles {reader / writer / compute: (compile-time args, {core: runtime args}, defines, config)}, cbs, io
        tensors, outputs (q, k, v, g, beta, gate_out, history_updated)) — the program's pieces, for composing it
        into a megakernel. signal = (sem id, [NoC (x, y) of head core h]): the writer of core kk increments it on
        the head cores of `consumers()[kk]` after its last write."""
        """qkv [1,1,B,24576] bf16, lr [1,1,B,320] bf16, hist = state.C [1,3,B,24576] bf16, active = the decoder's
        (mask fp32, mask bf16, lower_bound * mask fp32) [1,1,B,1] -> (q, k, v, g, beta, gate_out, history_updated)."""
        kda = self.kda
        b = rows
        # 7a GLM53_KDA_TILES_MIN = n (megakernel, TP shard): b >= n takes TILES mode with the history shift in the
        # reader (KDA_TILES_HIST) — CONV_PER_CORE / spread whole column tiles per core (3 at spread 4) instead of b / 2
        # row pairs; 0 / unset = ROWS up to rows_max (today)
        tiles_min = int(os.environ.get("GLM53_KDA_TILES_MIN", "0"))
        roww = b <= self.rows_max and not (tiles_min and b >= tiles_min)
        hist_tiles = not roww and tiles_min > 0
        assert roww or self.rsplit == 1, "KDA_RSPLIT needs ROWS mode (rows <= rows_max; GLM53_KDA_TILES_MIN: RSPLIT 1)"
        units = (b + 1) // 2 if roww else self.conv
        # Stage 14c GLM53_KDA_MODE containing "batch" (ROWS mode): every row pair's reads behind one barrier, the CBs
        # hold every pair (kernels/kda_pre/reader.cpp KDA_BATCH)
        batch = roww and "batch" in os.environ.get("GLM53_KDA_MODE", "").split(",")
        assert not (batch and self.pack != 2), "GLM53_KDA_MODE=batch: pair layout only (GLM53_KDA_PACK 2)"
        mk = lambda shape: ttnn.empty(shape, FP32, ttnn.TILE_LAYOUT, self.device, out_memory_config)
        q, k, v, g, go = (mk((1, 1, b, self.width)) for _ in range(5))
        beta = mk((1, 1, b, self.heads))
        act0, act2 = active[0], active[2]
        taps = self.taps_packed if roww else kda.taps_hist
        reader_ct = [int(roww)]
        for t in (hist, qkv, taps, kda.tap_cur, lr, kda.w_fb, kda.w_gb, kda.dt_bias, kda.decay_rate, act0, act2):
            reader_ct += self._acc(t)
        addrs = [t.buffer_address() for t in (hist, qkv, taps, kda.tap_cur, lr, kda.w_fb, kda.w_gb, kda.dt_bias,
                                             kda.decay_rate, act0, act2)]  # fmt: skip
        vsel = self.verify_sel
        if vsel is not None:
            # Stage 17: <= 8 rows (the t = 8 verify of DFlash2 / K = 7 on the native w8 programs: 4 row pairs)
            assert roww and b <= 8 and not batch and not hist_tiles, (
                "KDA_VERIFY: ROWS mode, <= 8 rows, no batch",
                b,
                batch,
            )
        reader = (reader_ct, self._rtd(lambda kk: addrs + [kk, b, int(roww or hist_tiles)]
                                       + ([vsel.buffer_address()] if vsel is not None else [])),
                  self.defines + ([("KDA_BATCH", "1")] if batch else []) + ([("KDA_TILES_HIST", "1")] if hist_tiles else [])
                  + ([("KDA_VERIFY", "1")] if vsel is not None else []),
                  ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                    noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        writer_ct = [int(roww)]
        for t in (q, k, v, g, go, beta):
            writer_ct += self._acc(t)
        waddrs = [t.buffer_address() for t in (q, k, v, g, go, beta)]
        sig, wdefs = (lambda kk: []), self.defines
        if signal is not None:
            cons = self.consumers()
            sig = lambda kk: [len(cons[kk])] + [v for h in cons[kk] for v in signal[1][h]]
            wdefs = self.defines + [("MK_KDA_SIGNAL_SEM", str(signal[0]))]
        writer = (writer_ct, self._rtd(lambda kk: waddrs + [kk, units] + sig(kk)), wdefs,
                  ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                    noc=ttnn.NOC.RISCV_1_default))  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (CB["DT"], CB["RATE"], CB["ACT"]) + (() if roww else (CB["T"],)):
            modes[self.cb_base + cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        compute = ([int(roww)], self._rtd(lambda kk: [kk, units]), self.defines, cc)
        t16, t32 = 2048, 4096
        # the verify reader's history / qkv pages hold every row pair (kernels/kda_pre/reader.cpp KDA_VERIFY)
        unit_pages = max(2, units) if (batch or vsel is not None) else 2
        cbs = [
            self._cb(CB["H0"], BF16, t16, unit_pages), self._cb(CB["H1"], BF16, t16, unit_pages),
            self._cb(CB["H2"], BF16, t16, unit_pages), self._cb(CB["Q"], BF16, t16, unit_pages),
            self._cb(CB["T"], BF16, t16, 4) if roww else self._cb(CB["T"], FP32, t32, 8),
            self._cb(CB["CONV"], FP32, t32, 2),
            self._cb(CB["FA"], BF16, t16, 4), self._cb(CB["GA"], BF16, t16, 4),
            self._cb(CB["WFB"], BF16, t16, 16), self._cb(CB["WGB"], BF16, t16, 16),
            self._cb(CB["DT"], FP32, t32, 4), self._cb(CB["RATE"], FP32, t32, 4), self._cb(CB["ACT"], FP32, t32, 2),
            self._cb(CB["G"], FP32, t32, 4), self._cb(CB["GO"], FP32, t32, 4),
            self._cb(CB["B"], BF16, t16, 1), self._cb(CB["BETA"], FP32, t32, 1), self._cb(CB["SCR"], BF16, 384, 1),
        ]  # fmt: skip
        io = [qkv, lr, hist, taps, kda.tap_cur, kda.w_fb, kda.w_gb, kda.dt_bias, kda.decay_rate, act0, act2,
              q, k, v, g, go, beta] + ([vsel] if vsel is not None else [])  # fmt: skip
        return dict(reader=reader, writer=writer, compute=compute), cbs, io, (q, k, v, g, beta, go, roww)
