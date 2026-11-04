# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a lever A (doc/ttnn/07a-fused-streaming-layer/prefill_overlap.md): the PREFILL mHC "pre" of one sublayer as
ONE multi-core program over T rows of the flat residual.

Replaces the prefill chain of tt/prefill_tp.py (mhc_coeffs + mhc_pre_prefill + the sublayer's rms_norm: the fn matmul,
square, sum, multiply / add / rsqrt / multiply, mhc_split_sinkhorn, the pre-mix program, rms_norm + typecast = 7-9
programs, 680 us per sublayer at TP8 / 1,413 us at TP2 for 2,048 rows), reading the fp32 residual once.

Math per token (tt/mhc_pre.py, the decode program's arithmetic; `reference_step` is the oracle):
    m = (x @ fn^T) / rms(x); z = m * scale + bias; pre = sigmoid(z[0:4]) + eps; post = 2 sigmoid(z[4:8]);
    comb = Sinkhorn(softmax(z[8:24] as 4 x 4)); h = sum_s pre_s x_s; hn = h / rms(h) * w
Inputs: x [1, 1, T, 16384] fp32 TILE (T % 32 == 0). Outputs: hn [1, 1, T, 4096] (bf16 or fp32), post [1, 1, T, 4] and
comb [1, 1, T, 16] fp32 TILE (row t, column i / 4j + k) -- the layouts tt/prefill_tp.py's mhc_post_prefill reads.

Program (kernels/mhc_pre_pf/, one source per RISC; roles by define):
    G groups x (P members + 1 root). Group g handles row blocks g, g + G, ... (32 tokens each).
    member j   owns column tiles [j CPC, (j + 1) CPC) of all 4 streams (CPC = 128 / P): fn^T (bf16) and the norm weight
               of its columns resident; per row block it reads its 4 CPC x tiles (one stream at a time, up to XBUF row
               blocks ahead), M = x @ fn^T (FPU, fp32 DST, HiFi4) and S = sum x^2 (SFPU, exact fp32) -> root; then
               h = sum_s pre_s x_s (exact fp32 SFPU), S2 = sum h^2 -> root; hn = h * 1/rms(h) * w -> hn tiles.
    root       sums the P partials, 1/rms(x), z, the decode root's plane-layout coefficient + Sinkhorn SFPU pass
               (kernels/mhc_pre/mhc_pre_sfpu.hpp), sends pre, writes post / comb of the row block; sums the S2
               partials, sends 1/rms(h).
DRAM per call: the residual once (T x 64 KB) + hn + the coefficients; fn^T / w once per member (128 + 64 KB at P = 8).
L1 per member at P = 8, XBUF = 1: ~600 KB (x 256 KB, fn^T 128 KB, w 64 KB, h 64 KB, gather 64 KB, small CBs).
"""

from __future__ import annotations

import os
import struct

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import reference_step  # noqa: F401  (the oracle)

KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "mhc_pre_pf")
FP32, BF16 = ttnn.float32, ttnn.bfloat16
DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32
N_STREAMS = 4
COL_TILES = 128
N_MIX = 24

# mirrors mhc_pre_pf_common.hpp
CB = dict(GATHER=0, MSG=1, X=2, X32=3, FN=4, W=5, PART=6, PRE=7, H=8, S2=9, INV=10, OUT=11, H16=12, RED=14, PLANE=15,
          COEF=16, SB=17, INVR=18, SCR=19)  # fmt: skip
SEMAPHORES = 4
PF_REV = "2"  # kernel revision define of kernels/mhc_pre_pf: bump on every kernel source edit (2: PF_SKIP)
# fp32 CBs read by copy_tile: unpacked straight to DST (exact fp32 on the SFPU)
FP32_COPY_CBS = (CB["GATHER"], CB["X32"], CB["W"], CB["PRE"], CB["H"], CB["INV"], CB["PLANE"], CB["SB"])


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class MhcPrePf:
    """One sublayer's fused prefill mHC pre (its fn / biases / scales / norm weight); call per prefill chunk."""

    def __init__(
        self,
        mesh_device,
        fn_t: torch.Tensor,  # [n*D, 24] host
        scale: torch.Tensor,  # [24] per-mix scale (pre x4 | post x4 | comb x16)
        bias: torch.Tensor,  # [24] (hc base)
        norm_w: torch.Tensor,  # [D] the sublayer's RMSNorm weight
        *,
        hc_eps: float,
        x_eps: float,
        h_eps: float,
        iters: int,
        hidden: int = 4096,
        n: int = N_STREAMS,
        members: int | None = None,
        max_groups: int | None = None,
        xbuf: int | None = None,
        fpu: bool | None = None,
        skip: int = 0,
    ):
        assert n == N_STREAMS and hidden == COL_TILES * TILE and tuple(fn_t.shape) == (n * hidden, N_MIX), fn_t.shape
        self.device = mesh_device
        self.iters, self.hc_eps, self.x_eps, self.h_eps = iters, hc_eps, x_eps, h_eps
        self.hidden, self.width = hidden, n * hidden
        # GLM53_PF_MHC_P members per group (column split), GLM53_PF_MHC_G groups at most, GLM53_PF_MHC_XBUF row blocks
        # of x in flight per member
        self.p = int(members or os.environ.get("GLM53_PF_MHC_P", "8"))
        assert COL_TILES % self.p == 0, self.p
        self.cpc = COL_TILES // self.p
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(grid.x), int(grid.y))
        cap = (self.grid[0] * self.grid[1]) // (self.p + 1)
        self.max_groups = min(cap, int(max_groups or os.environ.get("GLM53_PF_MHC_G", str(cap))))
        assert self.max_groups >= 1, (self.grid, self.p)
        self.xbuf = int(xbuf or os.environ.get("GLM53_PF_MHC_XBUF", "1"))
        # GLM53_PF_MHC_FPU=1 (PF_FPU): sum x^2 / sum h^2 / the collapse as FPU matmuls (tf32 operands, the op chain's
        # reduce / rms_norm precision) instead of the exact-fp32 SFPU passes
        self.fpu = bool(int(fpu if fpu is not None else os.environ.get("GLM53_PF_MHC_FPU", "0")))
        self.skip = int(skip)  # PF_SKIP timing bisect (wrong numerics): compute.cpp
        put = lambda t, dt: ttnn.from_torch(  # noqa: E731
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=DRAM
        )  # (replicated on a mesh)
        fn_pad = torch.zeros(n * hidden, TILE)
        fn_pad[:, :N_MIX] = fn_t.float()
        self.fn = put(fn_pad.to(torch.bfloat16), BF16)  # [16384, 32]: page = K tile (s * 128 + c)
        sb = torch.zeros(3 * TILE, TILE)
        sb[0, :N_MIX] = scale.float()
        sb[TILE, :N_MIX] = bias.float()
        sb[2 * TILE :] = torch.eye(TILE)
        self.sb = put(sb, FP32)  # page 0 row 0 = scale, page 1 row 0 = bias, page 2 = identity (PF_FPU diag mask)
        wt = torch.zeros(TILE, hidden)
        wt[0] = norm_w.float()
        self.w = put(wt, FP32)  # page c row 0 = w[32c : 32c + 32]
        self.host = dict(fn_t=fn_t.float(), scale=scale.float(), bias=bias.float(), w=norm_w.float())

    @classmethod
    def from_source(cls, mesh_device, source, cfg, L: int, which: str, norm_name: str, **kw):
        """The sublayer `which` ("attn" / "ffn") of layer L from a weight source (host copies), as
        optimized_decoder._build_fused_mhc builds the decode program."""
        n = cfg.hc_mult
        scale = source.host(L, f"hc_{which}_scale").float().reshape(-1)
        per_mix = torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)])
        return cls(
            mesh_device,
            source.host(L, f"hc_{which}_fn").reshape(n * cfg.hidden_size, (2 + n) * n),
            per_mix,
            source.host(L, f"hc_{which}_base").float().reshape(-1),
            source.host(L, norm_name).float().reshape(-1),
            hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters,
            hidden=cfg.hidden_size, n=n, **kw)  # fmt: skip

    # ---------------------------------------------------------------------------------------------- placement
    def _layout(self, n_rb: int):
        """G = min(max groups, row blocks) groups of P members + a root on the first G (P + 1) worker cores (row-major
        over the compute grid). Returns (G, [(root, [members])]) of logical (x, y)."""
        g = min(self.max_groups, n_rb)
        gx = self.grid[0]
        cores = [(i % gx, i // gx) for i in range(g * (self.p + 1))]
        groups = []
        for k in range(g):
            blk = cores[k * (self.p + 1) : (k + 1) * (self.p + 1)]
            groups.append((blk[-1], blk[:-1]))
        return g, groups

    def _noc(self, xy):
        c = self.device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        return int(c.x), int(c.y)

    @staticmethod
    def _crs(cores):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])

    @staticmethod
    def _acc(t) -> list:
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    @staticmethod
    def _cb(index, dtype, page, pages, cores, alias=None):
        fmts = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
        if alias is not None:
            fmts.append(ttnn.CBFormatDescriptor(buffer_index=alias, data_format=dtype, page_size=page))
        return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=fmts)

    def l1_bytes(self, hn_dtype=BF16) -> dict:
        """CB bytes per member / root core (the L1 a caller must leave free below its buffers)."""
        t32, xt = 4096, N_STREAMS * self.cpc
        common = 2 * self.p * t32 + 512
        member = common + self.xbuf * xt * t32 + xt * 2048 + self.cpc * t32 + 2 * t32 + N_STREAMS * t32
        member += self.cpc * t32 + 2 * t32 + self.cpc * (t32 if hn_dtype == FP32 else 2048)
        root = common + 6 * t32 + 3 * t32
        return dict(member=member, root=root)

    def descriptor(self, x, hn, post, comb, n_rb: int):
        g, groups = self._layout(n_rb)
        members = [m for _, ms in groups for m in ms]
        roots = [r for r, _ in groups]
        m_crs, r_crs, all_crs = self._crs(members), self._crs(roots), self._crs(members + roots)
        t32, t16 = 4096, 2048
        hn_page = t32 if hn.dtype == FP32 else t16
        xt = N_STREAMS * self.cpc
        cbs = [
            self._cb(CB["GATHER"], FP32, t32, 2 * self.p, all_crs),
            self._cb(CB["MSG"], FP32, 512, 1, all_crs),
            self._cb(CB["X"], FP32, t32, self.xbuf * xt, m_crs, alias=CB["X32"]),
            self._cb(CB["FN"], BF16, t16, xt, m_crs),
            self._cb(CB["W"], FP32, t32, self.cpc, m_crs),
            self._cb(CB["PART"], FP32, t32, 2, m_crs),
            self._cb(CB["PRE"], FP32, t32, N_STREAMS, m_crs),
            self._cb(CB["H"], FP32, t32, self.cpc, m_crs, alias=CB["H16"] if self.fpu else None),
            self._cb(CB["S2"], FP32, t32, 1, m_crs),
            self._cb(CB["INV"], FP32, t32, 1, m_crs),
            self._cb(CB["OUT"], hn.dtype, hn_page, self.cpc, m_crs),
            self._cb(CB["RED"], FP32, t32, 1, r_crs),
            self._cb(CB["PLANE"], FP32, t32, 1, r_crs),
            self._cb(CB["COEF"], FP32, t32, 1, r_crs),
            self._cb(CB["SB"], FP32, t32, 3, r_crs),
            self._cb(CB["INVR"], FP32, t32, 1, r_crs),
            self._cb(CB["SCR"], FP32, t32, 2, r_crs),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(SEMAPHORES)]

        ms_rt, my_rt, mc_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rs_rt, ry_rt, rc_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, (root, ms) in enumerate(groups):
            rnoc = self._noc(root)
            for j, (cx, cy) in enumerate(ms):
                ms_rt[cx][cy] = [x.buffer_address(), self.fn.buffer_address(), self.w.buffer_address(),
                                 hn.buffer_address(), j * self.cpc, k, g, n_rb]  # fmt: skip
                my_rt[cx][cy] = [rnoc[0], rnoc[1], j, k, g, n_rb]
                mc_rt[cx][cy] = [k, g, n_rb]
            rx, ry = root
            rs_rt[rx][ry] = [self.sb.buffer_address()]
            ry_rt[rx][ry] = [post.buffer_address(), comb.buffer_address(), k, g, n_rb] + [
                v for m in ms for v in self._noc(m)]  # fmt: skip
            rc_rt[rx][ry] = [k, g, n_rb]

        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in FP32_COPY_CBS:
            if self.fpu and cb == CB["PRE"]:  # PF_FPU: the diagonal pre tiles are a matmul operand
                continue
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        compute_ct = [self.iters, f32_bits(self.hc_eps), f32_bits(1.0 / self.width), f32_bits(self.x_eps),
                      f32_bits(1.0 / self.hidden), f32_bits(self.h_eps), self.p, self.cpc]  # fmt: skip
        dm0 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                noc=ttnn.NOC.RISCV_0_default)  # fmt: skip
        dm1 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                noc=ttnn.NOC.RISCV_1_default)  # fmt: skip

        extra = (
            [("PF_REV", PF_REV)]
            + ([("PF_FPU", "1")] if self.fpu else [])
            + ([("PF_SKIP", str(self.skip))] if self.skip else [])
        )

        def kd(name, crs, ct, rt, role, cfg):
            return ttnn.KernelDescriptor(
                kernel_source=os.path.join(KERNEL_DIR, name),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs, compile_time_args=ct, defines=[(role, "1")] + extra, runtime_args=rt, config=cfg)  # fmt: skip

        kernels = [
            kd(
                "stream.cpp",
                m_crs,
                [self.cpc, self.xbuf] + self._acc(x) + self._acc(self.fn) + self._acc(self.w) + self._acc(hn),
                ms_rt,
                "PF_MEMBER",
                dm0,
            ),  # fmt: skip
            kd("sync.cpp", m_crs, [], my_rt, "PF_MEMBER", dm1),
            kd("compute.cpp", m_crs, compute_ct, mc_rt, "PF_MEMBER", cc),
            kd("stream.cpp", r_crs, self._acc(self.sb), rs_rt, "PF_ROOT", dm0),
            kd("sync.cpp", r_crs, [self.p] + self._acc(post) + self._acc(comb), ry_rt, "PF_ROOT", dm1),
            kd("compute.cpp", r_crs, compute_ct, rc_rt, "PF_ROOT", cc),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    def __call__(self, x, hn_dtype=BF16):
        """x [1, 1, T, 16384] fp32 TILE (DRAM) -> (hn [1, 1, T, 4096] hn_dtype, post [1, 1, T, 4], comb [1, 1, T, 16])"""
        t = int(x.shape[-2])
        assert x.dtype == FP32 and int(x.shape[-1]) == self.width and t % TILE == 0, (x.dtype, tuple(x.shape))
        dev = x.device()
        hn = ttnn.empty((1, 1, t, self.hidden), hn_dtype, ttnn.TILE_LAYOUT, dev, DRAM)
        post = ttnn.empty((1, 1, t, N_STREAMS), FP32, ttnn.TILE_LAYOUT, dev, DRAM)
        comb = ttnn.empty((1, 1, t, N_STREAMS * N_STREAMS), FP32, ttnn.TILE_LAYOUT, dev, DRAM)
        ttnn.generic_op([x, self.fn, self.w, self.sb, post, comb, hn], self.descriptor(x, hn, post, comb, t // TILE))
        return hn, post, comb


# ====================================================================================================== post-mix
POST_KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "mhc_post_pf7a")
# kernel revision define: bump on every kernel source edit (the JIT cache keys file-path kernels by path + defines)
# 2: coefficient columns 0 / 1 / 16 / 17 (lane 0 of each SFPLOAD slot; rev 1 put them at 0 / 8 / 16 / 24)
MPF7_REV = "5"  # 3: MPF7_DBG 5 (timing floor); 4: reader / writer BATCH units per barrier (variable pushes: watcher
# assert on d0 05:43Z); 5: fixed-size, BATCH-aligned batches


def mhc_post_prefill_7a(y, x, post, comb, dbg: int = 0, batch: int | None = None):
    """Stage 16's mhc_post_prefill (tt/prefill_tp.py: x' = post_s * y + sum_i C[i][s] x_i of the flat residual
    x [1, 1, T, 16384] fp32, y [1, 1, T, 4096] bf16 / fp32, post [1, 1, T, 4], comb [1, 1, T, 16] fp32) with the
    one-pass compute of kernels/mhc_post_pf7a (full-sync fp32 DST, one SFPU pass per output tile: fp32 FMAs) -> the new
    flat residual [1, 1, T, 16384] fp32. Same unit split and data movement as Stage 16's program. dbg (bisect, compute
    define MPF7_DBG): 1 every output = x_0, 2 / 3 output s = post_s * y (the custom pass / the LLK's broadcast multiply),
    5 no copies / math (timing floor: data movement + pack)."""
    t = int(x.shape[-2])
    hidden = int(x.shape[-1]) // N_STREAMS
    ct = hidden // TILE
    units = (t // TILE) * ct
    dev = x.device()
    grid = dev.compute_with_storage_grid_size()
    # GLM53_PF_MHC_POST7A_BATCH units per reader / writer barrier (the CBs hold two batches); every core gets a
    # BATCH-aligned multiple of BATCH units (fixed-size pushes that tile the CB capacity)
    b_ = int(batch or os.environ.get("GLM53_PF_MHC_POST7A_BATCH", "4"))
    assert ct % b_ == 0 and units % b_ == 0, (ct, units, b_)
    ncores = max(1, min(units // b_, grid.x * grid.y))
    per = -(-units // (ncores * b_)) * b_
    ncores = -(-units // per)
    cores = [(i % grid.x, i // grid.x) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(a, b)) for a, b in cores])
    out = ttnn.empty((1, 1, t, N_STREAMS * hidden), FP32, ttnn.TILE_LAYOUT, dev, DRAM)
    ypage = 2048 if y.dtype == BF16 else 4096
    rr, rc, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (a, b) in enumerate(cores):
        u0 = min(i * per, units)
        n = min(per, units - u0)
        rr[a][b] = [y.buffer_address(), x.buffer_address(), post.buffer_address(), comb.buffer_address(), u0, n]
        rc[a][b] = [u0, n]
        rw[a][b] = [out.buffer_address(), u0, n]
    src = lambda name: os.path.join(POST_KERNEL_DIR, f"{name}.cpp")  # noqa: E731
    dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                      fp32_dest_acc_en=True, dst_full_sync_en=True)  # fmt: skip
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb_ in (1, 2) + ((0,) if y.dtype == FP32 else ()):  # CB_X, CB_COEF (+ an fp32 y): exact fp32 copies
        modes[cb_] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
    kd = lambda name, ct_, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=src(name), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct_, runtime_args=rt, defines=[("MPF7_REV", MPF7_REV)] + ([("MPF7_DBG", str(dbg))] if dbg else []),
        config=cfg)  # fmt: skip
    kernels = [kd("reader", [ct, ypage, b_], rr, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
               kd("compute", [ct], rc, cc),
               kd("writer", [ct, b_], rw, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))]  # fmt: skip
    cb = lambda i, dt, page, n: ttnn.CBDescriptor(total_size=page * n, core_ranges=crs, format_descriptors=[  # noqa: E731
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
    cbs = [cb(0, y.dtype, ypage, 2 * b_), cb(1, FP32, 4096, 2 * N_STREAMS * b_), cb(2, FP32, 4096, 2 * N_STREAMS),
           cb(3, FP32, 4096, 2), cb(4, FP32, 4096, 2 * N_STREAMS * b_)]  # fmt: skip
    ttnn.generic_op([y, x, post, comb, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out
