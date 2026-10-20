# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MhcPreBlock``: the fused exact fp32 mHC pre-block for one token-tile (decode, T <= 32) as a model-local
``ttnn.generic_op`` (kernels in ``kernels/``; contract in README.md).

One program on the whole compute grid does what the composite's 20-odd ops do per wrap:

    mixes = (x @ fn^T) * rsqrt(sum_c x^2 + eps*nC)       exact: x split into 3 bf16 parts, fp32 accumulation
    pre, post, comb = Sinkhorn(mixes)                    the mhc_split_sinkhorn tile sequence, on one core
    h = sum_i pre_i x_i                                   exact fp32 (SFPU FMAs)
    hn = h * rsqrt(mean_c h^2 + norm_eps) * w             -> bf16, the sublayer input

Core k < C/32 owns column tile k of the C-wide streams; the two reductions over C (the projection and the sum
of squares, then the sum of h^2) run up a two-level tree of exact SFPU adds (column leaders, coordinator), the
coordinator finishes mixes, runs the Sinkhorn, writes post / comb in the post-block kernel's lane layout and
fans the pre tile (then the rsqrt tile) back out. Every core is independent per row (pad rows stay garbage).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import ttnn
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import KERNEL_DIR as POST_KERNEL_DIR
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import NUM_CBS, f32_bits

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
FP32_TILE = TILE * TILE * 4
BF16_TILE = TILE * TILE * 2


def sinkhorn_reference(logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


class MhcPreBlock:
    """Builds and launches the fused pre-block program; the per-wrap constants are prepared once in ``__init__``.
    Call ``__call__`` with pre-allocated outputs (``allocate_outputs``)."""

    # CB plan (kept in sync with the kernels; ids are the compute kernel's compile-time args in this order)
    CB = dict(
        X=0,
        FN=1,
        HI=2,
        LO=3,
        LO2=4,
        ZERO=5,
        PART=6,
        GATHER1=7,
        GATHER2=8,
        PRE=9,
        RS2=10,
        W=11,
        CONSTS=12,
        MIXES=13,
        SPRE=14,
        SPOST=15,
        SCOMB=17,
        MA=18,
        MB=19,
        RS=20,
        RECIP=21,
        TMP=22,
        COEF_OUT=23,
        H=24,
        HSQ=25,
        RS2OUT=26,
        OUT=16,
        PARTS=27,
        DBG=28,
        MASK=29,
    )
    FREP = 3  # row-replicated fn tiles per stream tile (exact SFPU projection)
    MAX_COLS = 2  # column tiles per core (128 tiles over a 120-core grid)
    SEM_ARRIVE1, SEM_ARRIVE2, SEM_PRE, SEM_RS2 = 0, 1, 2, 3

    def __init__(
        self,
        device,
        fn: torch.Tensor,
        consts: ttnn.Tensor,
        norm_weight: torch.Tensor,
        *,
        n: int,
        C: int,
        sinkhorn_iters: int,
        eps: float,
        norm_eps: float,
        rms_eps: float,
        debug: bool = False,
        exact: bool = False,
    ):
        """``fn [(2+n)*n, n*C]`` (bf16-exact values), ``consts`` the Sinkhorn constants ``[8,32,32]`` fp32 on
        device (``build_consts``), ``norm_weight [C]`` the sublayer's RMSNorm weight; ``norm_eps`` is the
        hyper-connection's projection eps, ``rms_eps`` the sublayer norm's."""
        self.device, self.n, self.C = device, n, C
        self.nC = n * C
        self.Ct = C // TILE
        self.iters, self.eps, self.norm_eps, self.rms_eps = sinkhorn_iters, eps, norm_eps, rms_eps
        self.debug = debug  # comb tiles 2, 3 carry the coordinator's raw sum and mixes tiles instead
        self.exact = exact  # projection on the SFPU (exact fp32) instead of the FPU (composite-class 5e-5)
        mix_hc = (2 + n) * n
        assert n == 4 and C % TILE == 0 and tuple(fn.shape) == (mix_hc, self.nC), (n, C, fn.shape)
        fn_s = fn.float() * math.sqrt(self.nC)  # sqrt(nC) folded in (a power of two for nC = 16384: exact)
        assert torch.equal(fn_s.to(torch.bfloat16).float(), fn_s), "fn must be bf16-exact (checkpoint bf16)"
        if exact:
            # row-replicated layout for the SFPU projection: per stream tile (i, c) three tiles q, band r of tile q =
            # four copies of fn_s[:, 8q + r] over that tile's 32 features -> [n*Ct*3 tiles, 32]
            frep = torch.zeros(n, self.Ct, self.FREP, 8, 4, TILE)
            fs = fn_s.t().reshape(n, self.Ct, TILE, mix_hc)  # [i, c, f, m]
            for q in range(self.FREP):
                for r in range(8):
                    m = 8 * q + r
                    if m < mix_hc:
                        frep[:, :, q, r, :, :] = fs[:, :, :, m].reshape(n, self.Ct, 1, TILE)
            self.fn = ttnn.from_torch(
                frep.reshape(1, 1, n * self.Ct * self.FREP * TILE, TILE),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            mask = torch.zeros(TILE, TILE)
            for k in range(8):
                mask[4 * k : 4 * k + 4, 2 * k] = 1.0  # band k: the lane-k mask (even columns, lane k <-> column 2k)
            self.mask = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        else:
            fn_t = torch.zeros(self.nC, TILE)
            fn_t[:, :mix_hc] = fn_s.t()
            self.fn = ttnn.from_torch(
                fn_t.reshape(1, 1, self.nC, TILE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            self.mask = ttnn.from_torch(
                torch.zeros(TILE, TILE), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
            )
        self.frep = self.FREP if exact else 1
        self.consts = consts
        w_rows = norm_weight.float().reshape(1, C).repeat(TILE, 1)
        self.w = ttnn.from_torch(
            w_rows.reshape(1, 1, TILE, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.grid = device.compute_with_storage_grid_size()
        self.gx, self.gy = self.grid.x, self.grid.y
        ncores = self.gx * self.gy
        assert ncores * self.MAX_COLS >= self.Ct, f"grid {self.gx}x{self.gy} < {self.Ct} column tiles / {self.MAX_COLS}"
        # column tiles per core: the first `extra` cores take two (128 tiles over 120 cores -> 8 cores with two)
        extra = max(0, self.Ct - ncores)
        self.work = []  # (count, first column tile) per core k = x*gy + y
        c = 0
        for k in range(ncores):
            cnt = 2 if k < extra else (1 if c < self.Ct else 0)
            self.work.append((cnt, c))
            c += cnt
        assert c == self.Ct, (c, self.Ct)
        self.noc_x = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x for x in range(self.gx)]
        self.noc_y = [device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y for y in range(self.gy)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, self.gy - 1))])

    # ------------------------------------------------------------------ reference
    def golden(self, x: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor):
        """float64 HF math on packed ``x [T, n*C]``: mixes [T,24], pre, post [T,n], comb [T,n*n], h, hn [T,C]."""
        n, C = self.n, self.C
        xd = x.double()
        rs = torch.rsqrt(xd.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = (xd @ fn.double().t()) * rs
        b = base.double()
        pre = torch.sigmoid(mixes[:, :n] * float(scale[0]) + b[:n]) + self.eps
        post = 2.0 * torch.sigmoid(mixes[:, n : 2 * n] * float(scale[1]) + b[n : 2 * n])
        comb = sinkhorn_reference(
            (mixes[:, 2 * n :] * float(scale[2]) + b[2 * n :]).reshape(-1, n, n), self.iters, self.eps
        )
        h = (pre[:, :, None] * xd.reshape(-1, n, C)).sum(1)
        w = ttnn.to_torch(self.w)[0, 0, 0].double()
        hn = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.rms_eps) * w
        return mixes, pre, post, comb.reshape(-1, n * n), h, hn

    # ------------------------------------------------------------------ launch
    def allocate_outputs(self, x: ttnn.Tensor):
        dev = x.device()
        hn = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, self.C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, x.memory_config()
        )
        post = ttnn.allocate_tensor_on_device(
            ttnn.Shape([TILE, TILE]), ttnn.float32, ttnn.TILE_LAYOUT, dev, x.memory_config()
        )
        comb = ttnn.allocate_tensor_on_device(
            ttnn.Shape([TILE, self.n * TILE]), ttnn.float32, ttnn.TILE_LAYOUT, dev, x.memory_config()
        )
        return hn, post, comb

    def __call__(self, x: ttnn.Tensor, hn: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor) -> ttnn.Tensor:
        n, Ct, gx, gy = self.n, self.Ct, self.gx, self.gy
        assert x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT and not x.memory_config().is_sharded()
        assert list(x.padded_shape)[-2:] == [TILE, self.nC], x.padded_shape
        assert hn.dtype == ttnn.bfloat16 and list(hn.padded_shape)[-2:] == [TILE, self.C], hn.padded_shape
        assert post.dtype == ttnn.float32 and list(post.padded_shape)[-2:] == [TILE, TILE], post.padded_shape
        assert comb.dtype == ttnn.float32 and list(comb.padded_shape)[-2:] == [TILE, n * TILE], comb.padded_shape
        CB = self.CB

        def cb(index, tiles, dtype=ttnn.float32):
            page = FP32_TILE if dtype == ttnn.float32 else BF16_TILE
            return ttnn.CBDescriptor(
                total_size=tiles * page,
                core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
            )

        mc = self.MAX_COLS
        cbs = [
            cb(CB["X"], n * mc),
            cb(CB["FN"], n * mc * self.frep, ttnn.bfloat16),
            cb(CB["MASK"], 1),
            cb(CB["HI"], n, ttnn.bfloat16),
            cb(CB["LO"], n, ttnn.bfloat16),
            cb(CB["LO2"], n, ttnn.bfloat16),
            cb(CB["ZERO"], 1),
            cb(CB["PART"], 2),
            cb(CB["GATHER1"], gy),
            cb(CB["GATHER2"], gx),
            cb(CB["PRE"], 1),
            cb(CB["RS2"], 1),
            cb(CB["W"], mc, ttnn.bfloat16),
            cb(CB["CONSTS"], 8),
            cb(CB["MIXES"], 1),
            cb(CB["SPRE"], 1),
            cb(CB["SPOST"], 1),
            cb(CB["SCOMB"], 1),
            cb(CB["MA"], 2),
            cb(CB["MB"], 2),
            cb(CB["RS"], 2),
            cb(CB["RECIP"], 2),
            cb(CB["TMP"], 2),
            cb(CB["COEF_OUT"], n + 2),
            cb(CB["H"], mc),
            cb(CB["HSQ"], 2),
            cb(CB["RS2OUT"], 1),
            cb(CB["OUT"], 2, ttnn.bfloat16),
            cb(CB["PARTS"], mc),
            cb(CB["DBG"], 3),
        ]
        sems = [
            ttnn.SemaphoreDescriptor(s, ttnn.CoreType.WORKER, self.cores, 0)
            for s in (self.SEM_ARRIVE1, self.SEM_ARRIVE2, self.SEM_PRE, self.SEM_RS2)
        ]

        reader_ct = [
            CB["X"],
            CB["FN"],
            CB["ZERO"],
            CB["GATHER1"],
            CB["GATHER2"],
            CB["PRE"],
            CB["RS2"],
            CB["W"],
            CB["CONSTS"],
            n,
            gy,
            gx,
            Ct,
            self.SEM_ARRIVE1,
            self.SEM_ARRIVE2,
            self.SEM_PRE,
            self.SEM_RS2,
        ] + self.noc_y
        reader_ct += [self.frep, CB["MASK"]]
        for t in (x, self.fn, self.consts, self.w, self.mask):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = (
            [
                CB["PART"],
                CB["ZERO"],
                CB["COEF_OUT"],
                CB["HSQ"],
                CB["RS2OUT"],
                CB["OUT"],
                CB["PRE"],
                CB["RS2"],
                CB["GATHER1"],
                CB["GATHER2"],
                n,
                gy,
                gx,
                Ct,
                self.SEM_ARRIVE1,
                self.SEM_ARRIVE2,
                self.SEM_PRE,
                self.SEM_RS2,
            ]
            + self.noc_x
            + [self.noc_y[0]]
        )
        for t in (post, comb, hn):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        order = [
            "X",
            "FN",
            "HI",
            "LO",
            "LO2",
            "ZERO",
            "PART",
            "GATHER1",
            "GATHER2",
            "PRE",
            "RS2",
            "W",
            "CONSTS",
            "MIXES",
            "SPRE",
            "SPOST",
            "SCOMB",
            "MA",
            "MB",
            "RS",
            "RECIP",
            "TMP",
            "COEF_OUT",
            "H",
            "HSQ",
            "RS2OUT",
            "OUT",
        ]
        compute_ct = [CB[k] for k in order] + [
            n,
            gy,
            gx,
            self.iters,
            f32_bits(self.eps),
            f32_bits(self.norm_eps * self.nC),
            f32_bits(1.0 / self.C),
            f32_bits(self.rms_eps),
            CB["PARTS"],
            CB["DBG"],
            int(self.debug),
            int(self.exact),
            CB["MASK"],
        ]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for xi in range(gx):
            for yi in range(gy):
                count, c0 = self.work[xi * gy + yi]
                leader = int(yi == 0)
                coord = int(xi == 0 and yi == 0)
                reader_rt[xi][yi] = [
                    x.buffer_address(),
                    self.fn.buffer_address(),
                    self.consts.buffer_address(),
                    self.w.buffer_address(),
                    count,
                    c0,
                    leader,
                    coord,
                    self.noc_x[xi],
                    self.mask.buffer_address(),
                ]
                writer_rt[xi][yi] = [
                    post.buffer_address(),
                    comb.buffer_address(),
                    hn.buffer_address(),
                    count,
                    c0,
                    leader,
                    coord,
                    self.noc_x[xi],
                    self.noc_y[0],
                    yi,
                    self.noc_x[0],
                    self.noc_y[0],
                    xi,
                ]
                compute_rt[xi][yi] = [count, leader, coord]

        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        for k in (
            "X",
            "ZERO",
            "GATHER1",
            "GATHER2",
            "PRE",
            "RS2",
            "H",
            "SPRE",
            "SPOST",
            "SCOMB",
            "PARTS",
            "DBG",
            "MASK",
        ):
            modes[CB[k]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_mhc_pre_block.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_mhc_pre_block.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_mhc_pre_block.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
                compiler_include_paths=[str(POST_KERNEL_DIR)],
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        ttnn.generic_op([x, self.fn, self.consts, self.w, self.mask, post, comb, hn], program)
        return hn
