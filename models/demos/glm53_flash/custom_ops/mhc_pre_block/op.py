# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MhcPreBlock``: the fused exact fp32 mHC pre-block for one token-tile (decode, T <= 32) as a model-local
``ttnn.generic_op`` (kernels in ``kernels/``; contract in README.md).

One program on the whole compute grid does what the composite's 20-odd ops do per wrap:

    mixes = (x @ fn^T) * rsqrt(sum_c x^2 + eps*nC)       exact: x split into 3 bf16 parts, fp32 accumulation
    pre, post, comb = Sinkhorn(mixes)                    the mhc_split_sinkhorn tile sequence, on one core
    h = sum_i pre_i x_i                                   exact fp32 (SFPU FMAs)
    hn = h * rsqrt(mean_c h^2 + norm_eps) * w             -> bf16, the sublayer input

Core k owns column tile(s) of the C-wide streams (``plan_work``); the two reductions over C (the projection and
the sum of squares, then the sum of h^2) run up a two-level tree of exact SFPU adds (column leaders, coordinator),
the coordinator finishes mixes, runs the Sinkhorn, writes post / comb in the post-block kernel's lane layout and
fans the pre tile (then the rsqrt tile) back out. Every core is independent per row (pad rows stay garbage).

Mesh notes (audit 2026-10-22, _agents/mhc_a1_audit/AUDIT.md): every NoC target of the kernels comes from the
coordinate tables built here (``worker_core_from_logical_core`` = Blackhole translated coordinates, identical on
every chip of a mesh and validated equal across the mesh's devices by ``MeshDevice``), every remote address is a
circular-buffer or semaphore L1 address of the same program (one address on every core), and every op-held tensor
is a mesh-replicated ``from_torch`` upload (one address on every device). The grid and the tables belong to the
device object the op was built for (``__call__`` checks the device id).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import ttnn
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import KERNEL_DIR as POST_KERNEL_DIR
from models.demos.glm53_flash.custom_ops.mhc_post_mix.op import NUM_CBS, delegate_units, f32_bits

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
    # the compute kernel reads the CB ids in this order (compile-time args 0..26), then the parameters
    COMPUTE_CB_ORDER = (
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
    )
    FREP = 3  # row-replicated fn tiles per stream tile (exact SFPU projection)
    MAX_COLS = (
        2  # column tiles per core (C/32 = 128 column tiles need >= 64 cores; a 12x10 grid gives 8 two-column cores)
    )
    # the "extras" work split (`split="extras"`, kernels compiled with MHC_MAX_EXTRA): a two-column core projects only
    # its first column; the second column's n stream tiles are projected as single "extra" tiles on single-column,
    # non-leader cores (at most MAX_EXTRA per core; `plan_work_extras`). The h / norm phases keep both columns on the
    # owner. Cost model in split_x / project passes: a column = its 4 tiles + a fixed part, an extra = 1 tile + the
    # fixed part -> 12x10: the critical path 10 -> 7 (32 extras on 32 of the 112 single-column cores); 13x10 (no
    # two-column cores) and 8x8 (every core two columns, no recipient) keep the column plan.
    MAX_EXTRA = 2
    COL_COST, EXTRA_COST = 5, 2
    SINKHORN_MODES = ("llk", "fused")
    SPLIT_MODES = ("cols", "extras")
    SEM_ARRIVE1, SEM_ARRIVE2, SEM_PRE, SEM_RS2 = 0, 1, 2, 3
    # kernel ABI (checked against the kernel sources by tests/custom_ops/test_mhc_pre_block_host.py):
    # reader compile-time args: 9 CB ids + n, gy, gx, Ct + 4 semaphore ids = 17, then the NoC y table (gy entries),
    # then frep, the mask CB id, then the five TensorAccessorArgs; writer: 10 CB ids + 4 + 4 = 18, then the NoC x
    # table (gx entries), the NoC y of row 0, then three TensorAccessorArgs; compute: 27 CB ids, 8 parameters,
    # PARTS, DBG, debug, exact, MASK.
    READER_NOC_Y_BASE = 17
    WRITER_NOC_X_BASE = 18
    COMPUTE_PARAM_BASE = 27

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
        sinkhorn: str = "llk",
        split: str = "cols",
    ):
        """``fn [(2+n)*n, n*C]`` (bf16-exact values), ``consts`` the Sinkhorn constants ``[8,32,32]`` fp32 on
        device (``build_consts``), ``norm_weight [C]`` the sublayer's RMSNorm weight; ``norm_eps`` is the
        hyper-connection's projection eps, ``rms_eps`` the sublayer norm's. Builds device uploads: never call
        inside a trace capture (the model builds the op in its eager warm-up step).
        ``sinkhorn``: ``"llk"`` (the mhc_split_sinkhorn tile sequence, one LLK op per CB hop) or ``"fused"`` (one dest
        pass per normalisation, compile define MHC_SINKHORN_FUSED); ``split``: ``"cols"`` (column tiles only) or
        ``"extras"`` (delegated stream tiles, define MHC_MAX_EXTRA; FPU projection only). Both change the compiled
        program (its defines), neither its tensors or ABI; the defaults are today's program."""
        assert sinkhorn in self.SINKHORN_MODES and split in self.SPLIT_MODES, (sinkhorn, split)
        self.device, self.n, self.C = device, n, C
        self.sinkhorn = sinkhorn
        self.split = split if not exact else "cols"  # the exact SFPU projection has no extras path
        self.device_id = int(device.id())
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
        # the norm weight: kept on the host for the golden (a device read-back of a mesh tensor needs a composer)
        self.w_host = norm_weight.float().reshape(-1).clone()
        assert self.w_host.numel() == C, (self.w_host.shape, C)
        w_rows = self.w_host.reshape(1, C).repeat(TILE, 1)
        self.w = ttnn.from_torch(
            w_rows.reshape(1, 1, TILE, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.grid = device.compute_with_storage_grid_size()
        self.gx, self.gy = self.grid.x, self.grid.y
        ncores = self.gx * self.gy
        assert ncores * self.MAX_COLS >= self.Ct, f"grid {self.gx}x{self.gy} < {self.Ct} column tiles / {self.MAX_COLS}"
        self.work = self.plan_work(ncores, self.Ct, self.MAX_COLS)  # (count, first column tile) per core k = x*gy + y
        # extras split: (count, c0, count_proj, [(stream, column), ...]) per core; max_extra = the kernels' define
        self.work_ext = None
        self.max_extra = 0
        if self.split == "extras":
            crit, plan = self.plan_work_extras(ncores, self.Ct, self.MAX_COLS, self.gy, self.MAX_EXTRA, n)
            if any(ex for _, _, _, ex in plan):
                self.work_ext, self.max_extra = plan, self.MAX_EXTRA
        # NoC coordinates of the logical grid rows / columns (translated coordinates: the same on every chip)
        self.noc_x = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x for x in range(self.gx)]
        self.noc_y = [device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y for y in range(self.gy)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, self.gy - 1))])

    # ------------------------------------------------------------------ host-side plan (device-free, testable)
    @staticmethod
    def plan_work(ncores: int, Ct: int, max_cols: int) -> list[tuple[int, int]]:
        """``(count, first column tile)`` per core ``k`` (column-major ``k = x*gy + y``): the first ``Ct - ncores``
        cores take two column tiles, the next take one, the rest (``k >= Ct``) are idle and contribute zero tiles
        to the gather tree. ``max_cols`` bounds the per-core count (the CB plan)."""
        assert ncores * max_cols >= Ct, (ncores, Ct, max_cols)
        extra = max(0, Ct - ncores)
        work, c = [], 0
        for k in range(ncores):
            cnt = 2 if k < extra else (1 if c < Ct else 0)
            assert cnt <= max_cols
            work.append((cnt, c))
            c += cnt
        assert c == Ct, (c, Ct)
        return work

    @classmethod
    def plan_work_extras(cls, ncores: int, Ct: int, max_cols: int, gy: int, max_extra: int, n: int):
        """``plan_work`` plus the delegation of the two-column cores' second columns: ``(critical path, plan)`` with
        ``plan[k] = (count, c0, count_proj, extras)`` -- ``count`` / ``c0`` as ``plan_work`` (the h / norm phases),
        ``count_proj`` the columns core ``k`` projects itself, ``extras`` the ``(stream i, column c)`` tiles it
        projects for others. Recipients: single-column cores that are neither a column leader (row 0: they sum their
        column's partials next) nor the coordinator, from the end of the grid; ``delegate_units`` picks how many
        two-column cores delegate (0 when it would not shorten the critical path)."""
        work = cls.plan_work(ncores, Ct, max_cols)
        counts = [c for c, _ in work]
        eligible = [counts[k] == 1 and (k % gy) != 0 for k in range(ncores)]
        crit, new_counts, extras = delegate_units(
            counts, n, cls.COL_COST, cls.EXTRA_COST, max_extra, eligible, prefer_last=True
        )
        plan = []
        for k, (count, c0) in enumerate(work):
            ex = [(part, work[owner][1] + work[owner][0] - 1) for owner, part in extras[k]]
            assert new_counts[k] in (count, count - 1) and (new_counts[k] == count or count == 2)
            plan.append((count, c0, new_counts[k], ex))
        return crit, plan

    @classmethod
    def cb_plan(cls, n: int, gx: int, gy: int, frep: int, max_extra: int = 0) -> dict[str, tuple[int, object]]:
        """CB name -> (tiles, dtype): the per-core circular buffers (every core allocates the whole plan).
        ``max_extra``: the extras split's per-core extra tiles (X / FN / PARTS grow by them)."""
        mc = cls.MAX_COLS
        return {
            "X": (n * mc + max_extra, ttnn.float32),
            "FN": ((n * mc + max_extra) * frep, ttnn.bfloat16),
            "MASK": (1, ttnn.float32),
            "HI": (n, ttnn.bfloat16),
            "LO": (n, ttnn.bfloat16),
            "LO2": (n, ttnn.bfloat16),
            "ZERO": (1, ttnn.float32),
            "PART": (2, ttnn.float32),
            "GATHER1": (gy, ttnn.float32),
            "GATHER2": (gx, ttnn.float32),
            "PRE": (1, ttnn.float32),
            "RS2": (1, ttnn.float32),
            "W": (mc, ttnn.bfloat16),
            "CONSTS": (8, ttnn.float32),
            "MIXES": (1, ttnn.float32),
            "SPRE": (1, ttnn.float32),
            "SPOST": (1, ttnn.float32),
            "SCOMB": (1, ttnn.float32),
            "MA": (2, ttnn.float32),
            "MB": (2, ttnn.float32),
            "RS": (2, ttnn.float32),
            "RECIP": (2, ttnn.float32),
            "TMP": (2, ttnn.float32),
            "COEF_OUT": (n + 2, ttnn.float32),
            "H": (mc, ttnn.float32),
            "HSQ": (2, ttnn.float32),
            "RS2OUT": (1, ttnn.float32),
            "OUT": (2, ttnn.bfloat16),
            "PARTS": (mc + (1 if max_extra else 0), ttnn.float32),
            "DBG": (3, ttnn.float32),
        }

    @classmethod
    def compile_time_args(
        cls,
        *,
        n: int,
        gx: int,
        gy: int,
        Ct: int,
        noc_x: list[int],
        noc_y: list[int],
        frep: int,
        iters: int,
        eps: float,
        norm_eps: float,
        C: int,
        rms_eps: float,
        debug: bool,
        exact: bool,
        reader_accessors: list[int],
        writer_accessors: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """The three kernels' compile-time arg vectors (without the device: the TensorAccessorArgs are passed in)."""
        CB = cls.CB
        sems = [cls.SEM_ARRIVE1, cls.SEM_ARRIVE2, cls.SEM_PRE, cls.SEM_RS2]
        reader_ct = (
            [CB[k] for k in ("X", "FN", "ZERO", "GATHER1", "GATHER2", "PRE", "RS2", "W", "CONSTS")]
            + [n, gy, gx, Ct]
            + sems
        )
        assert len(reader_ct) == cls.READER_NOC_Y_BASE
        reader_ct = reader_ct + list(noc_y) + [frep, CB["MASK"]] + list(reader_accessors)
        writer_ct = (
            [CB[k] for k in ("PART", "ZERO", "COEF_OUT", "HSQ", "RS2OUT", "OUT", "PRE", "RS2", "GATHER1", "GATHER2")]
            + [n, gy, gx, Ct]
            + sems
        )
        assert len(writer_ct) == cls.WRITER_NOC_X_BASE
        writer_ct = writer_ct + list(noc_x) + [noc_y[0]] + list(writer_accessors)
        compute_ct = [CB[k] for k in cls.COMPUTE_CB_ORDER]
        assert len(compute_ct) == cls.COMPUTE_PARAM_BASE
        compute_ct += [
            n,
            gy,
            gx,
            iters,
            f32_bits(eps),
            f32_bits(norm_eps * n * C),
            f32_bits(1.0 / C),
            f32_bits(rms_eps),
            CB["PARTS"],
            CB["DBG"],
            int(debug),
            int(exact),
            CB["MASK"],
        ]
        return reader_ct, writer_ct, compute_ct

    @staticmethod
    def runtime_args_for_core(
        *,
        xi: int,
        yi: int,
        count: int,
        c0: int,
        noc_x: list[int],
        noc_y: list[int],
        x_addr: int,
        fn_addr: int,
        consts_addr: int,
        w_addr: int,
        mask_addr: int,
        post_addr: int,
        comb_addr: int,
        hn_addr: int,
        count_proj: int | None = None,
        extras: tuple = (),
        max_extra: int = 0,
    ) -> tuple[list[int], list[int], list[int]]:
        """(reader, writer, compute) runtime args of logical core ``(xi, yi)``: buffer addresses, the work item, the
        leader / coordinator flags, the NoC coordinates of this column's leader (row 0) and of the coordinator
        (0, 0), and this core's slot in their gather buffers (its logical row / column). With ``max_extra > 0`` (the
        extras split) the reader gets ``[count_proj, n_extra, i0, c0, ...]`` (padded to ``max_extra`` pairs) and the
        compute kernel ``[count_proj, n_extra]`` appended; without, the vectors are the pre-split ones."""
        leader = int(yi == 0)
        coord = int(xi == 0 and yi == 0)
        reader = [x_addr, fn_addr, consts_addr, w_addr, count, c0, leader, coord, noc_x[xi], mask_addr]
        writer = [
            post_addr,
            comb_addr,
            hn_addr,
            count,
            c0,
            leader,
            coord,
            noc_x[xi],
            noc_y[0],
            yi,
            noc_x[0],
            noc_y[0],
            xi,
        ]
        compute = [count, leader, coord]
        if max_extra:
            assert len(extras) <= max_extra, (extras, max_extra)
            cp = count if count_proj is None else int(count_proj)
            tail = [cp, len(extras)]
            for i, c in extras:
                tail += [int(i), int(c)]
            tail += [0, 0] * (max_extra - len(extras))
            reader = reader + tail
            compute = compute + [cp, len(extras)]
        return reader, writer, compute

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
        w = self.w_host.to(torch.bfloat16).double()  # the device holds the bf16 rows of the same values
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
        n, gx, gy = self.n, self.gx, self.gy
        assert x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT and not x.memory_config().is_sharded()
        assert list(x.padded_shape)[-2:] == [TILE, self.nC], x.padded_shape
        assert hn.dtype == ttnn.bfloat16 and list(hn.padded_shape)[-2:] == [TILE, self.C], hn.padded_shape
        assert post.dtype == ttnn.float32 and list(post.padded_shape)[-2:] == [TILE, TILE], post.padded_shape
        assert comb.dtype == ttnn.float32 and list(comb.padded_shape)[-2:] == [TILE, n * TILE], comb.padded_shape
        # the grid, the NoC tables and the op's uploads belong to the device the op was built for
        assert int(x.device().id()) == self.device_id, (x.device().id(), self.device_id)
        CB = self.CB

        def cb(name):
            tiles, dtype = plan[name]
            page = FP32_TILE if dtype == ttnn.float32 else BF16_TILE
            return ttnn.CBDescriptor(
                total_size=tiles * page,
                core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB[name], data_format=dtype, page_size=page)],
            )

        plan = self.cb_plan(n, gx, gy, self.frep, self.max_extra)
        cbs = [cb(name) for name in plan]
        defines = []
        if self.sinkhorn == "fused":
            defines.append(("MHC_SINKHORN_FUSED", "1"))
        if self.max_extra:
            defines.append(("MHC_MAX_EXTRA", str(self.max_extra)))
        sems = [
            ttnn.SemaphoreDescriptor(s, ttnn.CoreType.WORKER, self.cores, 0)
            for s in (self.SEM_ARRIVE1, self.SEM_ARRIVE2, self.SEM_PRE, self.SEM_RS2)
        ]
        reader_acc, writer_acc = [], []
        for t in (x, self.fn, self.consts, self.w, self.mask):
            reader_acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        for t in (post, comb, hn):
            writer_acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_ct, writer_ct, compute_ct = self.compile_time_args(
            n=n,
            gx=gx,
            gy=gy,
            Ct=self.Ct,
            noc_x=self.noc_x,
            noc_y=self.noc_y,
            frep=self.frep,
            iters=self.iters,
            eps=self.eps,
            norm_eps=self.norm_eps,
            C=self.C,
            rms_eps=self.rms_eps,
            debug=self.debug,
            exact=self.exact,
            reader_accessors=reader_acc,
            writer_accessors=writer_acc,
        )

        addrs = dict(
            x_addr=x.buffer_address(),
            fn_addr=self.fn.buffer_address(),
            consts_addr=self.consts.buffer_address(),
            w_addr=self.w.buffer_address(),
            mask_addr=self.mask.buffer_address(),
            post_addr=post.buffer_address(),
            comb_addr=comb.buffer_address(),
            hn_addr=hn.buffer_address(),
        )
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for xi in range(gx):
            for yi in range(gy):
                k = xi * gy + yi
                count, c0 = self.work[k]
                ext = {}
                if self.work_ext is not None:
                    _, _, count_proj, extras = self.work_ext[k]
                    ext = dict(count_proj=count_proj, extras=tuple(extras), max_extra=self.max_extra)
                r, w, c = self.runtime_args_for_core(
                    xi=xi, yi=yi, count=count, c0=c0, noc_x=self.noc_x, noc_y=self.noc_y, **addrs, **ext
                )
                reader_rt[xi][yi] = r
                writer_rt[xi][yi] = w
                compute_rt[xi][yi] = c

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
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_mhc_pre_block.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_mhc_pre_block.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
                compiler_include_paths=[str(POST_KERNEL_DIR)],
                defines=defines,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        ttnn.generic_op([x, self.fn, self.consts, self.w, self.mask, post, comb, hn], program)
        return hn
