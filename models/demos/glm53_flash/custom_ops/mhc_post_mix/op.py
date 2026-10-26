# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MhcPostMix``: the fused exact fp32 mHC residual update as a model-local ``ttnn.generic_op`` (kernels in
``kernels/``; contract in README.md).

One program replaces the composite post-block (block-diagonal matmul + addcmul + the collapse masks):

    out[t, j, c] = sum_i comb[t, i*n + j] * x[t, i, c] + post[t, j] * y[t, c]          (streams)
    out[t, c]    = (1/n) * sum_j out[t, j, c]                                          (collapse)

on the packed residual streams ``x [1,1,T,n*C]`` (fp32 TILE, stream i in columns ``i*C:(i+1)*C``), the
sublayer output ``y [1,1,T,C]`` and the Sinkhorn coefficients ``post [T,n]`` / ``comb [T,n*n]`` (fp32 TILE,
one tile wide). Every multiply/add runs on the SFPU in fp32 with the inputs unpacked straight into the fp32
dest (no 19-bit SrcA rounding), so the result matches the composite fp32 chain to ~3e-8. The work unit is a
(token-tile, column-tile) position, dealt over the compute grid like ``split_work_to_cores``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
NUM_CBS = 64  # unpack_to_dest_mode entries (>= the number of CBs the build knows)
FP32_TILE_BYTES = TILE * TILE * 4
# the "extras" work split (v2, streams mode): a core's work unit is a (token-tile, column-tile) position = n output
# streams; when the positions do not divide over the grid (128 on a 12x10 grid: 8 cores take two), the heaviest cores
# hand the n output streams of their last position to lightly loaded cores as (position, stream) "extras", each read
# like a position but evaluated and written for one stream. The define MHC_MAX_EXTRA bounds a core's extras (the runtime
# args are padded to it: one program per split mode, whatever the plan); the cost model deals them (`delegate_units`).
POST_MAX_EXTRA = 4
POST_UNIT_COST = 6  # one position: n = 4 stream passes + ~2 passes' worth of reads / writes / dest loads
POST_EXTRA_COST = 3  # one extra: one stream pass + the same reads / loads
SPLIT_MODES = ("tile", "extras")


def delegate_units(
    counts: list[int],
    n_parts: int,
    unit_cost: int,
    extra_cost: int,
    max_extra: int,
    eligible: list[bool],
    *,
    prefer_last: bool = True,
):
    """Rebalance a per-core unit count: the cores with the most units ("donors") hand the ``n_parts`` parts of their
    LAST unit to eligible non-donor cores (at most ``max_extra`` parts per recipient, the cheapest recipient first,
    ties to the last / first core by ``prefer_last``). The number of donating cores ``d`` is chosen to minimise the
    critical path ``max_k(count_k * unit_cost + extras_k * extra_cost)`` (ties: the fewest donors), so a plan that
    does not help (prefill-sized unit counts, a grid of all-donor cores) stays whole.
    Returns ``(critical_path, new_counts, extras)``: ``extras[k]`` = list of ``(donor core, part)``; ``new_counts[k]``
    is ``counts[k] - 1`` for a donor. Device-free; the pre-block and post-mix planners share it."""
    ncores = len(counts)
    assert len(eligible) == ncores and n_parts >= 1 and max_extra >= 0
    base = [c * unit_cost for c in counts]
    best = (max(base) if base else 0, list(counts), [[] for _ in range(ncores)])
    if ncores == 0 or max(counts) == 0 or max_extra == 0:
        return best
    top = max(counts)
    donors = [k for k in range(ncores) if counts[k] == top]
    donor_set = set(donors)
    for d in range(1, len(donors) + 1):
        cost = list(base)
        new_counts = list(counts)
        extras = [[] for _ in range(ncores)]
        for owner in donors[:d]:
            cost[owner] -= unit_cost
            new_counts[owner] -= 1
        ok = True
        for owner in donors[:d]:
            for part in range(n_parts):
                cands = [k for k in range(ncores) if eligible[k] and k not in donor_set and len(extras[k]) < max_extra]
                if not cands:
                    ok = False
                    break
                k = min(cands, key=(lambda k: (cost[k], -k)) if prefer_last else (lambda k: (cost[k], k)))
                extras[k].append((owner, part))
                cost[k] += extra_cost
            if not ok:
                break
        if not ok:
            break
        crit = max(cost)
        if crit < best[0]:
            best = (crit, new_counts, extras)
    return best


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def split_work(total: int, grid) -> list[tuple[ttnn.CoreCoord, int, int]]:
    """``(core, start, count)`` per core, column-major over the grid, the first ``total % ncores`` cores
    taking one unit more -- the same deal as tt-metal's ``split_work_to_cores``."""
    ncores = min(total, grid.x * grid.y)
    base, rem = divmod(total, ncores)
    out, start = [], 0
    for i in range(ncores):
        count = base + (1 if i < rem else 0)
        out.append((ttnn.CoreCoord(i // grid.y, i % grid.y), start, count))
        start += count
    return out


_PLAN_CACHE: dict[tuple, tuple] = {}


def plan_positions(total: int, grid, n: int, split: str = "tile", allow_extras: bool = True):
    """The v2 work plan: ``(core, start, count, extras)`` per core, ``extras`` = list of ``(position, stream j)``.
    ``split="tile"``: ``split_work`` as is; ``"extras"``: ``delegate_units`` over it (``POST_*`` cost model) when
    ``allow_extras`` (streams mode) -- a position's n streams go to n cores at most ``POST_MAX_EXTRA`` each, from the
    last cores of the grid. Returns the plan and its critical path in cost units (``(plan, crit)``). Cached per
    ``(total, grid, n, split, allow_extras)``: the plan is a pure function of the static shape, computed once per shape
    on the host (the eager warm-up), never inside a captured call."""
    assert split in SPLIT_MODES, split
    key = (int(total), int(grid.x), int(grid.y), int(n), split, bool(allow_extras))
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        return hit
    plan = _plan_positions(total, grid, n, split, allow_extras)
    _PLAN_CACHE[key] = plan
    return plan


def _plan_positions(total: int, grid, n: int, split: str, allow_extras: bool):
    work = split_work(total, grid)
    counts = [w[2] for w in work]
    if split == "tile" or not allow_extras or not work:
        return [(core, start, count, []) for core, start, count in work], (
            max(counts) if counts else 0
        ) * POST_UNIT_COST
    crit, new_counts, extras = delegate_units(
        counts, n, POST_UNIT_COST, POST_EXTRA_COST, POST_MAX_EXTRA, [True] * len(work), prefer_last=True
    )
    plan = []
    for k, (core, start, count) in enumerate(work):
        ex = [(work[owner][1] + work[owner][2] - 1, part) for owner, part in extras[k]]
        plan.append((core, start, new_counts[k], ex))
    return plan, crit


def extra_args(extras: list[tuple[int, int]], max_extra: int) -> list[int]:
    """``[n_extra, p0, j0, p1, j1, ...]`` padded to ``max_extra`` pairs (a fixed runtime-arg count per program)."""
    assert len(extras) <= max_extra, (len(extras), max_extra)
    flat = [len(extras)]
    for p, j in extras:
        flat += [int(p), int(j)]
    flat += [0, 0] * (max_extra - len(extras))
    return flat


def core_range_set(cores: list[ttnn.CoreCoord], grid) -> ttnn.CoreRangeSet:
    """The column-major prefix of the grid as (at most) two rectangles."""
    n = len(cores)
    full_cols, rem = divmod(n, grid.y)
    ranges = []
    if full_cols:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_cols - 1, grid.y - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(full_cols, 0), ttnn.CoreCoord(full_cols, rem - 1)))
    return ttnn.CoreRangeSet(ranges)


class MhcPostMix:
    """Builds and launches the fused post-block program. Call ``__call__`` with a pre-allocated output
    (``allocate_output`` gives one in the residual streams' memory config)."""

    # CB plan (kept in sync with the kernels): fp32 tiles, the work unit is one (token-tile r, column-tile c)
    CB_X = 0  # n stream tiles x[r, i, c] per position, double buffered
    CB_Y = 1  # y[r, c] per position, double buffered
    CB_RAW = 2  # reader scratch: the comb and post tiles of token-tile r
    CB_COL = 3  # n*n + n coefficient column tiles (coefficient in column 0)
    CB_BCAST = 4  # the same, broadcast across the 32 columns (made by the compute kernel)
    CB_OUT = 16  # n (or 1) output tiles per position, double buffered

    # v2 CB plan: X, Y as above; RAW (c_2) the raw comb + post tiles of a token-tile (expanded by the compute
    # kernel); COEF (c_3) the n + 1 lane-layout coefficient tiles (comb of stream j: comb[:, i*n+j] in column
    # group i, post[:, j] in column group j; group m = face m//2 (columns 0-15 / 16-31), column parity m%2 --
    # the column sets one SFPU slot load covers, see kernels/mhc_sfpu_x8.h); OUT as above.
    CB_COEF = 3

    def __init__(self, n: int = 4, math_fidelity=ttnn.MathFidelity.HiFi4, version: int = 1, split: str = "tile"):
        """``split``: ``"tile"`` (one work unit per (token-tile, column-tile) position, today's deal) or ``"extras"``
        (v2 streams mode: the heaviest cores' last positions split by output stream over lightly loaded cores,
        ``plan_positions``; the kernels are compiled with ``MHC_MAX_EXTRA`` and take the extra units after their
        positions). ``GLM53_MHC_POST_SPLIT`` in tt/mhc.py."""
        assert split in SPLIT_MODES, split
        self.n = n
        self.fidelity = math_fidelity
        self.version = version
        self.split = split

    # ------------------------------------------------------------------ reference
    @staticmethod
    def golden(x: torch.Tensor, y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, n: int, collapse: bool):
        """x [T, n*C], y [T, C], post [T, n], comb [T, n*n] (column i*n+j = comb[i,j]) -> [T, n*C] or [T, C]
        in float64."""
        T, C = y.shape
        xs = x.double().reshape(T, n, C)
        cm = comb.double().reshape(T, n, n)
        out = torch.einsum("tij,tic->tjc", cm, xs) + post.double()[:, :, None] * y.double()[:, None, :]
        return out.mean(1) if collapse else out.reshape(T, n * C)

    # ------------------------------------------------------------------ launch
    def output_shape(self, x: ttnn.Tensor, collapse: bool) -> list[int]:
        shape = list(x.shape)
        if collapse:
            shape[-1] //= self.n
        return shape

    def allocate_output(self, x: ttnn.Tensor, collapse: bool = False) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(self.output_shape(x, collapse)), ttnn.float32, ttnn.TILE_LAYOUT, x.device(), x.memory_config()
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        y: ttnn.Tensor,
        post: ttnn.Tensor,
        comb: ttnn.Tensor,
        out: ttnn.Tensor,
        collapse: bool = False,
    ) -> ttnn.Tensor:
        if self.version == 2:
            return self._call_v2(x, y, post, comb, out, collapse)
        n = self.n
        for name, t in (("x", x), ("y", y), ("post", post), ("comb", comb), ("out", out)):
            assert t.dtype == ttnn.float32, f"mhc_post_mix: {name} must be float32 (got {t.dtype})"
            assert t.layout == ttnn.TILE_LAYOUT, f"mhc_post_mix: {name} must be TILE layout"
            assert not t.memory_config().is_sharded(), f"mhc_post_mix: {name} must be interleaved"
        T, Cn = x.padded_shape[-2], x.padded_shape[-1]
        C = y.padded_shape[-1]
        assert T % TILE == 0 and C % TILE == 0, f"mhc_post_mix: T={T}, C={C} must be tile multiples"
        assert Cn == n * C, f"mhc_post_mix: x width {Cn} != n*C = {n * C}"
        assert y.padded_shape[-2] == T and post.padded_shape[-2] == T and comb.padded_shape[-2] == T
        assert post.padded_shape[-1] == TILE and comb.padded_shape[-1] == TILE, "post/comb must be one tile wide"
        assert post.shape[-1] == n and comb.shape[-1] == n * n, "post [T,n], comb [T,n*n]"
        assert list(out.padded_shape)[-2:] == [T, C if collapse else Cn], f"out {out.padded_shape} shape mismatch"

        R, Ct = T // TILE, C // TILE
        n_coef = n * n + n
        out_tiles = 1 if collapse else n
        grid = x.device().compute_with_storage_grid_size()
        work = split_work(R * Ct, grid)
        cores = core_range_set([w[0] for w in work], grid)

        def cb(index, tiles):
            return ttnn.CBDescriptor(
                total_size=tiles * FP32_TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=FP32_TILE_BYTES)
                ],
            )

        cbs = [
            cb(self.CB_X, 2 * n),
            cb(self.CB_Y, 2),
            cb(self.CB_RAW, 2),
            cb(self.CB_COL, n_coef),
            cb(self.CB_BCAST, n_coef),
            cb(self.CB_OUT, 2 * out_tiles),
        ]

        reader_ct = [self.CB_X, self.CB_Y, self.CB_RAW, self.CB_COL, n, Ct]
        for t in (x, y, post, comb):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [self.CB_OUT, n, Ct, int(collapse)] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        compute_ct = [
            self.CB_X,
            self.CB_Y,
            self.CB_COL,
            self.CB_BCAST,
            self.CB_OUT,
            n,
            Ct,
            int(collapse),
            f32_bits(1.0 / n),
        ]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, start, count in work:
            reader_rt[core.x][core.y] = [
                x.buffer_address(),
                y.buffer_address(),
                post.buffer_address(),
                comb.buffer_address(),
                start,
                count,
            ]
            writer_rt[core.x][core.y] = [out.buffer_address(), start, count]
            compute_rt[core.x][core.y] = [start, count]

        # The fp32 inputs must be unpacked straight into the fp32 dest (SrcA/SrcB hold 19 bits): this drives
        # copy_tile's unpack-to-dest and the 32-bit path of the column-broadcast datacopy.
        compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        for index in (self.CB_X, self.CB_Y, self.CB_COL, self.CB_BCAST):
            modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_mhc_post_mix.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_mhc_post_mix.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_mhc_post_mix.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, y, post, comb, out], program)

    # ------------------------------------------------------------------ v2
    def h_mix(self, x: ttnn.Tensor, pre: ttnn.Tensor, out: ttnn.Tensor) -> ttnn.Tensor:
        """The pre-mix h = sum_i pre_i x_i (the sublayer input before its norm), exact fp32: ``x [1,1,T,n*C]``,
        ``pre [T,n]`` (raw, padded tile) or its lane layout ``[T,32]``, ``out [1,1,T,C]`` fp32 pre-allocated.
        The v2 kernels in HMODE (one output tile per position, no y)."""
        return self._call_v2(x, x, pre, x, out, collapse=True, hmode=True)

    def allocate_h(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.allocate_output(x, collapse=True)

    def _call_v2(self, x, y, post, comb, out, collapse, hmode=False):
        """v2: resident operands in the 8-slot fp32 dest, lane-layout coefficient tiles, hand-written SFPU mix;
        y may be the bf16 sublayer output. ``post``/``comb`` either raw (``[T,n]`` / ``[T,n*n]``, expanded on
        device once per token-tile) or already in the lane layout (``[T,32]`` / ``[T,n*32]``)."""
        n = self.n
        assert n in (2, 4), "v2 pairs coefficients in the expansion (n even, <= 4)"
        for name, t in (("x", x), ("post", post), ("comb", comb), ("out", out)):
            assert t.dtype == ttnn.float32, f"mhc_post_mix: {name} must be float32 (got {t.dtype})"
        assert y.dtype in (ttnn.float32, ttnn.bfloat16), f"mhc_post_mix: y must be float32 or bfloat16 ({y.dtype})"
        for name, t in (("x", x), ("y", y), ("post", post), ("comb", comb), ("out", out)):
            assert t.layout == ttnn.TILE_LAYOUT, f"mhc_post_mix: {name} must be TILE layout"
            assert not t.memory_config().is_sharded(), f"mhc_post_mix: {name} must be interleaved"
        T, Cn = x.padded_shape[-2], x.padded_shape[-1]
        C = Cn // n if hmode else y.padded_shape[-1]  # hmode: no y (x stands in for the unused accessors)
        assert T % TILE == 0 and C % TILE == 0, f"mhc_post_mix: T={T}, C={C} must be tile multiples"
        assert Cn == n * C, f"mhc_post_mix: x width {Cn} != n*C = {n * C}"
        assert y.padded_shape[-2] == T and post.padded_shape[-2] == T and comb.padded_shape[-2] == T
        if hmode:
            expand = post.shape[-1] == n  # raw pre [T,n] (expanded on device) or its lane layout [T,32]
            assert post.padded_shape[-1] == TILE, "pre must be one tile wide"
        elif post.shape[-1] == n and comb.shape[-1] == n * n:
            expand = True
            assert post.padded_shape[-1] == TILE and comb.padded_shape[-1] == TILE, "raw post/comb: one tile wide"
        else:
            expand = False
            assert (
                post.padded_shape[-1] == TILE and comb.padded_shape[-1] == n * TILE
            ), f"lane-layout post/comb must be [T,32] / [T,{n * TILE}] (got {post.padded_shape}, {comb.padded_shape})"
        assert list(out.padded_shape)[-2:] == [T, C if collapse else Cn], f"out {out.padded_shape} shape mismatch"

        R, Ct = T // TILE, C // TILE
        out_tiles = 1 if collapse else n
        y_bytes = FP32_TILE_BYTES if y.dtype == ttnn.float32 else FP32_TILE_BYTES // 2
        grid = x.device().compute_with_storage_grid_size()
        # the extras split serves the streams mode only (a collapsed / h-mode position has one output tile)
        use_extras = self.split == "extras" and not collapse and not hmode
        work, _crit = plan_positions(R * Ct, grid, n, self.split, allow_extras=use_extras)
        cores = core_range_set([w[0] for w in work], grid)
        defines = [("MHC_MAX_EXTRA", str(POST_MAX_EXTRA))] if use_extras else []

        def cb(index, tiles, dtype=ttnn.float32, page=FP32_TILE_BYTES):
            return ttnn.CBDescriptor(
                total_size=tiles * page,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
            )

        cbs = [
            cb(self.CB_X, 2 * n),
            cb(self.CB_Y, 2, y.dtype, y_bytes),
            cb(self.CB_RAW, 1 if hmode else 2),
            cb(self.CB_COEF, 1 if hmode else n + 1),
            cb(self.CB_OUT, 2 * out_tiles),
        ]

        reader_ct = [self.CB_X, self.CB_Y, self.CB_RAW, self.CB_COEF, n, Ct, int(expand), int(hmode)]
        for t in (x, y, post, comb):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [self.CB_OUT, n, Ct, int(collapse)] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        compute_ct = [
            self.CB_X,
            self.CB_Y,
            self.CB_RAW,
            self.CB_COEF,
            self.CB_OUT,
            n,
            Ct,
            int(collapse),
            f32_bits(1.0 / n),
            int(expand),
            int(hmode),
        ]

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, start, count, extras in work:
            tail = extra_args(extras, POST_MAX_EXTRA) if use_extras else []
            reader_rt[core.x][core.y] = [
                x.buffer_address(),
                y.buffer_address(),
                post.buffer_address(),
                comb.buffer_address(),
                start,
                count,
            ] + tail
            writer_rt[core.x][core.y] = [out.buffer_address(), start, count] + tail
            compute_rt[core.x][core.y] = [start, count] + tail

        # fp32 operands unpack straight into the fp32 dest (SrcA holds 19 bits); dst_full_sync_en: the whole
        # 8-slot fp32 dest per acquire, kept resident across the per-stream packs
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=self.fidelity, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        for index in (self.CB_X, self.CB_RAW, self.CB_COEF):
            modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if y.dtype == ttnn.float32:
            modes[self.CB_Y] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_mhc_post_mix_v2.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_mhc_post_mix_v2.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_mhc_post_mix_v2.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
                compiler_include_paths=[str(KERNEL_DIR)],
                defines=defines,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, y, post, comb, out], program)
