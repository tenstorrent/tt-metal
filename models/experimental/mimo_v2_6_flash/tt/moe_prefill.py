# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo MoE prefill experts on one TP=8 row (``TtMiMoMoE.forward``): x [1, 1, T, hidden] and every token's top-8
experts and routing weights -> this chip's partial output [1, 1, T, hidden], the routing-weighted sum over the
token's experts of its 256-wide slice of the intermediate dim. The caller all-reduces it across the row.

The experts are the layer's BFP4 tensors, shared with decode: w_gate_up [1, 256, hidden, 2 * 256] per chip
(columns [gate | up] of this chip's slice) and w_down [1, 256, 256, hidden], DRAM interleaved tiles.

Expert-major: the T x 8 (token, k) pairs are sorted by expert on the host (the routing is read back once per
call) and each expert's pairs are padded to whole 32-row groups, so every group belongs to one expert. The token
rows are gathered into that order (``embedding``) and every group runs gate|up -> SwiGLU x routing weight -> down
against its own expert; the pairs' rows are gathered back k-major and summed over k.

GroupedExperts runs the groups in one ``generic_op`` program: teams of 4 cores in a row, each team a list of
(expert, groups) jobs balanced on the host. A team core holds its slice of the job's expert in L1 for all of the
job's groups (weights are read once per job, not per group) and computes two intermediate columns and 32 output
columns; the team swaps h tiles over the NoC (kernels in tt/kernels/prefill/). Layouts it does not cover fall back to
an indexed sparse_matmul per group (<= 256 groups per call), which re-reads the expert's weights for every group.
"""

import heapq
import math

import torch

import ttnn

from .common import TILE
from .moe_stream import is_dram_interleaved, noc_coords
from .profiling import zone

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels/prefill"


class GroupedExperts:
    """The groups of every expert in one launch on teams of TEAM cores (see the module docstring).

    Bank b of a w_gate_up per-chip tensor holds gate column b and up column b of every expert, 2 * kt contiguous
    tiles per expert ([kt][gate, up]); bank b of w_down holds output columns b, b + 8, ..., ikt * cols contiguous
    tiles per expert. Team core s takes banks s and s + TEAM: intermediate columns s and s + TEAM and their output
    columns. Per job it reads those four contiguous runs into L1, per group the group's 128 x tiles and its routing
    weights, computes its two h tiles, sends them to every team core, and once all 8 are in, its 32 down tiles."""

    TEAM = 4
    PLAN_WORDS = 1024  # one fp32-tile-sized page per team: [jobs, (expert, first group, groups) ...]
    CB_PLAN, CB_WGU, CB_WD, CB_X, CB_RW, CB_H_OWN, CB_H_ALL, CB_Y, CB_WROW, CB_XRM, CB_XQ, CB_TOK, CB_DST = range(13)
    SEM_H, SEM_X = 0, 1

    def __init__(self, moe):
        device = moe.plan.mesh_device
        self.moe = moe
        self.banks = device.dram_grid_size().x
        assert self.banks == 2 * self.TEAM, self.banks
        self.kt = moe.hidden // TILE
        self.ikt = moe.inter_local // TILE
        assert self.ikt == self.banks, "intermediate tile column b comes from bank b"
        assert self.kt % self.banks == 0 and self.kt % self.TEAM == 0
        self.quarter_tiles = self.kt // self.TEAM  # x tiles each team core tilizes per group
        self.cols = self.kt // self.banks  # down output tile columns per bank
        assert self.cols % 4 == 0
        grid = device.compute_with_storage_grid_size()
        per_row = grid.x // self.TEAM
        assert per_row > 0
        self.cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(per_row * self.TEAM)]
        self.num_teams = len(self.cores) // self.TEAM
        self.noc = noc_coords(device, self.cores)
        assert self.noc is not None
        self.ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(per_row * self.TEAM - 1, grid.y - 1))]
        )
        self.semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.ranges, initial_value=0) for i in (0, 1)]
        self.w_dtype = moe.w_gate_up.dtype
        self.w_tile_bytes = ttnn.Tile([TILE, TILE]).get_tile_size(self.w_dtype)
        self.gu_tiles = 2 * self.kt
        self.wd_tiles = self.ikt * self.cols
        # LoFi: matmul unpacks the weights (in1) to srcA, and LoFi already takes srcA's hidden bit + 4 mantissa MSBs, all
        # of BFP4's 3 bits; HiFi2's second phase only adds srcA's lower bits (zero here). x / h in srcB get 6 mantissa
        # bits at both fidelities, so LoFi is bit-identical to HiFi2 with BFP4 weights (not with BFP8).
        lofi = self.w_dtype == ttnn.bfloat4_b
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi if lofi else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )
        self.defines = []  # (name, value) preprocessor defines for all three kernels (benchmark ablations)

    def _cbs(self):
        def cb(index, dtype, pages):
            tile = ttnn.Tile([TILE, TILE])
            page = tile.get_tile_size(dtype)
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=dtype, page_size=page, tile=ttnn.TileDescriptor(tile)
            )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.ranges, format_descriptors=[fmt])

        bf = ttnn.bfloat16
        # Two groups of x rows and of y rows with BFP4 weights. BFP8 weight slices take 400 KB more: one of each keeps
        # the static CBs below their end on dev (1.47 MB; the chunked prefill tests hold L1 buffers from 1.50 MB).
        slots = 2 if self.w_dtype == ttnn.bfloat4_b else 1
        # 16 decode users hold ~128 KB more persistent L1 (the fabric row gather of 16-row reductions): one group of x
        # rows keeps these CBs below it.
        x_slots = 1 if (self.moe.decode_users or 0) > 8 else slots
        return [
            cb(self.CB_PLAN, ttnn.float32, 1),
            cb(self.CB_WGU, self.w_dtype, 2 * self.gu_tiles),
            cb(self.CB_WD, self.w_dtype, 2 * self.wd_tiles),
            cb(self.CB_X, bf, self.kt),
            cb(self.CB_XRM, bf, x_slots * self.quarter_tiles),
            cb(self.CB_XQ, bf, self.quarter_tiles),
            cb(self.CB_RW, bf, 2),
            cb(self.CB_H_OWN, bf, 4),
            cb(self.CB_H_ALL, bf, 2 * self.ikt),
            # 32 rows x this core's 32 output tile columns, row-major
            cb(self.CB_Y, bf, slots * 2 * self.cols),
            ttnn.CBDescriptor(
                total_size=64,
                core_ranges=self.ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=self.CB_WROW, data_format=bf, page_size=64)],
            ),
            ttnn.CBDescriptor(
                total_size=128,
                core_ranges=self.ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=self.CB_TOK, data_format=ttnn.uint32, page_size=128)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=128,
                core_ranges=self.ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=self.CB_DST, data_format=ttnn.uint32, page_size=128)
                ],
            ),
        ]

    def plan(self, groups):
        """groups [E] (32-row groups per expert, expert-major) -> int32 [teams, PLAN_WORDS] team job lists: experts
        split into chunks of at most ceil(G / teams) groups, longest first onto the least loaded team."""
        total = int(groups.sum())
        cap = max(1, math.ceil(total / self.num_teams))
        first = (torch.cumsum(groups, 0) - groups).tolist()
        jobs = []
        for e, n in enumerate(groups.tolist()):
            for off in range(0, n, cap):
                jobs.append((min(cap, n - off), e, first[e] + off))
        jobs.sort(reverse=True)
        heap = [(0.0, t) for t in range(self.num_teams)]
        lists = [[0] for _ in range(self.num_teams)]
        for n, e, g0 in jobs:
            load, t = heapq.heappop(heap)
            lists[t] += (e, g0, n)
            lists[t][0] += 1
            heapq.heappush(heap, (load + n + 0.5, t))  # + about half a group to load the expert's weights
        table = [lst + [0] * (self.PLAN_WORDS - len(lst)) for lst in lists]
        assert all(len(lst) == self.PLAN_WORDS for lst in table)
        return torch.tensor(table, dtype=torch.int32)

    def __call__(self, x_rm, row_tokens, row_weights, row_dest, plan, out):
        """x_rm [T, H] bf16 row-major; row_tokens / row_dest [1, R] uint32 and row_weights [1, R] bf16 row-major (each
        packed row's token, its row k * T + t of ``out`` (0xFFFFFFFF: padding) and routing weight); plan uint32
        row-major [teams, PLAN_WORDS] (DRAM) -> fills ``out`` ([K * T, H] bf16 row-major) with every pair's weighted
        expert output."""
        moe = self.moe
        w_gu, w_d = moe.w_gate_up, moe.w_down
        for t in (w_gu, w_d):
            assert is_dram_interleaved(t) and t.layout == ttnn.TILE_LAYOUT
        for t in (x_rm, row_tokens, row_weights, row_dest, out):
            assert is_dram_interleaved(t) and t.layout == ttnn.ROW_MAJOR_LAYOUT
        assert x_rm.dtype == ttnn.bfloat16 and x_rm.shape[-1] == moe.hidden == out.shape[-1]
        assert row_weights.dtype == ttnn.bfloat16 and row_tokens.dtype == row_dest.dtype == ttnn.uint32
        assert row_weights.shape[-1] == row_tokens.shape[-1] == row_dest.shape[-1]
        assert w_gu.dtype == self.w_dtype and w_d.dtype == self.w_dtype

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(self.cores):
            team, s = divmod(i, self.TEAM)
            b0, b1 = s, s + self.TEAM
            reader_rt[core.x][core.y] = [
                plan.buffer_address(),
                team,
                b0,
                b1,
                w_gu.buffer_address(),
                w_d.buffer_address(),
                x_rm.buffer_address(),
                row_weights.buffer_address(),
                row_tokens.buffer_address(),
                s,
                row_tokens.shape[-1] * 4,
            ]
            peers = self.noc[team * self.TEAM : (team + 1) * self.TEAM]
            writer_rt[core.x][core.y] = (
                [b0, b1, out.buffer_address()]
                + [v for c in peers for v in (c.x, c.y)]
                + [s, row_dest.buffer_address(), row_dest.shape[-1] * 4]
            )
            compute_rt[core.x][core.y] = []

        def dm(name, ct, rt, processor, noc):
            return ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_DIR}/{name}",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.ranges,
                compile_time_args=ct,
                defines=self.defines,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )

        reader = dm(
            "experts_reader.cpp",
            [
                self.CB_PLAN,
                self.CB_WGU,
                self.CB_WD,
                self.CB_X,
                self.CB_RW,
                self.kt,
                0,
                self.gu_tiles,
                self.wd_tiles,
                self.w_tile_bytes,
                self.PLAN_WORDS * 4,
                self.CB_WROW,
                row_weights.shape[-1] * 2,
                self.CB_XRM,
                self.TEAM,
                self.CB_TOK,
            ],
            reader_rt,
            ttnn.DataMovementProcessor.RISCV_1,
            ttnn.NOC.RISCV_0_default,
        )
        writer = dm(
            "experts_writer.cpp",
            [
                self.CB_PLAN,
                self.CB_H_OWN,
                self.CB_H_ALL,
                self.CB_Y,
                self.TEAM,
                self.ikt,
                self.kt,
                self.cols,
                self.banks,
                self.SEM_H,
                self.CB_XQ,
                self.CB_X,
                self.SEM_X,
                self.CB_DST,
            ],
            writer_rt,
            ttnn.DataMovementProcessor.RISCV_0,
            ttnn.NOC.RISCV_1_default,
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/experts_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.ranges,
            compile_time_args=[
                self.CB_PLAN,
                self.CB_WGU,
                self.CB_WD,
                self.CB_X,
                self.CB_RW,
                self.CB_H_OWN,
                self.CB_H_ALL,
                self.CB_Y,
                self.kt,
                self.CB_XRM,
                self.gu_tiles,
                self.wd_tiles,
                self.ikt,
                self.cols,
                self.CB_XQ,
                self.quarter_tiles,
            ],
            defines=self.defines,
            runtime_args=compute_rt,
            config=self.compute_config,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=self._cbs(), semaphores=self.semaphores)
        ttnn.generic_op([plan, w_gu, w_d, x_rm, row_tokens, row_weights, row_dest, out], program)
        return out


class DevicePlan:
    """The expert-major plan from the device's top-k on CORES cores, split by expert (tt/kernels/prefill/plan.cpp): no
    read back. Shapes are static per T, sized for the most groups the routing can need (8T / 32 + E): rows past the used
    groups are simply not in the team plan."""

    CB_TILES, CB_STATE, CB_TOKENS, CB_WEIGHTS, CB_DEST, CB_PLAN = range(6)
    GRID = (8, 2)  # CORES = 16

    def __init__(self, moe, grouped):
        self.moe, self.grouped = moe, grouped
        gx, gy = self.GRID
        self.cores = [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(gx)]
        self.ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    @staticmethod
    def max_rows(num_tokens, k, experts):
        return (num_tokens * k // TILE + experts) * TILE

    def _cb(self, index, size):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)
        return ttnn.CBDescriptor(total_size=size, core_ranges=self.ranges, format_descriptors=[fmt])

    def __call__(self, indices, weights):
        """indices uint32 / weights fp32 [1, 1, T, K] tiles -> (row_token uint32 [1, R], row_weight bf16 [1, R],
        row_dest uint32 [1, R] (k * T + t, 0xFFFFFFFF on padding rows), plan uint32 [teams, PLAN_WORDS]), all
        row-major DRAM."""
        moe, grouped = self.moe, self.grouped
        T, K, E = indices.shape[-2], indices.shape[-1], moe.num_experts
        assert indices.dtype == ttnn.uint32 and weights.dtype == ttnn.float32
        for t in (indices, weights):
            assert t.layout == ttnn.TILE_LAYOUT and is_dram_interleaved(t) and K <= 16
        R = self.max_rows(T, K, E)
        device = moe.plan.mesh_device

        def empty(shape, dtype):
            return ttnn.empty(
                shape, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        row_token = empty([1, R], ttnn.uint32)
        row_weight = empty([1, R], ttnn.bfloat16)
        row_dest = empty([1, R], ttnn.uint32)
        plan = empty([grouped.num_teams, grouped.PLAN_WORDS], ttnn.uint32)
        rt = ttnn.RuntimeArgs()
        for i, core in enumerate(self.cores):
            rt[core.x][core.y] = [
                indices.buffer_address(),
                weights.buffer_address(),
                row_token.buffer_address(),
                row_weight.buffer_address(),
                row_dest.buffer_address(),
                plan.buffer_address(),
                i,
            ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/plan.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.ranges,
            compile_time_args=[
                T,
                K,
                E,
                grouped.num_teams,
                grouped.PLAN_WORDS,
                R,
                self.CB_TILES,
                self.CB_STATE,
                self.CB_TOKENS,
                self.CB_WEIGHTS,
                self.CB_DEST,
                self.CB_PLAN,
                len(self.cores),
            ],
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )
        cbs = [
            self._cb(self.CB_TILES, 2 * 4096),
            self._cb(self.CB_STATE, (3 * E + 4) * 4),
            self._cb(self.CB_TOKENS, R * 4),
            self._cb(self.CB_WEIGHTS, R * 2),
            self._cb(self.CB_DEST, R * 4),
            self._cb(self.CB_PLAN, grouped.num_teams * grouped.PLAN_WORDS * 4),
        ]
        program = ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs, semaphores=[])
        ttnn.generic_op([indices, weights, row_token, row_weight, row_dest, plan], program)
        return row_token, row_weight, row_dest, plan


# DevicePlanSplit from this many tokens: 0.60 vs 1.55 ms at 4096, 0.34 vs 0.55 at 1056, 0.18 vs 0.14 at 32
# (tests/perf/moe_plan_bench.py, device time on one row); the two cross near 210 tokens.
SPLIT_PLAN_MIN_TOKENS = 256


class DevicePlanSplit(DevicePlan):
    """DevicePlan's plan in two programs, so no core reads all T x K pairs. plan_slice.cpp: core i counts tile rows
    [i * TR / cores, (i + 1) * TR / cores) per expert and writes them sorted by expert. plan_gather.cpp: core i reads
    every slice's counts and, from each slice, the records of its experts, and fills their rows (and core 0 the team
    plan) as plan.cpp does. Same outputs as DevicePlan, bit for bit."""

    def _kernel(self, name, ct, rt_of):
        rt = ttnn.RuntimeArgs()
        for i, core in enumerate(self.cores):
            rt[core.x][core.y] = rt_of(i)
        return ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/{name}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        )

    def __call__(self, indices, weights):
        moe, grouped = self.moe, self.grouped
        T, K, E = indices.shape[-2], indices.shape[-1], moe.num_experts
        assert indices.dtype == ttnn.uint32 and weights.dtype == ttnn.float32
        for t in (indices, weights):
            assert t.layout == ttnn.TILE_LAYOUT and is_dram_interleaved(t) and K <= 16
        R, n = self.max_rows(T, K, E), len(self.cores)
        rows_per_core = -(-(T // TILE) // n)
        slice_pairs = rows_per_core * TILE * K
        device = moe.plan.mesh_device

        def empty(shape, dtype):
            return ttnn.empty(
                shape, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        counts = empty([n, E], ttnn.uint32)
        rtok, rdest = empty([n, slice_pairs], ttnn.uint32), empty([n, slice_pairs], ttnn.uint32)
        rw = empty([n, slice_pairs], ttnn.bfloat16)
        slice_kernel = self._kernel(
            "plan_slice.cpp",
            [T, K, E, n, slice_pairs, 0, 1, 2],
            lambda i: [
                indices.buffer_address(),
                weights.buffer_address(),
                counts.buffer_address(),
                rtok.buffer_address(),
                rw.buffer_address(),
                rdest.buffer_address(),
                i,
            ],
        )
        slice_cbs = [
            self._cb(0, rows_per_core * 2 * 4096),
            self._cb(1, 2 * E * 4),
            self._cb(2, slice_pairs * 10),
        ]
        ttnn.generic_op(
            [indices, weights, counts, rtok, rw, rdest],
            ttnn.ProgramDescriptor(kernels=[slice_kernel], cbs=slice_cbs, semaphores=[]),
        )

        row_token = empty([1, R], ttnn.uint32)
        row_weight = empty([1, R], ttnn.bfloat16)
        row_dest = empty([1, R], ttnn.uint32)
        plan = empty([grouped.num_teams, grouped.PLAN_WORDS], ttnn.uint32)
        gather_kernel = self._kernel(
            "plan_gather.cpp",
            [E, grouped.num_teams, grouped.PLAN_WORDS, R, n, slice_pairs, 0, 1, 2, 3, 4, 5, 6],
            lambda i: [
                counts.buffer_address(),
                rtok.buffer_address(),
                rw.buffer_address(),
                rdest.buffer_address(),
                row_token.buffer_address(),
                row_weight.buffer_address(),
                row_dest.buffer_address(),
                plan.buffer_address(),
                i,
            ],
        )
        gather_cbs = [
            self._cb(0, n * E * 4),
            self._cb(1, (3 * E + 4) * 4),
            self._cb(2, 3 * (slice_pairs + 16) * 4),
            self._cb(3, R * 4),
            self._cb(4, R * 2),
            self._cb(5, R * 4),
            self._cb(6, grouped.num_teams * grouped.PLAN_WORDS * 4),
        ]
        ttnn.generic_op(
            [counts, rtok, rw, rdest, row_token, row_weight, row_dest, plan],
            ttnn.ProgramDescriptor(kernels=[gather_kernel], cbs=gather_cbs, semaphores=[]),
        )
        for t in (counts, rtok, rw, rdest):
            t.deallocate(True)
        return row_token, row_weight, row_dest, plan


class CombineK:
    """The k-major row-major expert output [K * T, H] -> sum over k -> [1, 1, T, H] bf16 tiles, on every core
    (tt/kernels/prefill/combine_*.cpp): per 32 tokens x BLOCK tile columns, the K row blocks summed in fp32, then
    tilized.
    """

    BLOCK = 16  # tile columns per item: 1 KB of each row per read
    CB_IN, CB_SUM, CB_OUT = range(3)

    def __init__(self, moe, cores, ranges):
        self.moe, self.cores, self.ranges = moe, cores, ranges
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,  # 8 fp32 dest tiles: 4 sums and 4 for the next k
        )

    def _cbs(self):
        def cb(index, pages):
            tile = ttnn.Tile([TILE, TILE])
            page = tile.get_tile_size(ttnn.bfloat16)
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=index, data_format=ttnn.bfloat16, page_size=page, tile=ttnn.TileDescriptor(tile)
            )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=self.ranges, format_descriptors=[fmt])

        return [cb(self.CB_IN, self.k * self.BLOCK), cb(self.CB_SUM, self.BLOCK), cb(self.CB_OUT, 2 * self.BLOCK)]

    def __call__(self, y, num_tokens, k):
        moe = self.moe
        H = moe.hidden
        nt = H // TILE
        assert (
            y.layout == ttnn.ROW_MAJOR_LAYOUT and is_dram_interleaved(y) and tuple(y.shape)[-2:] == (k * num_tokens, H)
        )
        out = ttnn.empty(
            [1, 1, num_tokens, H],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=moe.plan.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.k = k
        items = (num_tokens // TILE) * (nt // self.BLOCK)
        per, extra = divmod(items, len(self.cores))
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        first = 0
        for i, core in enumerate(self.cores):
            count = per + (1 if i < extra else 0)
            reader_rt[core.x][core.y] = [y.buffer_address(), first, count]
            writer_rt[core.x][core.y] = [out.buffer_address(), first, count]
            compute_rt[core.x][core.y] = [count]
            first += count

        def kernel(name, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=f"{KERNEL_DIR}/{name}",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )

        dm = ttnn.DataMovementConfigDescriptor
        program = ttnn.ProgramDescriptor(
            kernels=[
                kernel(
                    "combine_reader.cpp",
                    [self.CB_IN, num_tokens, k, nt, self.BLOCK],
                    reader_rt,
                    dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default),
                ),
                kernel(
                    "combine_writer.cpp",
                    [self.CB_OUT, nt, self.BLOCK],
                    writer_rt,
                    dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default),
                ),
                kernel(
                    "combine_compute.cpp",
                    [self.CB_IN, self.CB_SUM, self.CB_OUT, k, self.BLOCK],
                    compute_rt,
                    self.compute_config,
                ),
            ],
            cbs=self._cbs(),
            semaphores=[],
        )
        ttnn.generic_op([y, out], program)
        return out


class PrefillExperts:
    def __init__(self, moe):
        self.moe = moe
        self.all_experts = None  # the sparsity operand indexed sparse_matmul takes (not read)
        self.grouped = None
        self.grouped_failed = False
        self.device_plan = None
        self.combine = None
        self.host_plan = False  # True: build the plan on the host (reads the routing back; for testing)

    def _upload(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        moe = self.moe
        return ttnn.from_torch(
            t, device=moe.plan.mesh_device, dtype=dtype, layout=layout, mesh_mapper=moe.plan.replicate()
        )

    def _grouped(self):
        if self.grouped is None and not self.grouped_failed:
            try:
                self.grouped = GroupedExperts(self.moe)
            except AssertionError:
                self.grouped_failed = True
        return self.grouped

    def plan(self, indices, weights, num_tokens):
        """Host plan from chip 0's routing (identical on every chip): per packed row its token and routing weight
        (padding rows: token 0, weight 0) and destination row k * T + t (padding: 0xFFFFFFFF), the 32-row groups per
        expert, and each (k, token) pair's packed row (k-major)."""
        E, K = self.moe.num_experts, indices.shape[-1]
        ids = ttnn.to_torch(ttnn.get_device_tensors(indices)[0]).reshape(-1).long()
        wts = ttnn.to_torch(ttnn.get_device_tensors(weights)[0]).reshape(-1).float()
        n = ids.numel()
        order = torch.argsort(ids, stable=True)
        counts = torch.bincount(ids, minlength=E)
        groups = (counts + TILE - 1) // TILE
        first = torch.cumsum(counts, 0) - counts  # first sorted pair of each expert
        first_row = (torch.cumsum(groups, 0) - groups) * TILE  # first packed row of each expert
        sorted_ids = ids[order]
        dest = first_row[sorted_ids] + torch.arange(n) - first[sorted_ids]
        rows = int(groups.sum()) * TILE
        row_token = torch.zeros(rows, dtype=torch.int32)
        row_token[dest] = (order // K).to(torch.int32)
        row_weight = torch.zeros(rows)
        row_weight[dest] = wts[order]
        row_dest = torch.full((rows,), -1, dtype=torch.int64)  # 0xFFFFFFFF once uploaded as uint32
        row_dest[dest] = (order % K) * num_tokens + order // K
        pair_row = torch.empty(n, dtype=torch.int32)
        pair_row[order] = dest.to(torch.int32)
        pair_row = pair_row.reshape(num_tokens, K).T.reshape(1, -1)  # k-major: entry k * T + t
        return row_token, row_weight, row_dest, groups, pair_row

    def __call__(self, x, indices, weights):
        """x [1, 1, T, hidden] (T a multiple of 32), indices [1, 1, T, 8] (uint), weights [1, 1, T, 8] (fp32,
        normalized) -> partial [1, 1, T, hidden] bf16."""
        moe = self.moe
        T, H = x.shape[-2], moe.hidden
        K = indices.shape[-1]
        grouped = self._grouped()
        x_rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (T, H))
        if grouped is not None:
            with zone("plan"):
                if self.host_plan:
                    row_token, row_weight, row_dest, groups, _ = self.plan(indices, weights, T)
                    row_token_tt = self._upload(row_token.reshape(1, -1), ttnn.uint32)
                    row_weight_tt = self._upload(row_weight.reshape(1, -1).to(torch.bfloat16), ttnn.bfloat16)
                    row_dest_tt = self._upload(row_dest.to(torch.uint32).reshape(1, -1), ttnn.uint32)
                    plan = self._upload(grouped.plan(groups), ttnn.uint32)
                else:
                    if self.device_plan is None:
                        self.device_plan = {False: DevicePlan(moe, grouped), True: DevicePlanSplit(moe, grouped)}
                    device_plan = self.device_plan[T >= SPLIT_PLAN_MIN_TOKENS]
                    row_token_tt, row_weight_tt, row_dest_tt, plan = device_plan(indices, weights)
            # The team cores read their tokens' rows of x themselves and write every pair's output row into the
            # k-major [K * T, H]; CombineK sums over k.
            with zone("grouped_matmul"):
                y = ttnn.empty(
                    [K * T, H],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=moe.plan.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                grouped(x_rm, row_token_tt, row_weight_tt, row_dest_tt, plan, y)
            for t in (x_rm, row_token_tt, row_weight_tt, row_dest_tt, plan):
                t.deallocate(True)
            if self.combine is None:
                self.combine = CombineK(moe, grouped.cores, grouped.ranges)
            with zone("combine"):
                out = self.combine(y, T, K)
            y.deallocate(True)
            return out

        row_token, row_weight, _, groups, pair_row = self.plan(indices, weights, T)
        R = int(groups.sum()) * TILE
        row_token_tt = self._upload(row_token.reshape(1, -1), ttnn.uint32)
        packed = ttnn.embedding(row_token_tt, x_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, R, H]
        x_rm.deallocate(True)
        row_token_tt.deallocate(True)
        y = self._sparse_groups(packed, row_weight, groups)
        y_rm = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), (R, H))
        y.deallocate(True)
        pair_row_tt = self._upload(pair_row, ttnn.uint32)
        pairs = ttnn.embedding(pair_row_tt, y_rm, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [1, K * T, H]
        y_rm.deallocate(True)
        pair_row_tt.deallocate(True)
        out = ttnn.sum(ttnn.reshape(pairs, (K, T, H)), dim=0, keepdim=True)
        pairs.deallocate(True)
        return ttnn.reshape(out, (1, 1, T, H))

    def _sparse_groups(self, packed, row_weight, groups):
        """Fallback: an indexed sparse_matmul per group against the group's expert (re-reads its weights per group).
        packed [1, R, H] -> [1, G, 32, H]."""
        moe = self.moe
        H, I, E = moe.hidden, moe.inter_local, moe.num_experts
        G = int(groups.sum())
        if self.all_experts is None:
            self.all_experts = self._upload(torch.ones(1, 1, 1, E), ttnn.bfloat16)
        group_expert = torch.repeat_interleave(torch.arange(E, dtype=torch.int32), groups)
        packed = ttnn.reshape(packed, (1, G, TILE, H))
        row_weight_tt = self._upload(row_weight.reshape(1, G, TILE, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        outs = []
        for g0 in range(0, G, E):  # indexed sparse_matmul takes at most E groups per call
            g1 = min(G, g0 + E)
            ids = self._upload(group_expert[g0:g1].reshape(1, 1, 1, -1), ttnn.uint16)
            mm = dict(
                sparsity=self.all_experts,
                indices=ids,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=ttnn.Tile([TILE, TILE]),
                dtype=ttnn.bfloat16,
                compute_kernel_config=moe.expert_config,
            )
            xa = ttnn.slice(packed, (0, g0, 0, 0), (1, g1, TILE, H))
            gate_up = ttnn.sparse_matmul(xa, moe.w_gate_up, program_config=moe.gate_up_program, **mm)
            xa.deallocate(True)
            act = moe._swiglu(ttnn.reshape(gate_up, (1, g1 - g0, TILE, 2 * I)))
            w = ttnn.slice(row_weight_tt, (0, g0, 0, 0), (1, g1, TILE, 1))
            act_w = ttnn.mul(act, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            act.deallocate(True)
            w.deallocate(True)
            outs.append(ttnn.sparse_matmul(act_w, moe.w_down, program_config=moe.down_program, **mm))
            act_w.deallocate(True)
            ids.deallocate(True)
        packed.deallocate(True)
        row_weight_tt.deallocate(True)
        y = ttnn.concat(outs, dim=1) if len(outs) > 1 else outs[0]
        if len(outs) > 1:
            for t in outs:
                t.deallocate(True)
        return y
