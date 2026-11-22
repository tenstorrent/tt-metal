# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Bank-local DRAM weight reads for batch-1 decode on one Blackhole card (7 live DRAM banks on p100a).

Two tools:

1. Dense decode linears (``x [1,1,1,K] @ W [K,N]``): ``DramShardedLinear`` keeps W WIDTH_SHARDED over the DRAM banks and
   runs tt-metal's DRAM-sharded matmul, whose readers sit next to their bank and read only it. Wide N (the unembed) is
   split into column chunks that fit L1; ``use_multicore_pipeline`` spreads each bank over ``workers`` cores.

2. Indexed expert matmuls (``y[s] = x[s or 0] @ W[ids[s]]``, ids on device): ``BankLocalExperts`` re-lays the expert
   weights so every worker core owns a few output columns of EVERY expert, stored in ONE bank as one contiguous run per
   expert. A selected expert is then one long read from the worker's own bank (vs. single-tile reads spread over all
   banks in the interleaved ``sparse_matmul``). Runs as a ``ttnn.generic_op`` with the kernels in ``tt/kernels``.
"""

import math
import os

import torch

import ttnn

TILE = 32
_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def dram_banks(device):
    g = device.dram_grid_size()
    assert g.y == 1, f"expected an Nx1 DRAM bank grid, got {g.x}x{g.y}"
    return g.x


def _dram_width_sharded(device, height, bank_width):
    banks = dram_banks(device)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    spec = ttnn.ShardSpec(grid, (height, bank_width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def bank_worker_cores(device, per_bank):
    """Per DRAM bank b, ``per_bank[b]`` distinct logical worker cores: the tt-metal optimal reader of bank b (NOC 0)
    first, then the free cores nearest to it (Manhattan), taken round-robin over banks so no bank gets the far ones."""
    opt = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.RISCV_0_default)
    opt = [(int(c.x), int(c.y)) for c in opt]
    grid = device.compute_with_storage_grid_size()
    free = {(x, y) for x in range(grid.x) for y in range(grid.y)}
    assert sum(per_bank) <= len(free), (per_bank, len(free))
    out = [[] for _ in opt]
    for b, c in enumerate(opt):
        out[b].append(c)
        free.discard(c)
    while any(len(out[b]) < per_bank[b] for b in range(len(opt))):
        for b, (ox, oy) in enumerate(opt):
            if len(out[b]) < per_bank[b]:
                c = min(free, key=lambda p: (abs(p[0] - ox) + abs(p[1] - oy), abs(p[1] - oy), p))
                out[b].append(c)
                free.discard(c)
    return [cores[:n] for cores, n in zip(out, per_bank)]


# ------------------------------------------------------------------------------------------------ dense linears


class DramShardedLinear:
    """``x [1,1,1,K] bf16 -> [1,1,1,N]`` with W WIDTH_SHARDED in DRAM, N split in ``chunk`` column pieces.

    ``weight``: host or device ttnn tensor [.., K, N] (any layout/memory; re-laid on load) or torch [K, N].
    Each chunk is a separate DRAM-sharded matmul (L1 width-sharded output); the pieces are moved to ``out_mem``
    and concatenated. ``chunk`` must be a multiple of banks*32; N is zero-padded to whole chunks and cut back.
    ``workers`` > 1 selects tt-metal's multi-core pipeline (num_workers_per_dram_bank, use_multicore_pipeline)."""

    def __init__(
        self,
        device,
        weight,
        *,
        dtype=ttnn.bfloat8_b,
        chunk=None,
        act_cores=8,
        workers=3,
        out_dtype=ttnn.bfloat16,
        compute_kernel_config=None,
    ):
        banks = dram_banks(device)
        w = weight if isinstance(weight, torch.Tensor) else ttnn.to_torch(weight)
        w = w.reshape(w.shape[-2], w.shape[-1])
        self.k, self.n = w.shape
        chunk = chunk or math.ceil(self.n / (banks * TILE)) * banks * TILE
        assert chunk % (banks * TILE) == 0 and self.k % (TILE * act_cores) == 0, (chunk, self.k, act_cores)
        n_chunks = math.ceil(self.n / chunk)
        if n_chunks * chunk != self.n:
            w = torch.nn.functional.pad(w, (0, n_chunks * chunk - self.n))
        wmc = _dram_width_sharded(device, self.k, chunk // banks)
        self.chunks = [
            ttnn.from_torch(
                w[:, i * chunk : (i + 1) * chunk].reshape(1, 1, self.k, chunk).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=wmc,
            )
            for i in range(n_chunks)
        ]
        self.chunk = chunk
        grid = ttnn.CoreGrid(x=act_cores, y=1) if act_cores <= 8 else ttnn.CoreGrid(x=8, y=act_cores // 8)
        assert act_cores <= 8 or act_cores % 8 == 0, act_cores
        self.act_mem = ttnn.create_sharded_memory_config(
            (TILE, self.k // act_cores),
            grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        kt = self.k // (TILE * act_cores)
        extra = {"num_workers_per_dram_bank": workers, "use_multicore_pipeline": True} if workers > 1 else {}
        self.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=next(b for b in range(8, 0, -1) if kt % b == 0),
            per_core_M=1,
            per_core_N=math.ceil(chunk / (TILE * act_cores)),
            fused_activation=None,
            **extra,
        )
        self.out_dtype = out_dtype
        self.cc = compute_kernel_config

    def __call__(self, x, out_mem=ttnn.DRAM_MEMORY_CONFIG):
        xs = ttnn.to_memory_config(x, self.act_mem)
        parts = []
        for w in self.chunks:
            y = ttnn.linear(
                xs,
                w,
                program_config=self.program,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=self.out_dtype,
                compute_kernel_config=self.cc,
            )
            parts.append(ttnn.to_memory_config(y, out_mem))
            ttnn.deallocate(y)
        ttnn.deallocate(xs)
        if len(parts) == 1:
            out = parts[0]
        else:
            out = ttnn.concat(parts, dim=-1, memory_config=out_mem)
            for p in parts:
                ttnn.deallocate(p)
        if out.shape[-1] != self.n:
            cut = ttnn.slice(out, (0, 0, 0, 0), (1, 1, out.shape[-2], self.n), memory_config=out_mem)
            ttnn.deallocate(out)
            out = cut
        return out


# ------------------------------------------------------------------------------------------------ indexed experts


class BankLocalExperts:
    """Indexed expert matmul with bank-local weights: ``y[0, s] = x[0, s if per_slot else 0] @ W[ids[s]]``.

    ``w``: torch [E, K, N] (or ttnn host [1, E, K, N]), ``dtype``: bfloat4_b / bfloat8_b. N tiles are split over the
    banks (contiguous column ranges), each bank's columns over ``ceil(cols/nw)`` workers of ``nw`` columns.
    Bank b stores, per expert, worker 0's [Kt][nw] tiles, then worker 1's, ...: one contiguous run per (expert,
    worker). The host tensor is a WIDTH_SHARDED [E*S*32, banks*32] with one tile column per bank (S tiles per expert).
    Call: x TILE bf16 interleaved ([1,1,1,K] shared, or [1,slots,1,K] per slot), ids ROW_MAJOR uint16 [1,1,1,slots]
    -> [1, slots, 1, N] bf16 TILE in ``out_mem``."""

    def __init__(self, device, w, *, dtype=ttnn.bfloat4_b, nw=1, slots=8, math_fidelity=ttnn.MathFidelity.HiFi2):
        if not isinstance(w, torch.Tensor):
            w = ttnn.to_torch(w)
        w = w.reshape(-1, w.shape[-2], w.shape[-1])
        self.device, self.dtype, self.nw, self.slots = device, dtype, nw, slots
        self.e, k, n = w.shape
        self.kt, self.nt = k // TILE, n // TILE
        banks = dram_banks(device)
        base, extra = divmod(self.nt, banks)
        cols = [base + (b < extra) for b in range(banks)]
        starts = [sum(cols[:b]) for b in range(banks)]
        workers = [math.ceil(c / nw) for c in cols]
        self.stride_tiles = max(workers) * self.kt * nw  # tiles per expert per bank (uniform over banks)
        tb = _TILE_BYTES[dtype]
        # host re-layout: [E, Kt, 32, Nt, 32] tiles -> per bank [E, worker, Kt, nw] tile rows
        wt = w.reshape(self.e, self.kt, TILE, self.nt, TILE).permute(0, 1, 3, 2, 4)  # [E, Kt, Nt, 32, 32]
        host = torch.zeros(self.e, self.stride_tiles, TILE, banks, TILE, dtype=w.dtype)
        self.work = []  # (bank, core, byte offset in the expert run, first output col tile, valid cols)
        cores = bank_worker_cores(device, workers)
        for b in range(banks):
            for j in range(workers[b]):
                c0 = starts[b] + j * nw
                nv = min(nw, cols[b] - j * nw)
                blk = torch.zeros(self.e, self.kt, nw, TILE, TILE, dtype=w.dtype)
                blk[:, :, :nv] = wt[:, :, c0 : c0 + nv]
                r0 = j * self.kt * nw
                host[:, r0 : r0 + self.kt * nw, :, b, :] = blk.reshape(self.e, self.kt * nw, TILE, TILE)
                self.work.append((b, cores[b][j], r0 * tb, c0, nv))
        rows = self.e * self.stride_tiles * TILE
        mem = _dram_width_sharded(device, rows, TILE)
        self.w = ttnn.from_torch(
            host.reshape(rows, banks * TILE), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )
        del host, wt
        self.expert_stride = self.stride_tiles * tb
        self.cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for _, c, _, _, _ in self.work]
        )
        self.math_fidelity = math_fidelity

    def __call__(self, x, ids, out_mem=ttnn.L1_MEMORY_CONFIG):
        per_slot = int(x.padded_shape[-3] * x.padded_shape[-4]) > 1
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.slots, 1, self.nt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, out_mem
        )
        kt, nw, slots = self.kt, self.nw, self.slots
        in0_tiles = kt * (slots if per_slot else 1)
        # The program spans the whole worker grid: the activation multicast lands in every core of the rectangle,
        # so every core holds the CBs (idle cores run kernels that return at once).
        g = self.device.compute_with_storage_grid_size()
        rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=rect,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(0, ttnn.bfloat16, 2048, in0_tiles),
            cb(1, self.dtype, _TILE_BYTES[self.dtype], 2 * kt * nw),
            cb(2, ttnn.uint16, 64, 1),
            cb(16, ttnn.bfloat16, 2048, 2 * nw),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=rect, initial_value=0) for i in (0, 1)]
        idx_page = 2 * slots  # one ROW_MAJOR uint16 row of the slot ids
        rd_ct = [kt, nw, slots, in0_tiles, self.e, idx_page, 0, 1]
        rd_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        rd_ct += ttnn.TensorAccessorArgs(ids).get_compile_time_args()
        wr_ct = [nw, slots, self.nt] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        rd_rt, wr_rt, cp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for cx in range(g.x):
            for cy in range(g.y):
                rd_rt[cx][cy] = [0] * 7
                wr_rt[cx][cy] = [0, 0, 0]
                cp_rt[cx][cy] = [0]
        w_addr, x_addr, i_addr, o_addr = (t.buffer_address() for t in (self.w, x, ids, out))
        phys = lambda cx, cy: self.device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))  # noqa: E731
        p0, p1 = phys(0, 0), phys(g.x - 1, g.y - 1)
        sender = self.work[0][1]
        ps = phys(*sender)
        for bank, (cx, cy), off, c0, nv in self.work:
            base = [w_addr, bank, self.expert_stride, off, x_addr, i_addr]
            if (cx, cy) == sender:
                rd_rt[cx][cy] = base + [2, p0.x, p0.y, p1.x, p1.y, g.x * g.y - 1, len(self.work) - 1]
            else:
                rd_rt[cx][cy] = base + [1, ps.x, ps.y]
            wr_rt[cx][cy] = [o_addr, c0, nv]
            cp_rt[cx][cy] = [1]

        def kernel(name, ct, rt, config):
            return ttnn.KernelDescriptor(
                kernel_source=os.path.join(_KDIR, name),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=rect,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )

        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            kernel(
                "bank_expert_reader.cpp",
                rd_ct,
                rd_rt,
                dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default),
            ),
            kernel(
                "bank_expert_writer.cpp",
                wr_ct,
                wr_rt,
                dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default),
            ),
            kernel("bank_expert_compute.cpp", [kt, nw, slots, int(per_slot)], cp_rt, self._compute_config()),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        return ttnn.generic_op([x, ids, self.w, out], prog)

    def _compute_config(self):
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = self.math_fidelity
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        return c
