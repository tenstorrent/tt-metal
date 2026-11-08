# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy decode head: the argmax over the row's vocab shards in three small launches.

The top-k path (per-chip top-32 of 19072 logits, two all_gathers, a top-1 and the untilize / reshape / typecast to
a token row) costs ~200 us per decode step. Greedy decoding only needs each user's maximum: GreedyScan reads the
decode users' rows of the [1, 1, 32, V] logits tiles on 32 cores and records each core's (sort key, index); one
row all_gather collects every chip's partials; GreedyMerge picks the first strict maximum in (chip, core) order
(the lowest global id on ties, like torch.argmax over the full vocab) and writes the token row into the persistent
``next_tokens`` buffer and, optionally, straight into the decode trace's token input.
"""

import torch

import ttnn

from .common import TILE

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels"
SCAN_CORES = 32
SCRATCH_CB = 0


def _is_dram(t):
    return int(t.memory_config().buffer_type == ttnn.BufferType.DRAM)


class GreedyHead:
    def __init__(self, plan, vocab_local, users, max_batch):
        assert vocab_local % TILE == 0 and 1 <= users <= 16 and max_batch * 4 <= 1024
        self.plan, self.vocab_local, self.users, self.max_batch = plan, vocab_local, users, max_batch
        device = plan.mesh_device
        grid = device.compute_with_storage_grid_size()
        self.scan_cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(SCAN_CORES)]
        self.merge_core = ttnn.CoreCoord(0, 0)
        tiles = vocab_local // TILE
        bounds = [tiles * i // SCAN_CORES for i in range(SCAN_CORES + 1)]
        self.ranges = list(zip(bounds[:-1], bounds[1:]))
        self.max_tiles = max(e - b for b, e in self.ranges)
        rep = plan.replicate()
        self.partials = ttnn.from_torch(
            torch.zeros(1, 1, TILE, TILE, dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.next_tokens = ttnn.from_torch(
            torch.zeros(1, max_batch, dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    @staticmethod
    def _scratch(cores, nbytes):
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cb = ttnn.CBDescriptor(
            total_size=nbytes,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(SCRATCH_CB, ttnn.uint32, nbytes)],
        )
        return crs, cb

    def _scan_program(self, logits):
        elem_bytes = 4 if logits.dtype == ttnn.float32 else 2
        crs, cb = self._scratch(self.scan_cores, self.max_tiles * 2 * self.users * 16 * elem_bytes + 64)
        rt = ttnn.RuntimeArgs()
        for i, (core, (b, e)) in enumerate(zip(self.scan_cores, self.ranges)):
            rt[core.x][core.y] = [logits.buffer_address(), self.partials.buffer_address(), b, e, i]
        kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/greedy_scan.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=[self.users, SCRATCH_CB, self.max_tiles, elem_bytes],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])

    def _merge_program(self, gathered, token_input):
        tp = self.plan.tp
        halves = 2 if self.users > 8 else 1  # users 8-15 in the partial rows' right faces
        crs, cb = self._scratch([self.merge_core], tp * halves * 2 * 1024 + self.max_batch * 4)
        rt = ttnn.RuntimeArgs()
        out1 = token_input.buffer_address() if token_input is not None else 0
        rt[self.merge_core.x][self.merge_core.y] = [
            gathered.buffer_address(),
            _is_dram(gathered),
            self.next_tokens.buffer_address(),
            _is_dram(self.next_tokens),
            out1,
            _is_dram(token_input) if token_input is not None else 0,
        ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/greedy_merge.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=[self.users, SCAN_CORES, tp, self.vocab_local, self.max_batch, SCRATCH_CB],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])

    def __call__(self, logits, token_input=None):
        """logits [1, 1, 32, vocab_local] bf16 or fp32 TILE (DRAM interleaved; users in rows 0..users-1).

        Returns the persistent ``next_tokens`` [1, max_batch] uint32 row-major (zeros past ``users``); also writes
        it into ``token_input`` (a [1, max_batch] uint32 row-major tensor) when given."""
        assert logits.layout == ttnn.TILE_LAYOUT and logits.dtype in (ttnn.bfloat16, ttnn.float32) and _is_dram(logits)
        assert logits.shape[-1] == self.vocab_local and logits.shape[-2] == TILE
        ttnn.generic_op([logits, self.partials], self._scan_program(logits))
        gathered = self.plan.all_gather(self.partials, dim=-1)
        io = [gathered, self.next_tokens] + ([token_input] if token_input is not None else [])
        ttnn.generic_op(io, self._merge_program(gathered, token_input))
        gathered.deallocate(True)
        return self.next_tokens
