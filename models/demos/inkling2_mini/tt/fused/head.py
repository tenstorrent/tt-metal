# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Decode logits tail: unembed chunks -> softcap -> logits row + on-device argmax, one small program per chunk.

Replaces, per decode step: one ShardedToInterleaved per unembed chunk, then concat, slice, multiply, tanh,
multiply, vocab-mask add, untilize and argmax, each over the 32-row padded [1, 1, 32, 200064] fp32 tile tensor
(~1.1 ms of the step). Each chunk's tail program runs on the cores that hold the chunk's L1 width-sharded matmul
output and touches only row 0 of its tiles.

Outputs (module buffers, allocated on the first call, which must be eager):
  logits [1, 1, 1, vocab_pad] fp32 ROW_MAJOR DRAM: softcap(x) = tanh(x * (1/cap)) * cap, bit-equal to the ttnn op
    chain; columns >= vocab_real are not masked (readers cut at vocab_real), they are left out of the argmax.
  token  [1, 1, 1, 1] uint32 ROW_MAJOR DRAM: argmax over the vocab_real columns, ttnn.argmax tie rule.
"""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

SLOT = 64  # bytes per (chunk, core) partial


class FusedLogitsTail:
    def __init__(self, device, unembed, vocab_pad, vocab_real, softcap):
        """unembed: tt/dram_sharded.DramShardedLinear with fp32 output (its chunks and program are reused)."""
        assert unembed.out_dtype == ttnn.float32, unembed.out_dtype
        self.device = device
        self.mm = unembed
        self.vocab_pad, self.vocab_real = vocab_pad, vocab_real
        assert len(unembed.chunks) * unembed.chunk >= vocab_pad
        cap = torch.tensor(softcap, dtype=torch.float32)
        self.scal = F.dense_tensor(device, torch.stack([torch.full((F.PAGE,), 1.0 / softcap), cap.expand(F.PAGE)]))
        dram = ttnn.DRAM_MEMORY_CONFIG
        rm = ttnn.ROW_MAJOR_LAYOUT
        self.logits = ttnn.from_torch(
            torch.zeros(1, 1, 1, vocab_pad), dtype=ttnn.float32, layout=rm, device=device, memory_config=dram
        )
        self.token = ttnn.from_torch(
            torch.zeros(1, 1, 1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=rm, device=device, memory_config=dram
        )
        self.parts = None  # [n_chunks * cores slots of 64 B], sized on the first call

    def __call__(self, x):
        """x [1,1,1,K] bf16 (final-norm output) -> (logits, token) module buffers."""
        mm = self.mm
        xs = ttnn.to_memory_config(x, mm.act_mem)
        n = len(mm.chunks)
        for i, w in enumerate(mm.chunks):
            y = ttnn.linear(
                xs,
                w,
                program_config=mm.program,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.float32,
                compute_kernel_config=mm.cc,
            )
            self._tail(y, i, n)
            ttnn.deallocate(y)
        ttnn.deallocate(xs)
        return self.logits, self.token

    def _tail(self, y, i, n):
        spec = y.memory_config().shard_spec
        cores = ttnn.corerange_to_cores(spec.grid, row_wise=True)
        h, w = spec.shape
        assert y.dtype == ttnn.float32 and h == 32 and w % 32 == 0, (y.dtype, spec.shape)
        assert spec.orientation == ttnn.ShardOrientation.ROW_MAJOR and len(cores) * w == self.mm.chunk
        nc, ntiles = len(cores), w // 32
        T = F.pages(w)
        n_slots = n * nc
        if self.parts is None:
            self.parts = ttnn.from_torch(
                torch.zeros(1, 1, 1, n_slots * SLOT // 4, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        grid = spec.grid
        m = self.device.worker_core_from_logical_core(cores[0])
        last = int(i == n - 1)
        r_args, w_args = {}, {}
        for k, c in enumerate(cores):
            col0 = i * self.mm.chunk + k * w
            n_write = max(0, min(w, self.vocab_pad - col0))
            n_arg = max(0, min(w, self.vocab_real - col0))
            r_args[c] = [y.buffer_address(), self.scal.buffer_address()]
            w_args[c] = [self.logits.buffer_address(), col0, n_write, n_arg, self.parts.buffer_address(),
                         i * nc + k, last, int(k == 0), m.x, m.y, nc - 1, self.token.buffer_address()]  # fmt: skip
        kernels = [
            F.kernel("head_reader.cpp", ttnn.ReaderConfigDescriptor(), dict(ntiles=ntiles, pages=T), r_args, grid),
            F.kernel(
                "head_writer.cpp",
                ttnn.WriterConfigDescriptor(),
                dict(pages=T, n_slots=n_slots, logits_page=self.vocab_pad * 4, sem=0),
                w_args,
                grid,
            ),
            F.kernel("head_compute.cpp", F.compute_config([0, 1]), dict(pages=T), None, grid),
        ]
        cbs = [
            F.cb(0, ttnn.float32, T, cores=grid),
            F.cb(1, ttnn.float32, 2, cores=grid),
            F.cb(16, ttnn.float32, T, cores=grid),
            F.cb(17, ttnn.float32, 1, SLOT + n_slots * SLOT, cores=grid),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=grid, initial_value=0)]
        io = [y, self.scal, self.parts, self.token, self.logits]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
