# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Decode logits tail: unembed chunks -> softcap -> logits row + on-device argmax.

Replaces, per decode step: one ShardedToInterleaved per unembed chunk, then concat, slice, multiply, tanh,
multiply, vocab-mask add, untilize and argmax, each over the 32-row padded [1, 1, 32, 200064] fp32 tile tensor
(~1.1 ms of the step). Two kinds of programs instead:
  1. after each chunk's matmul, a pack program on the cores holding its L1 width-sharded output copies row 0
     of their tiles into a raw fp32 row in DRAM (data movement only);
  2. after the last chunk, one program on ~100 cores: each takes two 1024-value pages of the raw row, applies
     tanh(x * (1/cap)) * cap with the same SFPU calls as the ttnn op chain, writes its part of the logits row,
     and sends its (max, index) to a master core, which writes the token.

Outputs (module buffers, allocated on the first call, which must be eager):
  logits [1, 1, 1, vocab_pad] fp32 ROW_MAJOR DRAM: softcap(x), bit-equal to the ttnn op chain; columns
    >= vocab_real are not masked (readers cut at vocab_real), they are left out of the argmax.
  token  [1, 1, 1, 1] uint32 ROW_MAJOR DRAM: argmax over the vocab_real columns, ttnn.argmax tie rule.
"""

import math

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

PAGES_PER_CORE = 2


class FusedLogitsTail:
    def __init__(self, device, unembed, vocab_pad, vocab_real, softcap):
        """unembed: tt/dram_sharded.DramShardedLinear with fp32 output (its chunks and program are reused)."""
        assert unembed.out_dtype == ttnn.float32, unembed.out_dtype
        self.device = device
        self.mm = unembed
        self.vocab_pad, self.vocab_real = vocab_pad, vocab_real
        width = len(unembed.chunks) * unembed.chunk  # raw row: whole chunks
        assert width >= vocab_pad
        self.n_pages = math.ceil(width / F.PAGE)
        self.raw_width = self.n_pages * F.PAGE
        cap = torch.tensor(softcap, dtype=torch.float32)
        self.scal = F.dense_tensor(device, torch.stack([torch.full((F.PAGE,), 1.0 / softcap), cap.expand(F.PAGE)]))
        dram = ttnn.DRAM_MEMORY_CONFIG
        rm = ttnn.ROW_MAJOR_LAYOUT

        def row(n, dtype=ttnn.float32, tdtype=torch.float32):
            z = torch.zeros(1, 1, 1, n, dtype=tdtype)
            return ttnn.from_torch(z, dtype=dtype, layout=rm, device=device, memory_config=dram)

        self.raw = row(self.raw_width)
        self.logits = row(vocab_pad)
        self.token = row(1, ttnn.uint32, torch.int32)
        n_cores = math.ceil(self.n_pages / PAGES_PER_CORE)
        self.cores, self.grid = F.core_list(device, n_cores)
        self.master = device.worker_core_from_logical_core(self.cores[0])

    def __call__(self, x):
        """x [1,1,1,K] bf16 (final-norm output) -> (logits, token) module buffers."""
        mm = self.mm
        xs = ttnn.to_memory_config(x, mm.act_mem)
        for i, w in enumerate(mm.chunks):
            y = ttnn.linear(
                xs,
                w,
                program_config=mm.program,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.float32,
                compute_kernel_config=mm.cc,
            )
            self._pack(y, i)
            ttnn.deallocate(y)
        ttnn.deallocate(xs)
        self._cap()
        return self.logits, self.token

    def _pack(self, y, i):
        spec = y.memory_config().shard_spec
        cores = ttnn.corerange_to_cores(spec.grid, row_wise=True)
        h, w = spec.shape
        assert y.dtype == ttnn.float32 and h == 32 and w % 32 == 0, (y.dtype, spec.shape)
        assert spec.orientation == ttnn.ShardOrientation.ROW_MAJOR and len(cores) * w == self.mm.chunk
        rt = {
            c: [y.buffer_address(), self.raw.buffer_address(), i * self.mm.chunk + k * w] for k, c in enumerate(cores)
        }
        named = dict(ntiles=w // 32, row_page=self.raw_width * 4)
        kernels = [F.kernel("head_pack.cpp", ttnn.ReaderConfigDescriptor(), named, rt, spec.grid)]
        cbs = [F.cb(0, ttnn.float32, 1, w * 4, cores=spec.grid)]
        ttnn.generic_op([y, self.raw], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))

    def _cap(self):
        T, n = PAGES_PER_CORE, len(self.cores)
        r_args, w_args = {}, {}
        for k, c in enumerate(self.cores):
            col0 = k * T * F.PAGE
            n_write = max(0, min(T * F.PAGE, self.vocab_pad - col0))
            n_arg = max(0, min(T * F.PAGE, self.vocab_real - col0))
            r_args[c] = [self.raw.buffer_address(), k * T, self.scal.buffer_address()]
            w_args[c] = [self.logits.buffer_address(), col0, n_write, n_arg, k, self.master.x, self.master.y, n,
                         self.token.buffer_address()]  # fmt: skip
        named = dict(pages=T, row_page=self.raw_width * 4, logits_page=self.vocab_pad * 4, sem=0)
        kernels = [
            F.kernel("head_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, self.grid),
            F.kernel("head_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, self.grid),
            F.kernel("head_compute.cpp", F.compute_config([0, 1]), dict(pages=T), None, self.grid),
        ]
        cbs = [
            F.cb(0, ttnn.float32, T, cores=self.grid),
            F.cb(1, ttnn.float32, 2, cores=self.grid),
            F.cb(16, ttnn.float32, T, cores=self.grid),
            F.cb(17, ttnn.float32, 1, 16 * (n + 1), cores=self.grid),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.grid, initial_value=0)]
        io = [self.raw, self.scal, self.token, self.logits]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
