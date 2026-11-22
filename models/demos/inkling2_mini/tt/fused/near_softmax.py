# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Global-layer near softmax in one program: replaces ~12 ops of attention_fast._global on 32-row padded tiles
(mask reshape, bias add, concat with the sink block, max with fill-pad, subtract, exp, row-sum matmul, slice,
reciprocal, multiply, log, add).

One core per (kv head h, q row r) = 24 cores. Inputs: near scores [1, 8, 3, 1024] fp32 (q.k, scale already in q),
the prep mask [1, 1, 24, 1024] bf16 (rel bias + near mask, row 3h + r), the sink block [1, 8, 3, 32] fp32.
Outputs (module buffers, zero padded): p_near [1, 8, 3, 1024] bf16 and lse [1, 8, 3, 1] fp32.
"""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

N_KV, GROUPS, WINDOW = 8, 3, 1024


class NearSoftmax:
    def __init__(self, device):
        self.device = device
        dram = ttnn.DRAM_MEMORY_CONFIG

        def zeros(shape, dtype):
            return ttnn.from_torch(
                torch.zeros(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
            )

        self.p = zeros([1, N_KV, GROUPS, WINDOW], ttnn.bfloat16)
        self.lse = zeros([1, N_KV, GROUPS, 1], ttnn.float32)
        self.consts = F.dense_tensor(device, torch.stack([torch.ones(F.PAGE), torch.full((F.PAGE,), float("-inf"))]))
        self.scaler = ttnn.from_torch(
            torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )
        self.cores, self.grid = F.core_list(device, N_KV * GROUPS)
        fp, bf, g = ttnn.float32, ttnn.bfloat16, self.grid
        self.cbs = [F.cb(i, fp, 1, cores=g) for i in (0, 1, 3, 16, 17, 18, 19, 20, 21, 22, 23, 29)]
        self.cbs += [F.cb(2, bf, 1, cores=g), F.cb(4, bf, 1, cores=g), F.cb(28, bf, 1, cores=g)]
        self.cbs += [F.stage_cb(5, WINDOW // 32, cores=g), F.cb(6, fp, 1, 64, cores=g)]
        self.compute_cfg = F.compute_config([0, 1, 16, 18, 19, 21, 23])

    def __call__(self, near, mask, sink_blk):
        """-> (p_near, lse). near / mask / sink_blk: DRAM interleaved TILE (see module doc)."""
        for t, dt in ((near, ttnn.float32), (mask, ttnn.bfloat16), (sink_blk, ttnn.float32)):
            assert t.dtype == dt and t.layout == ttnn.TILE_LAYOUT, (t.dtype, t.layout)
            assert t.memory_config().buffer_type == ttnn.BufferType.DRAM and not t.is_sharded(), t.memory_config()
        r_args, w_args = {}, {}
        for i, c in enumerate(self.cores):
            h, r = i // GROUPS, i % GROUPS
            r_args[c] = [near.buffer_address(), mask.buffer_address(), sink_blk.buffer_address(),
                         self.consts.buffer_address(), self.scaler.buffer_address(), h, r, GROUPS * h + r]  # fmt: skip
            w_args[c] = [self.p.buffer_address(), self.lse.buffer_address(), h, r]
        named = dict(nt=WINDOW // 32)
        kernels = [
            F.kernel("near_sm_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, self.grid),
            F.kernel("near_sm_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, self.grid),
            F.kernel("near_sm_compute.cpp", self.compute_cfg, named, None, self.grid),
        ]
        io = [near, mask, sink_blk, self.consts, self.scaler, self.p, self.lse]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=self.cbs, semaphores=[]))
        return self.p, self.lse
