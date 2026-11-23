# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill cross-pass messages of one loop layer in one program: m_k = n1 + diff * w[:, c0 + k], k < 4.

prefill_model._messages ran ttnn.multiply(diff, w column) and ttnn.add(n1, .) per message: 8 programs and ~5
passes over [T, 2304] fp32 each. Here n1 and diff are read once and the 4 messages written, with the same SFPU
ops binary_ng runs (mul_binary_tile, add_binary_tile with RNE narrowing; fp32 DST). Bit-equal.

Call: PfMessages(device)(n1, diff, w, c0) with n1 / diff [1,1,T,C] fp32, w [1,1,T,64] fp32 (TILE, DRAM), c0 a
multiple of 4 -> 4 fp32 [1,1,T,C] tensors.
"""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F


class PfMessages:
    def __init__(self, device):
        self.device = device

    def __call__(self, n1, diff, w, c0):
        t, c = n1.shape[-2], n1.shape[-1]
        assert c0 % 4 == 0 and tuple(w.shape)[-2:] == (t, 64), (c0, w.shape)
        for a in (n1, diff, w):
            assert a.dtype == ttnn.float32 and a.layout == ttnn.TILE_LAYOUT, a
            assert a.memory_config().buffer_type == ttnn.BufferType.DRAM, a
        assert tuple(diff.shape) == tuple(n1.shape), diff.shape
        dram = ttnn.DRAM_MEMORY_CONFIG
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, t, c]), ttnn.float32, ttnn.TILE_LAYOUT, self.device, dram)
                for _ in range(4)]  # fmt: skip
        wt, rows = c // 32, t // 32
        grid = self.device.compute_with_storage_grid_size()
        n_cores = grid.x * grid.y
        split = max(1, -(-4 * n_cores // rows))  # ~4 units per core; each unit spreads its w tile once
        blk = -(-wt // min(split, wt))
        units = [(R, j0, min(wt, j0 + blk)) for R in range(rows) for j0 in range(0, wt, blk)]
        n_cores = min(n_cores, len(units))
        cores, crs = F.core_list(self.device, n_cores)
        per = [units[i::n_cores] for i in range(n_cores)]
        r_args, w_args, c_args = {}, {}, {}
        for core, us in zip(cores, per):
            flat = [v for u in us for v in u]
            r_args[core] = [n1.buffer_address(), diff.buffer_address(), w.buffer_address(), c0, len(us)] + flat
            w_args[core] = [o.buffer_address() for o in outs] + [len(us)] + flat
            c_args[core] = [len(us)] + [j1 - j0 for _, j0, j1 in us]
        named = F.fixed_rt(dict(wt=wt), r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_msg_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_msg_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_msg_compute.cpp", F.compute_config([0, 1, 2]), named, c_args, crs),
        ]
        fp = ttnn.float32
        cbs = [F.cb(0, fp, 2, cores=crs), F.cb(1, fp, 2, cores=crs), F.cb(2, fp, 8, cores=crs),
               F.cb(3, fp, 1, cores=crs), F.cb(16, fp, 4, cores=crs)]  # fmt: skip
        F.run([n1, diff, w] + outs, kernels, cbs)
        return outs
