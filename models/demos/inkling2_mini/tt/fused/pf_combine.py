# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE combine in one program: out[t] = sum_k wk[k, t] * y[back[k, t]] -> [1, 1, T, HIDDEN] bf16 TILE.

Replaces ttnn.embedding(back, y) -> ttnn.multiply(., wk) -> ttnn.sum(dim=1) with the same rounding points (bf16
product, fp32 sum in k order, bf16 out) and no [K, T, HIDDEN] intermediates: each unit gathers its 8 x 32 expert
row segments straight from y, tilizes, multiplies and sums."""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

CW = 24  # output tiles per unit (1.5 KB row segments; 768 units at T = 8192)
NB = 2  # row-major blocks in flight per core (3, 4: no faster)


class PfCombine:
    def __init__(self, device, cw=CW):
        self.device = device
        self.cw = cw
        self.zero = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device)  # fmt: skip

    def __call__(self, y, back, wk):
        """y [rows, C] bf16 ROW_MAJOR; back [1, K T] uint32 (k-major buffer rows); wk [1, K, T, 1] bf16 TILE, or the
        router's [1, 1, T, K] (K <= 16: the reader builds the K weight tiles, no permute program)."""
        wt8 = wk.shape[1] == 1 and wk.shape[3] > 1
        k, t = (wk.shape[3], wk.shape[2]) if wt8 else (wk.shape[1], wk.shape[2])
        assert not wt8 or (k <= 16 and wk.layout == ttnn.TILE_LAYOUT), wk.shape
        c = y.shape[-1]
        assert y.dtype == ttnn.bfloat16 and y.layout == ttnn.ROW_MAJOR_LAYOUT and c % (32 * self.cw) == 0, y.shape
        assert back.shape[-1] == k * t and t % 32 == 0 and wk.dtype == ttnn.bfloat16, (back.shape, wk.shape)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, t, c]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device,
                                             ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        ot = c // 32
        units = [(r, j) for r in range(t // 32) for j in range(ot // self.cw)]
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(len(units), grid.x * grid.y)
        cores, crs = F.core_list(self.device, n_cores)
        per = [units[i::n_cores] for i in range(n_cores)]
        r_args, w_args, c_args = {}, {}, {}
        for core, us in zip(cores, per):
            flat = [v for u in us for v in u]
            r_args[core] = [y.buffer_address(), back.buffer_address(), wk.buffer_address(),
                            self.zero.buffer_address(), len(us)] + flat  # fmt: skip
            w_args[core] = [out.buffer_address(), len(us)] + flat
            c_args[core] = [len(us)]
        named = F.fixed_rt(dict(t=t, k=k, cw=self.cw, row=c * 2, ot=ot, wt8=int(wt8)), r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_combine_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_combine_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_combine_compute.cpp", F.compute_config([]), named, c_args, crs),
        ]
        bf = ttnn.bfloat16
        cw = self.cw
        cbs = [F.cb(0, bf, NB * cw, cores=crs), F.cb(1, bf, k, cores=crs), F.cb(2, bf, 1, cores=crs),
               F.cb(3, bf, cw, cores=crs), F.cb(4, bf, k * cw, cores=crs), F.cb(16, bf, 2 * cw, cores=crs),
               F.cb(5, bf, k * 128 // 2048 + 1, cores=crs), F.cb(6, bf, 1, cores=crs)]  # fmt: skip
        F.run([y, back, wk, self.zero, out], kernels, cbs)
        return out
