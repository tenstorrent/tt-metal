# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused weighted RMSNorm + row split: rows of a [1,1,R,C] tensor -> R separate bf16 [1,1,1,C] rows, each
rms-normalised with its own weight, in ONE program (one core per row). Replaces the model's attention-input norms:
global layers 3 x (slice + 3 rms ops), loop passes 1 rms + 3 slices, local layers 1 rms (3 ops).
Same math as model._rms (fp32 rms, fp32 weight multiply, bf16 out)."""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.attnres import f32_bits


class FusedRowNorm:
    def __init__(self, device, norm_weights, c=2304, eps=1e-6, out_dtype=ttnn.bfloat16, out_l1=False):
        """norm_weights: list of [C] weights (torch or ttnn); row i of the input uses weight min(i, len - 1)."""
        self.device = device
        self.c = c
        self.eps = eps
        self.out_dtype = out_dtype
        ws = [(ttnn.to_torch(w) if isinstance(w, ttnn.Tensor) else w).float().reshape(1, c) for w in norm_weights]
        self.w = [F.dense_tensor(device, w) for w in ws]
        m = torch.zeros(1, 1, 32, 32)
        m[..., 0, :] = 1.0
        m[..., 16, :] = 1.0
        self.scaler = ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.out = F.OutCache(device)
        self.out_l1 = out_l1  # L1-interleaved output rows (e.g. the MoE input), not DRAM

    def __call__(self, x, num_rows=None):
        """x: [1,1,R,C] row-form (fp32 or bf16). Returns a list of num_rows (default R) rows."""
        F.check_row_input(x)
        n = num_rows or x.shape[-2]
        T, ntc = F.pages(self.c), self.c // 32
        cores, crs = F.core_list(self.device, n)
        if self.out_l1:
            key = ("row_l1", n)
            if key not in self.out.bufs:
                self.out.bufs[key] = [
                    ttnn.from_torch(
                        torch.zeros(1, 1, 1, self.c),
                        dtype=self.out_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )  # fmt: skip
                    for _ in range(n)
                ]
            outs = self.out.bufs[key]
        else:
            outs = [self.out.get(("row", i), 1, self.c, self.out_dtype) for i in range(n)]
        rd, wr = {}, {}
        for i, core in enumerate(cores):
            w = self.w[min(i, len(self.w) - 1)]
            rd[core] = [*F.input_args(x, i), 0, 0, self.scaler.buffer_address(), w.buffer_address()]
            rd[core] += [outs[i].buffer_address(), F.tile_bytes(self.out_dtype), F.ES[self.out_dtype]]
            wr[core] = [outs[i].buffer_address(), F.tile_bytes(self.out_dtype), F.ES[self.out_dtype]]
        named = dict(ntc=ntc, pages=T, nq=0, k=0, has_weight=1, split_out=2 if self.out_l1 else 1)
        kernels = [
            F.kernel("attnres_reader.cpp", ttnn.ReaderConfigDescriptor(), named, rd, crs),
            F.kernel("rownorm_writer.cpp", ttnn.WriterConfigDescriptor(), named, wr, crs),
            F.kernel(
                "attnres_compute.cpp",
                F.compute_config([4], full_sync=True),
                dict(pages=T, nq=0, k=0, has_weight=1, inv_c=f32_bits(1.0 / self.c), eps=f32_bits(self.eps)),
                None,
                crs,
            ),
        ]
        f32 = ttnn.float32
        cbs = [
            F.cb(0, F.input_dtype(x), T, cores=crs),
            F.cb(4, f32, T, cores=crs),
            F.cb(16, self.out_dtype, T, cores=crs),
        ]
        cbs += [F.cb(i, f32, 1, cores=crs) for i in (1, 2, 3, 17, 25, 26, 27, 28)]
        cbs.append(F.stage_cb(24, ntc, crs))
        tw = F.twin(x)
        F.run([x, *([tw[0]] if tw else []), self.scaler, *self.w, *outs], kernels, cbs)
        return outs
