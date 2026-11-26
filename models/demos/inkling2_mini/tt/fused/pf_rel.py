# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill rel bias in the SDPA's skewed layout, in one program: rel states [1, n, T, 32] bf16 ROW_MAJOR (PfHeads'
16-wide heads, head h in columns 16 (h % 2) ..) -> [1, n, T, W + 32] bf16 ROW_MAJOR with row q holding
rel[q, d] = states[q] . proj[:, d] at column W - d + (q % 32) (scaled_dot_product_attention(rel_skewed=True)).

Rows with the same phase r = q % 32 share one skewed projection B_r (skew_proj), so each (head, phase) is a plain
matmul of one K tile; the result is packed untilized straight into the output rows. Same products and rounding as
ttnn.matmul(states, proj) (HiFi4, fp32 dest, bf16 out), so the bias values are bit-equal; it replaces that matmul
and the untilize of its [n, T, W] output."""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F


def skew_proj(proj):
    """proj [32, W] (half_proj rows) -> [1, 32, 32, W + 32] torch: B_r[j, y] = proj[j, W - y + r] for
    y in [r + 1, r + W], zero elsewhere."""
    k, w = proj.shape
    out = torch.zeros(32, k, w + 32)
    for r in range(32):
        y = torch.arange(r + 1, r + w + 1)
        out[r][:, y] = proj[:, w - y + r].float()
    return out.view(1, 32, k, w + 32)


class PfRel:
    def __init__(self, device):
        self.device = device

    def __call__(self, states, bsk):
        """states [1, n, T, 32] bf16 ROW_MAJOR DRAM; bsk: skew_proj on device (bf16 TILE)."""
        _, n, t, k = states.shape
        nb = bsk.shape[-1] // 32
        assert states.dtype == ttnn.bfloat16 and states.layout == ttnn.ROW_MAJOR_LAYOUT and k == 32, states
        assert t % 32 == 0 and bsk.shape[1] == 32 and nb % 3 == 0, (states.shape, bsk.shape)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, t, nb * 32]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                             self.device, ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        units = [(h, r) for h in range(n) for r in range(32)]
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(len(units), grid.x * grid.y)
        cores, crs = F.core_list(self.device, n_cores)
        per = [units[i::n_cores] for i in range(n_cores)]
        r_args, w_args, c_args = {}, {}, {}
        for core, us in zip(cores, per):
            flat = [v for u in us for v in u]
            r_args[core] = [states.buffer_address(), bsk.buffer_address(), len(us)] + flat
            w_args[core] = [out.buffer_address(), len(us)] + flat
            c_args[core] = [len(us)]
        named = F.fixed_rt(dict(t=t, nb=nb, bw=3), r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_rel_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_rel_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_rel_compute.cpp", F.compute_config([]), named, c_args, crs),
        ]
        bf = ttnn.bfloat16
        cbs = [F.cb(0, bf, 2, cores=crs), F.cb(1, bf, nb, cores=crs), F.cb(2, bf, 2, cores=crs),
               F.cb(16, bf, 2 * nb, cores=crs)]  # fmt: skip
        F.run([states, bsk, out], kernels, cbs)
        return out
