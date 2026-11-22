# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill per-head norm over T rows in one program: [1, 1, T, W] row tiles -> [1, n, T, 128] heads (bf16).

Replaces the prefill attention chain slice -> reshape -> permute (heads) -> typecast -> [* scale] -> L2 clip or RMS
-> typecast [-> * tau -> typecast] [-> * SCALE -> typecast]: ~12 passes over the q / k / v rows. Per (tile row,
head): y = x [* scale] (fp32), s = sum over the head's 128 columns of y^2 (fp32: a bf16 hi + lo split feeds the
row reduce), f = rsqrt(max(s / thr^2, 1)) (L2 clip,
= min(thr / |y|, 1)) or rsqrt(s / 128 + eps) (RMS), out = bf16(y * f) [-> bf16(* tau)] [-> bf16(* mult)]: the
op chain's rounding points. The head split is only a
tile-index remap: head h of tile row R is x tiles (R, c0 + 4h .. c0 + 4h + 3).

Not bit-equal to the op chain (fp32 sums in another order, and the decode kernels' clip form without the +1e-6):
use the forced-logits gate. Make the PfHeads in Inkling2Prefill.__init__ (constant scaler tile).
"""

import struct

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

HEAD_DIM = 128


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def scale_rows(scale):
    """Per-column scale [n, 128] (torch) -> the kernel's [1, 1, 32, n * 128] fp32 form (row 0)."""
    s = torch.zeros(32, scale.numel())
    s[0] = scale.reshape(-1).float()
    return s.view(1, 1, 32, -1)


class PfHeads:
    def __init__(self, device):
        self.device = device
        m = torch.zeros(1, 1, 32, 32)
        m[..., 0, :] = 1.0
        m[..., 16, :] = 1.0
        self.scaler = ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, x, n, col0=0, norm="l2", thr=1.0, eps=1e-6, scale=None, tau=None, mult=1.0):
        """x [1,1,T,W] bf16 TILE DRAM; heads h = 0..n-1 are columns [col0 + 128 h, col0 + 128 h + 128). norm: "l2"
        (clip to L2 norm thr) or "rms". scale: [1,1,32,n*128] fp32 device tensor (scale_rows). tau: [1,1,T,1] fp32.
        Returns [1, n, T, 128] bf16."""
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
        t, w = x.shape[-2], x.padded_shape[-1]  # tile row stride from the padded width
        assert t % 32 == 0 and col0 % 32 == 0 and col0 + n * HEAD_DIM <= w, (x.shape, col0, n)
        dram = ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, t, HEAD_DIM]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.device, dram)  # fmt: skip
        rt = t // 32
        units = [(r, h) for h in range(n) for r in range(rt)]
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(len(units), grid.x * grid.y)
        cores, crs = F.core_list(self.device, n_cores)
        per = [units[i::n_cores] for i in range(n_cores)]
        sc = scale.buffer_address() if scale is not None else 0
        ta = tau.buffer_address() if tau is not None else 0
        r_args, w_args, c_args = {}, {}, {}
        for core, us in zip(cores, per):
            flat = [v for u in us for v in u]
            r_args[core] = [x.buffer_address(), sc, ta, self.scaler.buffer_address(), len(us)] + flat
            w_args[core] = [out.buffer_address(), len(us)] + flat
            c_args[core] = [len(us)]
        inv = 1.0 / (thr * thr) if norm == "l2" else 1.0 / HEAD_DIM
        named = dict(wt=w // 32, c0=col0 // 32, rt=rt, has_scale=int(scale is not None), has_tau=int(tau is not None),
                     norm=0 if norm == "l2" else 1, inv=_bits(inv), eps=_bits(eps), mult=_bits(mult))  # fmt: skip
        named = F.fixed_rt(named, r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_heads_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_heads_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_heads_compute.cpp", F.compute_config([1, 2, 16, 17, 18]), named, c_args, crs),
        ]
        bf, fp = ttnn.bfloat16, ttnn.float32
        cbs = [F.cb(0, bf, 8, cores=crs), F.cb(1, fp, 8, cores=crs), F.cb(2, fp, 2, cores=crs), F.cb(3, fp, 1, cores=crs),
               F.cb(16, fp, 4, cores=crs), F.cb(17, fp, 4, cores=crs), F.cb(18, fp, 1, cores=crs),
               F.cb(19, bf, 8, cores=crs), F.cb(20, fp, 2, cores=crs)]  # fmt: skip
        io = [x, self.scaler] + ([scale] if scale is not None else []) + ([tau] if tau is not None else []) + [out]
        F.run(io, kernels, cbs)
        return out
