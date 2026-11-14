# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview iHC for PREFILL chunks (T = 32 .. a few hundred token rows per chip): composite ttnn ops on token-row streams [1,1,T,4*D] fp32 TILE
(row t = the 4 streams of token t concatenated). The fused decode kernel (hy4_ihc_fused.py) is built for T <= 16 (one 4x8 core rectangle, L1-resident chunks);
this class is the large-T path (same math, no comb / Sinkhorn): the DSV4 mHC kernels stop at T = 32.

    m    = (x * rsqrt(mean(x^2) + eps)) @ hc_fn^T      pre = sigmoid(m[:4] * s0 + b[:4]) + hc_eps     post = 2 * sigmoid(m[4:] * s1 + b[4:])
    h    = rms_norm(sum_i pre_i x_i) * w                x' = x + post_i * y   (per stream i)
"""
import torch

import ttnn

f32, bf16 = ttnn.float32, ttnn.bfloat16


class Hy4IHCPrefill:
    def __init__(self, device, fn8, base8, scale2, dim=6144, hc_eps=1e-6, norm_eps=1e-5):
        self.device, self.dim, self.hc_eps, self.norm_eps = device, dim, hc_eps, norm_eps
        rep = ttnn.ReplicateTensorToMesh(device)

        def up(t, dtype=f32):
            return ttnn.from_torch(
                t.contiguous(),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        fn = torch.zeros(4 * dim, 32)
        fn[:, :8] = fn8.float().t()
        self.fnT = up(fn.reshape(1, 1, 4 * dim, 32))
        sc = torch.cat([scale2[0].float().repeat(4), scale2[1].float().repeat(4)])
        row = torch.zeros(1, 1, 1, 32)
        row[..., :8] = sc
        self.sc = up(row)
        row = torch.zeros(1, 1, 1, 32)
        row[..., :8] = base8.float()
        self.base = up(row)
        off = torch.zeros(1, 1, 1, 32)
        off[..., :4] = hc_eps
        self.off = up(off)
        mul = torch.zeros(1, 1, 1, 32)
        mul[..., :4] = 1.0
        mul[..., 4:8] = 2.0
        self.mul = up(mul)
        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def pre(self, x, norm_w):
        """x [1,1,T,4D] fp32 TILE, norm_w fp32 [1,1,1,D] TILE -> (h bf16 [1,1,T,D], pp fp32 [1,1,T,32] (cols 0..3 pre, 4..7 post))."""
        T, D = int(x.shape[2]), self.dim
        sq = ttnn.multiply(x, x)
        ms = ttnn.mean(sq, dim=3, keepdim=True)  # [1,1,T,1]
        rs = ttnn.rsqrt(ttnn.add(ms, self.norm_eps))
        m = ttnn.matmul(x, self.fnT, compute_kernel_config=self.ckc)  # [1,1,T,32]
        m = ttnn.multiply(m, rs)
        z = ttnn.add(ttnn.multiply(m, self.sc), self.base)
        pp = ttnn.multiply(ttnn.add(ttnn.sigmoid(z), self.off), self.mul)
        h = None
        for i in range(4):
            xi = ttnn.slice(x, [0, 0, 0, i * D], [1, 1, T, (i + 1) * D])
            pi = ttnn.slice(pp, [0, 0, 0, i], [1, 1, T, i + 1])
            t_ = ttnn.multiply(xi, pi)
            h = t_ if h is None else ttnn.add(h, t_)
        h = ttnn.typecast(h, bf16)
        h = ttnn.rms_norm(h, weight=ttnn.typecast(norm_w, bf16), epsilon=self.norm_eps)
        return h, pp

    def expand(self, x, y, pp):
        """x' = x + post_i * y (y [1,1,T,D] bf16/fp32 token rows) -> [1,1,T,4D] fp32."""
        T, D = int(x.shape[2]), self.dim
        y32 = ttnn.typecast(y, f32) if y.dtype != f32 else y
        outs = []
        for i in range(4):
            xi = ttnn.slice(x, [0, 0, 0, i * D], [1, 1, T, (i + 1) * D])
            po = ttnn.slice(pp, [0, 0, 0, 4 + i], [1, 1, T, 5 + i])
            outs.append(ttnn.add(xi, ttnn.multiply(y32, po)))
        return ttnn.concat(outs, dim=3)
