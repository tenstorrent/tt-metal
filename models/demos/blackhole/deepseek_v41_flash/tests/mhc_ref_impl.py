# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""mHC (hyper-connection) pieces of one DeepSeek-V4.1-Flash block, on device.

V4.1 staggers the coefficients: each sub-block derives (pre, post, comb) from ITS OWN input stream, but the
`pre` used to collapse the streams going into a sub-block was produced one sub-block earlier (attention uses
the previous layer's FFN `pre`, the FFN uses the attention's `pre`). So the block calls

    pre, post, comb = mhc.mixes(x)        # from the stream x
    h = mhc.collapse(x, pre_in)           # pre_in: coefficient produced by the PREVIOUS sub-block
    y = mhc.expand(f(h), x, post, comb)   # new streams

Layout (token-major): streams x are [T, 1, n, C] fp32 (token t, stream i, hidden c), so collapse and expand are
batched matmuls over tokens:  collapse = pre[T,1,1,n] @ x,  expand = comb^T[T,1,n,n] @ x + post[T,1,n,1] * y.
Coefficients: pre [T,1,1,n], post [T,1,n,1], comb [T,1,n,n] (comb[i, j] = weight of stream i into new stream j).
The projection x_flat @ fn^T (K = n*C = 20480) is split into S chunks along K that run as a batch on a core grid.
The Sinkhorn kernel is the deepseek_prefill ``mhc_split_sinkhorn`` op (parametrisation of ``TtMHCWrap``).
"""

import ttnn
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap

CHUNKS = 32  # K-chunks of the projection (batched across cores)


class DSV41MHC:
    def __init__(self, device, fn, base, scale, dim=5120, n=4, iters=20, eps=1e-6, norm_eps=1e-20):
        cfg = MHCConfig(dim=dim, n=n, sinkhorn_iters=iters, eps=eps, norm_eps=norm_eps)
        self._w = TtMHCWrap(device, cfg, fn.float(), base.float(), scale.float(), tp_axis=None)
        self.n, self.dim = n, dim
        K = fn.shape[0]
        S = CHUNKS
        self.fn_chunks = ttnn.from_torch(
            fn.float().t().contiguous().reshape(S, 1, n * dim // S, K),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self.ckc = self._w.ckc
        self.grid = ttnn.CoreGrid(y=4, x=8)
        # selection matrix [32, 32]: [pre | post | comb^T...] column layout of the kernel output -> see _coeffs

    def mixes(self, x):
        """x [T,1,n,C] fp32 -> (pre [T,1,1,n], post [T,1,n,1], comb [T,1,n,n]) fp32."""
        n, S = self.n, CHUNKS
        T = x.shape[0]
        xr = ttnn.permute(ttnn.reshape(x, [T, 1, S, n * self.dim // S]), (2, 1, 0, 3))  # [S,1,T,K/S]
        mx = ttnn.matmul(xr, self.fn_chunks, compute_kernel_config=self.ckc, core_grid=self.grid)  # [S,1,T,K]
        ss = ttnn.sum(ttnn.multiply(xr, xr), dim=-1, keepdim=True)  # [S,1,T,1]
        both = ttnn.sum(ttnn.concat([mx, ss], dim=-1), dim=0, keepdim=True)  # [1,1,T,K+1]
        K = mx.shape[-1]
        w = self._w
        inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(both[:, :, :, K : K + 1], 1.0 / (n * self.dim)), w.norm_eps))
        mixes = ttnn.multiply(both[:, :, :, :K], inv)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mixes, w.consts, n, w.iters, w.eps)
        return (
            ttnn.reshape(pre, [T, 1, 1, n]),
            ttnn.reshape(post, [T, 1, n, 1]),
            ttnn.reshape(comb, [T, 1, n, n]),
        )

    def collapse(self, x, pre):
        """sum_i pre_i x_i: [T,1,1,n] @ [T,1,n,C] -> [T,1,1,C]"""
        return ttnn.matmul(pre, x, compute_kernel_config=self.ckc, core_grid=self.grid)

    def expand(self, y, residual, post, comb):
        """new_j = post_j * y + sum_i comb[i, j] residual_i; y [T,1,1,C] -> [T,1,n,C]"""
        mixed = ttnn.matmul(comb, residual, transpose_a=True, compute_kernel_config=self.ckc, core_grid=self.grid)
        return ttnn.add(mixed, ttnn.multiply(post, y))
