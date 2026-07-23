# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN composite implementation of Manifold-Constrained Hyper-Connections (mHC).

Built entirely from existing ttnn ops -- no custom kernel yet. This is the functional
baseline that the fused mHC kernel (tenstorrent/tt-metal#40707) will be validated
against, and that validates itself against the pure-torch ground truth in
../reference/mhc_reference.py by PCC.

Math mirrors mhc_reference.py exactly (which mirrors DeepSeek-V4-Pro model.py/kernel.py):
    parametrize:  mixes -> (pre, post, comb=Sinkhorn(...))
    hc_pre:       reduce n streams -> 1   (weighted by pre)
    hc_post:      expand 1 -> n streams   (post .* F  +  comb^T @ residual)
    hc_head:      collapse n streams -> 1 (pre-only)

Sinkhorn via fixed selection matrices
-------------------------------------
The Sinkhorn normalisations are the hard part on tile hardware: per token they run on a
tiny n x n (=4x4) matrix with alternating row / column reductions. We flatten each token's
matrix to a row of length n*n (entry (i,j) at column i*n+j) and pack all T tokens as
M[T, W] (W=32, one tile wide). Then

    row-sum broadcast  == M @ RB      col-sum broadcast == M @ CB

with constant 0/1 matrices RB, CB, so every normalisation is `M / (M @ K)` -- a same-shape
elementwise divide, no fragile sub-tile broadcast or reduction. The pad columns [n*n, W)
carry an identity block in RB/CB so their divisor is never zero (the softmax step has no eps
in its denominator).
"""

from __future__ import annotations

import torch

import ttnn
from models.common.utility_functions import is_blackhole


def _compute_kernel_config():
    """HiFi4 + fp32 accumulation -- the Sinkhorn is iterative and fp32-sensitive."""
    kwargs = dict(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return (
        ttnn.types.BlackholeComputeKernelConfig(**kwargs)
        if is_blackhole()
        else ttnn.types.WormholeComputeKernelConfig(**kwargs)
    )


def _selection_matrices(n: int, width: int):
    """Row-sum (RB) and col-sum (CB) broadcast matrices for the matmul-form Sinkhorn.

    For M[., width] holding a token's flattened n*n matrix (entry (i,j) at column i*n+j) in
    columns [0, n*n) and zeros in the pad, (M @ RB)[., p] equals the sum over p's row-group
    and (M @ CB)[., p] the sum over p's column-group. The pad carries an identity block so
    pad divisors stay 1 instead of 0.
    """
    nn = n * n
    assert nn <= width, f"n*n={nn} must fit in one tile width={width}"
    RB = torch.zeros(width, width, dtype=torch.float32)
    CB = torch.zeros(width, width, dtype=torch.float32)
    for p in range(nn):
        pi, pj = divmod(p, n)
        for q in range(nn):
            qi, qj = divmod(q, n)
            if qi == pi:
                RB[q, p] = 1.0  # q shares p's row -> contributes to p's row sum
            if qj == pj:
                CB[q, p] = 1.0  # q shares p's column -> contributes to p's column sum
    for p in range(nn, width):
        RB[p, p] = 1.0
        CB[p, p] = 1.0
    return RB, CB


class TtMHC:
    """One mHC-wrapped sublayer's parameters and ttnn computation.

    Trainable params (constant at inference), uploaded once:
        fn:    [mix_hc, n*dim]  fused projection P
        base:  [mix_hc]         bias b
        scale: [3]              scalars a = (a_pre, a_post, a_res)
    """

    def __init__(self, device, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, dtype=ttnn.float32):
        self.device = device
        self.cfg = cfg
        self.dtype = dtype
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        self.iters = int(cfg.sinkhorn_iters)
        self.ckc = _compute_kernel_config()

        n = cfg.n
        self.n = n
        self.width = 32  # one tile; requires n*n <= 32

        RB, CB = _selection_matrices(n, self.width)
        self.RB = ttnn.from_torch(
            RB.reshape(1, 1, self.width, self.width), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype
        )
        self.CB = ttnn.from_torch(
            CB.reshape(1, 1, self.width, self.width), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype
        )
        # identity embed n*n -> width, so exp() of the pad never divides by a zero row-sum
        emb = torch.zeros(n * n, self.width, dtype=torch.float32)
        for i in range(n * n):
            emb[i, i] = 1.0
        self.EMB = ttnn.from_torch(
            emb.reshape(1, 1, n * n, self.width), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype
        )

        # transposed so mixes = xnorm @ fn_T  (matches reference F.linear(xf, fn))
        self.fn_T = ttnn.from_torch(
            fn.t().contiguous().reshape(1, 1, fn.shape[1], fn.shape[0]),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=dtype,
        )

        self.a_pre, self.a_post, self.a_res = (float(scale[0]), float(scale[1]), float(scale[2]))
        self.b_pre = ttnn.from_torch(base[0:n].reshape(1, 1, 1, n), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
        self.b_post = ttnn.from_torch(
            base[n : 2 * n].reshape(1, 1, 1, n), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype
        )
        self.b_comb = ttnn.from_torch(
            base[2 * n :].reshape(1, 1, 1, n * n), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype
        )

    # ----- parametrization: mixes -> H matrices (the fused-kernel target) -----
    def _sinkhorn(self, logits16):
        """logits16: [1,1,T,n*n] -> doubly-stochastic comb [1,T,n,n]."""
        n, T = self.n, logits16.shape[-2]
        m = ttnn.matmul(logits16, self.EMB, compute_kernel_config=self.ckc)  # [1,1,T,width], pad=0
        e = ttnn.exp(m)
        # row softmax: e / rowsum   (pad rowsum = e_pad via identity -> stays finite)
        m = ttnn.divide(e, ttnn.matmul(e, self.RB, compute_kernel_config=self.ckc))
        m = ttnn.add(m, self.eps)
        # first column normalise
        m = ttnn.divide(m, ttnn.add(ttnn.matmul(m, self.CB, compute_kernel_config=self.ckc), self.eps))
        for _ in range(self.iters - 1):
            m = ttnn.divide(m, ttnn.add(ttnn.matmul(m, self.RB, compute_kernel_config=self.ckc), self.eps))
            m = ttnn.divide(m, ttnn.add(ttnn.matmul(m, self.CB, compute_kernel_config=self.ckc), self.eps))
        # extract the n*n real columns and fold into [1,T,n,n]
        m = ttnn.slice(m, [0, 0, 0, 0], [1, 1, T, n * n])
        m = ttnn.to_layout(m, ttnn.ROW_MAJOR_LAYOUT)
        m = ttnn.reshape(m, [1, T, n, n])
        return ttnn.to_layout(m, ttnn.TILE_LAYOUT)

    def parametrize(self, mixes):
        """mixes: [1,1,T,mix_hc] -> (pre [1,1,T,n], post [1,1,T,n], comb [1,T,n,n])."""
        n, T = self.n, mixes.shape[-2]
        pre_raw = ttnn.slice(mixes, [0, 0, 0, 0], [1, 1, T, n])
        post_raw = ttnn.slice(mixes, [0, 0, 0, n], [1, 1, T, 2 * n])
        comb_raw = ttnn.slice(mixes, [0, 0, 0, 2 * n], [1, 1, T, 2 * n + n * n])

        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.mul(pre_raw, self.a_pre), self.b_pre)), self.eps)
        post = ttnn.mul(ttnn.sigmoid(ttnn.add(ttnn.mul(post_raw, self.a_post), self.b_post)), 2.0)
        comb_logits = ttnn.add(ttnn.mul(comb_raw, self.a_res), self.b_comb)
        comb = self._sinkhorn(comb_logits)
        return pre, post, comb

    # ----- projection: expanded X -> mixes -----
    def _flatten_streams(self, x_streams):
        """[1,T,n,C] -> [1,1,T,n*C] via a row-major reinterpret (n is sub-tile)."""
        _, T, n, C = x_streams.shape
        rm = ttnn.to_layout(x_streams, ttnn.ROW_MAJOR_LAYOUT)
        rm = ttnn.reshape(rm, [1, 1, T, n * C])
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    def project(self, x_streams):
        """x_streams: [1,T,n,C] -> mixes [1,1,T,mix_hc]  (RMSNorm rsqrt applied post-linear)."""
        xf = self._flatten_streams(x_streams)
        mixes_un = ttnn.matmul(xf, self.fn_T, compute_kernel_config=self.ckc)
        ms = ttnn.mean(ttnn.multiply(xf, xf), dim=-1, keepdim=True)
        rsqrt = ttnn.rsqrt(ttnn.add(ms, self.norm_eps))
        return ttnn.multiply(mixes_un, rsqrt)

    # ----- computation: apply H around a sublayer F -----
    def _row_to_batch(self, t, shape):
        """Move the token dim from tile-rows to a batch dim (row-major reinterpret)."""
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.reshape(t, shape)
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT)

    def hc_pre(self, x_streams):
        """[1,T,n,C] -> (y [1,T,1,C], post, comb). y = sum_i pre_i * x_i, fed to F."""
        T = x_streams.shape[1]
        mixes = self.project(x_streams)
        pre, post, comb = self.parametrize(mixes)
        pre_row = self._row_to_batch(pre, [1, T, 1, self.n])  # [1,T,1,n]
        y = ttnn.matmul(pre_row, x_streams, compute_kernel_config=self.ckc)  # [1,T,1,C]
        return y, post, comb

    def hc_post(self, f_out, x_streams, post, comb):
        """f_out: [1,T,1,C]; residual x_streams: [1,T,n,C] -> [1,T,n,C].

        new_j = post_j * F  +  sum_i comb[i,j] * residual_i   (== comb^T @ residual).
        """
        T = x_streams.shape[1]
        post_col = self._row_to_batch(post, [1, T, self.n, 1])  # [1,T,n,1]
        term1 = ttnn.matmul(post_col, f_out, compute_kernel_config=self.ckc)  # outer: [1,T,n,C]
        comb_t = ttnn.transpose(comb, -2, -1)
        term2 = ttnn.matmul(comb_t, x_streams, compute_kernel_config=self.ckc)  # [1,T,n,C]
        return ttnn.add(term1, term2)


class TtMHCHead:
    """Collapse n streams -> 1 at the output (pre-only). Params: fn [n, n*dim], base [n], scale [1]."""

    def __init__(self, device, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, dtype=ttnn.float32):
        self.device = device
        self.cfg = cfg
        self.n = cfg.n
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        self.ckc = _compute_kernel_config()
        self.a = float(scale[0])
        self.fn_T = ttnn.from_torch(
            fn.t().contiguous().reshape(1, 1, fn.shape[1], fn.shape[0]),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=dtype,
        )
        self.base = ttnn.from_torch(base.reshape(1, 1, 1, cfg.n), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    def _flatten_streams(self, x_streams):
        _, T, n, C = x_streams.shape
        rm = ttnn.to_layout(x_streams, ttnn.ROW_MAJOR_LAYOUT)
        rm = ttnn.reshape(rm, [1, 1, T, n * C])
        return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)

    def __call__(self, x_streams):
        """[1,T,n,C] -> [1,T,1,C]."""
        T = x_streams.shape[1]
        xf = self._flatten_streams(x_streams)
        mixes_un = ttnn.matmul(xf, self.fn_T, compute_kernel_config=self.ckc)
        ms = ttnn.mean(ttnn.multiply(xf, xf), dim=-1, keepdim=True)
        rsqrt = ttnn.rsqrt(ttnn.add(ms, self.norm_eps))
        mixes = ttnn.multiply(mixes_un, rsqrt)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.mul(mixes, self.a), self.base)), self.eps)  # [1,1,T,n]
        pre = ttnn.to_layout(pre, ttnn.ROW_MAJOR_LAYOUT)
        pre_row = ttnn.to_layout(ttnn.reshape(pre, [1, T, 1, self.n]), ttnn.TILE_LAYOUT)
        return ttnn.matmul(pre_row, x_streams, compute_kernel_config=self.ckc)  # [1,T,1,C]
