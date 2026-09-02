# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Manifold-Constrained Hyper-Connections (mHC) on device:

    X' = H_res @ X + H_post.T @ F(H_pre @ X)

Mirrors the pure-torch ground truth in ../../reference/mhc/mhc_reference.py class for class
(MHCWrap -> TtMHCWrap, MHCHead -> TtMHCHead, mhc_expand -> mhc_expand) so the PCC tests can
compare method against method.

The implementation splits along the design boundary:
  - Parametrization (X -> the H matrices, including the Sinkhorn projection of H_res onto the
    doubly-stochastic manifold) is the fused kernel
    ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn, fed by `project`.
  - Computation (apply the H matrices around the sublayer F) stays composite ttnn; see hc_post
    for why.

Tensor convention: the n residual streams occupy their own dim, x [1, T, n, C], with the
T = B*S tokens down the tile rows. F is any callable [1,T,1,C] -> [1,T,1,C] (attention,
MLP, ...) and owns its own pre-norm, exactly like DeepSeek's Block.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole

W = 32  # one tile wide; the packed Sinkhorn state requires n*n <= W


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


def _selection_row_col(n):
    """Row-sum (RB) and col-sum (CB) broadcast matrices for the matmul-form Sinkhorn.

    Sinkhorn alternates row and column normalisations of a tiny n x n matrix per token --
    awkward on tile hardware. Flattening each token's matrix to a row of length n*n (entry
    (i,j) at column i*n+j) and packing the T tokens as M[T, W] turns

        row-sum broadcast == M @ RB        col-sum broadcast == M @ CB

    so every normalisation becomes `M / (M @ K)`: a same-shape elementwise divide, with no
    sub-tile broadcast or reduction. The pad columns [n*n, W) carry an identity block so their
    divisor is never zero (the softmax step has no eps in its denominator).
    """
    RB = torch.zeros(W, W, dtype=torch.float32)
    CB = torch.zeros(W, W, dtype=torch.float32)
    for p in range(n * n):
        pi, pj = divmod(p, n)
        for q in range(n * n):
            qi, qj = divmod(q, n)
            if qi == pi:
                RB[q, p] = 1.0  # q shares p's row -> contributes to p's row sum
            if qj == pj:
                CB[q, p] = 1.0  # q shares p's column -> contributes to p's column sum
    for p in range(n * n, W):
        RB[p, p] = 1.0
        CB[p, p] = 1.0
    return RB, CB


def build_consts(cfg, scale, base) -> torch.Tensor:
    """Bake a, b and the Sinkhorn selection matrices into the kernel's constant tensor.

    -> [8, 32, 32] fp32. The kernel is pure tile ops, so the host owns this layout; tile order
    is the host <-> kernel contract:
        0 SEL_pre   1 SEL_post   2 SEL_comb   3 base_pre   4 base_post   5 base_comb   6 RB   7 CB

    SEL_* extract and left-align each group out of the mix_hc-wide mixes and fold in the scalar
    a, so `mixes @ SEL_g` yields `a_g * raw_g` left-aligned. RB/CB are the Sinkhorn row/col-sum
    matrices (see _selection_row_col).
    """
    n = cfg.n
    a_pre, a_post, a_res = (float(scale[0]), float(scale[1]), float(scale[2]))
    base = base.float()

    sel_pre = torch.zeros(W, W)
    sel_post = torch.zeros(W, W)
    sel_comb = torch.zeros(W, W)
    for p in range(n):
        sel_pre[p, p] = a_pre
        sel_post[n + p, p] = a_post
    for p in range(n * n):
        sel_comb[2 * n + p, p] = a_res

    # per-column bias replicated down every token row (the kernel's add is tile + tile)
    base_pre = torch.zeros(W, W)
    base_post = torch.zeros(W, W)
    base_comb = torch.zeros(W, W)
    base_pre[:, :n] = base[0:n]
    base_post[:, :n] = base[n : 2 * n]
    base_comb[:, : n * n] = base[2 * n :]

    RB, CB = _selection_row_col(n)
    return torch.stack([sel_pre, sel_post, sel_comb, base_pre, base_post, base_comb, RB, CB], dim=0)


def _upload(device, t, shape, dtype):
    return ttnn.from_torch(t.contiguous().reshape(shape), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)


def _flatten_streams(x):
    """[1,T,n,C] -> [1,1,T,n*C] via a row-major reinterpret (n is sub-tile)."""
    _, T, n, C = x.shape
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    rm = ttnn.reshape(rm, [1, 1, T, n * C])
    return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)


def _row_to_batch(t, shape):
    """Move the token dim from tile-rows to a batch dim (row-major reinterpret)."""
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, shape)
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def _project(x, fn_T, norm_eps, ckc):
    """[1,T,n,C] -> [1,1,T,K] = RMSNorm(flatten(x)) @ fn_T.

    RMSNorm carries no learned weight, so its rsqrt commutes with the linear and is applied
    after it -- one [T,1] broadcast instead of a full [T,n*C] scale.
    """
    xf = _flatten_streams(x)
    mixes_un = ttnn.matmul(xf, fn_T, compute_kernel_config=ckc)
    ms = ttnn.mean(ttnn.multiply(xf, xf), dim=-1, keepdim=True)
    rsqrt = ttnn.rsqrt(ttnn.add(ms, norm_eps))
    return ttnn.multiply(mixes_un, rsqrt)


def mhc_expand(h, n: int):
    """Embedding [1,1,T,C] -> n identical residual streams [1,T,n,C]."""
    T, C = h.shape[-2], h.shape[-1]
    rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
    rm = ttnn.repeat(ttnn.reshape(rm, [1, T, 1, C]), [1, 1, n, 1])
    return ttnn.to_layout(rm, ttnn.TILE_LAYOUT)


class TtMHCWrap(LightweightModule):
    """One mHC-wrapped sublayer. DeepSeek-V4 has two per transformer block -- one around
    attention, one around the MoE FFN -- each with independent parameters.

    Parameters (constant at inference), uploaded once:
        fn:    [mix_hc, n*dim]  fused projection P
        base:  [mix_hc]         bias b
        scale: [3]              scalars a = (a_pre, a_post, a_res)
    The H matrices are not stored: they are recomputed from X every forward pass.
    """

    def __init__(self, device, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, dtype=ttnn.float32):
        # the fused parametrization op is fp32-only; reject here rather than fail deep in hc_pre
        assert dtype == ttnn.float32, f"TtMHCWrap is fp32-only (fused op requires FLOAT32), got {dtype}"
        self.cfg = cfg
        self.n = cfg.n
        self.iters = int(cfg.sinkhorn_iters)
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        self.ckc = _compute_kernel_config()
        # transposed so mixes = xnorm @ fn_T, matching the reference F.linear(xf, fn)
        self.fn_T = _upload(device, fn.t(), (1, 1, fn.shape[1], fn.shape[0]), dtype)
        self.consts = _upload(device, build_consts(cfg, scale, base), (8, W, W), dtype)

    def project(self, x):
        """[1,T,n,C] -> mixes [1,1,T,mix_hc], the fused kernel's input."""
        return _project(x, self.fn_T, self.norm_eps, self.ckc)

    def hc_pre(self, x):
        """[1,T,n,C] -> (y [1,T,1,C], post [T,n], comb [1,T,n,n]).

        y = sum_i pre_i * x_i is the single stream handed to F. The reduction uses the raw
        stream values; only the projection input is normalised.
        """
        T = x.shape[1]
        mixes = self.project(x)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.eps
        )
        pre_row = _row_to_batch(pre, [1, T, 1, self.n])
        y = ttnn.matmul(pre_row, x, compute_kernel_config=self.ckc)
        return y, post, _row_to_batch(comb, [1, T, self.n, self.n])

    def hc_post(self, x, residual, post, comb):
        """x: F's output [1,T,1,C]; residual: [1,T,n,C] -> [1,T,n,C].

        Per output stream j:  new_j = post_j * x + sum_i comb[i,j] * residual_i,
        i.e. the residual mixing applies comb^T (still doubly-stochastic).
        """
        T = residual.shape[1]
        post_col = _row_to_batch(post, [1, T, self.n, 1])
        # Both matmuls are shape-bound, not bandwidth-bound: n=4 pads to a full tile row and the
        # contraction axis is 1 here and 4 below, so they hold 0.005% FPU and 213 GB/s while the
        # add over the same tensors reaches 445 GB/s. A fused kernel would inherit that padding --
        # the win is packing the n axis, which is a layout change rather than a kernel.
        term1 = ttnn.matmul(post_col, x, compute_kernel_config=self.ckc)  # outer product
        term2 = ttnn.matmul(ttnn.transpose(comb, -2, -1), residual, compute_kernel_config=self.ckc)
        return ttnn.add(term1, term2)

    def forward(self, x, sublayer):
        """x: [1,T,n,C]; sublayer: [1,T,1,C] -> [1,T,1,C]. Returns [1,T,n,C]."""
        residual = x
        h, post, comb = self.hc_pre(x)
        h = sublayer(h)
        return self.hc_post(h, residual, post, comb)


class TtMHCHead(LightweightModule):
    """Collapse the n streams back to 1 at the model output.

    Parameters: fn [n, n*dim], base [n], scale [1]. Pre-only -- there is no H_res here, so no
    Sinkhorn and no fused kernel: this runs once per model, not once per layer.
    """

    def __init__(self, device, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, dtype=ttnn.float32):
        self.cfg = cfg
        self.n = cfg.n
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        self.ckc = _compute_kernel_config()
        self.a = float(scale[0])
        self.fn_T = _upload(device, fn.t(), (1, 1, fn.shape[1], fn.shape[0]), dtype)
        self.base = _upload(device, base, (1, 1, 1, cfg.n), dtype)

    def forward(self, x):
        """[1,T,n,C] -> [1,T,1,C]."""
        T = x.shape[1]
        mixes = _project(x, self.fn_T, self.norm_eps, self.ckc)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.mul(mixes, self.a), self.base)), self.eps)
        pre_row = _row_to_batch(pre, [1, T, 1, self.n])
        return ttnn.matmul(pre_row, x, compute_kernel_config=self.ckc)
