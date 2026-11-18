# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview independent Hyper-Connections (iHC) on device (tenstorrent/tt-metal#65891):

    X'_j = post_j * F(sum_i pre_i * X_i) + X_j

Mirrors the pure-torch ground truth in ../reference/ihc.py class for class
(IHCWrap -> TtIHCWrap, IHCHead -> TtIHCHead, ihc_expand -> ihc_expand).

Tensor convention: the n residual streams are packed along the last dim, x [1, 1, T, n*C] with
stream i at columns [i*C, (i+1)*C) and the T tokens (prefill) or users (decode) down the tile
rows. Giving n a dim of its own would pad it 4 -> 32 in every tile. F is any callable
[1,1,T,C] -> [1,1,T,C] and owns its own pre-norm, as in HYV4DecoderLayer.

Hidden split (tt/common.py MeshPlan). With the hidden dimension split ``tp`` ways, each chip
holds its own hidden slice of every stream, [1, 1, T, n*C/tp]. The projection is then a partial
inner product and the RMS a partial sum of squares, so both are summed across the hidden axis --
in ONE all-reduce, with the sum of squares riding as an extra column of the mixes. That
all-reduce is the only cross-chip traffic iHC adds per sublayer; the two mixes are chip-local.

iHC has no stream-mixing matrix, so nothing here is iterative: it is composite ttnn on tensors
that are either one tile wide (the mixes) or the streams themselves.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.tt.common import MeshPlan, compute_config


def _upload(plan, t, shape, dtype):
    """A small constant, the same on every chip."""
    return ttnn.from_torch(
        t.float().contiguous().reshape(shape), layout=ttnn.TILE_LAYOUT, device=plan.mesh_device, dtype=dtype
    )


def _streams(x, n):
    """The n per-stream slices of a packed [1,1,T,n*C] tensor."""
    T, C = x.shape[-2], x.shape[-1] // n
    return [ttnn.slice(x, [0, 0, 0, i * C], [1, 1, T, (i + 1) * C]) for i in range(n)]


def _cols(t, k, start=0):
    """k single-column slices of [1,1,T,*] from column ``start``, each broadcastable across a stream's C."""
    T = t.shape[-2]
    return [ttnn.slice(t, [0, 0, 0, start + j], [1, 1, T, start + j + 1]) for j in range(k)]


def _mix(streams, coeffs):
    """sum_i coeffs[i] * streams[i], with each [1,1,T,1] coefficient broadcast down its stream."""
    y = ttnn.multiply(coeffs[0], streams[0])
    for s, c in zip(streams[1:], coeffs[1:]):
        y = ttnn.addcmul(y, s, c)
    return y


def ihc_expand(h, n: int):
    """Embedding [1,1,T,C] -> n identical residual streams [1,1,T,n*C]."""
    return ttnn.repeat(h, [1, 1, 1, n])


class _TtIHCProjection(LightweightModule):
    """The part TtIHCWrap and TtIHCHead share: mixes = RMSNorm(x) @ fn^T, reduced across the hidden split."""

    def __init__(self, plan: MeshPlan, cfg, fn: torch.Tensor, dtype):
        self.plan = plan
        self.cfg = cfg
        self.n = cfg.n
        self.dtype = dtype
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        # HiFi4 with fp32 accumulation: the projection reduces over n*C = 24576 terms.
        self.ckc = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)

        k, tp = fn.shape[0], plan.hidden_tp
        assert fn.shape[1] == self.n * cfg.dim, f"fn is {tuple(fn.shape)}, expected [*, {self.n * cfg.dim}]"
        assert cfg.dim % tp == 0, f"hidden {cfg.dim} does not split {tp} ways"
        # Transposed so mixes = x @ fn_T, matching the reference F.linear(flat, fn). Sharded in the
        # 3D (stream, hidden, out) form so fn's rows land in the order x is split across chips: each
        # chip's own hidden slice of every stream, not one whole stream.
        self.fn_T = ttnn.from_torch(
            fn.float().t().contiguous().reshape(1, self.n, cfg.dim, k),
            layout=ttnn.TILE_LAYOUT,
            device=plan.mesh_device,
            dtype=dtype,
            mesh_mapper=plan.shard_hidden(2),
        )
        self.fn_T = ttnn.reshape(self.fn_T, [1, 1, self.n * (cfg.dim // tp), k])
        self.k = k

    def project(self, x):
        """[1,1,T,n*C/tp] -> mixes [1,1,T,K] = RMSNorm(x) @ fn_T, identical on every chip of a hidden group.

        RMSNorm carries no learned weight, so its rsqrt commutes with the linear and is applied
        after it: one [T,1] broadcast instead of a full [T,n*C] scale. The RMS divisor is the
        global width n*C, not this chip's slice.
        """
        T, k, tp = x.shape[-2], self.k, self.plan.hidden_tp
        mixes_un = ttnn.matmul(x, self.fn_T, compute_kernel_config=self.ckc)
        ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        if tp > 1:
            both = self.plan.all_reduce_hidden(ttnn.concat([mixes_un, ss], dim=-1))
            mixes_un = ttnn.slice(both, [0, 0, 0, 0], [1, 1, T, k])
            ss = ttnn.slice(both, [0, 0, 0, k], [1, 1, T, k + 1])
        ms = ttnn.multiply(ss, 1.0 / (x.shape[-1] * tp))
        return ttnn.multiply(mixes_un, ttnn.rsqrt(ttnn.add(ms, self.norm_eps)))


class TtIHCWrap(_TtIHCProjection):
    """One iHC-wrapped sublayer. Hy4 has two per decoder layer -- around attention and around the
    MoE / dense FFN -- each with independent parameters.

    Parameters (constant at inference), uploaded once:
        fn:    [2n, n*dim]  fused projection, rows [0, n) -> pre, [n, 2n) -> post
        base:  [2n]         bias
        scale: [2]          scalars (a_pre, a_post)

    ``dtype`` is the dtype of the streams and of every iHC intermediate. fp32 is what HF computes
    the parametrization and both mixes in; it is the only dtype the PCC tests cover.
    """

    def __init__(
        self, plan: MeshPlan, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, dtype=ttnn.float32
    ):
        n = cfg.n
        assert fn.shape[0] == 2 * n, f"fn is {tuple(fn.shape)}, expected [{2 * n}, {n * cfg.dim}]"
        assert tuple(base.shape) == (2 * n,), f"base is {tuple(base.shape)}, expected [{2 * n}]"
        assert tuple(scale.shape) == (2,), f"scale is {tuple(scale.shape)}, expected [2]"
        super().__init__(plan, cfg, fn, dtype)
        # One row each, broadcast down the T rows: columns [0, n) are pre, [n, 2n) are post.
        ones = torch.ones(n)
        scale = scale.float()
        self.scale_row = _upload(plan, torch.cat([ones * scale[0], ones * scale[1]]), (1, 1, 1, 2 * n), dtype)
        self.base_row = _upload(plan, base, (1, 1, 1, 2 * n), dtype)
        self.gain_row = _upload(plan, torch.cat([ones, ones * float(cfg.magnitude)]), (1, 1, 1, 2 * n), dtype)

    def parametrize(self, mixes):
        """mixes [1,1,T,2n] -> [1,1,T,2n]: columns [0, n) = pre, [n, 2n) = post.

        pre = sigmoid(a_pre * m + b) + eps;  post = magnitude * sigmoid(a_post * m + b) + eps.
        """
        g = ttnn.sigmoid(ttnn.add(ttnn.multiply(mixes, self.scale_row), self.base_row))
        return ttnn.add(ttnn.multiply(g, self.gain_row), self.eps)

    def hc_pre(self, x):
        """[1,1,T,n*C] -> (y [1,1,T,C], post [1,1,T,n]).

        y = sum_i pre_i * x_i is the single stream handed to F. The reduction uses the raw stream
        values; only the projection input is normalised.
        """
        T, n = x.shape[-2], self.n
        g = self.parametrize(self.project(x))
        post = ttnn.slice(g, [0, 0, 0, n], [1, 1, T, 2 * n])
        return _mix(_streams(x, n), _cols(g, n)), post

    def hc_post(self, x, residual, post):
        """x: F's output [1,1,T,C]; residual: [1,1,T,n*C] -> [1,1,T,n*C].

        Per output stream j:  new_j = post_j * x + residual_j. There is no stream mixing, so the
        residual is added packed, without being sliced into streams.
        """
        return ttnn.add(residual, ttnn.concat([ttnn.multiply(p, x) for p in _cols(post, self.n)], dim=-1))

    def forward(self, x, sublayer, sublayer_dtype=None):
        """x: [1,1,T,n*C]; sublayer: [1,1,T,C] -> [1,1,T,C]. Returns [1,1,T,n*C].

        The sublayer has its own dtype: ``sublayer_dtype`` casts the collapsed hidden to it going
        in, and back to ``x``'s dtype coming out.
        """
        residual = x
        h, post = self.hc_pre(x)
        if sublayer_dtype is None:
            h = sublayer(h)
        else:
            cast = ttnn.typecast(h, sublayer_dtype)
            ttnn.deallocate(h)
            out = sublayer(cast)
            h = ttnn.typecast(out, x.dtype)
            ttnn.deallocate(out)
            ttnn.deallocate(cast)
        return self.hc_post(h, residual, post)


class TtIHCHead(_TtIHCProjection):
    """Collapse the n streams back to 1 at the model output, before the final norm.

    Parameters: fn [n, n*dim], base [n], scale [1]. Pre-only. Runs once per model.
    """

    def __init__(
        self, plan: MeshPlan, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, dtype=ttnn.float32
    ):
        n = cfg.n
        assert fn.shape[0] == n, f"fn is {tuple(fn.shape)}, expected [{n}, {n * cfg.dim}]"
        assert tuple(base.shape) == (n,), f"base is {tuple(base.shape)}, expected [{n}]"
        super().__init__(plan, cfg, fn, dtype)
        self.a = float(scale.flatten()[0])
        self.base = _upload(plan, base, (1, 1, 1, n), dtype)

    def forward(self, x):
        """[1,1,T,n*C] -> [1,1,T,C]."""
        mixes = self.project(x)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mixes, self.a), self.base)), self.eps)
        return _mix(_streams(x, self.n), _cols(pre, self.n))
