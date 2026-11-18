# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for Hy4-preview's independent Hyper-Connections (iHC).

Pure torch, runs on CPU. Mirrors HF transformers ``models/hy_v4/modeling_hy_v4.py``:

  - HYV4HyperConnection.forward        -> IHCWrap.hc_pre     (reduce n streams -> 1, emit post)
  - HYV4DecoderLayer.forward (residual)-> IHCWrap.hc_post    (expand 1 -> n streams, add residual)
  - HYV4HyperHead.forward              -> IHCHead            (collapse n streams -> 1 at the output)
  - HYV4Model.forward (expand)         -> ihc_expand         (replicate the embedding to n streams)

iHC is DeepSeek-V4's mHC without the n x n stream-mixing matrix, so there is no Sinkhorn:

    mHC:  new_j = post_j * F(sum_i pre_i x_i) + sum_i comb[i, j] * x_i
    iHC:  new_j = post_j * F(sum_i pre_i x_i) + x_j

Each stream keeps its own identity residual; the streams only meet in the weighted sum that
feeds the sublayer. The remaining differences from mHC are constants:

    fn / base   [2n, n*dim] / [2n]   (pre, post)
    scale       [2] = (a_pre, a_post)
    post        = magnitude * sigmoid(.) + eps   (config ``hc_magnitude``; mHC hard-codes 2 and adds no eps)
    norm eps    = config ``rms_norm_eps`` (1e-5 for Hy4-preview), not ``hc_eps``

Precision, as HF runs it: the parametrization and both mixes are computed in fp32 with autocast
off, and each result is cast back to the streams' dtype. With a BF16 checkpoint the streams are
therefore rounded to BF16 after every sublayer; they are fp32 end to end only if the caller holds
them in fp32. Both behaviours fall out of this module from the dtype of ``x``.

Checkpoint names (tencent/Hy4-preview):
    model.layers.{L}.hc_attn_layer.hc_pre.{hc_fn, hc_base, hc_scale}   attention site
    model.layers.{L}.hc_mlp_layer.hc_pre.{hc_fn, hc_base, hc_scale}    MoE / dense FFN site
    model.hc_head.{hc_head_fn, hc_head_base, hc_head_scale}            head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IHCConfig:
    """iHC hyperparameters. Defaults are the tencent/Hy4-preview config.json values."""

    dim: int = 6144  # C -- model hidden size                      ("hidden_size")
    n: int = 4  # number of residual streams                   ("hc_mult")
    eps: float = 1e-6  # added to pre and post                    ("hc_eps")
    norm_eps: float = 1e-5  # RMSNorm denominator epsilon          ("rms_norm_eps")
    magnitude: float = 2.0  # post = magnitude * sigmoid(.) + eps  ("hc_magnitude")

    @property
    def mix_hc(self) -> int:
        # n (pre) + n (post)
        return 2 * self.n


def _rsqrt_mean_square(xf: torch.Tensor, eps: float) -> torch.Tensor:
    """HYV4UnweightedRMSNorm: the inverse RMS only, [.., 1]; the caller applies it."""
    return torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)


class IHCWrap(nn.Module):
    """Wrap one sublayer F with iHC. Hy4 has two per decoder layer, one around attention and one
    around the MoE / dense FFN, each with independent parameters.

    Parameters (constant at inference):
        fn:    [2n, n*dim]  fused projection, rows [0, n) -> pre, [n, 2n) -> post
        base:  [2n]         bias
        scale: [2]          scalars (a_pre, a_post)
    """

    def __init__(self, cfg: IHCConfig):
        super().__init__()
        self.cfg = cfg
        self.fn = nn.Parameter(torch.empty(cfg.mix_hc, cfg.n * cfg.dim))
        self.base = nn.Parameter(torch.zeros(cfg.mix_hc))
        self.scale = nn.Parameter(torch.full((2,), 0.01))
        nn.init.normal_(self.fn, std=0.006)

    def parametrize(self, x: torch.Tensor):
        """x: [b, s, n, d] -> (pre [b, s, n], post [b, s, n]), fp32."""
        cfg, n = self.cfg, self.cfg.n
        flat = x.flatten(2).float()  # [b, s, n*d]
        mixes = F.linear(flat, self.fn.float()) * _rsqrt_mean_square(flat, cfg.norm_eps)  # [b, s, 2n]
        pre_b, post_b = self.base.float().split(n, dim=-1)
        pre_scale, post_scale = self.scale.float().unbind(0)
        pre_logits, post_logits = mixes.split(n, dim=-1)
        pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + cfg.eps
        post = cfg.magnitude * torch.sigmoid(post_logits * post_scale + post_b) + cfg.eps
        return pre, post

    def hc_pre(self, x: torch.Tensor):
        """Reduce n streams -> 1 and produce post for the matching hc_post.

        x: [b, s, n, d] -> (y [b, s, d] in x's dtype, post [b, s, n] fp32). The projection input
        is RMS-normalised (no learned weight); the reduction uses the raw stream values.
        """
        pre, post = self.parametrize(x)
        y = torch.sum(pre.unsqueeze(-1) * x, dim=2)
        return y.to(x.dtype), post

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """x: F's output [b, s, d]; residual: [b, s, n, d] -> [b, s, n, d] in residual's dtype.

        Per output stream j:  new_j = post_j * x + residual_j.
        """
        y = post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()
        return y.to(residual.dtype)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """x: [b, s, n, d]; ``sublayer``: callable [b, s, d] -> [b, s, d] that owns its pre-norm."""
        residual = x
        h, post = self.hc_pre(x)
        return self.hc_post(sublayer(h), residual, post)


def ihc_expand(h: torch.Tensor, n: int) -> torch.Tensor:
    """Embedding [b, s, d] -> n identical residual streams [b, s, n, d]."""
    return h.unsqueeze(2).expand(-1, -1, n, -1).contiguous()


class IHCHead(nn.Module):
    """Collapse the n streams back to 1 at the output, before the final norm and LM head.

    Parameters: fn [n, n*dim], base [n], scale [1]. Pre-only: there is no residual to rejoin.
    """

    def __init__(self, cfg: IHCConfig):
        super().__init__()
        self.cfg = cfg
        self.fn = nn.Parameter(torch.empty(cfg.n, cfg.n * cfg.dim))
        self.base = nn.Parameter(torch.zeros(cfg.n))
        self.scale = nn.Parameter(torch.full((1,), 0.01))
        nn.init.normal_(self.fn, std=0.006)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [b, s, n, d] -> [b, s, d] in x's dtype."""
        cfg = self.cfg
        flat = x.flatten(2).float()
        mixes = F.linear(flat, self.fn.float()) * _rsqrt_mean_square(flat, cfg.norm_eps)  # [b, s, n]
        pre = torch.sigmoid(mixes * self.scale.float() + self.base.float()) + cfg.eps
        y = (pre.unsqueeze(-1) * x).sum(dim=2)
        return y.to(x.dtype)
