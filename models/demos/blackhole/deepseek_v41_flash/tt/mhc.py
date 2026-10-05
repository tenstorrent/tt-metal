# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""mHC (hyper-connection) pieces of one DeepSeek-V4.1-Flash block, on device.

V4.1 staggers the coefficients: each sub-block derives (pre, post, comb) from ITS OWN input stream, but the
`pre` used to collapse the streams going into a sub-block was produced one sub-block earlier (attention uses
the previous layer's FFN `pre`, the FFN uses the attention's `pre`). So the block calls

    pre, post, comb = mhc.mixes(x)                          # from the stream x
    h = mhc.collapse_norm(x, pre_in, norm_weight, eps)      # == rms_norm(collapse(x, pre_in)) * w, bf16 [1,1,T,C]
    y = mhc.expand(f(h), x, post, comb)                     # new streams

(pre_in: coefficient produced by the PREVIOUS sub-block; ``collapse(x, pre_in)`` alone is still available.)

Layout (token-major): streams x are [T, 1, n, C] fp32 (token t, stream i, hidden c). Coefficients: pre [T,1,1,n],
post [T,1,n,1], comb [T,1,n,n] (comb[i, j] = weight of stream i into new stream j). n = 4, T <= 32.

Each of the three pieces is a small number of fused ``ttnn.generic_op`` kernels (tt/mhc_*.py, kernels in tt/mhc_kernels/):
  mixes:         mhc_proj  (32 cores: x chunk @ fn chunk and sum of squares, fp32) + mhc_post (1 core: sum of partials,
                 RMS scale, Sinkhorn = the deepseek_prefill mhc_split_sinkhorn math, token-major outputs)
  collapse_norm: mhc_collapse (32 cores: h = sum_i pre_i x_i as bf16 token rows + partial sum of squares)
                 + mhc_norm_apply (32 cores: gather partials, rsqrt, * weight)
  expand:        mhc_expand (32 cores: post*y + comb^T x as ONE tile matmul per (token, column tile))
Composite-ttnn versions cost 0.207 / 0.125 / 0.065 ms per call (traced, T=4); the fused ones ~0.08 / 0.03 / 0.02 ms.
"""

import math
import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import mhc_collapse, mhc_collapse_norm, mhc_norm_apply
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_ep import mhc_expand_proj
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_expand import mhc_expand
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_expand2 import mhc_expand2
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_mixes import mhc_post, mhc_proj
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_mixes2 import mhc_post2, mhc_proj2, proj_plan


def _flag(name, default="0"):
    return os.environ.get(name, default) == "1"


# Each fused variant is behind its own env flag (read at call time):
#   DSV41_MHC_MIXES_V2   packed column-tile projection + packed-partial post kernel with the SFPU Sinkhorn (mixes: 68 -> ~25 us at T=4)
#   DSV41_MHC_EXPAND_V2  expand with the comb/post tiles built in the writer, fewer reader pages (T >= 16 faster)
#   DSV41_MHC_EP         expand fused with the NEXT sub-block's projection (implies MIXES_V2's post); layer.py uses expand_mixes()
#   DSV41_MHC_CN_FUSED   collapse + norm as ONE program (30 -> 16 us at T=4)
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap, build_consts

KC = 640  # K-chunk of the projection: S = n*C/KC = 32 cores


class DSV41MHC:
    def __init__(
        self,
        device,
        fn,
        base,
        scale,
        dim=5120,
        n=4,
        iters=20,
        eps=1e-6,
        norm_eps=1e-20,
        kc=KC,
        post_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        assert n == 4 and fn.shape[0] == (2 + n) * n < 32
        cfg = MHCConfig(dim=dim, n=n, sinkhorn_iters=iters, eps=eps, norm_eps=norm_eps)
        self._w = TtMHCWrap(device, cfg, fn.float(), base.float(), scale.float(), tp_axis=None)  # fp32 consts + fn_T
        self.device = device
        self.post_fidelity = post_fidelity
        self.n, self.dim = n, dim
        self.ckc = self._w.ckc  # fp32 HiFi4
        self.grid = ttnn.CoreGrid(y=4, x=8)
        rep = ttnn.ReplicateTensorToMesh(device)

        def up(t):
            return ttnn.from_torch(
                t.contiguous(),
                device=device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        mix_hc = fn.shape[0]  # 24
        self._fn, self._up, self.kc = fn.float(), up, kc
        self._wts = {}
        self.mix_col = mix_hc
        # the post kernel scales the mixes by rsqrt(ss + N*eps) with ss = sum of squares of the flat stream, i.e. a factor
        # sqrt(N) too small vs rsqrt(ss/N + eps): fold sqrt(N) into the SEL tiles (they are applied to the scaled mixes)
        N = n * dim
        consts = build_consts(cfg, scale.float(), base.float())
        consts[0:3] *= math.sqrt(N)
        sel_ss = torch.zeros(32, 32)
        sel_ss[mix_hc, :] = 1.0  # replicates column mix_hc (the sum of squares) to every column
        self.consts10 = up(torch.cat([consts.reshape(8, 32, 32), sel_ss[None], torch.eye(32)[None]], dim=0))
        sel_pre, sel_post, sel_comb, base_pre, base_post, base_comb, rb, cb_col = consts
        # v2: pre | post share one tile: columns 0..3 pre, 4..7 post;  out = (sigmoid(mixes @ SEL_PP + base_PP) + OFF_PP) * SCALE_PP
        sel_pp = sel_pre.clone()
        sel_pp[:, n : 2 * n] = sel_post[:, 0:n]
        base_pp = torch.zeros(32, 32)
        base_pp[:, 0:n] = base_pre[:, 0:n]
        base_pp[:, n : 2 * n] = base_post[:, 0:n]
        scale_pp = torch.zeros(32, 32)
        scale_pp[:, 0:n] = 1.0
        scale_pp[:, n : 2 * n] = 2.0
        off_pp = torch.zeros(32, 32)
        off_pp[:, 0:n] = eps
        self.consts9 = up(
            torch.stack([sel_comb, base_comb, rb, cb_col, sel_pp, base_pp, scale_pp, off_pp, sel_ss], dim=0)
        )
        self.sq_eps = N * norm_eps
        self._norm_cache = {}

    # ------------------------------------------------------------------------------------------------------
    def _wt(self, plan):
        """fn^T chunks [S,1,4*TPJ*32,32]: tile (i*TPJ + j) of chunk r = fn[:, i*C + (r*TPJ + j)*32 + c]^T (rows c, columns = mixes)."""
        key = (plan.S, plan.TPJ)
        wt = self._wts.get(key)
        if wt is None:
            S, TPJ, n, C = plan.S, plan.TPJ, self.n, self.dim
            fT = self._fn.t().contiguous().reshape(n, C // 32, 32, self.mix_col)  # [stream, column tile, c, m]
            w = torch.zeros(S, n, TPJ, 32, 32)
            w[..., : self.mix_col] = fT.reshape(n, S, TPJ, 32, self.mix_col).permute(1, 0, 2, 3, 4)
            wt = self._wts[key] = self._up(
                w.reshape(S, 1, n * TPJ * 32, 32)
            )  # first call per T class must be outside a trace
        return wt

    def mixes(self, x):
        """x [T,1,n,C] fp32 -> (pre [T,1,1,n], post [T,1,n,1], comb [T,1,n,n]) fp32."""
        T = x.shape[0]
        if _flag("DSV41_MHC_MIXES_V2", "1"):
            if (
                T % 8 and T != 4
            ):  # T=12/20 give wrong mixes in the packed kernel (pad to a multiple of 8); T<4 pads to 4; T=5..7 (B=4 spec: drafter 5 rows) pads to 8
                Tp = 4 if T < 4 else -(-T // 8) * 8
                xp = ttnn.pad(x, [(0, Tp - T), (0, 0), (0, 0), (0, 0)], 0.0)
                outs = self.mixes(xp)
                return tuple(ttnn.slice(o, [0, 0, 0, 0], [T, o.shape[1], o.shape[2], o.shape[3]]) for o in outs)
            plan = proj_plan(T, self.dim, self.device)
            part = mhc_proj2(x, self._wt(plan), self.mix_col, plan)
            return mhc_post2(part, self.consts9, plan, self._w.iters, self._w.eps, self.sq_eps, self.post_fidelity)
        kc = self.kc if T <= 8 else self.kc // 2  # more projection cores (fewer gathered rows per core) for big T
        wt = self._wts.get(kc)
        if wt is None:
            S = self.n * self.dim // kc
            wfn = torch.zeros(S, 1, kc, 32)
            wfn[:, 0, :, : self.mix_col] = self._fn.t().contiguous().reshape(S, kc, self.mix_col)
            wt = self._wts[kc] = self._up(wfn)  # first call per T class must be outside a trace
        part = mhc_proj(x, wt, self.mix_col)
        return mhc_post(part, self.consts10, x.shape[0], self._w.iters, self._w.eps, self.sq_eps, self.post_fidelity)

    def collapse(self, x, pre):
        """sum_i pre_i x_i: [T,1,1,n] @ [T,1,n,C] -> [T,1,1,C] (fp32). (The layer wants collapse_norm.)"""
        return ttnn.matmul(pre, x, compute_kernel_config=self.ckc, core_grid=self.grid)

    def _norm_weight(self, w, T):
        """fp32 norm weight (tile row [1,1,1,C]) * sqrt(C), rows replicated to [1,1,T,C]; cached per source tensor. The
        first call for a given weight must happen outside a trace (the layer's compile/warm-up call does)."""
        key = (id(w), T)
        ent = self._norm_cache.get(key)
        if ent is None:
            w4 = ttnn.repeat(ttnn.multiply(w, math.sqrt(self.dim)), [1, 1, T, 1])
            ent = (w, w4)  # keep `w` alive so id(w) stays unique
            self._norm_cache[key] = ent
        return ent[1]

    def collapse_norm(self, x, pre, w, eps=1e-20):
        """== layer._norm(layer._to_row(collapse(x, pre)), w): bf16(sum_i pre_i x_i) normalised in fp32 (RMS, eps) times
        the fp32 weight ``w`` ([1,1,1,C] tile row), returned as bf16 [1,1,T,C]."""
        w4 = self._norm_weight(w, x.shape[0])
        if _flag("DSV41_MHC_CN_FUSED", "1") and x.shape[0] in (4, 16, 32):
            return mhc_collapse_norm(x, pre, w4, eps)
        h, part = mhc_collapse(x, pre)
        return mhc_norm_apply(h, part, w4, eps)

    def collapse_norm_rm(self, x, pre, w, eps=1e-20):
        """collapse_norm that also returns the bf16 ROW_MAJOR [T,1,1,C] copy the MoE dispatch wants:
        -> (h [1,1,T,C] bf16 tile, h_tok [T,1,1,C] bf16 row-major)."""
        w4 = self._norm_weight(w, x.shape[0])
        if _flag("DSV41_MHC_CN_FUSED", "1") and x.shape[0] in (4, 16, 32):
            return mhc_collapse_norm(x, pre, w4, eps, emit_rm=True)
        h, part = mhc_collapse(x, pre)
        return mhc_norm_apply(h, part, w4, eps, emit_rm=True)

    def expand(self, y, residual, post, comb, y2=None):
        """new_j = post_j * (y [+ y2]) + sum_i comb[i, j] residual_i -> [T,1,n,C] fp32. y: [T,1,1,C] fp32 or [1,1,T,C]
        (token rows) fp32/bf16; optional y2 [1,1,T,C] fp32 (added to y inside the kernel)."""
        if _flag("DSV41_MHC_EXPAND_V2", "1"):
            return mhc_expand2(y, residual, post, comb, y2)
        return mhc_expand(y, residual, post, comb, y2)

    def expand_mixes(self, y, residual, post, comb, y2, nxt):
        """expand(y, residual, post, comb, y2) fused with ``nxt.mixes`` of the new streams (nxt: the DSV41MHC of the sub-block that
        consumes them) -> (new streams, (pre, post, comb) of nxt).  Falls back to the two separate calls."""
        T = residual.shape[0]
        if _flag("DSV41_MHC_EP") and y.shape[2] == T and y.shape[0] == 1 and (y2 is None or y2.shape[2] == T):
            self.ep_used = getattr(self, "ep_used", 0) + 1
            plan = proj_plan(T, self.dim, self.device)
            x_new, part = mhc_expand_proj(y, residual, post, comb, y2, nxt._wt(plan), nxt.mix_col, plan)
            return x_new, mhc_post2(part, nxt.consts9, plan, nxt._w.iters, nxt._w.eps, nxt.sq_eps, nxt.post_fidelity)
        x_new = self.expand(y, residual, post, comb, y2)
        return x_new, nxt.mixes(x_new)
