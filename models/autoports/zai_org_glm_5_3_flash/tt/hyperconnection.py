# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC hyper-connections (hc_mult = 4 streams) for GLM-5.3-Flash, composite TTNN path.

Streams are packed `[1, 1, T, 4 * 4096]` float32 (stream i at columns [4096 i, 4096 (i+1))), tokens on tile rows.
Per sublayer (modeling_glm5_next.py:296-329):

    m    = rsqrt(mean(x^2) + rms_norm_eps) * (x @ fn^T)                       # unweighted RMSNorm folded after the matmul
    pre  = sigmoid(scale0 * m[0:4] + base[0:4]) + hc_eps
    post = 2 * sigmoid(scale1 * m[4:8] + base[4:8])
    C    = Sinkhorn_20(softmax_row(scale2 * m[8:24].view(4,4) + base[8:24]) + hc_eps)
    h    = sum_i pre_i X_i                       -> sublayer input (before its RMSNorm)
    X'_j = post_j * y + sum_i C[i, j] X_i        -> after the sublayer

The split/affine/Sinkhorn part is the fused `ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn` kernel (same
formulas as DeepSeek-V4: models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py::parametrize, verified equal to
GLM modeling:313-324); the projection, collapse and mix are composite fp32 ttnn ops.
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.precision import hifi4_fp32

_W = 32  # packed Sinkhorn state width (n*n = 16 <= 32)


def _selection_row_col(n: int):
    """Row-sum / column-sum broadcast matrices for the flattened [T, n*n] Sinkhorn state (identity on pad cols)."""
    rb = torch.zeros(_W, _W)
    cb = torch.zeros(_W, _W)
    for p in range(n * n):
        pi, pj = divmod(p, n)
        for q in range(n * n):
            qi, qj = divmod(q, n)
            if qi == pi:
                rb[q, p] = 1.0
            if qj == pj:
                cb[q, p] = 1.0
    for p in range(n * n, _W):
        rb[p, p] = 1.0
        cb[p, p] = 1.0
    return rb, cb


def build_sinkhorn_consts(n: int, scale: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """[8, 32, 32] fp32 constants of mhc_split_sinkhorn: SEL_pre/post/comb (with the 3 scales folded in), the three
    bias tiles (replicated down the rows) and the Sinkhorn row/col-sum matrices (tile order is the kernel contract)."""
    a_pre, a_post, a_res = (float(s) for s in scale.float())
    base = base.float()
    sel_pre, sel_post, sel_comb = torch.zeros(_W, _W), torch.zeros(_W, _W), torch.zeros(_W, _W)
    for p in range(n):
        sel_pre[p, p] = a_pre
        sel_post[n + p, p] = a_post
    for p in range(n * n):
        sel_comb[2 * n + p, p] = a_res
    base_pre, base_post, base_comb = torch.zeros(_W, _W), torch.zeros(_W, _W), torch.zeros(_W, _W)
    base_pre[:, :n] = base[0:n]
    base_post[:, :n] = base[n : 2 * n]
    base_comb[:, : n * n] = base[2 * n :]
    rb, cb = _selection_row_col(n)
    return torch.stack([sel_pre, sel_post, sel_comb, base_pre, base_post, base_comb, rb, cb], dim=0)


class HyperConnection:
    """One mHC-wrapped sublayer (attn_hc or ffn_hc). All runtime work is device-only."""

    def __init__(
        self,
        mesh_device,
        fn_t: ttnn.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        *,
        hc_mult: int,
        hidden_size: int,
        sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float,
    ):
        self.n = hc_mult
        self.hidden = hidden_size
        self.iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.norm_eps = norm_eps
        self.ckc = hifi4_fp32(mesh_device)
        # fn^T [16384, 24] (bf16 in the checkpoint) used in fp32 (modeling:303)
        fn32 = ttnn.typecast(fn_t, ttnn.float32)
        self.fn_t = ttnn.reshape(fn32, (1, 1, fn32.shape[-2], fn32.shape[-1]))
        self.consts = ttnn.from_torch(
            build_sinkhorn_consts(self.n, scale, base),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _streams(self, x):
        t = x.shape[-2]
        return [ttnn.slice(x, (0, 0, 0, i * self.hidden), (1, 1, t, (i + 1) * self.hidden)) for i in range(self.n)]

    @staticmethod
    def _cols(t, k):
        rows = t.shape[-2]
        return [ttnn.slice(t, (0, 0, 0, j), (1, 1, rows, j + 1)) for j in range(k)]

    def pre(self, x: ttnn.Tensor):
        """x [1,1,T,4*4096] fp32 -> (h [1,1,T,4096] fp32, post [1,1,T,4], comb [1,1,T,16]) (comb[i*4+j] = C[i,j])."""
        t = x.shape[-2]
        mixes = ttnn.matmul(x, self.fn_t, compute_kernel_config=self.ckc, dtype=ttnn.float32)
        ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / x.shape[-1]), self.norm_eps))
        mixes = ttnn.multiply(mixes, inv)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.hc_eps
        )
        r4 = lambda u: ttnn.reshape(u, (1, 1, t, u.shape[-1]))
        pre, post, comb = r4(pre), r4(post), r4(comb)
        streams = self._streams(x)
        coeffs = self._cols(pre, self.n)
        h = ttnn.multiply(streams[0], coeffs[0])
        for s, c in zip(streams[1:], coeffs[1:]):
            h = ttnn.addcmul(h, s, c)
        return h, post, comb

    def post(self, y: ttnn.Tensor, residual: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor) -> ttnn.Tensor:
        """y [1,1,T,4096] (sublayer output), residual [1,1,T,4*4096] fp32 -> new streams [1,1,T,4*4096] fp32."""
        if y.dtype != ttnn.float32:
            y = ttnn.typecast(y, ttnn.float32)
        n = self.n
        res = self._streams(residual)
        cmb = self._cols(comb, n * n)
        out = [ttnn.multiply(p, y) for p in self._cols(post, n)]
        for i in range(n):
            for j in range(n):
                out[j] = ttnn.addcmul(out[j], res[i], cmb[i * n + j])
        return ttnn.concat(out, dim=-1)
