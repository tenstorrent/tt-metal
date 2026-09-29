# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""mHC residual (attn_residual / ffn_residual) on device, replicated, no CCL, no weights.

out_m = post[:, m] * y + sum_n comb[:, n, m] * x_n (glm_ref.hc_residual: post * y + comb^T @ x).
x [1, 1, S, n*H] (streams packed along the last dim, the reference's token-major [S * n, H]), hc [1, 1, S, (2+n)*n]
fp32 = [pre n | post n | comb n*n row-major], y [1, 1, S, H]. Accumulates in fp32, rounds to bf16 once.
Reuses DeepSeek's TtMHCWrap.hc_post pattern (_streams / _cols, multiply + addcmul).
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _streams


def _f32(t):
    return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32)


class TtHcResidual(LightweightModule):
    def __init__(self, n=4, out_dtype=ttnn.bfloat16):
        self.n, self.out_dtype = n, out_dtype

    def __call__(self, x, hc, y):
        """x [1, 1, S, n*H], hc [1, 1, S, K >= (2+n)*n] fp32, y [1, 1, S, H] -> [1, 1, S, n*H] out_dtype."""
        n = self.n
        S = x.shape[-2]
        res = _streams(x, n)
        res32 = [_f32(r) for r in res]
        if res32[0] is not res[0]:
            for r in res:
                ttnn.deallocate(r)
        y32 = _f32(y)
        h32 = _f32(hc)
        col = lambda k: ttnn.slice(h32, [0, 0, 0, k], [1, 1, S, k + 1])  # noqa: E731
        out = []
        for m in range(n):
            p = col(n + m)
            o = ttnn.multiply(p, y32)
            ttnn.deallocate(p)
            for i in range(n):
                c = col(2 * n + i * n + m)
                o2 = ttnn.addcmul(o, res32[i], c)
                ttnn.deallocate(o)
                ttnn.deallocate(c)
                o = o2
            if self.out_dtype != ttnn.float32:
                o2 = ttnn.typecast(o, self.out_dtype)
                ttnn.deallocate(o)
                o = o2
            out.append(o)
        for t in res32:
            ttnn.deallocate(t)
        if y32 is not y:
            ttnn.deallocate(y32)
        if h32 is not hc:
            ttnn.deallocate(h32)
        z = ttnn.concat(out, dim=-1)
        for t in out:
            ttnn.deallocate(t)
        return z


def build_residual(cfg) -> TtHcResidual:
    return TtHcResidual(n=cfg.hc_mult)
