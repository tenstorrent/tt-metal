# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 mHC collapse (attn_collapse / ffn_collapse, reference/xing_ref.py:hc_collapse) on the 4x2 mesh.

    out [S, H] = sum_n pre[:, n] * stream_n            (pre = hc columns 0..3, from attn_hc / ffn_hc)

Per chip (r, c): streams [1, 1, S/4, 4 x 1792] fp32 (tt/layout.py, stream-major), hc [1, 1, S/4, 24] fp32
(replicated over axis 1) -> out [1, 1, S/4, 1792] fp32, the chip's rows and hidden columns. The column split carries
through, so there is no collective and no weight. fp32 multiply + 3 x addcmul (glm53 tt/collapse.py:TtHcCollapse,
deepseek_v3_d_p tt/mhc/tt_mhc.py:_mix), no host work.

XING_HC_IMPL (tt/mhc.py) = fused (default): one ttnn.bringup.mhc_pre_xing(hc, x, coefficients_given=True) program
(y = sum_i pre_i x_i, fp32 SFPU multiply-adds, exact fp32 unpack); composed: the slice / multiply / addcmul chain.
"""

from __future__ import annotations

import ttnn

from .layout import HC
from .mhc import hc_impl


class TtHcCollapse:
    def __init__(self, n: int = HC, out_dtype=ttnn.float32, impl: str | None = None):
        self.n, self.out_dtype = n, out_dtype
        self.impl = impl or hc_impl()

    def _fused(self, x: ttnn.Tensor, hc: ttnn.Tensor) -> ttnn.Tensor:
        xs = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
        hs = hc if hc.dtype == ttnn.float32 else ttnn.typecast(hc, ttnn.float32)
        _, y = ttnn.bringup.mhc_pre_xing(hs, xs, n=self.n, coefficients_given=True)
        for a, b in ((xs, x), (hs, hc)):
            if a is not b:
                ttnn.deallocate(a)
        if self.out_dtype != ttnn.float32:
            y32 = y
            y = ttnn.typecast(y32, self.out_dtype)
            ttnn.deallocate(y32)
        return y

    def __call__(self, x: ttnn.Tensor, hc: ttnn.Tensor) -> ttnn.Tensor:
        if self.impl == "fused":
            return self._fused(x, hc)
        s4, w = x.shape[-2], x.shape[-1] // self.n
        dram = ttnn.DRAM_MEMORY_CONFIG
        y = None
        for i in range(self.n):
            s = ttnn.slice(x, [0, 0, 0, i * w], [1, 1, s4, (i + 1) * w], memory_config=dram)
            if s.dtype != ttnn.float32:
                s32 = ttnn.typecast(s, ttnn.float32)
                ttnn.deallocate(s)
                s = s32
            c = ttnn.slice(hc, [0, 0, 0, i], [1, 1, s4, i + 1], memory_config=dram)
            if c.dtype != ttnn.float32:
                c32 = ttnn.typecast(c, ttnn.float32)
                ttnn.deallocate(c)
                c = c32
            if y is None:
                y = ttnn.multiply(s, c, dtype=ttnn.float32, memory_config=dram)
            else:
                nxt = ttnn.addcmul(y, s, c, memory_config=dram)
                ttnn.deallocate(y)
                y = nxt
            ttnn.deallocate(s)
            ttnn.deallocate(c)
        if self.out_dtype != ttnn.float32:
            y32 = y
            y = ttnn.typecast(y32, self.out_dtype)
            ttnn.deallocate(y32)
        return y


def build_collapse(cfg) -> TtHcCollapse:
    return TtHcCollapse(n=cfg.hc_mult)
