# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 mHC residual (attn_residual / ffn_residual; reference/ds41_ref.py:hc_post, inference Block.hc_post)
on the SP x TP mesh, fp32:

    out stream k = post[:, k] * y + sum_j comb[:, j, k] * x_j          (j = input stream, k = output stream)

mixes = [pre 4 | post 4 | comb 16 row-major (j, k)] (tt/mhc.py:TtHcMixes), so post_k is column 4 + k and comb[j, k]
is column 8 + 4 j + k. That is ttnn.bringup.mhc_post's DEFAULT (comb applied transposed: X'_k = post_k F +
sum_j comb[j n + k] X_j), unlike xing40 (comb_transposed=False).

Per chip (r, c): x [1, 1, S/SP, 4 x H/TP] fp32 streams (tt/layout.py, stream-major), mixes [1, 1, S/SP, 24] fp32
(row-split, replicated over TP), y [1, 1, S/SP, H/TP] fp32 (attn_out: rows over SP, hidden columns over TP) ->
[1, 1, S/SP, 4 x H/TP] fp32. Every chip mixes its own rows and hidden columns: no collective, no weight, no host work.

RESIDUAL_MIX (tt/settings.py) selects the path; both are fp32 SFPU math (no FPU reads, K-30 does not apply):
- fused (default): ttnn.bringup.mhc_post in one program (fp32 weighted sum in SFPU registers).
- addcmul: per output stream multiply + 4 x addcmul (the op chain, kept for comparison).
"""

from __future__ import annotations

import ttnn

from .common import DRAM
from .layout import HC
from .settings import settings

MIX_MODES = ("fused", "addcmul")


class TtHcResidual:
    """Call with (x streams [1, 1, R, n x w] fp32, mixes [1, 1, R, >= (2 + n) n] fp32, y [1, 1, R, w] fp32); returns
    the new streams [1, 1, R, n x w] fp32. Consumes no input."""

    def __init__(self, n: int = HC, mode: str | None = None):
        self.n = n
        self.mode = mode or settings.get("RESIDUAL_MIX")
        assert self.mode in MIX_MODES, (self.mode, MIX_MODES)

    def __call__(self, x: ttnn.Tensor, mix: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor:
        n = self.n
        assert x.dtype == ttnn.float32 and mix.dtype == ttnn.float32 and y.dtype == ttnn.float32, (
            x.dtype,
            mix.dtype,
            y.dtype,
        )
        rows, w = x.shape[-2], y.shape[-1]
        assert x.shape[-1] == n * w and mix.shape[-2] == rows and y.shape[-2] == rows, (x.shape, mix.shape, y.shape)
        assert mix.shape[-1] >= (2 + n) * n, mix.shape
        if self.mode == "fused":
            return self._fused(x, mix, y, rows)
        return self._addcmul(x, mix, y, rows, w)

    def _fused(self, x, mix, y, rows):
        n = self.n
        post = ttnn.slice(mix, [0, 0, 0, n], [1, 1, rows, 2 * n], memory_config=DRAM)
        comb = ttnn.slice(mix, [0, 0, 0, 2 * n], [1, 1, rows, 2 * n + n * n], memory_config=DRAM)
        out = ttnn.bringup.mhc_post(y, x, post, comb)  # comb_transposed=True (default): output k reads comb[j n + k]
        ttnn.deallocate(post)
        ttnn.deallocate(comb)
        return out

    def _addcmul(self, x, mix, y, rows, w):
        n = self.n
        col = lambda k: ttnn.slice(mix, [0, 0, 0, k], [1, 1, rows, k + 1], memory_config=DRAM)  # noqa: E731
        xs = [ttnn.slice(x, [0, 0, 0, j * w], [1, 1, rows, (j + 1) * w], memory_config=DRAM) for j in range(n)]
        outs = []
        for k in range(n):
            p = col(n + k)
            o = ttnn.multiply(y, p, dtype=ttnn.float32, memory_config=DRAM)
            ttnn.deallocate(p)
            for j in range(n):
                c = col(2 * n + j * n + k)
                o2 = ttnn.addcmul(o, xs[j], c, memory_config=DRAM)
                ttnn.deallocate(o)
                ttnn.deallocate(c)
                o = o2
            outs.append(o)
        for t in xs:
            ttnn.deallocate(t)
        z = ttnn.concat(outs, dim=-1, memory_config=DRAM)
        for t in outs:
            ttnn.deallocate(t)
        return z


def build_residual(cfg, mode: str | None = None) -> TtHcResidual:
    return TtHcResidual(n=cfg.hc, mode=mode)
