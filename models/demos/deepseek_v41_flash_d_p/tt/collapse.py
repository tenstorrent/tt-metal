# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 mHC collapse (attn_collapse; reference/ds41_ref.py:hc_pre, inference Block.hc_pre's sum) on the
SP x TP mesh, fp32:

    out [S, H] = sum_n pre[:, n] * stream_n

Single-pass mHC (DS41-3): the attention collapses with the PENDING pre (the previous layer's ffn_hc pre carried in the
block boundary; layer 0: one-hot on stream 0), not attn_hc's. ffn_collapse uses the same module with attn_hc's
columns 0..3 as ``pre`` (bringup/hooks.py:_ffn_collapse_host_fn passes all 24 mixes).

Per chip (r, c): streams [1, 1, S/SP, 4 x H/TP] fp32 TILE (tt/layout.py, stream-major), pre [1, 1, S/SP, >= 4] fp32
TILE (row-split, replicated over TP; only columns 0..3 are read) -> out [1, 1, S/SP, H/TP] fp32, the chip's rows and
hidden columns. The column split carries through, so there is no collective and no weight. ttnn.multiply + 3 x
ttnn.addcmul with a [S/SP, 1] column broadcast: both take the SFPU path at fp32 (ternary_program_factory.cpp: the FPU
addcmul is only for non-fp32), so every product and sum is an exact fp32 operation (K-30 does not apply). No host work.

MHC_FUSED=on (P.14) and a whole [S/SP, 24] mixes tensor as ``pre``: ONE ttnn.bringup.mhc_pre_xing(pre, x,
coefficients_given=True) program instead (the same fp32 SFPU multiply + fused multiply-adds in stream order). The
all-device model then folds attn / ffn collapse into the hc steps (tt/mhc.py:TtHcMixes.fused); this path is its final
collapse (and the hybrid's ffn_collapse).
"""

from __future__ import annotations

import ttnn

from .ccl import dealloc
from .common import DRAM
from .cores import max_grid, sub_cores
from .layout import HC
from .settings import settings


class TtHcCollapse:
    """Call with (streams [1, 1, R, n x w] fp32 or bf16, pre [1, 1, R, >= n] fp32); returns [1, 1, R, w] in the
    streams' dtype (bf16 streams: BF16_ACTIVATIONS=on, P.20). Consumes neither input."""

    def __init__(self, n: int = HC, fused: bool | None = None):
        self.n = n
        self.fused = settings.get("MHC_FUSED") == "on" if fused is None else bool(fused)

    def __call__(self, x: ttnn.Tensor, pre: ttnn.Tensor) -> ttnn.Tensor:
        assert x.dtype in (ttnn.float32, ttnn.bfloat16) and pre.dtype == ttnn.float32, (x.dtype, pre.dtype)
        rows, w = x.shape[-2], x.shape[-1] // self.n
        assert x.shape[-1] == self.n * w and pre.shape[-1] >= self.n and pre.shape[-2] == rows, (x.shape, pre.shape)
        ng = self.n * (self.n + 2)
        if self.fused and pre.shape[-1] == ng:
            _, y = ttnn.bringup.mhc_pre_xing(pre, x, n=self.n, coefficients_given=True, max_grid=max_grid())
            return y
        if x.dtype == ttnn.bfloat16:
            # BF16_ACTIVATIONS=on (P.20) with a bare [R, n] pre (the hybrid harness): the fork's collapse on the pre
            # padded to the mixes' width (only columns 0..n-1 are read); y bf16, one rounding of the fp32 sum
            p = ttnn.pad(pre, [(0, 0), (0, 0), (0, 0), (0, ng - pre.shape[-1])], 0.0) if pre.shape[-1] < ng else pre
            _, y = ttnn.bringup.mhc_pre_xing(p, x, n=self.n, coefficients_given=True, max_grid=max_grid())
            if p is not pre:
                dealloc(p)
            return y
        y = None
        for i in range(self.n):
            s = ttnn.slice(
                x, [0, 0, 0, i * w], [1, 1, rows, (i + 1) * w], memory_config=DRAM, sub_core_grids=sub_cores(x)
            )
            c = ttnn.slice(pre, [0, 0, 0, i], [1, 1, rows, i + 1], memory_config=DRAM, sub_core_grids=sub_cores(pre))
            if y is None:
                y = ttnn.multiply(s, c, dtype=ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores(s))
            else:
                nxt = ttnn.addcmul(y, s, c, memory_config=DRAM)
                dealloc(y)
                y = nxt
            dealloc(s)
            dealloc(c)
        return y


def build_collapse(cfg) -> TtHcCollapse:
    return TtHcCollapse(n=cfg.hc)
