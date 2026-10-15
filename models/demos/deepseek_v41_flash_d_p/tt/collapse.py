# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 mHC collapse (attn_collapse; reference/ds41_ref.py:hc_pre, inference Block.hc_pre's sum) on the
SP x TP mesh, fp32:

    out [S, H] = sum_n pre[:, n] * stream_n

Single-pass mHC (DS41-3): the attention collapses with the PENDING pre (the previous layer's ffn_hc pre carried in the
block boundary; layer 0: one-hot on stream 0), not attn_hc's. ffn_collapse uses the same module with attn_hc's
columns 0..3 as ``pre``.

Per chip (r, c): streams [1, 1, S/SP, 4 x H/TP] fp32 TILE (tt/layout.py, stream-major), pre [1, 1, S/SP, >= 4] fp32
TILE (row-split, replicated over TP; only columns 0..3 are read) -> out [1, 1, S/SP, H/TP] fp32, the chip's rows and
hidden columns. The column split carries through, so there is no collective and no weight. ttnn.multiply + 3 x
ttnn.addcmul with a [S/SP, 1] column broadcast: both take the SFPU path at fp32 (ternary_program_factory.cpp: the FPU
addcmul is only for non-fp32), so every product and sum is an exact fp32 operation (K-30 does not apply). No host work.
"""

from __future__ import annotations

import ttnn

from .common import DRAM
from .layout import HC


class TtHcCollapse:
    """Call with (streams [1, 1, R, n x w] fp32, pre [1, 1, R, >= n] fp32); returns [1, 1, R, w] fp32. Consumes
    neither input."""

    def __init__(self, n: int = HC):
        self.n = n

    def __call__(self, x: ttnn.Tensor, pre: ttnn.Tensor) -> ttnn.Tensor:
        assert x.dtype == ttnn.float32 and pre.dtype == ttnn.float32, (x.dtype, pre.dtype)
        rows, w = x.shape[-2], x.shape[-1] // self.n
        assert x.shape[-1] == self.n * w and pre.shape[-1] >= self.n and pre.shape[-2] == rows, (x.shape, pre.shape)
        y = None
        for i in range(self.n):
            s = ttnn.slice(x, [0, 0, 0, i * w], [1, 1, rows, (i + 1) * w], memory_config=DRAM)
            c = ttnn.slice(pre, [0, 0, 0, i], [1, 1, rows, i + 1], memory_config=DRAM)
            if y is None:
                y = ttnn.multiply(s, c, dtype=ttnn.float32, memory_config=DRAM)
            else:
                nxt = ttnn.addcmul(y, s, c, memory_config=DRAM)
                ttnn.deallocate(y)
                y = nxt
            ttnn.deallocate(s)
            ttnn.deallocate(c)
        return y


def build_collapse(cfg) -> TtHcCollapse:
    return TtHcCollapse(n=cfg.hc)
