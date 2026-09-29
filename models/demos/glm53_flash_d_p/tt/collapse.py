# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""mHC collapse (attn_collapse / ffn_collapse) on device, replicated, no CCL, no weights.

attn_in [1, 1, S, H] = sum_n pre[:, n] * x_n, x [1, 1, S, n*H] (streams packed along the last dim, the same memory as
the reference's token-major [S * n, H]), hc [1, 1, S, >= n] fp32 with pre in columns 0..n-1.
Accumulates in fp32 and rounds to bf16 once (glm_ref.hc_collapse). Reuses DeepSeek's _streams / _cols / _mix.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _cols, _mix, _streams


class TtHcCollapse(LightweightModule):
    def __init__(self, n=4, out_dtype=ttnn.bfloat16):
        self.n, self.out_dtype = n, out_dtype

    def __call__(self, x, hc):
        """x [1, 1, S, n*H], hc [1, 1, S, K] (K >= n, fp32) -> [1, 1, S, H] out_dtype."""
        streams = _streams(x, self.n)
        if x.dtype != ttnn.float32:
            s32 = [ttnn.typecast(s, ttnn.float32) for s in streams]
            for s in streams:
                ttnn.deallocate(s)
            streams = s32
        pre = hc if hc.dtype == ttnn.float32 else ttnn.typecast(hc, ttnn.float32)
        cols = _cols(pre, self.n)
        y = _mix(streams, cols)
        for t in streams + cols:
            ttnn.deallocate(t)
        if pre is not hc:
            ttnn.deallocate(pre)
        if self.out_dtype != ttnn.float32:
            y32 = y
            y = ttnn.typecast(y32, self.out_dtype)
            ttnn.deallocate(y32)
        return y


def build_collapse(cfg) -> TtHcCollapse:
    return TtHcCollapse(n=cfg.hc_mult)
