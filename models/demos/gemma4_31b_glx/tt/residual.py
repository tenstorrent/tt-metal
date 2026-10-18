# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 residual adds on the residual layout ([1, 1, rows, 5376] bf16 per chip, the TP row split; no collective).

attn_residual: h_mid = x + attn_post. ffn_residual: out = (h_mid + mlp_post) * layer_scalar.
As in models/demos/gemma4_d_p/tt/layer.py: one ttnn.add, bf16 out in DRAM; the layer_scalar is fused into it as a
MUL_UNARY_SFPU activation (a Python float read once at load).
RESIDUAL_FP32_SCALE: the scaled add runs with an fp32 output and one typecast to bf16. With a bf16 output the fused
scale lowers row norms -0.26 % per layer (layer 3 on golden data; bf16 rounding of the exact result: +0.00004 %), which
compounds over the stack (assembled 60 layers: median row norm bias 0.022 at layer 52 against 0.005).
"""

import ttnn

from .settings import settings


class ResidualAdd:
    def __init__(self, scale: float | None = None, dtype=ttnn.bfloat16):
        """dtype: the output (the device model's residual stream, RESIDUAL_STREAM); bf16 for the component steps."""
        self.activations = [] if scale is None else [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale)]
        self.dtype = dtype
        self.fp32 = bool(self.activations) and settings.get("RESIDUAL_FP32_SCALE")

    def __call__(self, x, y):
        if self.dtype == ttnn.float32:
            y32 = ttnn.typecast(y, ttnn.float32) if y.dtype != ttnn.float32 else y
            x32 = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
            out = ttnn.add(
                x32, y32, activations=self.activations, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for t, src in ((x32, x), (y32, y)):
                if t is not src:
                    ttnn.deallocate(t)
            return out
        if not self.fp32:
            return ttnn.add(
                x, y, activations=self.activations, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        out = ttnn.add(x, y, activations=self.activations, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y16 = ttnn.typecast(out, ttnn.bfloat16)
        ttnn.deallocate(out)
        return y16
