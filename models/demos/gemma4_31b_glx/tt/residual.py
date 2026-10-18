# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 residual adds on the residual layout ([1, 1, rows, 5376] bf16 per chip, the TP row split; no collective).

attn_residual: h_mid = x + attn_post. ffn_residual: out = (h_mid + mlp_post) * layer_scalar.
As in models/demos/gemma4_d_p/tt/layer.py: one ttnn.add, bf16 out in DRAM; the layer_scalar is fused into it as a
MUL_UNARY_SFPU activation (a Python float read once at load).
"""

import ttnn


class ResidualAdd:
    def __init__(self, scale: float | None = None):
        self.activations = [] if scale is None else [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale)]

    def __call__(self, x, y):
        return ttnn.add(x, y, activations=self.activations, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
