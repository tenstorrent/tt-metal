# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 residual adds on the residual layout ([1, 1, rows, 5376] bf16 per chip, the TP row split; no collective).

attn_residual: h_mid = x + attn_post. ffn_residual: out = (h_mid + mlp_post) * layer_scalar (its own task).
As in models/demos/gemma4_d_p/tt/layer.py: one ttnn.add, bf16 out in DRAM.
"""

import ttnn


class ResidualAdd:
    def __call__(self, x, y):
        return ttnn.add(x, y, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
