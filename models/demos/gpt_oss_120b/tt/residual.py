# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS residual add (attn_residual, mlp_residual): out = a + b.

Both operands are the residual stream's layout (plan.md: [1, 1, S/8, 2880] bf16 TILE per chip, split by sequence over
axis 0, replicated over axis 1), so the add is local: one ttnn.add, no collective, bf16 out in DRAM.
"""

import ttnn


class TtResidualAdd:
    def __call__(self, a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(a, b, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
