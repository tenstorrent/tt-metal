# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Residual add (attn_residual, mlp_residual; the same op serves moe_combine).

Both inputs are the TP-split residual layout ([1, 1, S/8, H/4] bf16 per chip, layout.py), so the add is per chip,
with no collective (as tt_prefill_block.py does). Output bf16, DRAM interleaved.
"""

import ttnn


class TtResidualAdd:
    def __call__(self, a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(a, b, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
