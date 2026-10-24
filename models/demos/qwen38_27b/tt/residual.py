# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Residual add of Qwen3.5 (attn_residual, mlp_residual): out = a + b.

Both operands are the residual stream layout, [1, 1, 640, 5120] bf16 per chip (sequence split over mesh axis 0,
replicated over axis 1), so the add is local: one ttnn.add, no collective.
"""

from __future__ import annotations

import ttnn


class TtResidualAdd:
    def __call__(self, a, b):
        return ttnn.add(a, b, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
