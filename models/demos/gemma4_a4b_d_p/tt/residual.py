# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Residual add on a 1xN mesh, replicated: y = a + b (optionally * scale).

As in models/demos/gemma4/tt/layer.py (`ttnn.add(residual, attn_output)`). Both operands are replicated [1, 1, S, H]
TILE tensors, so the add needs no collective.
"""

from __future__ import annotations

import ttnn


class TtResidualAdd:
    def __init__(self, mesh, scale: float | None = None):
        self.mesh = mesh
        self.scale = scale

    def __call__(self, a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        y = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.scale is not None and self.scale != 1.0:
            z = ttnn.multiply(y, float(self.scale), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(y)
            y = z
        return y
