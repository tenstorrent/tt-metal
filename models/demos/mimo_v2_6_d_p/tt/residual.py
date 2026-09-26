# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Residual add on a 1xN mesh, replicated: y = a + b.

From models/demos/gemma4_a4b_d_p/tt/residual.py (MiMo has no layer scalar). Both operands are replicated
[1, 1, S, H] bf16 TILE tensors, so the add needs no collective. Output stays bf16 (bfp8 on the residual stream is
not budgeted by the component test).
"""

from __future__ import annotations

import ttnn


class TtResidualAdd:
    def __init__(self, mesh):
        self.mesh = mesh

    def __call__(self, a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(a, b, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
