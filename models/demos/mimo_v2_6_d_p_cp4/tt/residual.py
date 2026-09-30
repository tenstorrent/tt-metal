# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Residual add on a 1x4 mesh with CP=4: y = a + b.

Copied from models/demos/mimo_v2_6_d_p/tt/residual.py. Both operands are the chip's CP slice [1, 1, S/4, H] bf16 TILE
(the same rows on each chip), so the add needs no collective. Output stays bf16 (bfp8 on the residual stream is not
budgeted by the component test).
"""

from __future__ import annotations

import ttnn


class TtResidualAdd:
    def __init__(self, mesh):
        self.mesh = mesh

    def __call__(self, a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(a, b, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
