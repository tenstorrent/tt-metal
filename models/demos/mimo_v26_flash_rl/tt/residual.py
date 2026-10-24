# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Residual add on the device: ``x + y`` on each chip's local rows of the residual stream.

Both operands are [1, 1, 160, 4096] bf16 on the 8x4 mesh in the same row layout (the attention and MLP modules end
with a reduce_scatter over TP back to the local rows), so the add is elementwise with no collective.
"""

from __future__ import annotations

import ttnn


class ResidualAdd:
    def __call__(self, x, y, memory_config=None):
        return ttnn.add(x, y, dtype=ttnn.bfloat16, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
