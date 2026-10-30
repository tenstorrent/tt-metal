# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Residual add of Qwen3.8-27B on the device: h_mid = in + attn_out, out = h_mid + mlp_out.

Token-local: both operands are in the residual layout [1, 1, 640, 5120] per chip (SP rows over axis 0, replicated
over the 4 TP columns), so the add is elementwise on each chip with no collective.
"""

import ttnn


class TtResidualAdd:
    """attn_residual / mlp_residual (both block types)."""

    def __call__(self, x: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.add(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
