# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro ``UpSample1d`` from ``reference/istftnet.py``.

Nearest-neighbour doubling along the sequence axis for activations in **NLC** layout ``[B, L, C]``,
matching ``F.interpolate(..., scale_factor=2, mode="nearest")`` on ``[B, C, L]``.
"""

from __future__ import annotations

import ttnn


class TTUpSample1d:
    """``layer_type == "none"`` is identity; otherwise length doubles (factor ``2``)."""

    __slots__ = ("_layer_type",)

    def __init__(self, layer_type: str) -> None:
        self._layer_type = layer_type

    def forward(
        self,
        x_nlc: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        if self._layer_type == "none":
            return x_nlc
        return ttnn.repeat_interleave(x_nlc, 2, 1, memory_config=memory_config)

    __call__ = forward
