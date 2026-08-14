# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.modules.LinearNorm`.

``LinearNorm`` is a thin wrapper around ``nn.Linear`` (xavier init at construction time only),
so the device port is just a ``ttnn.linear`` with stored weight ``transpose_b=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

import ttnn


@dataclass(frozen=True)
class TTLinearNormParams:
    """Device-resident weights for :class:`TTLinearNorm`."""

    weight: ttnn.Tensor
    bias: Optional[ttnn.Tensor]
    in_dim: int
    out_dim: int


def preprocess_tt_linear_norm(
    module: nn.Module,
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
) -> TTLinearNormParams:
    """Upload the inner ``nn.Linear`` weight (and optional bias)."""
    lin = module.linear_layer
    w = ttnn.from_torch(
        lin.weight.detach().cpu(),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = None
    if lin.bias is not None:
        b = ttnn.from_torch(
            lin.bias.detach().cpu().reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return TTLinearNormParams(
        weight=w,
        bias=b,
        in_dim=int(lin.in_features),
        out_dim=int(lin.out_features),
    )


class TTLinearNorm:
    """``ttnn.linear`` with stored weight (``transpose_b=True``)."""

    __slots__ = ("params",)

    def __init__(self, params: TTLinearNormParams) -> None:
        self.params = params

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        compute_kernel_config,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """Apply ``y = x @ W^T + b``. Any input rank is supported via ``ttnn.linear``."""
        return ttnn.linear(
            x,
            self.params.weight,
            bias=self.params.bias,
            transpose_b=True,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )

    __call__ = forward
