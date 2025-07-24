# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .fun_conv2d import vae_conv2d, TtConv2dParameters
from ..parallel_config import VAEParallelConfig


@dataclass
class TtUpsample2DParameters:
    conv: TtConv2dParameters

    @classmethod
    def from_torch(
        cls,
        torch_upsample: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
    ) -> TtUpsample2DParameters:
        return cls(
            conv=TtConv2dParameters.from_torch(torch_upsample.conv, dtype=dtype, parallel_config=parallel_config),
        )


def vae_upsample2d(x, parameters):
    in_layout = x.layout
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Upsample requires row major.
    x = ttnn.upsample(x, scale_factor=2)
    x = ttnn.to_layout(x, in_layout)
    x = vae_conv2d(x, parameters.conv)
    return x
