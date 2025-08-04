# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .fun_conv2d import vae_conv2d, TtConv2dParameters
from ..parallel_config import VAEParallelConfig


# Asumption: If the input is sharded, the output is sharded. If the input is not sharded, the output is replicated across mesh.
# TODO: See if there is any benefits of paralellizing upsample2d since we have to go between layouts
@dataclass
class TtUpsample2DParameters:
    conv: TtConv2dParameters
    parallel_config: VAEParallelConfig
    mesh_sharded_input: bool

    @classmethod
    def from_torch(
        cls,
        torch_upsample: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
        mesh_sharded_input: bool = True,  # Indicates if the input should be left sharded across the mesh devices
    ) -> TtUpsample2DParameters:
        return cls(
            conv=TtConv2dParameters.from_torch(
                torch_upsample.conv,
                dtype=dtype,
                parallel_config=parallel_config,
                mesh_sharded_input=mesh_sharded_input,
                mesh_sharded_output=mesh_sharded_input,
            ),
            parallel_config=parallel_config,
            mesh_sharded_input=mesh_sharded_input,
        )


def vae_upsample2d(x, parameters):
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  # Upsample requires row major.
    x = ttnn.upsample(x, scale_factor=2)

    x = vae_conv2d(x, parameters.conv)
    return x
