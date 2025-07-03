# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .fun_linear import TtLinearParameters, sd_linear
from .substate import substate
from .parallel_config import DiTParallelConfig

if TYPE_CHECKING:
    import torch


@dataclass
class TtFeedForwardParameters:
    in_proj: TtLinearParameters
    out_proj: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
    ) -> TtFeedForwardParameters:
        # Note: Implied sharding here
        # TODO: Add 2D sharding when tensor+sequence parallel supported
        # Implements Megatron-style MLP parallelism
        return cls(
            in_proj=TtLinearParameters.from_torch(
                substate(state, "net.0.proj"), dtype=dtype, device=device, shard_dim=-1
            ),
            out_proj=TtLinearParameters.from_torch(substate(state, "net.2"), dtype=dtype, device=device, shard_dim=-2),
        )


def sd_feed_forward(
    x: ttnn.Tensor, parameters: TtFeedForwardParameters, parallel_config: DiTParallelConfig
) -> ttnn.Tensor:
    grid_size = x.device().compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)
    # NOTE: With activation fused into linear, unclear whether it's using approx mode.
    x3 = sd_linear(x, parameters.in_proj, core_grid=core_grid, activation="gelu")
    # Turning on fast_and_approximate_mode leads to big changes in the generated image.
    # The image quality might still be okay.
    # x3 = ttnn.gelu(x2, fast_and_approximate_mode=False)
    # ttnn.deallocate(x2)

    result = sd_linear(x3, parameters.out_proj, core_grid=core_grid)
    ttnn.deallocate(x3)

    if parallel_config.tensor_parallel.factor > 1:
        result = ttnn.reduce_scatter(
            result,
            dim=-1,
            math_op=ttnn.ReduceType.Sum,
            num_links=1,
            memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            topology=ttnn.Topology.Ring,
            # TODO: set cluster axis based on parallel config
        )

    return result
