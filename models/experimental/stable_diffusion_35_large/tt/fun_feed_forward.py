# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .fun_linear import TtLinearParameters, sd_linear
from .substate import substate
from .parallel_config import DiTParallelConfig, StableDiffusionParallelManager

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
        hidden_dim_padding: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
    ) -> TtFeedForwardParameters:
        # Note: Implied sharding here
        # Implements Megatron-style MLP parallelism
        return cls(
            in_proj=TtLinearParameters.from_torch(
                substate(state, "net.0.proj"),
                dtype=dtype,
                device=device,
                shard_dim=-1,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
            out_proj=TtLinearParameters.from_torch(
                substate(state, "net.2"),
                dtype=dtype,
                device=device,
                shard_dim=-2,
                hidden_dim_padding=hidden_dim_padding,
                parallel_config=parallel_config,
            ),
        )


def sd_feed_forward(
    x: ttnn.Tensor,
    parameters: TtFeedForwardParameters,
    parallel_manager: StableDiffusionParallelManager,
    cfg_index: int,
    is_spatial: bool = True,
) -> ttnn.Tensor:
    device = x.device()

    grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)
    # NOTE: With activation fused into linear, unclear whether it's using approx mode.
    x3 = sd_linear(x, parameters.in_proj, core_grid=core_grid, activation="gelu")
    # Turning on fast_and_approximate_mode leads to big changes in the generated image.
    # The image quality might still be okay.
    # x3 = ttnn.gelu(x2, fast_and_approximate_mode=False)
    # ttnn.deallocate(x2)

    result = sd_linear(x3, parameters.out_proj, core_grid=core_grid)
    # ttnn.deallocate(x3)

    if parallel_manager.is_tensor_parallel:
        intermediate_buffer_name = (
            "spatial_rs_feed_forward_intermediate" if is_spatial else "prompt_rs_feed_forward_intermediate"
        )
        output_buffer_name = "spatial_rs_feed_forward_output" if is_spatial else "prompt_rs_feed_forward_output"
        result = ttnn.experimental.reduce_scatter_minimal_async(
            result,
            persistent_intermediate_buffer=parallel_manager.get_ping_pong_buffer(cfg_index, intermediate_buffer_name),
            persistent_output_buffer=parallel_manager.get_ping_pong_buffer(cfg_index, output_buffer_name),
            dim=3,
            multi_device_global_semaphore=parallel_manager.get_rs_ping_pong_semaphore(cfg_index),
            num_links=parallel_manager.num_links,
            memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            topology=parallel_manager.dit_parallel_config.topology,
            cluster_axis=parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis,
            # mesh_device=device,
            # from_remote_multi_device_global_semaphore=parallel_manager.cfg_semaphores[cfg_index]["rs_from"],
            # to_remote_multi_device_global_semaphore=parallel_manager.cfg_semaphores[cfg_index]["rs_to"],
            # math_op=ttnn.ReduceType.Sum,
        )
        # Reshape to set padding again
        result_unpadded_shape = list(result.shape)
        result_unpadded_shape[2] = x.shape[2]
        result = ttnn.reshape(result, result_unpadded_shape, result.shape)

    return result
