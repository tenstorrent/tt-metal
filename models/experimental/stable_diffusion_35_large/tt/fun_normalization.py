# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .utils import from_torch_fast, from_torch_fast_2d
from .parallel_config import StableDiffusionParallelManager, DiTParallelConfig


@dataclass
class TtRmsNormParameters:
    weight: ttnn.Tensor
    eps: float = 1e-5

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        eps: float = 1e-5,
    ) -> TtRmsNormParameters:
        return cls(
            weight=from_torch_fast(state["weight"].unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device),
            eps=eps,
        )


@dataclass
class TtLayerNormParameters:
    weight: ttnn.Tensor | None = None
    bias: ttnn.Tensor | None = None
    distributed: bool = False
    eps: float = 1e-5

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        distributed: bool = True,
        weight_shape: list[int] | None = None,
        eps: float = 1e-5,
        parallel_config: DiTParallelConfig,
    ) -> TtLayerNormParameters:
        distributed = distributed and parallel_config.tensor_parallel.factor > 1
        mesh_width = parallel_config.tensor_parallel.factor

        weight = state.get("weight")
        bias = state.get("bias")
        if distributed:
            # ttnn.layer_norm_post_all_gather currently requires weight and bias
            if weight is None:
                assert weight_shape is not None, "weight_shape is required when weight is missing"
                weight = torch.ones(weight_shape)
            if bias is None:
                bias = torch.zeros_like(weight)

            h = 32 * mesh_width
            _, mesh_width = device.shape
            weight = weight.reshape([-1, h])
            bias = bias.reshape([-1, h])

        dims = [None, None]
        dims[parallel_config.tensor_parallel.mesh_axis] = -1
        mesh_mapper = ttnn.ShardTensor2dMesh(device, tuple(device.shape), dims) if distributed else None
        layout = ttnn.ROW_MAJOR_LAYOUT if distributed else ttnn.TILE_LAYOUT
        if distributed and dtype != ttnn.float32:
            dtype = ttnn.bfloat16

        return cls(
            weight=from_torch_fast_2d(
                weight,
                mesh_device=device,
                mesh_shape=tuple(device.shape),
                dims=[None, None],
                layout=layout,
                dtype=dtype,
                mesh_mapper=mesh_mapper,
            )
            if weight is not None
            else None,
            bias=from_torch_fast_2d(
                bias,
                mesh_device=device,
                mesh_shape=tuple(device.shape),
                dims=[None, None],
                layout=layout,
                dtype=dtype,
                mesh_mapper=mesh_mapper,
            )
            if bias is not None
            else None,
            distributed=distributed,
            eps=eps,
        )


def sd_rms_norm(x: ttnn.Tensor, parameters: TtRmsNormParameters, deallocate: bool = False) -> ttnn.Tensor:
    output = ttnn.rms_norm(x, weight=parameters.weight, epsilon=parameters.eps)

    # if deallocate:
    #     ttnn.deallocate(x)

    return output


def sd_layer_norm(
    x: ttnn.Tensor,
    parameters: TtLayerNormParameters,
    parallel_manager: StableDiffusionParallelManager,
    memory_config: ttnn.MemoryConfig | None = None,
    program_config: ttnn.ProgramConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    cfg_index: int = 0,
    is_spatial: bool = True,
) -> ttnn.Tensor:
    if not parameters.distributed:
        return ttnn.layer_norm(
            x,
            weight=parameters.weight,
            bias=parameters.bias,
            epsilon=parameters.eps,
            memory_config=memory_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    rank = len(x.shape)
    if rank < 4:
        shape = [1] * (4 - rank) + list(x.shape)
        x = ttnn.reshape(x, shape)

    stats = ttnn.layer_norm_pre_all_gather(
        x,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )

    buffer_name = "spatial_layernorm_buffer" if is_spatial else "prompt_layernorm_buffer"
    stats_gathered = ttnn.experimental.all_gather_async(
        stats,
        dim=len(x.shape) - 1,
        cluster_axis=parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis,
        mesh_device=x.device(),
        topology=parallel_manager.dit_parallel_config.topology,
        multi_device_global_semaphore=parallel_manager.get_ping_pong_semaphore(cfg_index),
        persistent_output_tensor=parallel_manager.get_ping_pong_buffer(cfg_index, buffer_name),
        memory_config=memory_config,
        num_links=parallel_manager.num_links,
    )

    x = ttnn.layer_norm_post_all_gather(
        x,
        stats_gathered,
        weight=parameters.weight,
        bias=parameters.bias,
        epsilon=parameters.eps,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    if rank < 4:
        shape = list(x.shape)[4 - rank :]
        x = ttnn.reshape(x, shape)

    return x
