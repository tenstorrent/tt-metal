# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device weight loading for Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.experimental.kimi_delta_attention.checkpoint import normalize_kda_state_dict
from models.experimental.kimi_delta_attention.config import KDAConfig


@dataclass(frozen=True)
class KDAWeights:
    input_projection: ttnn.Tensor
    decay_output_projection: ttnn.Tensor
    output_projection: ttnn.Tensor
    decay_scale_flat: ttnn.Tensor
    decay_bias_flat: ttnn.Tensor
    norm: ttnn.Tensor
    convolution_taps: tuple[ttnn.Tensor, ...]
    tensor_parallel_size: int


def load_kda_weights(
    device: ttnn.Device | ttnn.MeshDevice,
    config: KDAConfig,
    state_dict: Mapping[str, torch.Tensor],
    tensor_cache_path: Path | None = None,
    *,
    tensor_parallel_axis: int = 1,
) -> KDAWeights:
    """Fuse compatible projections and place whole-head shards on device."""
    state_dict = normalize_kda_state_dict(state_dict, config)
    if tensor_parallel_axis not in (0, 1):
        raise ValueError(f"tensor_parallel_axis must be 0 or 1, got {tensor_parallel_axis}")
    if isinstance(device, ttnn.MeshDevice):
        mesh_shape = tuple(device.shape)
        tensor_parallel_size = mesh_shape[tensor_parallel_axis]
    else:
        mesh_shape = (1, 1)
        tensor_parallel_size = 1
    if config.num_heads % tensor_parallel_size != 0:
        raise ValueError(
            f"num_heads {config.num_heads} must be divisible by tensor parallel size {tensor_parallel_size}"
        )

    def device_tensor(
        tensor: torch.Tensor,
        name: str,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
        shard_dim: int | None = None,
    ) -> ttnn.Tensor:
        cache_name = f"{name}.tp{tensor_parallel_size}.axis{tensor_parallel_axis}" if tensor_parallel_size > 1 else name
        cache_file = tensor_cache_path / cache_name if tensor_cache_path is not None else None
        mesh_mapper = None
        if tensor_parallel_size > 1:
            mesh_dims = [None, None]
            mesh_dims[tensor_parallel_axis] = shard_dim
            mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(mesh_dims), mesh_shape=mesh_shape)
        return ttnn.as_tensor(
            tensor.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file,
        )

    def group_output_shards(*weights: torch.Tensor) -> torch.Tensor:
        """Group every projection tensor corresponding head slice on the same device."""
        grouped = []
        for device_index in range(tensor_parallel_size):
            device_weights = []
            for weight in weights:
                shard_width = weight.shape[0] // tensor_parallel_size
                start = device_index * shard_width
                device_weights.append(weight[start : start + shard_width])
            grouped.append(torch.cat(device_weights, dim=0))
        return torch.cat(grouped, dim=0)

    common_input_weights = (
        state_dict["q_proj.weight"],
        state_dict["k_proj.weight"],
        state_dict["v_proj.weight"],
        state_dict["f_a_proj.weight"].repeat(tensor_parallel_size, 1),
    )
    if config.use_full_rank_gate:
        input_projection = group_output_shards(
            *common_input_weights,
            state_dict["g_proj.weight"],
            state_dict["b_proj.weight"],
        ).T
    else:
        output_gate_projection = state_dict["g_b_proj.weight"].reshape(
            config.num_heads, config.head_v_dim, config.head_v_dim
        )
        output_gate_direct = torch.matmul(
            output_gate_projection,
            state_dict["g_a_proj.weight"],
        ).reshape(config.v_dim, config.hidden_size)
        input_projection = group_output_shards(
            *common_input_weights,
            output_gate_direct,
            state_dict["b_proj.weight"],
        ).T

    decay_scale = state_dict["A_log"].float().exp()
    if config.gate_lower_bound is None:
        decay_scale = -decay_scale
    decay_bias = state_dict["dt_bias"].reshape(1, 1, config.num_heads, config.head_k_dim)
    decay_scale_flat = decay_scale.expand(-1, -1, -1, config.head_k_dim).reshape(1, 1, config.q_dim)
    decay_bias_flat = decay_bias.reshape(1, 1, config.q_dim)
    convolution_taps = []
    for tap in range(config.conv_kernel_size):
        fused_tap = torch.cat(
            (
                state_dict["q_conv1d.weight"][:, 0, tap],
                state_dict["k_conv1d.weight"][:, 0, tap],
                state_dict["v_conv1d.weight"][:, 0, tap],
            )
        ).reshape(1, 1, config.q_dim + config.k_dim + config.v_dim)
        if tensor_parallel_size > 1:
            fused_tap = group_output_shards(
                state_dict["q_conv1d.weight"][:, 0, tap],
                state_dict["k_conv1d.weight"][:, 0, tap],
                state_dict["v_conv1d.weight"][:, 0, tap],
            ).reshape(1, 1, -1)
        convolution_taps.append(device_tensor(fused_tap, f"conv_tap_{tap}", shard_dim=-1))

    return KDAWeights(
        input_projection=device_tensor(input_projection, "input_projection_head_major", shard_dim=-1),
        decay_output_projection=device_tensor(
            state_dict["f_b_proj.weight"].T,
            "decay_output_projection",
            shard_dim=-1,
        ),
        output_projection=device_tensor(
            state_dict["o_proj.weight"].T,
            "output_projection",
            shard_dim=-2,
        ),
        decay_scale_flat=device_tensor(decay_scale_flat, "decay_scale_flat", shard_dim=-1),
        decay_bias_flat=device_tensor(decay_bias_flat, "decay_bias_flat", shard_dim=-1),
        norm=device_tensor(state_dict["o_norm.weight"], "norm"),
        convolution_taps=tuple(convolution_taps),
        tensor_parallel_size=tensor_parallel_size,
    )
