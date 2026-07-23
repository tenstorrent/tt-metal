# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device weight loading for Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import validate_reference_weights


@dataclass(frozen=True)
class KDAWeights:
    qkv_projection: ttnn.Tensor
    auxiliary_projection: ttnn.Tensor
    decay_output_projection: ttnn.Tensor
    output_gate_projection: ttnn.Tensor
    output_projection: ttnn.Tensor
    decay_scale: ttnn.Tensor
    decay_bias: ttnn.Tensor
    decay_scale_flat: ttnn.Tensor
    decay_bias_flat: ttnn.Tensor
    norm: ttnn.Tensor
    convolution_taps: tuple[ttnn.Tensor, ...]


def load_kda_weights(
    device: ttnn.Device,
    config: KDAConfig,
    state_dict: Mapping[str, torch.Tensor],
    tensor_cache_path: Path | None = None,
) -> KDAWeights:
    """Fuse compatible projections and place all runtime weights on device."""
    validate_reference_weights(state_dict, config)

    def device_tensor(
        tensor: torch.Tensor,
        name: str,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> ttnn.Tensor:
        cache_file = tensor_cache_path / name if tensor_cache_path is not None else None
        return ttnn.as_tensor(
            tensor.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file,
        )

    qkv = torch.cat(
        (
            state_dict["q_proj.weight"],
            state_dict["k_proj.weight"],
            state_dict["v_proj.weight"],
        ),
        dim=0,
    ).T
    auxiliary = torch.cat(
        (
            state_dict["f_a_proj.weight"],
            state_dict["g_a_proj.weight"],
            state_dict["b_proj.weight"],
        ),
        dim=0,
    ).T

    decay_scale = -state_dict["A_log"].float().exp()
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
        convolution_taps.append(device_tensor(fused_tap, f"conv_tap_{tap}"))

    return KDAWeights(
        qkv_projection=device_tensor(qkv, "qkv_projection"),
        auxiliary_projection=device_tensor(auxiliary, "auxiliary_projection"),
        decay_output_projection=device_tensor(
            state_dict["f_b_proj.weight"].T,
            "decay_output_projection",
        ),
        output_gate_projection=device_tensor(
            state_dict["g_b_proj.weight"].T,
            "output_gate_projection",
        ),
        output_projection=device_tensor(
            state_dict["o_proj.weight"].T,
            "output_projection",
        ),
        decay_scale=device_tensor(decay_scale, "decay_scale"),
        decay_bias=device_tensor(decay_bias, "decay_bias"),
        decay_scale_flat=device_tensor(decay_scale_flat, "decay_scale_flat"),
        decay_bias_flat=device_tensor(decay_bias_flat, "decay_bias_flat"),
        norm=device_tensor(state_dict["o_norm.weight"], "norm"),
        convolution_taps=tuple(convolution_taps),
    )
