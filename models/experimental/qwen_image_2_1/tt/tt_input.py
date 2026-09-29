# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen Image 2.1 DiT input and timestep-conditioning projections on TTNN."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import ttnn

from .tt_dit_components import to_device


@dataclass
class InputWeights:
    image: ttnn.Tensor
    text_norm: ttnn.Tensor
    text_in: ttnn.Tensor
    text_out: ttnn.Tensor
    time_in: ttnn.Tensor
    time_out: ttnn.Tensor
    modulation: ttnn.Tensor
    text_norm_compute: object


def prepare_weights(state: dict[str, torch.Tensor], device) -> InputWeights:
    def projection(key: str) -> ttnn.Tensor:
        return to_device(state[key].T.contiguous(), device)

    effective_text_scale = (state["txt_in.text_norm.weight"].float() + 1.0).to(torch.bfloat16)
    return InputWeights(
        image=projection("img_in.weight"),
        text_norm=to_device(effective_text_scale.reshape(1, 1, -1), device),
        text_in=projection("txt_in.in_layer.weight"),
        text_out=projection("txt_in.out_layer.weight"),
        time_in=projection("time_text_embed.timestep_embedder.linear_1.weight"),
        time_out=projection("time_text_embed.timestep_embedder.linear_2.weight"),
        modulation=projection("modulation.1.weight"),
        text_norm_compute=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )


def image_projection(latents: ttnn.Tensor, weights: InputWeights, compute_kernel_config=None) -> ttnn.Tensor:
    return ttnn.matmul(latents, weights.image, compute_kernel_config=compute_kernel_config)


def text_projection_stages(
    context: ttnn.Tensor, weights: InputWeights, matmul_compute_kernel_config=None
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    normalized = ttnn.rms_norm(
        context,
        weight=weights.text_norm,
        epsilon=1e-6,
        compute_kernel_config=weights.text_norm_compute,
    )
    linear1 = ttnn.matmul(normalized, weights.text_in, compute_kernel_config=matmul_compute_kernel_config)
    activated = ttnn.gelu(linear1, fast_and_approximate_mode=False)
    output = ttnn.matmul(activated, weights.text_out, compute_kernel_config=matmul_compute_kernel_config)
    return normalized, linear1, activated, output


def text_projection(context: ttnn.Tensor, weights: InputWeights, matmul_compute_kernel_config=None) -> ttnn.Tensor:
    return text_projection_stages(context, weights, matmul_compute_kernel_config)[-1]


def sinusoidal_timestep(timestep: torch.Tensor) -> torch.Tensor:
    """Prepare the 256-feature trigonometric input; the large MLP runs on TT."""
    half = 128
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    phases = (timestep.float() * 1000.0)[:, None] * frequencies[None]
    return torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1).to(timestep.dtype)


def timestep_and_modulation(
    timestep_features: ttnn.Tensor, weights: InputWeights, compute_kernel_config=None
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    embedded = ttnn.matmul(timestep_features, weights.time_in, compute_kernel_config=compute_kernel_config)
    embedded = ttnn.silu(embedded)
    embedded = ttnn.matmul(embedded, weights.time_out, compute_kernel_config=compute_kernel_config)
    modulation = ttnn.matmul(ttnn.silu(embedded), weights.modulation, compute_kernel_config=compute_kernel_config)
    return embedded, modulation
