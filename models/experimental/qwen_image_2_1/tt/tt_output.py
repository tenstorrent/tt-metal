# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Final adaptive normalization and latent projection on TTNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .tt_dit_components import select_rows_by_mask, to_device


@dataclass
class OutputWeights:
    norm_scale: ttnn.Tensor
    projection: ttnn.Tensor


def prepare_weights(norm_weight: torch.Tensor, projection_weight: torch.Tensor, device) -> OutputWeights:
    return OutputWeights(
        norm_scale=to_device(norm_weight.T.contiguous(), device),
        projection=to_device(projection_weight.T.contiguous(), device),
    )


def select_timestep_embedding(temb: torch.Tensor, target_token_mask: torch.Tensor, device) -> ttnn.Tensor:
    """Select timestep rows from captured conditioning before device compute."""
    if temb.ndim != 2 or temb.shape[0] != 2:
        raise ValueError(f"expected timestep embedding [2, hidden], got {tuple(temb.shape)}")
    selected = torch.where(target_token_mask.reshape(1, -1, 1).bool(), temb[:1, None], temb[1:2, None])
    return to_device(selected, device)


def select_timestep_embedding_device(temb: ttnn.Tensor, target_token_mask: torch.Tensor, device) -> ttnn.Tensor:
    return select_rows_by_mask(temb, target_token_mask, 4096, device)


def output_head(
    hidden: ttnn.Tensor,
    selected_temb: ttnn.Tensor,
    weights: OutputWeights,
    compute_kernel_config=None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    scale = ttnn.matmul(
        ttnn.silu(selected_temb),
        weights.norm_scale,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    normalized = ttnn.layer_norm(hidden, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    normalized = ttnn.multiply(normalized, ttnn.add(scale, 1.0))
    output = ttnn.matmul(
        normalized,
        weights.projection,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    return normalized, output
