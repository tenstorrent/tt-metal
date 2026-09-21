# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host input patch embedding, parked for a later CPU vs P150 speed comparison.

This is the eval-mode residual block that used to live in ``tt/``. The device
matmuls are not implemented yet. The vendored ``ResidualBlock`` under
``reference/`` remains the Amazon CPU baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt.model_preprocessing import Chronos2PatchedInputs


@dataclass(frozen=True)
class ResidualBlockWeights:
    """Eval-mode input patch embedding weights (three linears, no layer norm).

    Shapes follow ``nn.Linear``: weight is ``(out_features, in_features)``.
    ``input_patch_embedding`` uses ``in_features = 3 * patch_size``.
    """

    hidden_weight: torch.Tensor
    hidden_bias: torch.Tensor
    output_weight: torch.Tensor
    output_bias: torch.Tensor
    residual_weight: torch.Tensor
    residual_bias: torch.Tensor
    act_fn_name: str = "relu"


@dataclass(frozen=True)
class Chronos2EmbeddedInputs:
    """Encoder inputs after the input patch embedding and optional REG token.

    Shapes:
        inputs_embeds: (B, num_context_patches + reg + num_output_patches, d_model)
        attention_mask: (B, num_context_patches + reg + num_output_patches)
    ``reg`` is 1 when a REG embedding row is provided, otherwise 0.
    """

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    loc_scale: tuple[torch.Tensor, torch.Tensor]
    group_ids: torch.Tensor
    target_idx_ranges: list[tuple[int, int]]


def residual_block(x: torch.Tensor, weights: ResidualBlockWeights) -> torch.Tensor:
    """Eval-mode ``ResidualBlock.forward`` (dropout is a no-op, no layer norm)."""
    if weights.act_fn_name != "relu":
        raise ValueError(f"residual_block only supports relu, found {weights.act_fn_name!r}")
    hidden = torch.relu(torch.nn.functional.linear(x, weights.hidden_weight, weights.hidden_bias))
    output = torch.nn.functional.linear(hidden, weights.output_weight, weights.output_bias)
    residual = torch.nn.functional.linear(x, weights.residual_weight, weights.residual_bias)
    return output + residual


def embed_patched_inputs(
    patched: Chronos2PatchedInputs,
    weights: ResidualBlockWeights,
    reg_embedding: torch.Tensor | None = None,
) -> Chronos2EmbeddedInputs:
    """Map patched context and future through the input patch embedding.

    Matches the embed sequence ``Chronos2Model.encode`` builds before the encoder:
    context embeds, optional REG token, then future embeds. ``reg_embedding`` is
    the ``shared.weight[reg_token_id]`` row of shape ``(d_model,)``.
    """
    context_embeds = residual_block(patched.patched_context, weights)
    future_embeds = residual_block(patched.patched_future, weights)
    batch_size, num_output_patches, _ = future_embeds.shape
    inputs_embeds = context_embeds
    attention_mask: torch.Tensor = patched.attention_mask

    if reg_embedding is not None:
        reg_row = reg_embedding.to(dtype=context_embeds.dtype, device=context_embeds.device).reshape(-1)
        reg_embeds = reg_row.view(1, 1, -1).expand(batch_size, 1, -1)
        inputs_embeds = torch.cat([inputs_embeds, reg_embeds], dim=-2)
        reg_mask = torch.ones(batch_size, 1, dtype=context_embeds.dtype, device=context_embeds.device)
        attention_mask = torch.cat([attention_mask.to(dtype=context_embeds.dtype), reg_mask], dim=-1)

    future_mask = torch.ones(
        batch_size, num_output_patches, dtype=context_embeds.dtype, device=context_embeds.device
    )
    inputs_embeds = torch.cat([inputs_embeds, future_embeds], dim=-2)
    attention_mask = torch.cat([attention_mask.to(dtype=context_embeds.dtype), future_mask], dim=-1)
    return Chronos2EmbeddedInputs(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        loc_scale=patched.loc_scale,
        group_ids=patched.group_ids,
        target_idx_ranges=patched.target_idx_ranges,
    )
