# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Single Qwen Image 2.1 transformer block on TTNN, first-step prefill."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .tt_attention import AttentionWeights, HIDDEN, prefill_attention, prepare_weights as prepare_attention_weights
from .tt_dit_components import select_rows_by_mask, to_device


@dataclass
class BlockWeights:
    attention: AttentionWeights
    mlp_gate: ttnn.Tensor
    mlp_proj: ttnn.Tensor
    mlp_out: ttnn.Tensor


def prepare_weights(state: dict[str, torch.Tensor], device) -> BlockWeights:
    def projection(name: str) -> ttnn.Tensor:
        return to_device(state[f"img_mlp.{name}.weight"].T.contiguous(), device)

    return BlockWeights(
        attention=prepare_attention_weights(state, device),
        mlp_gate=projection("gate_layer"),
        mlp_proj=projection("proj"),
        mlp_out=projection("out"),
    )


def select_modulation(modulation: torch.Tensor, target_token_mask: torch.Tensor, device) -> tuple[ttnn.Tensor, ...]:
    """Upload token-selected conditioning from the CUDA stage input.

    This is a validation boundary. The later full model must run the timestep
    embedding and shared modulation projection on TT before block execution.
    """
    if tuple(modulation.shape) != (2, 4 * HIDDEN):
        raise ValueError(f"expected modulation [2, {4 * HIDDEN}], got {tuple(modulation.shape)}")
    mask = target_token_mask.reshape(1, -1, 1).bool()
    result = []
    for chunk in modulation.split(HIDDEN, dim=-1):
        selected = torch.where(mask, chunk[:1, None], chunk[1:2, None])
        result.append(to_device(selected, device))
    return tuple(result)


def select_modulation_device(
    modulation: ttnn.Tensor, target_token_mask: torch.Tensor, device
) -> tuple[ttnn.Tensor, ...]:
    """Select all four modulation vectors entirely on the TT device."""
    selected = select_rows_by_mask(modulation, target_token_mask, 4 * HIDDEN, device)
    sequence = target_token_mask.numel()
    return tuple(
        ttnn.slice(selected, (0, 0, index * HIDDEN), (1, sequence, (index + 1) * HIDDEN)) for index in range(4)
    )


def prefill_block(
    hidden: ttnn.Tensor,
    weights: BlockWeights,
    modulation: tuple[ttnn.Tensor, ...],
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    sequence: int,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    scale1, gate1, scale2, gate2 = modulation
    norm1 = ttnn.layer_norm(hidden, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attended_input = ttnn.multiply(norm1, ttnn.add(scale1, 1.0))
    attention = prefill_attention(
        attended_input, weights.attention, cos, sin, attention_mask, sequence, compute_kernel_config
    )
    hidden = ttnn.add(hidden, ttnn.multiply(ttnn.tanh(gate1), attention))

    norm2 = ttnn.layer_norm(hidden, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mlp_input = ttnn.multiply(norm2, ttnn.add(scale2, 1.0))
    gate = ttnn.matmul(
        mlp_input,
        weights.mlp_gate,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    projection = ttnn.matmul(
        mlp_input,
        weights.mlp_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    activated = ttnn.multiply(ttnn.silu(gate), projection)
    mlp_output = ttnn.matmul(
        activated,
        weights.mlp_out,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    return ttnn.add(hidden, ttnn.multiply(ttnn.tanh(gate2), mlp_output))
