# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen Image 2.1 single-card TTNN attention for the prefill path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .tt_dit_components import rotary_split_half, split_half_indices, to_device


HEADS = 32
HEAD_DIM = 128
HIDDEN = HEADS * HEAD_DIM


@dataclass
class AttentionWeights:
    q: ttnn.Tensor
    k: ttnn.Tensor
    v: ttnn.Tensor
    out: ttnn.Tensor
    q_norm: ttnn.Tensor
    k_norm: ttnn.Tensor


def prepare_weights(state: dict[str, torch.Tensor], device) -> AttentionWeights:
    """Permute Q/K channels once so TT's half-split RoPE equals Qwen's pair RoPE."""
    indices = split_half_indices(HEAD_DIM)

    def projection(name: str, pair_order: bool) -> ttnn.Tensor:
        weight = state[f"attn.{name}.weight"]
        if pair_order:
            weight = weight.reshape(HEADS, HEAD_DIM, HIDDEN)[:, indices].reshape(HIDDEN, HIDDEN)
        return to_device(weight.T.contiguous(), device)

    def scale(name: str) -> ttnn.Tensor:
        return to_device(state[f"attn.norm_{name}.weight"][indices].reshape(1, 1, 1, HEAD_DIM), device)

    return AttentionWeights(
        q=projection("to_q", True),
        k=projection("to_k", True),
        v=projection("to_v", False),
        out=to_device(state["attn.to_out.0.weight"].T.contiguous(), device),
        q_norm=scale("q"),
        k_norm=scale("k"),
    )


def prefill_mask(sequence: int, segments: list[tuple[int, int, bool]], device, key_valid: torch.Tensor | None = None):
    """Build the additive block-causal mask from public token metadata."""
    padded = (sequence + 31) // 32 * 32
    allowed = torch.zeros(sequence, sequence, dtype=torch.bool)
    for start, end, is_text in segments:
        allowed[start:end, :start] = True
        allowed[start:end, start:end] = (
            torch.ones(end - start, end - start, dtype=torch.bool).tril() if is_text else True
        )
    prefix_len = segments[-1][1] if segments else 0
    allowed[prefix_len:, :] = True
    if key_valid is not None:
        allowed &= key_valid[0, :sequence].bool()[None, :]
    mask = torch.full((1, 1, padded, padded), -10000.0, dtype=torch.bfloat16)
    mask[0, 0, :sequence, :sequence] = torch.where(allowed, 0.0, -10000.0).to(torch.bfloat16)
    mask[0, 0, sequence:, 0] = 0
    return to_device(mask, device)


def prefill_attention(
    hidden: ttnn.Tensor,
    weights: AttentionWeights,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    mask: ttnn.Tensor,
    sequence: int,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Project QKV, normalize, rotate, attend, and project back on device."""
    padded = (sequence + 31) // 32 * 32

    def project(weight: ttnn.Tensor, norm: ttnn.Tensor | None = None) -> ttnn.Tensor:
        result = ttnn.matmul(
            hidden,
            weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        result = ttnn.reshape(result, (1, sequence, HEADS, HEAD_DIM))
        if norm is not None:
            result = ttnn.rms_norm(result, weight=norm, epsilon=1e-6, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = ttnn.permute(result, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if padded != sequence:
            result = ttnn.pad(
                result,
                ((0, 0), (0, 0), (0, padded - sequence), (0, 0)),
                0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return result

    q = rotary_split_half(project(weights.q, weights.q_norm), cos, sin)
    k = rotary_split_half(project(weights.k, weights.k_norm), cos, sin)
    v = project(weights.v)
    context = ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        is_causal=False,
        scale=HEAD_DIM**-0.5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    context = ttnn.slice(context, (0, 0, 0, 0), (1, HEADS, sequence, HEAD_DIM))
    context = ttnn.permute(context, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    context = ttnn.reshape(context, (1, sequence, HIDDEN))
    return ttnn.matmul(
        context,
        weights.out,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
