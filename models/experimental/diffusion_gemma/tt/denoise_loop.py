# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device denoise-loop helpers for DiffusionGemma.

This module starts with the single-step decision kernel composition used by the
full W3 loop: sample, entropy, entropy-budget accept, and renoise. The model
forward remains injected by callers so the same helper can be used by synthetic
tests and the real W2 denoise logits path.
"""

from __future__ import annotations

from typing import NamedTuple

import ttnn

from models.experimental.diffusion_gemma.tt import sampling as TS


class TtDenoiseStepResult(NamedTuple):
    canvas: ttnn.Tensor
    accept_mask: ttnn.Tensor
    entropy: ttnn.Tensor
    sampled: ttnn.Tensor
    argmax: ttnn.Tensor


def temperature_at_step(step: int, num_steps: int, t_start: float, t_end: float) -> float:
    """HF reversed-step linear temperature schedule."""
    if num_steps <= 0:
        return t_start
    cur_step = num_steps - step
    return t_end + (t_start - t_end) * (cur_step / num_steps)


def entropy_budget_accept(entropy, budget: float):
    """Device entropy-budget accept mask using the HF exclusive-prefix rule."""
    sorted_vals, sorted_idx = ttnn.sort(entropy, dim=-1)
    cum = ttnn.cumsum(sorted_vals, dim=-1)
    excl = ttnn.subtract(cum, sorted_vals)
    budget_t = ttnn.full(
        list(entropy.shape),
        float(budget),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=entropy.device(),
    )
    accept_sorted = ttnn.le(excl, budget_t)
    accept_sorted_bf = ttnn.typecast(accept_sorted, ttnn.bfloat16)
    zeros = ttnn.typecast(ttnn.zeros_like(entropy), ttnn.bfloat16)
    accept = ttnn.scatter(zeros, -1, sorted_idx, accept_sorted_bf)
    sorted_vals.deallocate(True)
    sorted_idx.deallocate(True)
    cum.deallocate(True)
    excl.deallocate(True)
    budget_t.deallocate(True)
    accept_sorted.deallocate(True)
    accept_sorted_bf.deallocate(True)
    zeros.deallocate(True)
    return accept


def renoise(accept_mask, sampled, noise_tokens):
    """Select sampled tokens where accepted, otherwise random renoise tokens."""
    accept_u32 = ttnn.typecast(accept_mask, ttnn.uint32)
    ones = ttnn.full(
        list(sampled.shape),
        1,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=sampled.device(),
    )
    reject_u32 = ttnn.subtract(ones, accept_u32)
    accepted = ttnn.multiply(sampled, accept_u32)
    rejected = ttnn.multiply(noise_tokens, reject_u32)
    canvas = ttnn.add(accepted, rejected)
    accept_u32.deallocate(True)
    ones.deallocate(True)
    reject_u32.deallocate(True)
    accepted.deallocate(True)
    rejected.deallocate(True)
    return canvas


def denoise_step(
    logits,
    *,
    temperature: float,
    entropy_budget: float,
    gumbel_noise,
    noise_tokens,
):
    """Run one device denoise decision step with injected noise.

    Returns token id tensors in `uint32` TILE layout except `accept_mask` and
    `entropy`, which are bf16/fp32 decision tensors.

    The renoise select is written as uint32 arithmetic because `ttnn.where`
    currently corrupts `uint32` TILE token tensors in this path.
    """
    sampled = TS.gumbel_max(logits, temperature, gumbel_noise)
    sampled = ttnn.typecast(sampled, ttnn.uint32)
    argmax = ttnn.argmax(logits, dim=-1, keepdim=True)
    argmax = ttnn.typecast(argmax, ttnn.uint32)
    entropy = TS.token_entropy(logits, temperature=temperature)
    entropy_for_accept = ttnn.reshape(entropy, (entropy.shape[0] * entropy.shape[1], entropy.shape[2]))
    accept_flat = entropy_budget_accept(entropy_for_accept, entropy_budget)
    accept_mask = ttnn.reshape(accept_flat, (entropy.shape[0], entropy.shape[1], 1, entropy.shape[2]))
    accept_for_where = ttnn.reshape(accept_mask, sampled.shape)
    canvas = renoise(accept_for_where, sampled, noise_tokens)
    entropy_for_accept.deallocate(True)
    accept_flat.deallocate(True)
    accept_for_where.deallocate(True)
    return TtDenoiseStepResult(
        canvas=canvas,
        accept_mask=accept_mask,
        entropy=entropy,
        sampled=sampled,
        argmax=argmax,
    )
