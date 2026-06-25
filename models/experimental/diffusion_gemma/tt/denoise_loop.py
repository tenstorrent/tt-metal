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


def denoise_step(
    logits,
    *,
    temperature: float,
    entropy_budget: float,
    gumbel_noise,
    noise_tokens,
    token_ids_fit_bf16: bool = False,
):
    """Run one device denoise decision step with injected noise.

    Returns token id tensors in `uint32` TILE layout except `accept_mask` and
    `entropy`, which are bf16/fp32 decision tensors.

    `ttnn.where` currently corrupts `uint32` TILE token tensors in this path.
    The bf16 where fallback is exact only for small-vocab smoke tests whose token
    ids fit in bf16 exactly; full-vocab DiffusionGemma needs a uint32-safe
    renoise path before this helper is production-ready.
    """
    if not token_ids_fit_bf16:
        raise ValueError("set token_ids_fit_bf16=True only for small-vocab smoke tests")

    sampled = TS.gumbel_max(logits, temperature, gumbel_noise)
    sampled = ttnn.typecast(sampled, ttnn.uint32)
    argmax = ttnn.argmax(logits, dim=-1, keepdim=True)
    argmax = ttnn.typecast(argmax, ttnn.uint32)
    entropy = TS.token_entropy(logits, temperature=temperature)
    entropy_for_accept = ttnn.reshape(entropy, (entropy.shape[0] * entropy.shape[1], entropy.shape[2]))
    accept_flat = entropy_budget_accept(entropy_for_accept, entropy_budget)
    accept_mask = ttnn.reshape(accept_flat, (entropy.shape[0], entropy.shape[1], 1, entropy.shape[2]))
    accept_for_where = ttnn.reshape(accept_mask, sampled.shape)
    sampled_bf = ttnn.typecast(sampled, ttnn.bfloat16)
    noise_tokens_bf = ttnn.typecast(noise_tokens, ttnn.bfloat16)
    canvas_bf = ttnn.where(accept_for_where, sampled_bf, noise_tokens_bf)
    canvas = ttnn.typecast(canvas_bf, ttnn.uint32)
    entropy_for_accept.deallocate(True)
    accept_flat.deallocate(True)
    accept_for_where.deallocate(True)
    sampled_bf.deallocate(True)
    noise_tokens_bf.deallocate(True)
    canvas_bf.deallocate(True)
    return TtDenoiseStepResult(
        canvas=canvas,
        accept_mask=accept_mask,
        entropy=entropy,
        sampled=sampled,
        argmax=argmax,
    )
