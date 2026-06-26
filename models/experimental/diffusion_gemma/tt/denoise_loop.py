# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device denoise-loop helpers for DiffusionGemma.

This module starts with the single-step decision kernel composition used by the
full W3 loop: sample, entropy, entropy-budget accept, and renoise. The model
forward remains injected by callers so the same helper can be used by synthetic
tests and the real W2 denoise logits path.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional

import torch
import ttnn

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory, StepRecord
from models.experimental.diffusion_gemma.tt import sampling as TS

TtLogitsFn = Callable[[ttnn.Tensor, int], ttnn.Tensor]
TtNoiseFn = Callable[[int], ttnn.Tensor]


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
    """Device entropy-budget accept mask using the HF exclusive-prefix rule.

    This path is load-bearing for the Part I decision-fidelity bar: device sort
    regressions must show up as accept-mask/count mismatches against the host
    oracle, not as silent trajectory drift.
    """
    sorted_vals, sorted_idx = ttnn.sort(entropy, dim=-1)
    cum = ttnn.cumsum(sorted_vals, dim=-1)
    excl = ttnn.subtract(cum, sorted_vals)
    budget_t = ttnn.full(
        list(entropy.shape),
        float(budget),
        dtype=entropy.get_dtype(),
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


def _to_host_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    try:
        return ttnn.to_torch(tensor)
    except RuntimeError as exc:
        if "distributed on MeshShape" not in str(exc):
            raise
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _ids_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return _to_host_torch(tensor).squeeze(1).squeeze(-1).to(torch.long)


def _entropy_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return _to_host_torch(tensor).squeeze(1).squeeze(-1).float()


def _accept_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return _to_host_torch(tensor).squeeze(1).squeeze(1) > 0.5


def _deallocate_decision_tensors(res: TtDenoiseStepResult) -> None:
    res.accept_mask.deallocate(True)
    res.entropy.deallocate(True)
    res.sampled.deallocate(True)
    res.argmax.deallocate(True)


def _reset_logits_fn(logits_fn: TtLogitsFn) -> None:
    reset = getattr(logits_fn, "reset", None)
    if callable(reset):
        reset()


def denoise_block(
    logits_fn: TtLogitsFn,
    init_canvas: ttnn.Tensor,
    config: DiffusionConfig,
    *,
    gumbel_noise_fn: Optional[TtNoiseFn] = None,
    noise_tokens_fn: Optional[TtNoiseFn] = None,
) -> DenoiseTrajectory:
    """Run a device denoise trajectory and return host decision records.

    The full ``[B, L, vocab]`` logits stay on device. For the data-dependent halt
    check and the trajectory harness, this reads back only per-step ``[B, L]``
    decision tensors: argmax, entropy, sampled ids, accept mask, and canvas.
    ``init_canvas`` is consumed; each superseded device canvas is deallocated.
    """
    canvas = init_canvas
    records: List[StepRecord] = []
    committed: Optional[torch.Tensor] = None
    argmax_history: List[torch.Tensor] = []
    n_stable = config.stable_steps_to_halt

    for step in range(config.max_denoise_steps):
        temperature = temperature_at_step(
            step, config.max_denoise_steps, config.temperature_start, config.temperature_end
        )
        res = denoise_step(
            logits_fn(canvas, step),
            temperature=temperature,
            entropy_budget=config.entropy_budget,
            gumbel_noise=gumbel_noise_fn(step) if gumbel_noise_fn else None,
            noise_tokens=noise_tokens_fn(step) if noise_tokens_fn else None,
        )

        argmax = _ids_to_torch(res.argmax)
        entropy = _entropy_to_torch(res.entropy)
        sampled = _ids_to_torch(res.sampled)
        accept_mask = _accept_to_torch(res.accept_mask)
        host_canvas = _ids_to_torch(res.canvas)
        entropy_mean = entropy.mean().item()
        records.append(
            StepRecord(
                step=step,
                temperature=temperature,
                entropy_mean=entropy_mean,
                num_accepted=int(accept_mask.sum()),
                argmax=argmax,
                entropy=entropy,
                sampled=sampled,
                accept_mask=accept_mask,
                canvas=host_canvas,
            )
        )

        committed = argmax
        stable = n_stable == 0 or (
            len(argmax_history) >= n_stable and all(torch.equal(argmax, h) for h in argmax_history[-n_stable:])
        )
        confident = entropy_mean < config.entropy_stop_threshold
        argmax_history.append(argmax)
        next_canvas = res.canvas
        if canvas is not next_canvas:
            canvas.deallocate(True)
        _deallocate_decision_tensors(res)

        if stable and confident:
            next_canvas.deallocate(True)
            _reset_logits_fn(logits_fn)
            return DenoiseTrajectory(committed, step + 1, True, records)

        canvas = next_canvas

    if config.max_denoise_steps > 0:
        canvas.deallocate(True)
    _reset_logits_fn(logits_fn)
    return DenoiseTrajectory(committed, config.max_denoise_steps, False, records)
