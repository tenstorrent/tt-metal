# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference per-block denoise trajectory (pure torch, #47463/#47468).

Assembles the sampling primitives (``reference/sampling.py``) into the full
per-canvas denoise loop: up to ``max_denoise_steps`` of
``forward -> sample -> accept -> renoise``, halting when the clean-argmax canvas
is stable AND mean entropy drops below threshold, committing the **clean
argmax** (never the noisy sample). This is the reference trajectory the PCC
harness (#47468) compares the device loop (#47463) against — it validates the
diffusion *decisions* (per-step entropy / argmax), not just final logits.

The model forward is injected as a ``logits_fn(canvas_tokens, step) -> logits``
callback so the loop is testable with a synthetic oracle now and the real
backbone forward later. The block-autoregressive *outer* loop (commit -> append
KV -> next block) is device-side and lives with the KV state machine (#47474) /
e2e (#47464).
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional

import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig

from . import sampling as S

LogitsFn = Callable[[torch.Tensor, int], torch.Tensor]
NoiseFn = Callable[[int], torch.Tensor]


class StepRecord(NamedTuple):
    step: int
    temperature: float
    entropy_mean: float
    num_accepted: int
    argmax: torch.Tensor  # [B, L] clean argmax this step


class DenoiseTrajectory(NamedTuple):
    committed: torch.Tensor  # [B, L] final clean-argmax tokens (the commit value)
    num_steps: int  # steps actually run
    halted: bool  # True if convergence halt fired; False if hit the step cap
    per_step: List[StepRecord]


def denoise_block(
    logits_fn: LogitsFn,
    init_canvas: torch.Tensor,
    config: DiffusionConfig,
    vocab_size: int,
    *,
    gumbel_noise_fn: Optional[NoiseFn] = None,
    noise_tokens_fn: Optional[NoiseFn] = None,
) -> DenoiseTrajectory:
    """Run one canvas's denoise trajectory.

    ``logits_fn(canvas, step)`` returns ``[B, L, vocab]`` for the current canvas.
    ``init_canvas`` is ``[B, L]`` (random token ids — the diffusion prior).
    Optional ``gumbel_noise_fn(step)`` / ``noise_tokens_fn(step)`` inject the
    torch run's exact noise for token-for-token PCC determinism (R5).
    """
    canvas = init_canvas
    prev_argmax: Optional[torch.Tensor] = None
    committed: Optional[torch.Tensor] = None
    records: List[StepRecord] = []
    stable_count = 0

    for step in range(config.max_denoise_steps):
        temperature = S.temperature_at_step(
            step, config.max_denoise_steps, config.temperature_start, config.temperature_end
        )
        logits = logits_fn(canvas, step)
        res = S.denoise_step(
            logits,
            temperature=temperature,
            entropy_budget=config.entropy_budget,
            vocab_size=vocab_size,
            gumbel_noise=gumbel_noise_fn(step) if gumbel_noise_fn else None,
            noise_tokens=noise_tokens_fn(step) if noise_tokens_fn else None,
        )

        entropy_mean = res.entropy.mean().item()
        records.append(StepRecord(step, temperature, entropy_mean, int(res.accept_mask.sum()), res.argmax))

        committed = res.argmax  # commit = clean argmax
        stable = prev_argmax is not None and torch.equal(res.argmax, prev_argmax)
        if stable and entropy_mean < config.entropy_stop_threshold:
            stable_count += 1
        else:
            stable_count = 0
        prev_argmax = res.argmax
        canvas = res.canvas  # carry the renoised canvas into the next step

        if stable_count >= config.stable_steps_to_halt:
            return DenoiseTrajectory(committed, step + 1, True, records)

    return DenoiseTrajectory(committed, config.max_denoise_steps, False, records)
