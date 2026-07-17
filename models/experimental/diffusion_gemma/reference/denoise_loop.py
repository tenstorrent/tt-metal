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
    """Per-step decision record for the harness — decisions, not just logits (#47468).

    The harness validates the diffusion *decisions* a candidate trajectory makes;
    keep every per-step tensor a candidate must reproduce: the clean argmax
    (commit value), per-token entropy values, Gumbel-sampled ids, the accept
    mask (entropy-budget acceptance scatter-back), and the renoised canvas
    that carries into the next step.
    """

    step: int
    temperature: float
    entropy_mean: float
    num_accepted: int
    argmax: torch.Tensor  # [B, L] clean argmax (the commit value if this is the last step)
    entropy: torch.Tensor  # [B, L] per-token entropy of the temperature-scaled logits
    sampled: torch.Tensor  # [B, L] Gumbel-max sampled token ids
    accept_mask: torch.Tensor  # [B, L] bool — entropy-budget accept decisions (scatter-back)
    canvas: torch.Tensor  # [B, L] renoised canvas after this step (input to the next step)


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
    sampler: str = S.SAMPLER_MULTINOMIAL,
    gumbel_noise_fn: Optional[NoiseFn] = None,
    noise_tokens_fn: Optional[NoiseFn] = None,
    generator: Optional[torch.Generator] = None,
) -> DenoiseTrajectory:
    """Run one canvas's denoise trajectory.

    ``logits_fn(canvas, step)`` returns ``[B, L, vocab]`` for the current canvas.
    ``init_canvas`` is ``[B, L]`` (random token ids — the diffusion prior).
    ``sampler`` is ``"multinomial"`` (HF-faithful oracle, default) or ``"gumbel"``
    (device path). Optional ``gumbel_noise_fn(step)`` / ``noise_tokens_fn(step)``
    inject the torch run's exact noise for token-for-token PCC determinism (R5,
    forces the gumbel path); ``generator`` seeds regenerated noise otherwise so a
    single seeded generator reproduces the whole trajectory.
    """
    canvas = init_canvas
    committed: Optional[torch.Tensor] = None
    records: List[StepRecord] = []
    # HF StableAndConfidentStoppingCriteria: halt when the clean-argmax canvas has
    # been stable across the last `stable_steps_to_halt` (HF `stability_threshold`)
    # steps AND the mean per-position entropy of the temperature-scaled logits is
    # below `entropy_stop_threshold` (HF `confidence_threshold`). HF keeps a rolling
    # argmax buffer (init -1); we keep the last N argmaxes.
    # NOTE: whole-batch collapsed — we halt the loop when ALL rows satisfy the
    # criterion, vs HF's PER-EXAMPLE finished-masking (a finished row is frozen and
    # padded while others continue). Equivalent for batch=1 (the current scope);
    # per-request halting/freezing for batch>1 is #47557. Parity vs the exact HF
    # criterion is locked by tests/test_upstream_parity (batch-1).
    n_stable = config.stable_steps_to_halt
    argmax_history: List[torch.Tensor] = []

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
            sampler=sampler,
            gumbel_noise=gumbel_noise_fn(step) if gumbel_noise_fn else None,
            noise_tokens=noise_tokens_fn(step) if noise_tokens_fn else None,
            generator=generator,
        )

        entropy_mean = res.entropy.mean().item()
        records.append(
            StepRecord(
                step=step,
                temperature=temperature,
                entropy_mean=entropy_mean,
                num_accepted=int(res.accept_mask.sum()),
                argmax=res.argmax,
                entropy=res.entropy,
                sampled=res.sampled,
                accept_mask=res.accept_mask,
                canvas=res.canvas,
            )
        )

        committed = res.argmax  # commit = clean argmax (never the noisy sample)
        # stable: current argmax equals each of the last N argmaxes (HF compares the
        # current canvas to all N rolling-buffer slots before storing it).
        stable = n_stable == 0 or (
            len(argmax_history) >= n_stable and all(torch.equal(res.argmax, h) for h in argmax_history[-n_stable:])
        )
        confident = entropy_mean < config.entropy_stop_threshold
        argmax_history.append(res.argmax)
        canvas = res.canvas  # carry the renoised canvas into the next step

        if stable and confident:
            return DenoiseTrajectory(committed, step + 1, True, records)

    return DenoiseTrajectory(committed, config.max_denoise_steps, False, records)
