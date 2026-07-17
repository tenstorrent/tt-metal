# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Discrete-diffusion sampling primitives (pure-torch reference).

This is the algorithmic oracle for the denoise loop (#47463) and on-device
canvas sampling (#47472), and the reference for the device-side **entropy-budget
acceptance spike** (plan.md §6 #47463, risk R1). It is intentionally pure torch
(CPU-runnable, no checkpoint / ttnn / hardware) so the exact semantics — most
importantly the **sort-by-confidence + cumulative-entropy cutoff + scatter-back**
the device path must replicate — are pinned and unit-tested before any kernel
work.

Per denoise step (reconciled against transformers `generation_diffusion_gemma.py`):
    temperature-scale (LinearTemperatureScheduleLogitsProcessor)
    -> sample a denoiser canvas
    -> entropy-bound acceptance (EntropyBoundSampler.accept_canvas)
    -> renoise the rejected positions (to RANDOM tokens, not [MASK]).
Acceptance, stopping, and the next-step self-conditioning signal all operate on
the **temperature-scaled** ``processed_logits``. Commit value is the **clean
argmax** of those logits, not the sampled canvas.

Sampler note: HF draws the denoiser canvas with ``torch.multinomial(softmax(
processed_logits))`` (see :func:`sample_canvas`). :func:`gumbel_max_sample` is the
**distributionally-equivalent** Gumbel-max form used for deterministic, injectable
noise — what the device path needs for token-for-token PCC, since on-device RNG
won't match torch's multinomial. The two only affect the *intermediate* canvas
carried between steps; the validated **decisions** (clean argmax commit, per-step
entropy, accept mask) are deterministic in the logits and identical either way.

Determinism (risk R5): for token-for-token PCC vs torch, the caller injects the
torch run's exact Gumbel noise (`gumbel_noise=`) and renoise token ids
(`noise_tokens=`) — on-device RNG will not match bit-exactly.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F


def temperature_at_step(step: int, num_steps: int, t_start: float, t_end: float) -> float:
    """Linear temperature schedule (HF ``LinearTemperatureScheduleLogitsProcessor``).

    HF runs the denoise step index in REVERSE (``cur_step = num_steps .. 1``) and
    computes ``temperature = t_min + (t_max - t_min) * (cur_step / num_steps)``.
    Here ``step`` is the forward index ``0 .. num_steps-1`` (as the loop counts
    up), so ``cur_step = num_steps - step`` and ``t_start`` is HF ``t_max`` (the
    hottest, first step), ``t_end`` is HF ``t_min``:

        temperature(step) = t_end + (t_start - t_end) * ((num_steps - step) / num_steps)

    => step 0 returns ``t_start`` (0.8); the last step (``num_steps-1``) returns
    ``t_end + (t_start - t_end)/num_steps`` (~0.408 for 0.8/0.4/48), NOT exactly
    ``t_end`` — the schedule never reaches ``t_min`` because HF's ``cur_step``
    bottoms out at 1, not 0. The trajectory is monotonically decreasing.
    """
    if num_steps <= 0:
        return t_start
    cur_step = num_steps - step  # HF iterates cur_step = N..1
    return t_end + (t_start - t_end) * (cur_step / num_steps)


def sample_gumbel_noise(
    shape,
    *,
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Standard Gumbel(0,1) noise: ``-log(-log(U))``, U ~ Uniform(0,1)."""
    u = torch.rand(shape, generator=generator, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_max_sample(
    logits: torch.Tensor,
    temperature: float,
    *,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Gumbel-max sampling: ``argmax(logits / T + gumbel)`` over the vocab axis.

    ``logits``: ``[..., vocab]`` -> returns token ids ``[...]``. Pass ``noise``
    to inject exact Gumbel noise for deterministic PCC; ``noise=0`` reduces to
    plain ``argmax(logits)`` (temperature scaling preserves the argmax).
    """
    scaled = logits / temperature
    if noise is None:
        noise = sample_gumbel_noise(logits.shape, generator=generator, device=logits.device, dtype=logits.dtype)
    return torch.argmax(scaled + noise, dim=-1)


def sample_canvas(
    logits: torch.Tensor,
    temperature: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Multinomial canvas sampling — exact HF ``_denoising_step`` mechanism.

    ``denoiser_canvas = multinomial(softmax(logits / T))`` over the vocab axis.
    ``logits``: ``[B, L, vocab]`` -> token ids ``[B, L]``. Uses fp32 softmax like
    HF. Equivalent in distribution to :func:`gumbel_max_sample`; use that instead
    when you need injectable, device-reproducible noise (R5).
    """
    probs = F.softmax(logits / temperature, dim=-1, dtype=torch.float32)
    flat = probs.reshape(-1, probs.shape[-1])
    ids = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    return ids.view(*logits.shape[:-1])


def token_entropy(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    """Per-position Shannon entropy ``H = -sum p log p`` of ``softmax(logits/T)``.

    ``logits``: ``[..., vocab]`` -> entropy ``[...]``. Uniform -> ``log(vocab)``;
    one-hot -> ~0. Low entropy == high confidence.
    """
    logp = F.log_softmax(logits / temperature, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def entropy_budget_accept(
    entropy: torch.Tensor,
    budget: float,
    *,
    min_accept: int = 1,
) -> torch.Tensor:
    """Entropy-bound acceptance — exact reproduction of HF ``EntropyBoundSampler.accept_canvas``.

    ``entropy``: ``[..., L]`` -> bool accept mask ``[..., L]``. Positions are
    sorted by ASCENDING entropy (most confident first); position ``i`` (in sorted
    order) is accepted iff the **exclusive** entropy prefix — the sum of the
    entropies of all *strictly more confident* positions — stays ``<= budget``::

        sorted_entropy, sort_idx = sort(entropy)              # ascending
        cum = cumsum(sorted_entropy)
        accept_sorted = (cum - sorted_entropy) <= budget      # EXCLUSIVE prefix
        accept = scatter_back(accept_sorted, sort_idx)        # to original positions

    This is the upper bound on the joint mutual information of the accepted set
    (sum_i^k H_i - max_i H_i <= bound, https://arxiv.org/pdf/2505.24857), so the
    accepted tokens are ~independent. The most-confident position always has an
    exclusive prefix of 0, so it is always accepted (>=1 accepted per step) — HF
    has no explicit ``min_accept``. ``min_accept`` is retained only for the
    device spike's API (#47463) and is a no-op for the HF-default ``<=1``.

    The scatter-back is the inverse-permutation the device path must replicate.
    """
    sorted_entropy, sort_idx = torch.sort(entropy, dim=-1)  # ascending: confident first
    cum = torch.cumsum(sorted_entropy, dim=-1)
    accept_sorted = (cum - sorted_entropy) <= budget  # exclusive prefix (HF accept_canvas)
    if min_accept > 0:
        accept_sorted[..., :min_accept] = True
    accept = torch.zeros_like(entropy, dtype=torch.bool)
    accept.scatter_(-1, sort_idx, accept_sorted)  # scatter-back to original positions
    return accept


def renoise(
    token_ids: torch.Tensor,
    accept_mask: torch.Tensor,
    vocab_size: int,
    *,
    noise_tokens: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Keep accepted token ids; renoise rejected positions to RANDOM tokens.

    Uniform discrete diffusion (no ``[MASK]`` / absorbing state). Pass
    ``noise_tokens`` to inject the torch run's exact renoise ids (PCC, R5).
    """
    if noise_tokens is None:
        noise_tokens = torch.randint(0, vocab_size, token_ids.shape, generator=generator, device=token_ids.device)
    return torch.where(accept_mask, token_ids, noise_tokens)


def random_canvas(
    shape,
    vocab_size: int,
    *,
    generator: Optional[torch.Generator] = None,
    device=None,
) -> torch.Tensor:
    """Initialize a canvas to random token ids (the diffusion noise prior)."""
    return torch.randint(0, vocab_size, shape, generator=generator, device=device)


def is_converged(
    prev_tokens: torch.Tensor,
    cur_tokens: torch.Tensor,
    entropy: torch.Tensor,
    entropy_threshold: float,
) -> bool:
    """Halt when the argmax canvas is stable AND mean entropy < threshold.

    Whole-canvas (batch-collapsed) convergence; per-request halting for batched
    decode is #47557.
    """
    stable = torch.equal(prev_tokens, cur_tokens)
    low_entropy = entropy.mean().item() < entropy_threshold
    return stable and low_entropy


class DenoiseStepResult(NamedTuple):
    canvas: torch.Tensor  # [B, L] updated canvas token ids (accepted=sampled, rejected=renoised)
    accept_mask: torch.Tensor  # [B, L] bool
    entropy: torch.Tensor  # [B, L]
    sampled: torch.Tensor  # [B, L] sampled ids (multinomial HF-faithful, or Gumbel-max device path)
    argmax: torch.Tensor  # [B, L] clean argmax (the commit value)


# Sampler choices for the denoiser canvas (the intermediate that feeds the next
# step). They are distribution-equivalent but NOT token-equal under a fixed seed,
# and the sampled canvas cascades into the next forward — so pick deliberately:
#   "multinomial" — HF-faithful: matches DiffusionGemma `_denoising_step`'s
#                   torch.multinomial(softmax(processed_logits)). Use for the HF
#                   reference / reconstructed torch oracle trajectory.
#   "gumbel"      — argmax(logits/T + gumbel). Use for the DEVICE-comparison
#                   trajectory with the torch run's *injected* Gumbel noise, so
#                   on-device decisions are token-for-token comparable (R5).
SAMPLER_MULTINOMIAL = "multinomial"
SAMPLER_GUMBEL = "gumbel"


def denoise_step(
    logits: torch.Tensor,
    *,
    temperature: float,
    entropy_budget: float,
    vocab_size: int,
    sampler: str = SAMPLER_MULTINOMIAL,
    gumbel_noise: Optional[torch.Tensor] = None,
    noise_tokens: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    min_accept: int = 1,
) -> DenoiseStepResult:
    """Compose one denoise step over a ``[B, L, vocab]`` logits tensor.

    Returns the updated canvas plus the intermediate decisions, and the clean
    argmax (committed at convergence, never the noisy sample). The clean argmax,
    entropy, and accept mask are deterministic in ``logits`` (sampler-independent);
    only ``sampled`` / ``canvas`` depend on the sampler.

    ``sampler`` selects the HF-faithful ``multinomial`` (default) or ``gumbel``.
    Passing ``gumbel_noise`` forces the gumbel path (device-injection). ``generator``
    seeds the regenerated noise when neither ``gumbel_noise`` nor ``noise_tokens``
    is injected, so a single seeded generator makes the trajectory reproducible.
    """
    if gumbel_noise is not None or sampler == SAMPLER_GUMBEL:
        sampled = gumbel_max_sample(logits, temperature, noise=gumbel_noise, generator=generator)
    elif sampler == SAMPLER_MULTINOMIAL:
        sampled = sample_canvas(logits, temperature, generator=generator)
    else:
        raise ValueError(f"unknown sampler {sampler!r}; expected {SAMPLER_MULTINOMIAL!r} or {SAMPLER_GUMBEL!r}")
    entropy = token_entropy(logits, temperature=temperature)
    accept = entropy_budget_accept(entropy, entropy_budget, min_accept=min_accept)
    canvas = renoise(sampled, accept, vocab_size, noise_tokens=noise_tokens, generator=generator)
    argmax = logits.argmax(dim=-1)
    return DenoiseStepResult(canvas=canvas, accept_mask=accept, entropy=entropy, sampled=sampled, argmax=argmax)
