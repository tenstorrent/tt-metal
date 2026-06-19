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

Per denoise step (plan.md §2.1):
    temperature-scale -> Gumbel-max -> token entropy -> entropy-budget acceptance
    -> renoise the rejected positions (to RANDOM tokens, not [MASK]).
Commit value is the **clean argmax** of the logits, not the noisy sample.

Determinism (risk R5): for token-for-token PCC vs torch, the caller injects the
torch run's exact Gumbel noise (`gumbel_noise=`) and renoise token ids
(`noise_tokens=`) — on-device RNG will not match bit-exactly.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F


def temperature_at_step(step: int, num_steps: int, t_start: float, t_end: float) -> float:
    """Linear temperature schedule across denoise steps (default 0.8 -> 0.4)."""
    if num_steps <= 1:
        return t_end
    frac = step / (num_steps - 1)
    return t_start + (t_end - t_start) * frac


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
    """Accept most->least confident positions until cumulative entropy > budget.

    ``entropy``: ``[..., L]`` -> bool accept mask ``[..., L]``. Positions are
    sorted by ascending entropy (most confident first); the inclusive cumulative
    entropy prefix that stays ``<= budget`` is accepted (the position that tips
    the sum over budget is rejected), then the decision is **scattered back** to
    the original canvas positions (the inverse-permutation the device path must
    replicate, #47463). ``min_accept`` force-accepts the N most-confident
    positions to guarantee per-step progress.

    Note: the inclusive-``<=`` cutoff and ``min_accept`` tie-break are to be
    reconciled against the HF reference once it is importable (#47468).
    """
    sorted_entropy, sort_idx = torch.sort(entropy, dim=-1)  # ascending: confident first
    cum = torch.cumsum(sorted_entropy, dim=-1)
    accept_sorted = cum <= budget
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
    sampled: torch.Tensor  # [B, L] Gumbel-max sampled ids
    argmax: torch.Tensor  # [B, L] clean argmax (the commit value)


def denoise_step(
    logits: torch.Tensor,
    *,
    temperature: float,
    entropy_budget: float,
    vocab_size: int,
    gumbel_noise: Optional[torch.Tensor] = None,
    noise_tokens: Optional[torch.Tensor] = None,
    min_accept: int = 1,
) -> DenoiseStepResult:
    """Compose one denoise step over a ``[B, L, vocab]`` logits tensor.

    Returns the updated canvas plus the intermediate decisions, and the clean
    argmax (committed at convergence, never the noisy sample).
    """
    sampled = gumbel_max_sample(logits, temperature, noise=gumbel_noise)
    entropy = token_entropy(logits, temperature=temperature)
    accept = entropy_budget_accept(entropy, entropy_budget, min_accept=min_accept)
    canvas = renoise(sampled, accept, vocab_size, noise_tokens=noise_tokens)
    argmax = logits.argmax(dim=-1)
    return DenoiseStepResult(canvas=canvas, accept_mask=accept, entropy=entropy, sampled=sampled, argmax=argmax)
