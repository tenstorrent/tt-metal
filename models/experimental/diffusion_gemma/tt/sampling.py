# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device (ttnn) entropy + Gumbel-max primitives for the PCC harness (#47468; cross-ref #47472).

The accuracy harness must validate the diffusion *decisions* — not just logits.
Two of those decisions have **no entropy / −Σ p·log p computation anywhere in
gemma4**, so they are net-new and built here on `ttnn.max/exp/log/div/mul/sum` (+
`argmax`):

  * :func:`token_entropy` — per-position Shannon entropy ``H = −Σ p·log p`` of
    ``softmax(logits / T)``. Mirrors ``reference/sampling.token_entropy`` /
    ``torch.distributions.Categorical(logits).entropy()``.
  * :func:`gumbel_max` — ``argmax(logits / T + gumbel)`` over the vocab axis, with
    the Gumbel noise **injected** (not regenerated) so device argmax decisions can
    be matched token-for-token against the torch oracle (on-device RNG won't
    reproduce torch's RNG bit-exactly — issue #47468 "Determinism requires noise
    injection").

These let the harness diff entropy *values* and Gumbel-max *argmax agreement*
device-vs-torch, including under **bfp8** where small-probability drift can flip
accept/renoise (the whole reason the harness validates decisions, not logits).
``tests/test_device_entropy_harness.py`` measures both on QB2.

Numerical note: entropy is computed as ``H = logsumexp(z) − Σ softmax(z)·z``.
This is algebraically equivalent to ``−Σ p·log p`` while avoiding ``log(p)``
underflow and reducing accept-boundary flips at the 256-token canvas length.
"""

from __future__ import annotations

from typing import Optional

import ttnn


def temperature_scale(logits, temperature: float):
    """``logits / T`` (no-op when T == 1.0)."""
    if temperature == 1.0:
        return logits
    return ttnn.multiply(logits, 1.0 / float(temperature))


def token_entropy(logits, temperature: float = 1.0):
    """Per-position Shannon entropy ``H = −Σ p·log p`` of ``softmax(logits / T)``.

    ``logits``: ``[..., vocab]`` (TILE_LAYOUT). Returns ``[..., 1]`` (reduced over
    the vocab axis). Uses the logsumexp form to avoid ``log(p)`` underflow.
    """
    z = temperature_scale(logits, temperature)
    zmax = ttnn.max(z, dim=-1, keepdim=True)
    shifted = ttnn.subtract(z, zmax)
    exp_shifted = ttnn.exp(shifted)
    sum_exp = ttnn.sum(exp_shifted, dim=-1, keepdim=True)
    log_sum_exp = ttnn.log(sum_exp)
    log_sum = ttnn.add(log_sum_exp, zmax)
    probs = ttnn.div(exp_shifted, sum_exp)
    expected_terms = ttnn.multiply(probs, z)
    expected_z = ttnn.sum(expected_terms, dim=-1, keepdim=True)
    entropy = ttnn.subtract(log_sum, expected_z)
    zmax.deallocate(True)
    shifted.deallocate(True)
    exp_shifted.deallocate(True)
    sum_exp.deallocate(True)
    log_sum_exp.deallocate(True)
    log_sum.deallocate(True)
    probs.deallocate(True)
    expected_terms.deallocate(True)
    expected_z.deallocate(True)
    return entropy  # H = logsumexp(z) - Σ softmax(z)·z


def gumbel_max(logits, temperature: float, noise):
    """Gumbel-max sample: ``argmax(logits / T + noise)`` over the vocab axis.

    ``logits`` / ``noise``: ``[..., vocab]`` (TILE_LAYOUT). ``noise`` is the torch
    run's exact injected Gumbel(0,1) noise (issue #47468 determinism). Returns
    argmax indices ``[..., 1]``. ``noise`` all-zeros reduces to plain
    ``argmax(logits)`` (temperature scaling preserves the argmax).
    """
    z = temperature_scale(logits, temperature)
    perturbed = ttnn.add(z, noise)
    return ttnn.argmax(perturbed, dim=-1, keepdim=True)


def softmax(logits, temperature: float = 1.0, *, compute_kernel_config: Optional[object] = None):
    """``softmax(logits / T)`` over the vocab axis (the self-conditioning soft-embed prob)."""
    z = temperature_scale(logits, temperature)
    if compute_kernel_config is not None:
        return ttnn.softmax(z, dim=-1, numeric_stable=True, compute_kernel_config=compute_kernel_config)
    return ttnn.softmax(z, dim=-1)
