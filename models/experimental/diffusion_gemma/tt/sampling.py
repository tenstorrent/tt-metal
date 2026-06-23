# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device (ttnn) entropy + Gumbel-max primitives for the PCC harness (#47468; cross-ref #47472).

The accuracy harness must validate the diffusion *decisions* — not just logits.
Two of those decisions have **no entropy / −Σ p·log p computation anywhere in
gemma4**, so they are net-new and built here on `ttnn.softmax/log/mul/sum` (+
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

Numerical note: ``−Σ p·log p`` is computed directly per the issue. For finite
logits ``softmax`` is > 0, so ``log`` is finite; with extreme underflow add a
small eps before ``log`` (the harness inputs do not underflow). A logsumexp form
(``H = logsumexp(z) − Σ p·z``) avoids ``log(p)`` entirely if needed downstream.
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
    the vocab axis). Built on ``ttnn.softmax`` / ``log`` / ``mul`` / ``sum``.
    """
    z = temperature_scale(logits, temperature)
    p = ttnn.softmax(z, dim=-1)
    logp = ttnn.log(p)
    plogp = ttnn.multiply(p, logp)
    neg_h = ttnn.sum(plogp, dim=-1, keepdim=True)  # Σ p·log p  (negative)
    p.deallocate(True)
    logp.deallocate(True)
    plogp.deallocate(True)
    return ttnn.neg(neg_h)  # H = −Σ p·log p


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
