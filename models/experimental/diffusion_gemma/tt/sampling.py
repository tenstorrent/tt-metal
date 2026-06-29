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


def _deallocate_scaled_if_temporary(scaled, logits) -> None:
    if scaled is not logits:
        scaled.deallocate(True)


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
    _deallocate_scaled_if_temporary(z, logits)
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
    sampled = ttnn.argmax(perturbed, dim=-1, keepdim=True)
    perturbed.deallocate(True)
    _deallocate_scaled_if_temporary(z, logits)
    return sampled


def canvas_sample(logits, temperature: float, gumbel_noise):
    """Deterministic canvas sampler for W4 using injected Gumbel noise.

    This is the released per-position canvas draw used by the diffusion loop:
    ``argmax(logits / T + gumbel)`` over every canvas position. The noise is
    supplied by the caller for torch/device token-exact validation.
    """
    return gumbel_max(logits, temperature, gumbel_noise)


def _gumbel_from_uniform(u):
    u_eps = ttnn.add(u, 1.0e-10)
    log_u = ttnn.log(u_eps)
    neg_log_u = ttnn.multiply(log_u, -1.0)
    neg_log_u_eps = ttnn.add(neg_log_u, 1.0e-10)
    log_neg_log_u = ttnn.log(neg_log_u_eps)
    gumbel = ttnn.multiply(log_neg_log_u, -1.0)
    u.deallocate(True)
    u_eps.deallocate(True)
    log_u.deallocate(True)
    neg_log_u.deallocate(True)
    neg_log_u_eps.deallocate(True)
    log_neg_log_u.deallocate(True)
    return gumbel


def _rand_mesh_mapper(device):
    if hasattr(device, "shape") and device.get_num_devices() > 1:
        return ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate()],
            mesh_shape_override=ttnn.MeshShape([device.get_num_devices()]),
        )
    return None


def _validate_ttnn_rand_seed(seed: int) -> int:
    seed = int(seed)
    if seed <= 0:
        raise ValueError("TTNN regenerated Gumbel noise requires a positive nonzero seed")
    return seed


def sample_gumbel_noise(shape, *, device, seed: int, dtype=ttnn.float32):
    """Generate device Gumbel(0,1) noise with a deterministic TTNN rand seed."""
    seed = _validate_ttnn_rand_seed(seed)
    u = ttnn.rand(
        shape,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        low=0.0,
        high=1.0,
        seed=seed,
        mesh_mapper=_rand_mesh_mapper(device),
    )
    return _gumbel_from_uniform(u)


def sample_gumbel_noise_with_permuted_vocab(shape, *, device, seed: int, dtype=ttnn.float32):
    """Generate regenerated Gumbel noise with vocab not produced as the rand innermost axis.

    QB2's single-call ``ttnn.rand(shape=[..., vocab])`` path currently shows
    last-dimension correlation that biases Gumbel-max distributions. Generating
    the vocab axis first and permuting back preserves one random draw per logits
    element while avoiding that correlation in W4 distributional validation.
    """
    seed = _validate_ttnn_rand_seed(seed)
    shape = tuple(shape)
    if len(shape) < 2:
        raise ValueError("shape must include at least a sample axis and a vocab axis")

    rand_shape = (shape[-1], *shape[1:-1], shape[0])
    permute_order = (len(shape) - 1, *range(1, len(shape) - 1), 0)
    u = ttnn.rand(
        rand_shape,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        low=0.0,
        high=1.0,
        seed=seed,
        mesh_mapper=_rand_mesh_mapper(device),
    )
    u = ttnn.permute(u, permute_order)
    return _gumbel_from_uniform(u)


def sample_gumbel_noise_by_vocab_chunks(shape, *, device, seed: int, vocab_chunk_size: int = 1, dtype=ttnn.float32):
    """Slow iid-by-vocab-chunk Gumbel generator for distributional validation.

    QB2 currently shows last-dimension correlation when one large ``ttnn.rand``
    call generates all vocab noise at once. Generating each vocab chunk with a
    distinct seed removes that toy-vocab bias, but this is intentionally a
    validation/diagnostic path rather than the full-vocab production sampler.
    """
    seed = _validate_ttnn_rand_seed(seed)
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    shape = tuple(shape)
    vocab_size = shape[-1]
    parts = []
    for offset in range(0, vocab_size, vocab_chunk_size):
        chunk_size = min(vocab_chunk_size, vocab_size - offset)
        chunk_shape = (*shape[:-1], chunk_size)
        u = ttnn.rand(
            chunk_shape,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            low=0.0,
            high=1.0,
            seed=seed + offset,
            mesh_mapper=_rand_mesh_mapper(device),
        )
        parts.append(_gumbel_from_uniform(u))

    if len(parts) == 1:
        return parts[0]
    return ttnn.concat(parts, dim=-1)


def softmax(logits, temperature: float = 1.0, *, compute_kernel_config: Optional[object] = None):
    """``softmax(logits / T)`` over the vocab axis (the self-conditioning soft-embed prob)."""
    z = temperature_scale(logits, temperature)
    if compute_kernel_config is not None:
        probs = ttnn.softmax(z, dim=-1, numeric_stable=True, compute_kernel_config=compute_kernel_config)
    else:
        probs = ttnn.softmax(z, dim=-1)
    _deallocate_scaled_if_temporary(z, logits)
    return probs
