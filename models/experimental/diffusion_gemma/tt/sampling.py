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

from typing import NamedTuple

import ttnn


class ChunkedGumbelNoise(NamedTuple):
    seed: int
    vocab_chunk_size: int = 1024
    dtype: object = ttnn.float32


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
    shifted = ttnn.subtract(z, zmax, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    exp_shifted = ttnn.exp(shifted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sum_exp = ttnn.sum(exp_shifted, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    log_sum_exp = ttnn.log(sum_exp)
    # H = logsumexp(z) - E[z].  Since shifted = z - zmax, compute the
    # algebraically equivalent log(sum(exp(shifted))) - E[shifted] to avoid
    # subtracting two large, nearly equal values for very confident logits.
    # Use Σ(exp(shifted) * shifted) / Σexp directly so a full probability tensor
    # is not live alongside the full shifted tensor in the production path.
    expected_terms = ttnn.multiply(exp_shifted, shifted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sum_weighted_shifted = ttnn.sum(expected_terms, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    expected_shifted = ttnn.div(sum_weighted_shifted, sum_exp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    entropy = ttnn.subtract(log_sum_exp, expected_shifted)
    zmax.deallocate(True)
    shifted.deallocate(True)
    exp_shifted.deallocate(True)
    sum_exp.deallocate(True)
    log_sum_exp.deallocate(True)
    expected_terms.deallocate(True)
    sum_weighted_shifted.deallocate(True)
    expected_shifted.deallocate(True)
    _deallocate_scaled_if_temporary(z, logits)
    return entropy  # H = logsumexp(z) - Σ softmax(z)·z


def gumbel_max(logits, temperature: float, noise):
    """Gumbel-max sample: ``argmax(logits / T + noise)`` over the vocab axis.

    ``logits`` / ``noise``: ``[..., vocab]`` (TILE_LAYOUT). ``noise`` is the torch
    run's exact injected Gumbel(0,1) noise (issue #47468 determinism). Returns
    argmax indices ``[..., 1]``. ``noise`` all-zeros reduces to plain
    ``argmax(logits)`` (temperature scaling preserves the argmax). ``noise=None``
    is an explicit RUN-first shortcut for argmax sampling without allocating the
    full-vocab Gumbel buffer.
    """
    if isinstance(noise, ChunkedGumbelNoise):
        return gumbel_max_with_chunked_noise(
            logits,
            temperature,
            seed=noise.seed,
            vocab_chunk_size=noise.vocab_chunk_size,
            dtype=noise.dtype,
        )
    z = temperature_scale(logits, temperature)
    if noise is None:
        sampled = ttnn.argmax(z, dim=-1, keepdim=True)
        _deallocate_scaled_if_temporary(z, logits)
        return sampled
    perturbed = ttnn.add(z, noise)
    sampled = ttnn.argmax(perturbed, dim=-1, keepdim=True)
    perturbed.deallocate(True)
    _deallocate_scaled_if_temporary(z, logits)
    return sampled


def _offset_argmax_indices(indices, offset: int):
    indices = ttnn.typecast(indices, ttnn.uint32)
    if offset == 0:
        return indices
    out = ttnn.add(indices, offset)
    indices.deallocate(True)
    return out


def _select_by_mask(mask, candidate, current):
    mask_t = ttnn.typecast(mask, candidate.get_dtype())
    ones = ttnn.full(
        list(candidate.shape),
        1,
        dtype=candidate.get_dtype(),
        layout=ttnn.TILE_LAYOUT,
        device=candidate.device(),
    )
    keep_t = ttnn.subtract(ones, mask_t)
    selected_candidate = ttnn.multiply(candidate, mask_t)
    selected_current = ttnn.multiply(current, keep_t)
    out = ttnn.add(selected_candidate, selected_current)
    mask_t.deallocate(True)
    ones.deallocate(True)
    keep_t.deallocate(True)
    selected_candidate.deallocate(True)
    selected_current.deallocate(True)
    return out


def gumbel_max_with_chunked_noise(
    logits, temperature: float, *, seed: int, vocab_chunk_size: int = 1024, dtype=ttnn.float32
):
    """Gumbel-max without materializing full-vocab noise or perturbed logits.

    Each vocab chunk computes local ``max`` and ``argmax`` for
    ``logits / T + Gumbel``; the per-chunk winners are reduced to the global
    winner with elementwise masks. This is the production-noise fit path for
    large canvases where a full ``[B, L, vocab]`` Gumbel tensor does not fit.
    """
    seed = _validate_ttnn_rand_seed(seed)
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    vocab_size = logits.shape[-1]
    best_values = None
    best_indices = None

    for offset in range(0, vocab_size, vocab_chunk_size):
        end = min(offset + vocab_chunk_size, vocab_size)
        starts = [0] * len(logits.shape)
        ends = list(logits.shape)
        starts[-1] = offset
        ends[-1] = end
        chunk = ttnn.slice(
            logits,
            starts,
            ends,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = temperature_scale(chunk, temperature)
        noise = sample_gumbel_noise(
            z.shape,
            device=logits.device(),
            seed=seed + offset,
            dtype=dtype,
        )
        perturbed = ttnn.add(z, noise)
        chunk_values = ttnn.max(perturbed, dim=-1, keepdim=True)
        chunk_indices = _offset_argmax_indices(ttnn.argmax(perturbed, dim=-1, keepdim=True), offset)

        perturbed.deallocate(True)
        noise.deallocate(True)
        _deallocate_scaled_if_temporary(z, chunk)
        chunk.deallocate(True)

        if best_values is None:
            best_values = chunk_values
            best_indices = chunk_indices
            continue

        take_chunk = ttnn.gt(chunk_values, best_values)
        next_values = _select_by_mask(take_chunk, chunk_values, best_values)
        take_chunk_u32 = ttnn.typecast(take_chunk, ttnn.uint32)
        next_indices = _select_by_mask(take_chunk_u32, chunk_indices, best_indices)

        take_chunk.deallocate(True)
        take_chunk_u32.deallocate(True)
        best_values.deallocate(True)
        best_indices.deallocate(True)
        chunk_values.deallocate(True)
        chunk_indices.deallocate(True)
        best_values = next_values
        best_indices = next_indices

    best_values.deallocate(True)
    return best_indices


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


def _validate_gumbel_noise_shape(shape, *, require_vocab_axis: bool = False) -> tuple[int, ...]:
    shape = tuple(shape)
    if not shape:
        raise ValueError("Gumbel noise shape must be non-empty")
    if any(dim <= 0 for dim in shape):
        raise ValueError("Gumbel noise shape dimensions must be positive")
    if require_vocab_axis and len(shape) < 2:
        raise ValueError("Gumbel noise shape must include at least a sample axis and a vocab axis")
    return shape


def sample_gumbel_noise(shape, *, device, seed: int, dtype=ttnn.float32):
    """Generate device Gumbel(0,1) noise with a deterministic TTNN rand seed."""
    seed = _validate_ttnn_rand_seed(seed)
    shape = _validate_gumbel_noise_shape(shape)
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
    shape = _validate_gumbel_noise_shape(shape, require_vocab_axis=True)

    rand_shape = (shape[-1], *shape[1:-1], shape[0])
    permute_order = (len(shape) - 1, *range(1, len(shape) - 1), 0)
    raw_u = ttnn.rand(
        rand_shape,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        low=0.0,
        high=1.0,
        seed=seed,
        mesh_mapper=_rand_mesh_mapper(device),
    )
    u = ttnn.permute(raw_u, permute_order)
    if u is not raw_u:
        raw_u.deallocate(True)
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

    shape = _validate_gumbel_noise_shape(shape, require_vocab_axis=True)
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
    gumbel = ttnn.concat(parts, dim=-1)
    for part in parts:
        part.deallocate(True)
    return gumbel
