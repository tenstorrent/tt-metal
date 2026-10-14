# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device (ttnn) entropy + Gumbel-max primitives for the PCC harness (#47468).

  * :func:`token_entropy` — per-position Shannon entropy ``H = −Σ p·log p`` of
    ``softmax(logits / T)``. Mirrors ``reference/sampling.token_entropy`` /
    ``torch.distributions.Categorical(logits).entropy()``.
  * :func:`gumbel_max` — ``argmax(logits / T + gumbel)`` over the vocab axis, with
    the Gumbel noise **injected** (not regenerated) so device argmax decisions can
    be matched token-for-token against the torch oracle (on-device RNG cannot
    reproduce torch's RNG bit-exactly).

Numerical note: entropy is computed as ``H = logsumexp(z) − Σ softmax(z)·z``,
algebraically equivalent to ``−Σ p·log p`` while avoiding ``log(p)`` underflow.
"""

from __future__ import annotations

import ttnn


def argmax_last_dim(x, *, keepdim: bool = True):
    """Multi-core argmax over the last (vocab) dim.

    ``ttnn.argmax`` runs **single-core** on TILE input but **multi-core** on
    ROW_MAJOR input for a last-dim reduction, and it always emits UINT32 ROW_MAJOR
    output — so converting the input to ROW_MAJOR first is far faster over the
    production vocab, bit-identical, and leaves the output contract unchanged.
    """
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.argmax(rm, dim=-1, keepdim=keepdim)
    if rm is not x:
        rm.deallocate(True)
    return out


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
    is an explicit shortcut for argmax sampling without allocating the full-vocab
    Gumbel buffer.
    """
    z = temperature_scale(logits, temperature)
    if noise is None:
        sampled = argmax_last_dim(z)
        _deallocate_scaled_if_temporary(z, logits)
        return sampled
    perturbed = ttnn.add(z, noise)
    sampled = argmax_last_dim(perturbed)
    perturbed.deallocate(True)
    _deallocate_scaled_if_temporary(z, logits)
    return sampled


def canvas_sample(logits, temperature: float, gumbel_noise):
    """Deterministic canvas sampler using injected Gumbel noise.

    This is the released per-position canvas draw used by the diffusion loop:
    ``argmax(logits / T + gumbel)`` over every canvas position. The noise is
    supplied by the caller for torch/device token-exact validation.
    """
    return gumbel_max(logits, temperature, gumbel_noise)


def _gumbel_from_uniform(u, *, deallocate_input: bool = True):
    # Consume the uniform draw in place: at the production [1, 1, 256, 256K] shape,
    # retaining full-shape intermediates costs ~1.5 GiB of device traffic and blocks
    # post-trace Gumbel refresh. The non-consuming path clones once.
    gumbel = u if deallocate_input else ttnn.clone(u)
    ttnn.add(gumbel, 1.0e-10, output_tensor=gumbel)
    ttnn.log(gumbel, output_tensor=gumbel)
    ttnn.multiply(gumbel, -1.0, output_tensor=gumbel)
    ttnn.add(gumbel, 1.0e-10, output_tensor=gumbel)
    ttnn.log(gumbel, output_tensor=gumbel)
    ttnn.multiply(gumbel, -1.0, output_tensor=gumbel)
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


# UNIMPLEMENTED design note: a TP-sharded denoise terminal (argmax / global-max / entropy on the
# per-device vocab shard, skipping the per-step full-vocab all-gather) is sketched on #47465 but
# none of it exists here; the entropy combine is not bf16-bit-identical, so it is decision-gated.
