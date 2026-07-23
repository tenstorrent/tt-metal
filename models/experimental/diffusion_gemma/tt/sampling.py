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

from dataclasses import dataclass
from typing import NamedTuple

import ttnn
from loguru import logger


class ChunkedGumbelNoise(NamedTuple):
    seed: int
    vocab_chunk_size: int = 1024
    dtype: object = ttnn.float32


@dataclass
class TraceChunkedGumbelState:
    """Persistent device inputs shared by every step in one traced request."""

    seed_tensor: object
    uniform_buffer: object | None = None
    uniform_shape: tuple[int, ...] | None = None
    value_ones: object | None = None
    index_ones: object | None = None

    def release(self) -> None:
        try:
            for name, tensor in (
                ("seed_tensor", self.seed_tensor),
                ("uniform_buffer", self.uniform_buffer),
                ("value_ones", self.value_ones),
                ("index_ones", self.index_ones),
            ):
                if tensor is not None and hasattr(tensor, "deallocate"):
                    try:
                        tensor.deallocate(True)
                    except BaseException as cleanup_error:
                        logger.error(f"failed to release trace chunked-Gumbel {name}: {cleanup_error}")
        finally:
            self.seed_tensor = None
            self.uniform_buffer = None
            self.uniform_shape = None
            self.value_ones = None
            self.index_ones = None


@dataclass(frozen=True)
class TraceChunkedGumbelNoise:
    """Chunked Gumbel descriptor whose base seed is a trace input tensor."""

    state: TraceChunkedGumbelState
    seed_offset: int
    vocab_chunk_size: int = 1024
    dtype: object = ttnn.float32


def argmax_last_dim(x, *, keepdim: bool = True):
    """Multi-core argmax over the last (vocab) dim.

    ``ttnn.argmax`` runs **single-core** on TILE input but **multi-core** on
    ROW_MAJOR input for a last-dim reduction, and it always emits UINT32 ROW_MAJOR
    output. Converting the input to ROW_MAJOR first is ~86x faster over the 262144
    production vocab (measured on QB2: 1240ms TILE -> 14.4ms ROW_MAJOR) and is
    bit-identical to the TILE result (verified exact match). The output layout/dtype
    contract (UINT32 ROW_MAJOR) is unchanged, so downstream consumers are unaffected.
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
    is an explicit RUN-first shortcut for argmax sampling without allocating the
    full-vocab Gumbel buffer.
    """
    if isinstance(noise, TraceChunkedGumbelNoise):
        return gumbel_max_with_chunked_noise(
            logits,
            temperature,
            vocab_chunk_size=noise.vocab_chunk_size,
            dtype=noise.dtype,
            trace_noise=noise,
        )
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
        sampled = argmax_last_dim(z)
        _deallocate_scaled_if_temporary(z, logits)
        return sampled
    perturbed = ttnn.add(z, noise)
    sampled = argmax_last_dim(perturbed)
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


def _select_by_mask(mask, candidate, current, *, ones=None):
    mask_t = ttnn.typecast(mask, candidate.get_dtype())
    owns_ones = ones is None
    if owns_ones:
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
    if owns_ones:
        ones.deallocate(True)
    keep_t.deallocate(True)
    selected_candidate.deallocate(True)
    selected_current.deallocate(True)
    return out


def gumbel_max_with_chunked_noise(
    logits,
    temperature: float,
    *,
    seed: int | None = None,
    vocab_chunk_size: int = 1024,
    dtype=ttnn.float32,
    trace_noise: TraceChunkedGumbelNoise | None = None,
):
    """Gumbel-max without materializing full-vocab noise or perturbed logits.

    Each vocab chunk computes local ``max`` and ``argmax`` for
    ``logits / T + Gumbel``; the per-chunk winners are reduced to the global
    winner with elementwise masks. This is the bounded-memory fit path for large
    canvases where a full ``[B, L, vocab]`` Gumbel tensor does not fit. The current
    QB2 1024-wide RNG has a known distribution bias, so this path is not the
    official-quality reference until that RNG issue is fixed.
    """
    if trace_noise is None:
        seed = _validate_ttnn_rand_seed(seed)
    elif trace_noise.state.seed_tensor is None:
        raise RuntimeError("trace chunked-Gumbel state was released")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    vocab_size = logits.shape[-1]
    chunk_width = min(vocab_chunk_size, vocab_size)
    if trace_noise is not None and vocab_size % chunk_width != 0:
        raise ValueError("trace chunked Gumbel requires vocab_size divisible by the effective chunk size")
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
        if trace_noise is None:
            noise = sample_gumbel_noise(
                z.shape,
                device=logits.device(),
                seed=seed + offset,
                dtype=dtype,
            )
        else:
            from models.experimental.diffusion_gemma.tt import trace_gumbel

            state = trace_noise.state
            shape = tuple(int(dim) for dim in z.shape)
            if state.uniform_buffer is None:
                state.uniform_buffer = trace_gumbel.allocate_uniform_buffer(logits.device(), shape, dtype=dtype)
                state.uniform_shape = shape
            elif state.uniform_shape != shape:
                raise ValueError(f"trace chunked-Gumbel shape changed from {state.uniform_shape} to {shape}")
            uniform = trace_gumbel.trace_seeded_uniform(
                state.seed_tensor,
                state.uniform_buffer,
                seed_offset=trace_noise.seed_offset + offset,
            )
            noise = _gumbel_from_uniform(uniform, deallocate_input=False)
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
        if trace_noise is not None:
            state = trace_noise.state
            if state.value_ones is None:
                state.value_ones = ttnn.full(
                    list(chunk_values.shape),
                    1,
                    dtype=chunk_values.get_dtype(),
                    layout=ttnn.TILE_LAYOUT,
                    device=chunk_values.device(),
                )
            if state.index_ones is None:
                state.index_ones = ttnn.full(
                    list(chunk_indices.shape),
                    1,
                    dtype=chunk_indices.get_dtype(),
                    layout=ttnn.TILE_LAYOUT,
                    device=chunk_indices.device(),
                )
        else:
            state = None
        next_values = _select_by_mask(
            take_chunk,
            chunk_values,
            best_values,
            ones=state.value_ones if state is not None else None,
        )
        take_chunk_u32 = ttnn.typecast(take_chunk, ttnn.uint32)
        next_indices = _select_by_mask(
            take_chunk_u32,
            chunk_indices,
            best_indices,
            ones=state.index_ones if state is not None else None,
        )

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


def _gumbel_from_uniform(u, *, deallocate_input: bool = True):
    u_eps = ttnn.add(u, 1.0e-10)
    log_u = ttnn.log(u_eps)
    neg_log_u = ttnn.multiply(log_u, -1.0)
    neg_log_u_eps = ttnn.add(neg_log_u, 1.0e-10)
    log_neg_log_u = ttnn.log(neg_log_u_eps)
    gumbel = ttnn.multiply(log_neg_log_u, -1.0)
    if deallocate_input:
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


# ---------------------------------------------------------------------------------------------
# TP-sharded denoise terminal (DG_TERMINAL_SHARDED) — argmax / global-max / entropy on the
# per-device vocab shard, skipping the per-step full-vocab all-gather (#47465, path to 100 t/s).
#
# The lm_head is column-parallel over vocab (``models/demos/gemma4/tt/model.py::_apply_lm_head``),
# so each of the ``TP`` devices already holds the SOFTCAPPED shard ``[1,1,S,vocab/TP]`` covering
# a contiguous, tile-aligned, unpadded vocab block ``[c*per_dev, (c+1)*per_dev)``. With
# ``return_sharded=True`` the head skips the ~128 MiB/step ``ccl_allgather`` and these helpers do
# the reduction on the shard + a tiny cross-shard combine:
#   * argmax/gumbel  — per-shard local max+argmax, add the per-device offset, all-gather the tiny
#     ``[S,TP]`` candidates, then a global max + lowest-index-among-winners fold. BIT-IDENTICAL to
#     the replicated ``argmax_last_dim`` (max is a selection, no bf16 accumulation; the tie rule —
#     lowest global index wins — is preserved via the offset ordering).
#   * global max     — exact (bf16 max of bf16 values does no rounding, order-independent).
#   * entropy        — distributed logsumexp with the exact shared max; fp32 per-shard partials +
#     fp32 all-reduce(SUM). NOT bf16-bit-identical (the 262144-length sum is re-associated as
#     ``TP`` partials), same #48291 class as ``DG_NORM_FULLCANVAS``; decision-gated.
#
# Trace-safe: no ``ttnn.full`` / ``zeros_like`` (host writes rejected in trace capture) is used in
# the combine — the tie fold is a masked-min in fp32; the offset constant is preallocated OUTSIDE
# capture (``build_vocab_shard_offsets``); all shapes are fixed (``S``, ``TP``, ``per_dev``). The
# ``all_gather`` / ``all_reduce`` are the same collectives gemma4 traced decode already captures.
# ---------------------------------------------------------------------------------------------

# Larger than any global vocab index (< 262144) so a non-winning shard's candidate index never
# wins the cross-shard min; exact-enough in fp32 (its exact value is irrelevant, only its rank).
_ARGMAX_TIE_PENALTY = 1.0e9


def build_vocab_shard_offsets(mesh_device, mesh_config, *, canvas_len, vocab_size=None, per_device_vocab=None):
    """Per-device global-vocab index offset for the sharded terminal (E3).

    Builds ``[1,1,canvas_len,TP]`` int32 with column ``c = c*per_device_vocab`` and shards it on
    ``tp_axis`` (``mesh_config.column_parallel``), so device ``c`` statically holds the scalar
    ``[1,1,canvas_len,1]`` offset for its own vocab block. Mirrors the decode offset template
    ``models/common/sampling/tt_sampling.py::_create_indices_tensors``. Allocate ONCE, OUTSIDE any
    trace (a persistent constant with a fixed device address across replays — the KV-cache
    pattern). Adding ``local_argmax_idx + offset`` reproduces the global vocab index the replicated
    ``argmax_last_dim`` reports, because the column-parallel lm_head lays vocab out in contiguous
    device-column order.
    """
    import torch

    tp = mesh_config.tp
    if per_device_vocab is None:
        if vocab_size is None:
            raise ValueError("build_vocab_shard_offsets requires vocab_size or per_device_vocab")
        if vocab_size % tp != 0:
            raise ValueError(f"vocab_size {vocab_size} not divisible by tp {tp}")
        per_device_vocab = vocab_size // tp
    offsets = torch.zeros(1, 1, canvas_len, tp, dtype=torch.int32)
    for c in range(tp):
        offsets[:, :, :, c] = c * per_device_vocab
    return ttnn.from_torch(
        offsets,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _sharded_local_candidates(logits_shard, offsets, *, mesh_config, ccl_manager):
    """Per-shard ``(max_value, global_argmax_index)`` gathered across TP.

    Returns ``(gathered_vals [1,1,S,TP] TILE, gathered_gidx_f [1,1,S,TP] fp32 TILE)`` replicated on
    every device. ``offsets`` is the persistent [1,1,S,1] per-device base from
    :func:`build_vocab_shard_offsets` and is NOT deallocated here.
    """
    from models.demos.gemma4.tt.ccl import ccl_allgather

    local_max = ttnn.max(logits_shard, dim=-1, keepdim=True)  # [1,1,S,1] per-device local max
    local_idx = argmax_last_dim(logits_shard)  # [1,1,S,1] uint32 ROW_MAJOR per-device local argmax
    local_idx_tile = ttnn.to_layout(local_idx, ttnn.TILE_LAYOUT)
    if local_idx_tile is not local_idx:
        local_idx.deallocate(True)
    # fp32 index arithmetic: local idx < per_dev and global idx < vocab (< 2^24) are exact in fp32.
    local_idx_f = ttnn.typecast(local_idx_tile, ttnn.float32)
    local_idx_tile.deallocate(True)
    offsets_f = ttnn.typecast(offsets, ttnn.float32)
    global_idx_f = ttnn.add(local_idx_f, offsets_f)
    local_idx_f.deallocate(True)
    offsets_f.deallocate(True)
    gathered_vals = ccl_allgather(local_max, mesh_config, ccl_manager, dim=3)  # deallocates local_max
    gathered_gidx_f = ccl_allgather(global_idx_f, mesh_config, ccl_manager, dim=3)  # deallocates global_idx_f
    return gathered_vals, gathered_gidx_f


def _combine_sharded_argmax(gathered_vals, gathered_gidx_f):
    """Global argmax index from gathered per-shard ``(value, global_index)`` (lowest index on ties).

    ``M = max`` over the TP candidate values is exact (selection, no bf16 accumulation). Among the
    shards whose value equals ``M`` (bf16-exact compare), the lowest global index wins — reproduced
    by penalising non-winners by :data:`_ARGMAX_TIE_PENALTY` and taking the min. This matches the
    replicated ``ttnn.argmax`` tie rule (lowest index) exactly. Uses no ``ttnn.full`` (trace-safe).
    """
    global_max = ttnn.max(gathered_vals, dim=-1, keepdim=True)  # [1,1,S,1] exact global max
    not_winner = ttnn.ne(gathered_vals, global_max)  # [1,1,S,TP] 1 where value != global max
    not_winner_f = ttnn.typecast(not_winner, ttnn.float32)
    penalty = ttnn.multiply(not_winner_f, _ARGMAX_TIE_PENALTY)
    masked = ttnn.add(gathered_gidx_f, penalty)  # winners keep gidx; losers pushed above any gidx
    min_gidx_f = ttnn.min(masked, dim=-1, keepdim=True)
    result = ttnn.typecast(min_gidx_f, ttnn.uint32)
    global_max.deallocate(True)
    not_winner.deallocate(True)
    not_winner_f.deallocate(True)
    penalty.deallocate(True)
    masked.deallocate(True)
    min_gidx_f.deallocate(True)
    return result


def argmax_last_dim_sharded(logits_shard, offsets, *, mesh_config, ccl_manager):
    """Global argmax over the TP-sharded vocab (E4). BIT-IDENTICAL to ``argmax_last_dim`` on the
    full replicated logits. Returns ``[1,1,S,1]`` uint32 (TILE)."""
    gathered_vals, gathered_gidx_f = _sharded_local_candidates(
        logits_shard, offsets, mesh_config=mesh_config, ccl_manager=ccl_manager
    )
    result = _combine_sharded_argmax(gathered_vals, gathered_gidx_f)
    gathered_vals.deallocate(True)
    gathered_gidx_f.deallocate(True)
    return result


def gumbel_max_sharded(logits_shard, temperature: float, noise_shard, offsets, *, mesh_config, ccl_manager):
    """Gumbel-max sample over the TP-sharded vocab (E4). ``argmax(logits/T + noise)`` where
    ``noise_shard`` is the vocab-sharded Gumbel slice aligned to ``logits_shard`` (device ``c`` <->
    vocab block ``c``); ``noise_shard=None`` reduces to :func:`argmax_last_dim_sharded` (temperature
    scaling preserves the argmax). BIT-IDENTICAL to the replicated ``gumbel_max`` because
    ``(z+noise)`` on device ``c`` is the identical bits of the replicated tensor's columns
    ``[c*per_dev, ...)`` (element-wise add, no cross-vocab interaction) and the combine is the exact
    same max + lowest-index tie rule."""
    z = temperature_scale(logits_shard, temperature)
    if noise_shard is None:
        result = argmax_last_dim_sharded(z, offsets, mesh_config=mesh_config, ccl_manager=ccl_manager)
        _deallocate_scaled_if_temporary(z, logits_shard)
        return result
    perturbed = ttnn.add(z, noise_shard)
    result = argmax_last_dim_sharded(perturbed, offsets, mesh_config=mesh_config, ccl_manager=ccl_manager)
    perturbed.deallocate(True)
    _deallocate_scaled_if_temporary(z, logits_shard)
    return result


def global_vocab_max(logits_shard, *, mesh_config, ccl_manager):
    """Exact global vocab max ``M = [1,1,S,1]`` from the TP-sharded logits (E5).

    Per-shard local max -> tiny ``all_gather`` -> ``ttnn.max`` over TP. bf16 max of bf16 values does
    no rounding and is order-independent, so ``M`` equals the replicated ``ttnn.max`` bit-for-bit;
    shared by :func:`token_entropy_sharded` and the sharded soft-embedding."""
    from models.demos.gemma4.tt.ccl import ccl_allgather

    local_max = ttnn.max(logits_shard, dim=-1, keepdim=True)
    gathered = ccl_allgather(local_max, mesh_config, ccl_manager, dim=3)  # deallocates local_max
    global_max = ttnn.max(gathered, dim=-1, keepdim=True)
    gathered.deallocate(True)
    return global_max


def token_entropy_sharded(logits_shard, temperature: float = 1.0, *, mesh_config, ccl_manager, global_max=None):
    """Per-position Shannon entropy over the TP-sharded vocab as a distributed logsumexp (E6).

    ``H = log(Σexp(z-M)) - Σ(exp(z-M)·(z-M))/Σexp(z-M)`` with the shared exact max ``M``. The two
    per-shard partial sums are accumulated and all-reduced in fp32; ``M`` is bit-identical to the
    replicated path so every ``exp`` argument matches element-for-element, and ONLY the 262144-length
    sums are re-associated (TP partials). NOT bf16-bit-identical (same #48291 class as
    ``DG_NORM_FULLCANVAS``); the fp32 partials + fp32 all-reduce minimise the drift. Decision-gated —
    validate accept/renoise decision-agreement before any default flip. Returns ``[1,1,S,1]`` in the
    logits dtype (drop-in for the replicated :func:`token_entropy`)."""
    from models.demos.gemma4.tt.ccl import ccl_allreduce

    out_dtype = logits_shard.get_dtype()
    z = temperature_scale(logits_shard, temperature)
    owns_max = global_max is None
    max_t = global_vocab_max(z, mesh_config=mesh_config, ccl_manager=ccl_manager) if owns_max else global_max
    shifted = ttnn.subtract(z, max_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    exp_shifted = ttnn.exp(shifted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # fp32 partial reductions to minimise the cross-shard sum re-association error.
    exp_f = ttnn.typecast(exp_shifted, ttnn.float32)
    sum_exp_local = ttnn.sum(exp_f, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    exp_f.deallocate(True)
    weighted = ttnn.multiply(exp_shifted, shifted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    weighted_f = ttnn.typecast(weighted, ttnn.float32)
    weighted.deallocate(True)
    sum_weighted_local = ttnn.sum(weighted_f, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    weighted_f.deallocate(True)
    exp_shifted.deallocate(True)
    shifted.deallocate(True)
    # all-reduce(SUM) the two tiny [1,1,S,1] fp32 partials across TP (deallocates its inputs).
    sum_exp = ccl_allreduce(sum_exp_local, mesh_config, ccl_manager)
    sum_weighted = ccl_allreduce(sum_weighted_local, mesh_config, ccl_manager)
    log_sum_exp = ttnn.log(sum_exp)
    expected_shifted = ttnn.div(sum_weighted, sum_exp)
    entropy = ttnn.subtract(log_sum_exp, expected_shifted)
    sum_exp.deallocate(True)
    sum_weighted.deallocate(True)
    log_sum_exp.deallocate(True)
    expected_shifted.deallocate(True)
    if owns_max:
        max_t.deallocate(True)
    _deallocate_scaled_if_temporary(z, logits_shard)
    if entropy.get_dtype() != out_dtype:
        entropy_cast = ttnn.typecast(entropy, out_dtype)
        entropy.deallocate(True)
        return entropy_cast
    return entropy
