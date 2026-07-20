# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device denoise-loop helpers for DiffusionGemma.

This module starts with the single-step decision kernel composition used by the
full W3 loop: sample, entropy, entropy-budget accept, and renoise. The model
forward remains injected by callers so the same helper can be used by synthetic
tests and the real W2 denoise logits path.
"""

from __future__ import annotations

import os
from typing import Callable, List, NamedTuple, Optional

import torch
import ttnn

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory, StepRecord
from models.experimental.diffusion_gemma.tt import sampling as TS

TtLogitsFn = Callable[[ttnn.Tensor, int], ttnn.Tensor]
TtNoiseFn = Callable[[int], ttnn.Tensor]


class ShardedTerminalContext(NamedTuple):
    """Preallocated TP-sharded-terminal constants threaded into :func:`denoise_step`.

    Carries the per-device vocab index ``offsets`` plus the ``mesh_config`` / ``ccl_manager``
    handles the sharded reductions (``denoise_terminal_reductions_sharded``) need. Built ONCE,
    OUTSIDE trace capture, by :meth:`DenoiseLogitsAdapter.prepare_sharded_terminal` and surfaced
    via :meth:`DenoiseLogitsAdapter.sharded_terminal_context`. ``None`` selects the replicated
    full-vocab terminal (``DG_TERMINAL_SHARDED`` off) and keeps :func:`denoise_step` and the
    trace controllers byte-identical.
    """

    offsets: ttnn.Tensor
    mesh_config: object
    ccl_manager: object


def _sharded_terminal_context(logits_fn) -> "Optional[ShardedTerminalContext]":
    """Return the ``logits_fn`` adapter's :class:`ShardedTerminalContext`, or ``None``.

    Mirrors the ``owns_logits`` / ``reset`` duck-typing already used for ``logits_fn``: a plain
    (non-adapter) callback has no ``sharded_terminal_context`` method, so the replicated terminal
    is selected. The adapter itself only returns a context when it is emitting a vocab SHARD
    (trace-safe self-cond + ``DG_TERMINAL_SHARDED`` on), keeping the reductions' input in lockstep
    with ``denoise_step``'s routing.
    """
    getter = getattr(logits_fn, "sharded_terminal_context", None)
    if callable(getter):
        return getter()
    return None


def dedup_argmax_enabled() -> bool:
    """Whether the opt-in argmax-sampling terminal dedup is on (``DG_DEDUP_ARGMAX``).

    Off by default so the released terminal path is byte-for-byte unchanged. When
    on, :func:`denoise_step` collapses the two redundant full-vocab argmax
    reductions into one in the argmax-sampling (``gumbel_noise is None``) regime —
    the opt-in greedy RUN-first mode (``"argmax"``; the serving default is now the
    stochastic ``"chunked"`` Gumbel-max = the model's reference sampler, so this
    dedup only fires when argmax is explicitly selected). See
    :func:`_sample_and_argmax` for the bit-exactness argument and
    ``doc/optimize_perf/verify_terminal_dedup.py`` for the device gate.
    """
    return os.environ.get("DG_DEDUP_ARGMAX", "0").lower() in ("1", "true", "yes", "on")


def _sample_and_argmax(logits, temperature: float, gumbel_noise, *, dedup_argmax: bool):
    """Return ``(sampled, argmax)`` uint32 token-index tensors for one denoise step.

    Default path (unchanged): an independent Gumbel-max draw and a clean argmax,
    each a full-vocab (262144) reduction — two ~14 ms ROW_MAJOR argmaxes/step. In
    the argmax regime the two reductions are over *different* tensors: ``sampled =
    argmax_last_dim(logits / T)`` (``gumbel_max`` scales first) and ``argmax =
    argmax_last_dim(logits)`` (raw).

    Dedup fast path — taken only when ``dedup_argmax`` **and** ``gumbel_noise is
    None`` **and** ``temperature > 0``. It computes the raw-logit argmax ONCE and
    clones the tiny ``[B, 1, L, 1]`` index for ``sampled``, dropping the second
    262144-wide reduction **and** the ``logits / T`` multiply.

    Correctness:

    * ``argmax`` (the committed clean-argmax token), ``entropy``, and the
      entropy-budget ``accept`` mask are **bit-identical** to the default path by
      construction — the dedup does not touch ``argmax_last_dim(logits)`` or
      ``token_entropy``, and ``accept`` is a pure function of ``entropy``.
    * In exact arithmetic ``argmax(logits / T) == argmax(logits)`` for any ``T > 0``
      (positive scaling is order-preserving), so ``sampled`` is the same token.
      On device the logits are **bf16**, so the default path's ``logits / T``
      multiply can, at a position whose top-2 logits are adjacent bf16 values,
      round them equal and flip that position's ``sampled`` index to the tie's
      first index. The dedup takes ``sampled = argmax(logits)`` (the un-rescaled
      ranking), so it differs from the default ``sampled`` **only at exactly those
      temperature-rescale rounding ties** — positions where the default path's own
      two argmaxes already disagree (``sampled != argmax``). There the dedup makes
      ``sampled`` equal the committed ``argmax`` (more self-consistent, and within
      the model's existing bf16 error). At ``T == 1.0`` the multiply is a no-op, so
      the dedup is fully bit-identical.

    The two returned tensors are distinct objects (independent ownership: callers
    deallocate ``sampled`` while keeping ``argmax`` as the commit candidate), so
    the free/readback contract is preserved with no aliasing.
    """
    if dedup_argmax and gumbel_noise is None and temperature > 0:
        argmax = TS.argmax_last_dim(logits)
        argmax = ttnn.typecast(argmax, ttnn.uint32)
        # Clone the [B,1,L,1] index (not a second 262144-wide argmax). Preserve the
        # ROW_MAJOR uint32 layout `gumbel_max`→`argmax_last_dim` would have emitted
        # so `renoise`'s downstream multiply/add sees the same tensor shape/layout.
        sampled = ttnn.clone(argmax, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return sampled, argmax
    sampled = TS.gumbel_max(logits, temperature, gumbel_noise)
    sampled = ttnn.typecast(sampled, ttnn.uint32)
    argmax = TS.argmax_last_dim(logits)
    argmax = ttnn.typecast(argmax, ttnn.uint32)
    return sampled, argmax


class TtDenoiseStepResult(NamedTuple):
    canvas: ttnn.Tensor
    accept_mask: ttnn.Tensor
    entropy: ttnn.Tensor
    sampled: ttnn.Tensor
    argmax: ttnn.Tensor


def temperature_at_step(step: int, num_steps: int, t_start: float, t_end: float) -> float:
    """HF reversed-step linear temperature schedule."""
    if num_steps <= 0:
        return t_start
    cur_step = num_steps - step
    return t_end + (t_start - t_end) * (cur_step / num_steps)


class DenoiseConstants(NamedTuple):
    """Preallocated per-step constants for a trace-safe denoise loop.

    ``ttnn.full`` / ``ttnn.zeros_like`` are host->device WRITES and are rejected
    inside ``begin_trace_capture`` (``TT_FATAL: Writes are not supported during
    trace capture``). The trace-safe loop allocates these once (outside the trace)
    and reuses them every step. ``budget_t`` / ``accept_zeros`` are the entropy
    accept-chain constants ``[B, L]``; ``renoise_ones`` is ``[B, 1, L, 1]``.
    """

    budget_t: ttnn.Tensor
    accept_zeros: ttnn.Tensor
    renoise_ones: ttnn.Tensor


def make_denoise_constants(device, *, batch: int, canvas_len: int, budget: float, entropy_dtype=ttnn.bfloat16):
    """Allocate the reusable accept/renoise constants once, outside any trace."""
    budget_t = ttnn.full(
        [batch, canvas_len], float(budget), dtype=entropy_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    accept_zeros = ttnn.full([batch, canvas_len], 0.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    renoise_ones = ttnn.full([batch, 1, canvas_len, 1], 1, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    return DenoiseConstants(budget_t=budget_t, accept_zeros=accept_zeros, renoise_ones=renoise_ones)


def entropy_budget_accept(entropy, budget: float, *, budget_t=None, zeros=None):
    """Device entropy-budget accept mask using the HF exclusive-prefix rule.

    This path is load-bearing for the Part I decision-fidelity bar: device sort
    regressions must show up as accept-mask/count mismatches against the host
    oracle, not as silent trajectory drift.

    ``budget_t`` / ``zeros`` may be preallocated (persistent) tensors so the op
    is trace-safe; when ``None`` they are allocated per call for the eager path.
    """
    sorted_vals, sorted_idx = ttnn.sort(entropy, dim=-1)
    cum = ttnn.cumsum(sorted_vals, dim=-1)
    excl = ttnn.subtract(cum, sorted_vals)
    own_budget = budget_t is None
    if own_budget:
        budget_t = ttnn.full(
            list(entropy.shape),
            float(budget),
            dtype=entropy.get_dtype(),
            layout=ttnn.TILE_LAYOUT,
            device=entropy.device(),
        )
    accept_sorted = ttnn.le(excl, budget_t)
    accept_sorted_bf = ttnn.typecast(accept_sorted, ttnn.bfloat16)
    own_zeros = zeros is None
    if own_zeros:
        zeros = ttnn.typecast(ttnn.zeros_like(entropy), ttnn.bfloat16)
    accept = ttnn.scatter(zeros, -1, sorted_idx, accept_sorted_bf)
    sorted_vals.deallocate(True)
    sorted_idx.deallocate(True)
    cum.deallocate(True)
    excl.deallocate(True)
    if own_budget:
        budget_t.deallocate(True)
    accept_sorted.deallocate(True)
    accept_sorted_bf.deallocate(True)
    if own_zeros:
        zeros.deallocate(True)
    return accept


def renoise(accept_mask, sampled, noise_tokens, *, ones=None):
    """Select sampled tokens where accepted, otherwise random renoise tokens.

    ``ones`` may be a preallocated (persistent) tensor for trace-safety; when
    ``None`` it is allocated per call for the eager path.
    """
    accept_u32 = ttnn.typecast(accept_mask, ttnn.uint32)
    own_ones = ones is None
    if own_ones:
        ones = ttnn.full(
            list(sampled.shape),
            1,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=sampled.device(),
        )
    reject_u32 = ttnn.subtract(ones, accept_u32)
    accepted = ttnn.multiply(sampled, accept_u32)
    rejected = ttnn.multiply(noise_tokens, reject_u32)
    canvas = ttnn.add(accepted, rejected)
    accept_u32.deallocate(True)
    if own_ones:
        ones.deallocate(True)
    reject_u32.deallocate(True)
    accepted.deallocate(True)
    rejected.deallocate(True)
    return canvas


def denoise_step(
    logits,
    *,
    temperature: float,
    entropy_budget: float,
    gumbel_noise,
    noise_tokens,
    constants: "Optional[DenoiseConstants]" = None,
    dedup_argmax: "Optional[bool]" = None,
    sharded_terminal: "Optional[ShardedTerminalContext]" = None,
):
    """Run one device denoise decision step with injected noise.

    Returns token id tensors in `uint32` TILE layout except `accept_mask` and
    `entropy`, which are bf16/fp32 decision tensors.

    The renoise select is written as uint32 arithmetic because `ttnn.where`
    currently corrupts `uint32` TILE token tensors in this path.

    ``constants`` supplies preallocated accept/renoise constants for trace-safe
    capture (no per-step ``ttnn.full``/``zeros_like`` host writes); ``None`` keeps
    the eager per-call allocation.

    ``dedup_argmax`` opts into the argmax-sampling terminal dedup (one full-vocab
    argmax instead of two); ``None`` consults ``DG_DEDUP_ARGMAX``. It is a no-op
    unless ``gumbel_noise is None`` (argmax sampling); there the committed
    ``argmax``, ``entropy``, and ``accept`` mask stay bit-identical and only
    ``sampled``/``canvas`` can move, at bf16 temperature-rescale ties — see
    :func:`_sample_and_argmax`.

    ``sharded_terminal`` (``DG_TERMINAL_SHARDED``) routes the argmax/gumbel/entropy vocab
    reductions through the TP-sharded ops (``denoise_terminal_reductions_sharded``) so ``logits``
    is the per-device vocab SHARD ``[1,1,C,vocab/TP]`` and nothing all-gathers the full 262144
    vocab. ``None`` (default) keeps the replicated full-vocab reductions byte-identical. The
    accept-mask / renoise / commit downstream are vocab-agnostic and identical either way.
    """
    if dedup_argmax is None:
        dedup_argmax = dedup_argmax_enabled()
    budget_t = constants.budget_t if constants is not None else None
    accept_zeros = constants.accept_zeros if constants is not None else None
    renoise_ones = constants.renoise_ones if constants is not None else None
    if sharded_terminal is not None:
        from models.experimental.diffusion_gemma.tt.denoise_forward import denoise_terminal_reductions_sharded

        sampled, argmax, entropy = denoise_terminal_reductions_sharded(
            logits,
            temperature=temperature,
            offsets=sharded_terminal.offsets,
            mesh_config=sharded_terminal.mesh_config,
            ccl_manager=sharded_terminal.ccl_manager,
            gumbel_noise_shard=gumbel_noise,
            dedup_argmax=dedup_argmax,
        )
    else:
        sampled, argmax = _sample_and_argmax(logits, temperature, gumbel_noise, dedup_argmax=dedup_argmax)
        entropy = TS.token_entropy(logits, temperature=temperature)
    entropy_for_accept = ttnn.reshape(entropy, (entropy.shape[0] * entropy.shape[1], entropy.shape[2]))
    accept_flat = entropy_budget_accept(entropy_for_accept, entropy_budget, budget_t=budget_t, zeros=accept_zeros)
    accept_mask = ttnn.reshape(accept_flat, (entropy.shape[0], entropy.shape[1], 1, entropy.shape[2]))
    accept_for_where = ttnn.reshape(accept_mask, sampled.shape)
    canvas = renoise(accept_for_where, sampled, noise_tokens, ones=renoise_ones)
    entropy_for_accept.deallocate(True)
    accept_flat.deallocate(True)
    accept_for_where.deallocate(True)
    return TtDenoiseStepResult(
        canvas=canvas,
        accept_mask=accept_mask,
        entropy=entropy,
        sampled=sampled,
        argmax=argmax,
    )


def denoise_step_next_canvas(
    logits,
    *,
    temperature: float,
    entropy_budget: float,
    gumbel_noise,
    noise_tokens,
    constants: "Optional[DenoiseConstants]" = None,
    dedup_argmax: "Optional[bool]" = None,
    sharded_terminal: "Optional[ShardedTerminalContext]" = None,
):
    """One device denoise step returning only the device-resident feedback tensors.

    Trace-safe variant of :func:`denoise_step`: it performs no host readback and no
    data-dependent control flow, so it can be captured inside a Metal trace. Returns
    ``(next_canvas, argmax)`` where ``next_canvas`` is the accepted/renoised canvas
    consumed by the next step and ``argmax`` is the clean-argmax commit candidate.
    The intermediate decision tensors (accept mask / entropy / sampled) are freed.

    ``sharded_terminal`` forwards to :func:`denoise_step` (``DG_TERMINAL_SHARDED``); ``None``
    keeps the replicated full-vocab terminal.
    """
    res = denoise_step(
        logits,
        temperature=temperature,
        entropy_budget=entropy_budget,
        gumbel_noise=gumbel_noise,
        noise_tokens=noise_tokens,
        constants=constants,
        dedup_argmax=dedup_argmax,
        sharded_terminal=sharded_terminal,
    )
    res.accept_mask.deallocate(True)
    res.entropy.deallocate(True)
    res.sampled.deallocate(True)
    return res.canvas, res.argmax


# --- Data-dependent early-halt: on-device halt-scalar reduction (dg-08 lever 8) ------
#
# Early-halt must NOT trace the whole variable-length loop (a static Metal trace fixes
# the step count). Instead the trace-safe single denoise step (or a fixed K-step window)
# computes ONE tiny halt scalar per step ON DEVICE, and the HOST reads that scalar after
# each replay and branches continue/stop. This is the anti-pattern-avoiding replacement
# for the retired 5-tensor/step host readback (``bench_loop_readback.py`` = 27.76 ms/step).
#
# The halt condition mirrors the eager reference (:func:`denoise_block` /
# ``reference/denoise_loop.denoise_block``, ``StableAndConfidentStoppingCriteria``):
#   * stable    — the clean-argmax canvas is unchanged vs the previous step
#                 (``stable_steps_to_halt == 1``, the released config value), and
#   * confident — the mean per-position entropy of the temperature-scaled logits is
#                 below ``entropy_stop_threshold``.
# Both reduce to a per-step scalar: ``mismatch`` = #positions whose argmax changed
# (0 ⟺ stable, integer-exact ⟺ ``torch.equal``), and ``mean_entropy`` = mean over the
# canvas of the per-position entropy (matches ``entropy.mean()``). The host applies the
# exact eager rule (``eval_halt``), so no fp threshold decision is baked into the device.


class HaltBuffers(NamedTuple):
    """Persistent device buffers for the trace-safe early-halt scalar (allocated once).

    ``prev_argmax`` holds the previous step's clean argmax as TILE fp32 (token ids
    ≤262144 < 2^24 are exact in fp32) for the stability compare. ``mean_entropy`` and
    ``mismatch`` are the two tiny ``[1,1,1,1]`` fp32 scalars the host reads each
    step/window. All three are trace-write targets, so — like ``canvas_buf`` /
    ``committed_buf`` — they MUST be allocated before ``begin_trace_capture`` and their
    in-trace ``ttnn.copy`` writes warmed once eagerly.
    """

    prev_argmax: ttnn.Tensor
    mean_entropy: ttnn.Tensor
    mismatch: ttnn.Tensor


def _argmax_to_tile_f32(argmax):
    """``argmax`` (UINT32 ROW_MAJOR from :func:`~...tt.sampling.argmax_last_dim`) → TILE fp32.

    The clean argmax is emitted ROW_MAJOR uint32; the halt reductions (``ne`` / ``sum``)
    want TILE fp32. Token ids ≤262144 (< 2^24) are represented exactly in fp32.
    """
    tiled = ttnn.to_layout(argmax, ttnn.TILE_LAYOUT)
    af = ttnn.typecast(tiled, ttnn.float32)
    if tiled is not argmax:
        tiled.deallocate(True)
    return af


def compute_halt_scalars(argmax, entropy, prev_argmax_f32, *, canvas_len: int):
    """Reduce one step's decision tensors to ``(mean_entropy, mismatch, cur_argmax_f32)``.

    * ``mean_entropy`` ``[1,1,1,1]`` — ``sum(entropy)/canvas_len`` over the canvas, matching
      the eager ``entropy.mean()`` (entropy upcast to fp32 first, as the host path does).
    * ``mismatch`` ``[1,1,1,1]`` — count of canvas positions whose argmax differs from the
      previous step (``ne`` then ``sum``); ``0`` ⟺ ``torch.equal(argmax, prev)`` (the eager
      stability gate at ``stable_steps_to_halt == 1``).
    * ``cur_argmax_f32`` ``[1,1,C,1]`` — this step's argmax as TILE fp32; the caller copies it
      into ``prev_argmax`` for the next step's compare.

    Trace-safe: only device ops (typecast / to_layout / sum / ne / scalar-multiply), no host
    writes and no data-dependent control flow. The caller owns copying the three results into
    the persistent :class:`HaltBuffers`.
    """
    ent_f32 = ttnn.typecast(entropy, ttnn.float32)
    ent_sum = ttnn.sum(ent_f32, dim=2, keepdim=True)
    mean_entropy = ttnn.multiply(ent_sum, 1.0 / float(canvas_len))
    ent_f32.deallocate(True)
    ent_sum.deallocate(True)
    cur_argmax_f32 = _argmax_to_tile_f32(argmax)
    diff = ttnn.ne(cur_argmax_f32, prev_argmax_f32)
    mismatch = ttnn.sum(diff, dim=2, keepdim=True)
    diff.deallocate(True)
    return mean_entropy, mismatch, cur_argmax_f32


def write_halt_scalars(argmax, entropy, halt_bufs: "HaltBuffers", *, canvas_len: int) -> None:
    """Compute the halt scalars for one step and copy them into the persistent buffers.

    Overwrites ``halt_bufs.mean_entropy`` / ``.mismatch`` with THIS step's scalars and
    updates ``halt_bufs.prev_argmax`` to THIS step's argmax (read by the next step). The
    ``ne`` reads ``prev_argmax`` (the previous step) before the update copy overwrites it,
    and the intermediate ``cur_argmax_f32`` is a distinct tensor, so there is no aliasing
    hazard. Trace-safe (all targets preallocated; device copies only).
    """
    mean_entropy, mismatch, cur_argmax_f32 = compute_halt_scalars(
        argmax, entropy, halt_bufs.prev_argmax, canvas_len=canvas_len
    )
    ttnn.copy(mean_entropy, halt_bufs.mean_entropy)
    ttnn.copy(mismatch, halt_bufs.mismatch)
    ttnn.copy(cur_argmax_f32, halt_bufs.prev_argmax)
    mean_entropy.deallocate(True)
    mismatch.deallocate(True)
    cur_argmax_f32.deallocate(True)


def denoise_step_next_canvas_and_halt(
    logits,
    *,
    temperature: float,
    entropy_budget: float,
    gumbel_noise,
    noise_tokens,
    halt_bufs: "HaltBuffers",
    canvas_len: int,
    constants: "Optional[DenoiseConstants]" = None,
    dedup_argmax: "Optional[bool]" = None,
    sharded_terminal: "Optional[ShardedTerminalContext]" = None,
):
    """:func:`denoise_step_next_canvas` + write the on-device halt scalars for this step.

    Returns ``(next_canvas, argmax)`` exactly like :func:`denoise_step_next_canvas` — the
    canvas thread and committed argmax are byte-identical (the halt scalars are a read-only
    side computation over ``argmax`` / ``entropy`` and never touch the canvas). The entropy
    (freed by :func:`denoise_step_next_canvas`) is consumed here for the mean-entropy scalar.

    ``sharded_terminal`` forwards to :func:`denoise_step` (``DG_TERMINAL_SHARDED``); ``None``
    keeps the replicated full-vocab terminal.
    """
    res = denoise_step(
        logits,
        temperature=temperature,
        entropy_budget=entropy_budget,
        gumbel_noise=gumbel_noise,
        noise_tokens=noise_tokens,
        constants=constants,
        dedup_argmax=dedup_argmax,
        sharded_terminal=sharded_terminal,
    )
    write_halt_scalars(res.argmax, res.entropy, halt_bufs, canvas_len=canvas_len)
    res.accept_mask.deallocate(True)
    res.entropy.deallocate(True)
    res.sampled.deallocate(True)
    return res.canvas, res.argmax


def read_halt_scalars(halt_bufs: "HaltBuffers") -> "tuple[float, float]":
    """Read the two tiny halt scalars to host: ``(mean_entropy, mismatch_count)``.

    This is the ONLY host interaction per step/window — an 8-byte-logical readback, not the
    retired 5×256-wide (argmax/entropy/sampled/accept/canvas) per-step readback.
    """
    mean_entropy = float(_to_host_torch(halt_bufs.mean_entropy).reshape(-1)[0].item())
    mismatch = float(_to_host_torch(halt_bufs.mismatch).reshape(-1)[0].item())
    return mean_entropy, mismatch


def eval_halt(mean_entropy: float, mismatch: float, step: int, *, threshold: float, n_stable: int = 1) -> bool:
    """The eager ``StableAndConfidentStoppingCriteria`` rule on the per-step scalars.

    Halts at ``step`` (0-indexed) when the argmax has been stable vs the previous step
    (``mismatch == 0``, ``n_stable == 1``) AND the mean entropy is below ``threshold``. A
    prior step is required (``step >= n_stable``), matching the eager ``len(history) >=
    n_stable`` guard. Only ``n_stable == 1`` (the released ``stable_steps_to_halt``) is
    supported here; the caller must guard other values.
    """
    if n_stable != 1:
        raise NotImplementedError(
            f"traced early-halt supports stable_steps_to_halt == 1 (released config); got {n_stable}"
        )
    if step < n_stable:
        return False
    return bool(mismatch == 0.0 and mean_entropy < threshold)


def run_fixed_denoise_steps(
    logits_fn: TtLogitsFn,
    init_canvas: ttnn.Tensor,
    config: DiffusionConfig,
    *,
    gumbel_noise_fn: Optional[TtNoiseFn] = None,
    noise_tokens_fn: Optional[TtNoiseFn] = None,
    constants: "Optional[DenoiseConstants]" = None,
    dedup_argmax: "Optional[bool]" = None,
):
    """Fixed-count, device-only denoise loop for trace-safe capture.

    Chosen static/fixed-count scheme for the optimized loop: always run exactly
    ``config.max_denoise_steps`` (≤48) steps and feed the accepted canvas from step
    N straight into step N+1 **on device** — no host readback of the argmax/entropy/
    cutoff, no ``torch.equal`` halt, no Python early-return. Early-halt is a
    data-dependent decision that cannot shorten a static trace, so the trace-safe
    shape runs the full step budget; the entropy-budget accept mask and the sorted
    scatter indices stay device-resident tensors (see :func:`entropy_budget_accept`).
    ``commit = clean argmax`` of the final step; because the argmax is stable after
    convergence, running the remaining fixed steps does not change the commit.

    Returns the device-resident committed argmax ``[B,1,L,1]``. ``init_canvas`` is
    consumed. Intended to be called inside ``begin_trace_capture``/``end_trace_capture``
    with a warmed program cache and device-resident per-step noise tensors.
    """
    canvas = init_canvas
    committed: Optional[ttnn.Tensor] = None
    # DG_TERMINAL_SHARDED: derived once from the adapter (None => replicated terminal). Constant
    # across steps (persistent offsets/mesh/ccl); only present when logits_fn emits a vocab shard.
    sharded_terminal = _sharded_terminal_context(logits_fn)
    for step in range(config.max_denoise_steps):
        temperature = temperature_at_step(
            step, config.max_denoise_steps, config.temperature_start, config.temperature_end
        )
        gumbel_noise = gumbel_noise_fn(step) if gumbel_noise_fn else None
        noise_tokens = noise_tokens_fn(step) if noise_tokens_fn else None
        logits = logits_fn(canvas, step)
        next_canvas, argmax = denoise_step_next_canvas(
            logits,
            temperature=temperature,
            entropy_budget=config.entropy_budget,
            gumbel_noise=gumbel_noise,
            noise_tokens=noise_tokens,
            constants=constants,
            dedup_argmax=dedup_argmax,
            sharded_terminal=sharded_terminal,
        )
        if gumbel_noise is not None and hasattr(gumbel_noise, "deallocate"):
            gumbel_noise.deallocate(True)
        if noise_tokens is not None and hasattr(noise_tokens, "deallocate"):
            noise_tokens.deallocate(True)
        _deallocate_logits_if_unowned(logits_fn, logits)
        if committed is not None:
            committed.deallocate(True)
        committed = argmax
        if canvas is not next_canvas:
            canvas.deallocate(True)
        canvas = next_canvas
    if config.max_denoise_steps > 0 and canvas is not committed:
        canvas.deallocate(True)
    _reset_logits_fn(logits_fn)
    return committed


def device_loop_denoise_block(
    logits_fn: TtLogitsFn,
    init_canvas: ttnn.Tensor,
    config: DiffusionConfig,
    *,
    gumbel_noise_fn: Optional[TtNoiseFn] = None,
    noise_tokens_fn: Optional[TtNoiseFn] = None,
    constants: "Optional[DenoiseConstants]" = None,
) -> DenoiseTrajectory:
    """``denoise_block``-compatible wrapper over the device-only fixed-step loop.

    Same signature / return type as :func:`denoise_block`, but runs
    :func:`run_fixed_denoise_steps` — the device-resident loop with **no per-step
    host readback** and **no data-dependent early-halt** — then reads back only the
    final committed argmax once. Removes the 5 host readbacks/step (argmax, entropy,
    sampled, accept, canvas) + the ``torch.equal`` halt check that the eager
    ``denoise_block`` pays every step (~179 ms/step of the 30L serving step).

    Behaviourally identical to ``denoise_block`` **whenever early-halt does not fire**
    (both commit the final step's clean argmax after the full budget). Early-halt is
    currently a no-op on RUN-first output (mean entropy never clears the 0.005
    threshold under #48291), so the committed tokens are bit-identical — verified by
    comparing committed tokens with the eager path. Returns empty per-step records
    (``num_steps = max`` steps, ``halted = False``); callers that need the trajectory
    records must use the eager :func:`denoise_block`.
    """
    committed_dev = run_fixed_denoise_steps(
        logits_fn,
        init_canvas,
        config,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
        constants=constants,
    )
    committed_host = _ids_to_torch(committed_dev)
    committed_dev.deallocate(True)
    return DenoiseTrajectory(committed_host, config.max_denoise_steps, False, [])


def _to_host_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    device = tensor.device()
    if device is not None and hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    try:
        return ttnn.to_torch(tensor)
    except RuntimeError as exc:
        if "distributed on MeshShape" not in str(exc):
            raise
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _ids_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return _to_host_torch(tensor).squeeze(1).squeeze(-1).to(torch.long)


def _entropy_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return _to_host_torch(tensor).squeeze(1).squeeze(-1).float()


def _accept_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    return _to_host_torch(tensor).squeeze(1).squeeze(1) > 0.5


def _deallocate_decision_tensors(res: TtDenoiseStepResult) -> None:
    res.accept_mask.deallocate(True)
    res.entropy.deallocate(True)
    res.sampled.deallocate(True)
    res.argmax.deallocate(True)


def _reset_logits_fn(logits_fn: TtLogitsFn) -> None:
    reset = getattr(logits_fn, "reset", None)
    if callable(reset):
        reset()


def _deallocate_logits_if_unowned(logits_fn: TtLogitsFn, logits) -> None:
    owns_logits = getattr(logits_fn, "owns_logits", None)
    if callable(owns_logits) and owns_logits(logits):
        return
    logits.deallocate(True)


def denoise_block(
    logits_fn: TtLogitsFn,
    init_canvas: ttnn.Tensor,
    config: DiffusionConfig,
    *,
    gumbel_noise_fn: Optional[TtNoiseFn] = None,
    noise_tokens_fn: Optional[TtNoiseFn] = None,
    dedup_argmax: "Optional[bool]" = None,
) -> DenoiseTrajectory:
    """Run a device denoise trajectory and return host decision records.

    The full ``[B, L, vocab]`` logits stay on device. For the data-dependent halt
    check and the trajectory harness, this reads back only per-step ``[B, L]``
    decision tensors: argmax, entropy, sampled ids, accept mask, and canvas.
    ``init_canvas`` is consumed; each superseded device canvas is deallocated.
    """
    if gumbel_noise_fn is None or noise_tokens_fn is None:
        raise ValueError("denoise_block requires injected gumbel_noise_fn and noise_tokens_fn")

    canvas = init_canvas
    records: List[StepRecord] = []
    committed: Optional[torch.Tensor] = None
    argmax_history: List[torch.Tensor] = []
    n_stable = config.stable_steps_to_halt
    # DG_TERMINAL_SHARDED: derived once from the adapter (None => replicated terminal).
    sharded_terminal = _sharded_terminal_context(logits_fn)

    for step in range(config.max_denoise_steps):
        temperature = temperature_at_step(
            step, config.max_denoise_steps, config.temperature_start, config.temperature_end
        )
        gumbel_noise = gumbel_noise_fn(step) if gumbel_noise_fn else None
        noise_tokens = noise_tokens_fn(step) if noise_tokens_fn else None
        logits = logits_fn(canvas, step)
        res = denoise_step(
            logits,
            temperature=temperature,
            entropy_budget=config.entropy_budget,
            gumbel_noise=gumbel_noise,
            noise_tokens=noise_tokens,
            dedup_argmax=dedup_argmax,
            sharded_terminal=sharded_terminal,
        )
        if gumbel_noise is not None and hasattr(gumbel_noise, "deallocate"):
            gumbel_noise.deallocate(True)
        if noise_tokens is not None:
            noise_tokens.deallocate(True)

        argmax = _ids_to_torch(res.argmax)
        entropy = _entropy_to_torch(res.entropy)
        sampled = _ids_to_torch(res.sampled)
        accept_mask = _accept_to_torch(res.accept_mask)
        host_canvas = _ids_to_torch(res.canvas)
        _deallocate_logits_if_unowned(logits_fn, logits)
        entropy_mean = entropy.mean().item()
        records.append(
            StepRecord(
                step=step,
                temperature=temperature,
                entropy_mean=entropy_mean,
                num_accepted=int(accept_mask.sum()),
                argmax=argmax,
                entropy=entropy,
                sampled=sampled,
                accept_mask=accept_mask,
                canvas=host_canvas,
            )
        )

        committed = argmax
        stable = n_stable == 0 or (
            len(argmax_history) >= n_stable and all(torch.equal(argmax, h) for h in argmax_history[-n_stable:])
        )
        confident = entropy_mean < config.entropy_stop_threshold
        argmax_history.append(argmax)
        next_canvas = res.canvas
        if canvas is not next_canvas:
            canvas.deallocate(True)
        _deallocate_decision_tensors(res)

        if stable and confident:
            next_canvas.deallocate(True)
            _reset_logits_fn(logits_fn)
            return DenoiseTrajectory(committed, step + 1, True, records)

        canvas = next_canvas

    if config.max_denoise_steps > 0:
        canvas.deallocate(True)
    _reset_logits_fn(logits_fn)
    return DenoiseTrajectory(committed, config.max_denoise_steps, False, records)
