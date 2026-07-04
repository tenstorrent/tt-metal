# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device denoise-loop helpers for DiffusionGemma.

This module starts with the single-step decision kernel composition used by the
full W3 loop: sample, entropy, entropy-budget accept, and renoise. The model
forward remains injected by callers so the same helper can be used by synthetic
tests and the real W2 denoise logits path.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional

import torch
import ttnn

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory, StepRecord
from models.experimental.diffusion_gemma.tt import sampling as TS

TtLogitsFn = Callable[[ttnn.Tensor, int], ttnn.Tensor]
TtNoiseFn = Callable[[int], ttnn.Tensor]


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
):
    """Run one device denoise decision step with injected noise.

    Returns token id tensors in `uint32` TILE layout except `accept_mask` and
    `entropy`, which are bf16/fp32 decision tensors.

    The renoise select is written as uint32 arithmetic because `ttnn.where`
    currently corrupts `uint32` TILE token tensors in this path.

    ``constants`` supplies preallocated accept/renoise constants for trace-safe
    capture (no per-step ``ttnn.full``/``zeros_like`` host writes); ``None`` keeps
    the eager per-call allocation.
    """
    budget_t = constants.budget_t if constants is not None else None
    accept_zeros = constants.accept_zeros if constants is not None else None
    renoise_ones = constants.renoise_ones if constants is not None else None
    sampled = TS.gumbel_max(logits, temperature, gumbel_noise)
    sampled = ttnn.typecast(sampled, ttnn.uint32)
    argmax = TS.argmax_last_dim(logits)
    argmax = ttnn.typecast(argmax, ttnn.uint32)
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
):
    """One device denoise step returning only the device-resident feedback tensors.

    Trace-safe variant of :func:`denoise_step`: it performs no host readback and no
    data-dependent control flow, so it can be captured inside a Metal trace. Returns
    ``(next_canvas, argmax)`` where ``next_canvas`` is the accepted/renoised canvas
    consumed by the next step and ``argmax`` is the clean-argmax commit candidate.
    The intermediate decision tensors (accept mask / entropy / sampled) are freed.
    """
    res = denoise_step(
        logits,
        temperature=temperature,
        entropy_budget=entropy_budget,
        gumbel_noise=gumbel_noise,
        noise_tokens=noise_tokens,
        constants=constants,
    )
    res.accept_mask.deallocate(True)
    res.entropy.deallocate(True)
    res.sampled.deallocate(True)
    return res.canvas, res.argmax


def run_fixed_denoise_steps(
    logits_fn: TtLogitsFn,
    init_canvas: ttnn.Tensor,
    config: DiffusionConfig,
    *,
    gumbel_noise_fn: Optional[TtNoiseFn] = None,
    noise_tokens_fn: Optional[TtNoiseFn] = None,
    constants: "Optional[DenoiseConstants]" = None,
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
