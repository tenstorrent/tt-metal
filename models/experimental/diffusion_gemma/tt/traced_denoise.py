# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Production up-front Metal tracing for DiffusionGemma denoise.

The supported path is deliberately narrow:

* capture all 48 released denoise steps during model startup;
* capture one Metal trace per step so materialized Gumbel noise can be
  refreshed between replays and the host can check one halt scalar;
* reuse the traces across requests through a fixed ``p_max`` reveal-mask
  prefix reader and a model-lifetime logits adapter (under
  ``DG_DENOISE_REVEAL_BUCKETS`` the wrapper re-captures at a per-request
  power-of-two ``p_max`` — each capture is still one fixed span);
* require the released one-step stability criterion.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import ttnn
from loguru import logger

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    _layer_type_for_denoise,
    _sliding_window_for_denoise,
    denoise_sliding_window_enabled,
    sliding_read_span,
)
from models.experimental.diffusion_gemma.tt.denoise_loop import (
    HaltBuffers,
    _argmax_to_tile_f32,
    _deallocate_logits_if_unowned,
    _ids_to_torch,
    compute_halt_scalars,
    denoise_step,
    denoise_step_next_canvas_and_halt,
    eval_halt,
    make_denoise_constants,
    read_halt_scalars,
    temperature_at_step,
    write_halt_scalars,
)


UPFRONT_DENOISE_STEPS = 48
_CONTROLLER_ATTR = "_upfront_traced_denoise_controller"

# Fixed reveal span registered by a caller that derives it from its own served-context
# bound instead of the operator setting DG_DENOISE_REVEAL_PMAX (the vLLM wrapper derives
# it from ``max_model_len``). The controller is built deep inside the denoise-block entry
# point, which cannot take extra arguments without changing the ``denoise_block_fn``
# protocol, so the derived value is registered here once at startup. An explicit
# DG_DENOISE_REVEAL_PMAX always wins; both paths get identical validation.
_DEFAULT_REVEAL_PMAX: int | None = None

# Bucketed reveal spans (DG_DENOISE_REVEAL_BUCKETS, generator_vllm.py) re-capture the
# controller at a power-of-two span sized to the live request instead of the one
# deployment-wide worst case. The active bucket must win over the env pin — in bucket
# mode DG_DENOISE_REVEAL_PMAX (or the registered default) is the CEILING the buckets
# are clipped to, not the capture span — and it reaches this module through the same
# registration side door as the default, for the same denoise_block_fn-protocol reason.
_ACTIVE_REVEAL_PMAX: int | None = None


def set_default_reveal_pmax(p_max: int | None) -> None:
    """Register the derived fixed reveal span used when DG_DENOISE_REVEAL_PMAX is unset."""
    global _DEFAULT_REVEAL_PMAX
    _DEFAULT_REVEAL_PMAX = None if p_max is None else int(p_max)


def set_active_reveal_pmax(p_max: int | None) -> None:
    """Register (or clear) the bucket span the next controller capture binds to.

    Only the reveal-bucket policy in ``generator_vllm.py`` sets this, immediately
    before a release-and-recapture; ``None`` restores env/default resolution.
    """
    global _ACTIVE_REVEAL_PMAX
    _ACTIVE_REVEAL_PMAX = None if p_max is None else int(p_max)


def upfront_capture_enabled() -> bool:
    """Return whether model-startup denoise capture is enabled (default ON).

    Up-front capture is the shipped serving path: capture the 48 released denoise
    steps once during startup and replay them for every request. Set
    ``DG_UPFRONT_CAPTURE=0`` to opt out and run the ordinary eager per-step loop
    (slower, and the supported choice when per-step trajectory records are
    needed — replayed traces do not produce them).
    """
    return os.environ.get("DG_UPFRONT_CAPTURE", "1").lower() in ("1", "true", "yes", "on")


def prefix_borrow_enabled() -> bool:
    """Whether the fixed full-span prefix read borrows the cache instead of cloning it.

    Default ON: the clone it replaces is ~2 whole-cache copies per layer per step of data that
    is bit-identical across all 48 steps of a block, and borrowing is bit-exact by construction
    (the downstream concat copies the bytes it needs). ``DG_PREFIX_BORROW=0`` restores the clone
    — it exists to A/B the change on device and as an escape hatch.
    """
    return os.environ.get("DG_PREFIX_BORROW", "1").lower() not in ("0", "false", "no", "off")


def _resolve_reveal_pmax(adapter) -> int:
    """Resolve and validate the fixed prefix span used by every captured trace."""
    raw = os.environ.get("DG_DENOISE_REVEAL_PMAX", "").strip()
    if _ACTIVE_REVEAL_PMAX is not None:
        p_max = _ACTIVE_REVEAL_PMAX
    elif not raw:
        if _DEFAULT_REVEAL_PMAX is None:
            raise RuntimeError("up-front denoise capture requires an explicit bounded DG_DENOISE_REVEAL_PMAX")
        p_max = _DEFAULT_REVEAL_PMAX
    else:
        try:
            p_max = int(raw)
        except ValueError as exc:
            raise RuntimeError("DG_DENOISE_REVEAL_PMAX must be a positive 32-token multiple") from exc
    if p_max <= 0 or p_max % ttnn.TILE_SIZE:
        raise RuntimeError("DG_DENOISE_REVEAL_PMAX must be a positive 32-token multiple")

    tt_model = getattr(adapter, "tt_model", None)
    caches = getattr(tt_model, "tt_kv_cache", None)
    if caches:
        allocated_span = (
            int(tt_model._dg_hybrid_max_seq_len)
            if bool(getattr(tt_model, "_dg_model_owned_hybrid_kv", False))
            else min(int(k_cache.shape[-2]) for k_cache, _v_cache in caches)
        )
        if p_max > allocated_span:
            raise RuntimeError(
                f"DG_DENOISE_REVEAL_PMAX={p_max} exceeds the smallest allocated model KV span " f"{allocated_span}"
            )
    return p_max


def _prepare_fixed_reveal(adapter, *, canvas_len: int) -> int:
    """Bind the adapter to its model-lifetime fixed-span reveal-mask buffers."""
    p_max = _resolve_reveal_pmax(adapter)
    prompt_len = int(getattr(adapter, "prompt_len", 0) or 0)
    if prompt_len + canvas_len > p_max:
        raise ValueError(
            f"up-front denoise prefix plus canvas exceeds fixed reveal span: " f"{prompt_len} + {canvas_len} > {p_max}"
        )

    reader = getattr(adapter, "prompt_hidden_by_layer", None)
    set_read_span = getattr(reader, "set_read_span", None)
    if not callable(set_read_span):
        raise RuntimeError("up-front denoise capture requires a MutablePrefixKVReader prefix source")
    set_read_span(p_max)
    # The fixed span is now bound and never changes, so when it covers the whole cache the
    # per-layer prefix read can hand back the model-owned cache instead of cloning it.
    # Borrowing is safe only because BOTH hold: denoise_hidden_forward consults
    # ``owns_result`` before freeing, and denoise_attention compares BUFFERS (not object
    # identity) before freeing its ``to_memory_config`` result — ``to_memory_config``
    # returns a fresh Tensor object that aliases the input buffer when no conversion is
    # needed, so an identity check would free the model KV cache.
    if hasattr(reader, "borrow_full_span"):
        reader.borrow_full_span = prefix_borrow_enabled()
        logger.info(
            f"[DiffusionGemma up-front] prefix read borrow_full_span={reader.borrow_full_span} "
            f"engaged={not reader.owns_result} (p_max={p_max})"
        )
    # HF's sliding layers retain only the last sliding_window-1 committed tokens; enforcing that
    # is a fidelity fix (#51080) but decision-changing above prompt_len 1024, so it is gated.
    # Both masks share one shape, so this changes content only and every trace stays valid.
    enforce_window = denoise_sliding_window_enabled()
    # Bounded sliding spans (perf half of #51080): sliding layers read only the retained window
    # instead of the full p_max. The read must move OUT of the trace because its offset slides with
    # the committed prefix, so allocate the block-resident buffers HERE, before begin_trace_capture.
    sliding_span = None
    window_layers = {}
    # Gated on the RETENTION mask, not on a flag of its own: the bounded read is bit-identical when
    # retention is enforced (see sliding_read_span) and would silently change visibility without it.
    #
    # Input contract, which the fallbacks below cannot catch: adapter.tt_model must expose
    # `.layers` (dereferenced before the no-sliding-layers fallback can fire) and the reader must
    # expose prepare_window_buffers / refresh_windows.
    if enforce_window:
        window = _sliding_window_for_denoise(adapter.tt_model, 0)
        if window:
            sliding_span = sliding_read_span(window, p_max)
            if sliding_span >= p_max:
                # Nothing to save; keep the single-shape path.
                sliding_span = None
            else:
                window_layers = {
                    layer_idx: sliding_span
                    for layer_idx in range(len(adapter.tt_model.layers))
                    if _layer_type_for_denoise(adapter.tt_model, layer_idx) == "sliding_attention"
                }
                if not window_layers:
                    sliding_span = None
    adapter.prepare_reveal_mask_buffers(
        canvas_len=canvas_len,
        p_max=p_max,
        prompt_len=prompt_len,
        enforce_window=enforce_window,
        sliding_span=sliding_span,
    )
    if window_layers:
        reader.prepare_window_buffers(window_layers)
        reader.refresh_windows(prompt_len)
        # Derive the report from the ACTUAL per-layer spans: window_layers can also carry
        # full-attention layers (canvas-tail path), so counting its keys as "sliding" would
        # misprice the key rows.
        n_layers = len(adapter.tt_model.layers)
        bounded = sum(1 for span in window_layers.values() if sliding_span and span == sliding_span)
        key_rows_before = n_layers * (p_max + canvas_len)
        key_rows_after = sum(window_layers.get(i, p_max) + canvas_len for i in range(n_layers))
        # "bounded sliding read:" is a stable, greppable engagement marker -- it is how a run's
        # log proves the bounded read actually took effect. It deliberately names no env var.
        logger.info(
            f"[DiffusionGemma up-front] bounded sliding read: {bounded}/{n_layers} layers "
            f"bounded to {sliding_span}, rest at {p_max}; "
            f"SDPA key rows/step {key_rows_before} -> {key_rows_after}"
        )
    if enforce_window:
        logger.info(
            f"[DiffusionGemma up-front] DG_DENOISE_SLIDING_WINDOW=1: enforcing HF sliding-layer "
            f"key retention (masks={sorted(adapter._reveal_mask_bufs)}, p_max={p_max}, "
            f"sliding_span={sliding_span})"
        )
    adapter.update_reveal_mask_buffer(prompt_len)
    if not getattr(adapter, "use_reveal_mask", False):
        raise RuntimeError("up-front denoise capture failed to enable the persistent reveal mask")
    return p_max


def _deallocate_tensor(tensor) -> None:
    if tensor is not None and hasattr(tensor, "deallocate"):
        tensor.deallocate(True)


def _copy_materialized_noise(fresh, destination=None, *, label: str):
    """Clone or copy one caller-owned materialized TTNN noise tensor, then consume it."""
    if fresh is None or not hasattr(fresh, "deallocate"):
        raise ValueError(
            f"up-front traced denoise requires materialized host noise for {label}; "
            "argmax and chunked/descriptor noise are unsupported"
        )
    try:
        if destination is None:
            return ttnn.clone(fresh)
        ttnn.copy(fresh, destination)
        return destination
    finally:
        fresh.deallocate(True)


def _vocab_noise_pool(adapter) -> dict:
    """Model-lifetime pool for the [canvas, vocab] noise buffers.

    These buffers are span-independent (~134 MB each at the production vocab), so
    reallocating them on every bucket recapture both churns DRAM and — once
    heterogeneous rebuilds have fragmented memory — can stop fitting. Allocated
    once per model, at startup, while memory is clean; released only with the mesh.
    """
    tt_model = getattr(adapter, "tt_model", None)
    if tt_model is None:
        return {}
    pool = getattr(tt_model, "_dg_vocab_noise_pool", None)
    if pool is None:
        pool = {}
        tt_model._dg_vocab_noise_pool = pool
    return pool


def _trace_metric(event: str, **fields) -> None:
    logger.info("DG_TRACE_METRIC " + json.dumps({"event": event, **fields}, sort_keys=True, default=str))


def _summarize_halt_trace(halt_trace, *, threshold: float) -> dict:
    """Reduce a per-step ``(steps_run, mean_entropy, mismatch)`` trace to the fields that say
    WHICH halt criterion blocked -- something the ``halted`` boolean alone cannot express.

    ``eval_halt`` fires only when ``mismatch == 0`` AND ``mean_entropy < threshold``, and only
    from step index 1 on (it needs a previous step to compare the argmax against). So a
    step-cap exit has exactly three shapes, and they call for opposite fixes:

    * ``entropy``  -- the argmax went stable but the entropy never got under the bar
      (content did not converge; the lever is fidelity or the bar itself);
    * ``mismatch`` -- the entropy got under the bar but the argmax never stopped moving
      (a few positions oscillate; the lever is the accept/renoise decisions);
    * ``both``     -- no eligible step satisfied either gate.

    ``halt_entropy_floor_ratio`` separates a *structural* entropy floor (min entropy many
    times the bar) from a *numerical* near-miss (ratio ~1) -- also invisible in ``halted``.
    """
    if not halt_trace:
        return {"halt_trace_steps": 0, "halt_blocking_gate": "none"}
    entropies = [float(entropy) for _, entropy, _ in halt_trace]
    mismatches = [float(mismatch) for _, _, mismatch in halt_trace]
    # eval_halt cannot fire on the first step (no previous argmax to compare against), so gate
    # accounting must skip it or it reports a gate as blocking on a step that was never eligible.
    eligible = [(entropy, mismatch) for (steps_run, entropy, mismatch) in halt_trace if steps_run >= 2]
    entropy_ok = sum(1 for entropy, _ in eligible if entropy < threshold)
    mismatch_ok = sum(1 for _, mismatch in eligible if mismatch == 0.0)
    both_ok = sum(1 for entropy, mismatch in eligible if entropy < threshold and mismatch == 0.0)
    if both_ok:
        blocking = "none"
    elif mismatch_ok and not entropy_ok:
        blocking = "entropy"
    elif entropy_ok and not mismatch_ok:
        blocking = "mismatch"
    elif entropy_ok and mismatch_ok:
        blocking = "never_simultaneous"
    else:
        blocking = "both"
    entropy_min = min(entropies)
    return {
        "halt_trace_steps": len(halt_trace),
        "halt_threshold": threshold,
        "halt_blocking_gate": blocking,
        "halt_entropy_per_step": [round(entropy, 6) for entropy in entropies],
        "halt_mismatch_per_step": mismatches,
        "halt_entropy_first": round(entropies[0], 6),
        "halt_entropy_final": round(entropies[-1], 6),
        "halt_entropy_min": round(entropy_min, 6),
        "halt_entropy_margin_final": round(entropies[-1] - threshold, 6),
        "halt_entropy_floor_ratio": (round(entropy_min / threshold, 3) if threshold > 0 else None),
        "halt_mismatch_final": mismatches[-1],
        "halt_mismatch_min": min(mismatches),
        "halt_eligible_steps": len(eligible),
        "halt_steps_entropy_under_threshold": entropy_ok,
        "halt_steps_mismatch_zero": mismatch_ok,
        "halt_steps_both_gates": both_ok,
    }


@contextmanager
def _trace_capture_guard(mesh_device, *, cq_id: int = 0):
    """Finalize a successful trace or release a partially captured trace."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=cq_id)
    try:
        yield trace_id
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=cq_id)
        except Exception as cleanup_error:
            logger.error(f"failed to end aborted Metal trace {trace_id}: {cleanup_error}")
        try:
            ttnn.release_trace(mesh_device, trace_id)
        except Exception as cleanup_error:
            logger.error(f"failed to release aborted Metal trace {trace_id}: {cleanup_error}")
        raise
    else:
        try:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=cq_id)
        except BaseException:
            try:
                ttnn.release_trace(mesh_device, trace_id)
            except Exception as cleanup_error:
                logger.error(f"failed to release unfinalized Metal trace {trace_id}: {cleanup_error}")
            raise


class UpfrontTracedDenoiseController:
    """One model-lifetime controller for the production up-front denoise path."""

    def __init__(self, mesh_device, config: DiffusionConfig, consts=None):
        if config.max_denoise_steps != UPFRONT_DENOISE_STEPS:
            raise ValueError(
                "up-front traced denoise captures the released 48-step schedule; "
                f"got max_denoise_steps={config.max_denoise_steps}"
            )
        if config.stable_steps_to_halt != 1:
            raise ValueError(
                "up-front traced denoise requires stable_steps_to_halt=1; " f"got {config.stable_steps_to_halt}"
            )

        self.mesh = mesh_device
        self.config = config
        self.consts = consts
        self._owns_consts = consts is None
        self.adapter = None
        self.captured = False
        self.released = False
        self.reveal_pmax = None
        self.captured_prompt_len = None
        self._last_prompt_len = None

        self.traces: list = []
        self.canvas_buf = None
        self.committed_buf = None
        self.gumbel_buf = None
        self._gumbel_refresh_reserve = None
        self.noise_buf = None
        self.halt_bufs: HaltBuffers | None = None

        self.capture_events = 0
        self.traces_captured = 0
        self.replay_blocks = 0
        self.execute_trace_calls = 0
        self.halt_checks = 0
        self.halted_blocks = 0
        self.adapter_rebinds = 0
        self.last_halt_trace: list[tuple[int, float, float]] = []

    def stats(self) -> dict:
        """Return model-lifetime capture, replay, halt, and rebind counters."""
        return {
            "controller": type(self).__name__,
            "captured": self.captured,
            "released": self.released,
            "capture_events": self.capture_events,
            "traces_captured": self.traces_captured,
            "replay_blocks": self.replay_blocks,
            "execute_trace_calls": self.execute_trace_calls,
            "halt_checks": self.halt_checks,
            "halted_blocks": self.halted_blocks,
            "adapter_rebinds": self.adapter_rebinds,
            "reveal_pmax": self.reveal_pmax,
            "captured_prompt_len": self.captured_prompt_len,
            "trace_ids": [str(trace_id) for trace_id in self.traces],
            "gumbel_mode": "materialized",
            "halt_window": 1,
        }

    def _bind_adapter(self, adapter) -> None:
        if self.adapter is None:
            self.adapter = adapter
            return
        if adapter is not self.adapter:
            raise RuntimeError("up-front traced denoise controller cannot be rebound to a different adapter")

    def _initialize_noise(self, noise_tokens_fn) -> None:
        if noise_tokens_fn is None:
            raise ValueError("up-front traced denoise requires a per-step materialized noise_tokens_fn")
        pool = _vocab_noise_pool(self.adapter)
        self.noise_buf = _copy_materialized_noise(noise_tokens_fn(0), pool.get("noise"), label="renoise step 0")
        pool["noise"] = self.noise_buf

    def _refresh_noise(self, noise_tokens_fn, step: int):
        if noise_tokens_fn is None:
            raise ValueError("up-front traced denoise requires a per-step materialized noise_tokens_fn")
        return _copy_materialized_noise(
            noise_tokens_fn(step),
            self.noise_buf,
            label=f"renoise step {step}",
        )

    def _initialize_gumbel(self, gumbel_noise_fn) -> None:
        if gumbel_noise_fn is None:
            raise ValueError("up-front traced denoise requires a per-step materialized gumbel_noise_fn")
        pool = _vocab_noise_pool(self.adapter)
        self.gumbel_buf = _copy_materialized_noise(gumbel_noise_fn(0), pool.get("gumbel"), label="Gumbel step 0")
        pool["gumbel"] = self.gumbel_buf

    def _refresh_gumbel(self, gumbel_noise_fn, step: int):
        if gumbel_noise_fn is None:
            raise ValueError("up-front traced denoise requires a per-step materialized gumbel_noise_fn")
        return _copy_materialized_noise(
            gumbel_noise_fn(step),
            self.gumbel_buf,
            label=f"Gumbel step {step}",
        )

    def _prepare_adapter_for_capture(self, adapter, *, start_pos: int) -> None:
        canvas_len = self.config.canvas_length
        adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_len)
        adapter.prepare_canvas_rope_buffers(canvas_len=canvas_len)
        adapter.update_canvas_rope_buffers(start_pos)
        adapter.use_canvas_rope = True
        self.reveal_pmax = _prepare_fixed_reveal(adapter, canvas_len=canvas_len)

    def _warm_persistent_outputs(self, adapter, init_canvas) -> None:
        cfg = self.config
        canvas_len = cfg.canvas_length
        adapter.reset_signal_buffer()
        warm_canvas = ttnn.clone(init_canvas)
        logits = adapter(warm_canvas, 0)
        result = denoise_step(
            logits,
            temperature=temperature_at_step(
                0,
                UPFRONT_DENOISE_STEPS,
                cfg.temperature_start,
                cfg.temperature_end,
            ),
            entropy_budget=cfg.entropy_budget,
            gumbel_noise=self.gumbel_buf,
            noise_tokens=self.noise_buf,
            constants=self.consts,
        )
        _deallocate_logits_if_unowned(adapter, logits)

        self.canvas_buf = ttnn.clone(result.canvas)
        self.committed_buf = ttnn.clone(result.argmax)
        current_argmax = _argmax_to_tile_f32(result.argmax)
        previous_argmax = ttnn.clone(current_argmax)
        mean_entropy, mismatch, current_for_halt = compute_halt_scalars(
            result.argmax,
            result.entropy,
            previous_argmax,
            canvas_len=canvas_len,
        )
        self.halt_bufs = HaltBuffers(
            prev_argmax=previous_argmax,
            mean_entropy=ttnn.clone(mean_entropy),
            mismatch=ttnn.clone(mismatch),
        )

        ttnn.copy(result.canvas, self.canvas_buf)
        ttnn.copy(result.argmax, self.committed_buf)
        write_halt_scalars(
            result.argmax,
            result.entropy,
            self.halt_bufs,
            canvas_len=canvas_len,
        )

        for tensor in (
            current_argmax,
            mean_entropy,
            mismatch,
            current_for_halt,
            result.canvas,
            result.argmax,
            result.accept_mask,
            result.entropy,
            result.sampled,
            warm_canvas,
        ):
            _deallocate_tensor(tensor)
        ttnn.synchronize_device(self.mesh)

    def capture(
        self,
        adapter,
        init_canvas,
        *,
        gumbel_noise_fn,
        noise_tokens_fn,
    ) -> None:
        """Capture the complete 48-trace set; intended for model startup only."""
        if self.released:
            raise RuntimeError("cannot capture a released up-front denoise controller")
        if self.captured:
            raise RuntimeError("up-front denoise traces are already captured")
        self._bind_adapter(adapter)

        cfg = self.config
        start_pos = int(getattr(adapter, "q_rope_offset", 0) or 0)
        self._prepare_adapter_for_capture(adapter, start_pos=start_pos)
        self._initialize_gumbel(gumbel_noise_fn)
        # Long-context traces heavily fragment DRAM during capture. Preserve the
        # contiguous hole left by the initial materialized Gumbel draw so every
        # replay can allocate its fresh RNG tensor, copy into gumbel_buf, and
        # return that same hole; without the reservation the first refresh after
        # a long-context capture can find no large-enough per-bank free block.
        # A bucketed startup capture may bind the 4K floor even though the
        # deployment ceiling can later upshift beyond 64K. Allocate the
        # span-independent reservation while startup memory is clean in that
        # case too, rather than waiting for the first fragmented recapture.
        reserve_span = max(int(self.reveal_pmax or 0), int(_DEFAULT_REVEAL_PMAX or 0))
        if reserve_span >= 65536:
            pool = _vocab_noise_pool(self.adapter)
            reserve = pool.get("gumbel_refresh_reserve")
            self._gumbel_refresh_reserve = reserve if reserve is not None else ttnn.clone(self.gumbel_buf)
            pool["gumbel_refresh_reserve"] = self._gumbel_refresh_reserve
        self._initialize_noise(noise_tokens_fn)
        if self.consts is None:
            self.consts = make_denoise_constants(
                self.mesh,
                batch=1,
                canvas_len=cfg.canvas_length,
                budget=cfg.entropy_budget,
            )
        self._warm_persistent_outputs(adapter, init_canvas)

        adapter.reset_signal_buffer()
        self.traces = []
        set_cache_misses_allowed = getattr(self.mesh, "set_program_cache_misses_allowed", None)
        if callable(set_cache_misses_allowed):
            set_cache_misses_allowed(False)
        try:
            for step in range(UPFRONT_DENOISE_STEPS):
                temperature = temperature_at_step(
                    step,
                    UPFRONT_DENOISE_STEPS,
                    cfg.temperature_start,
                    cfg.temperature_end,
                )
                with _trace_capture_guard(self.mesh, cq_id=0) as trace_id:
                    logits = adapter(self.canvas_buf, step)
                    next_canvas, argmax = denoise_step_next_canvas_and_halt(
                        logits,
                        temperature=temperature,
                        entropy_budget=cfg.entropy_budget,
                        gumbel_noise=self.gumbel_buf,
                        noise_tokens=self.noise_buf,
                        halt_bufs=self.halt_bufs,
                        canvas_len=cfg.canvas_length,
                        constants=self.consts,
                    )
                    _deallocate_logits_if_unowned(adapter, logits)
                    ttnn.copy(next_canvas, self.canvas_buf)
                    ttnn.copy(argmax, self.committed_buf)
                    next_canvas.deallocate(True)
                    argmax.deallocate(True)
                self.traces.append(trace_id)
        except BaseException:
            self.release()
            raise
        finally:
            if callable(set_cache_misses_allowed):
                set_cache_misses_allowed(True)

        ttnn.synchronize_device(self.mesh)
        self.captured = True
        self.captured_prompt_len = int(getattr(adapter, "prompt_len", 0) or 0)
        self._last_prompt_len = self.captured_prompt_len
        self.capture_events = 1
        self.traces_captured = len(self.traces)
        if self.traces_captured != UPFRONT_DENOISE_STEPS:
            self.release()
            raise RuntimeError(
                f"up-front denoise capture produced {self.traces_captured} traces; " f"expected {UPFRONT_DENOISE_STEPS}"
            )
        _trace_metric("upfront_capture", **self.stats())

    def denoise_block(
        self,
        adapter,
        init_canvas,
        config: DiffusionConfig,
        *,
        gumbel_noise_fn,
        noise_tokens_fn,
    ) -> DenoiseTrajectory:
        """Replay one trace at a time, checking the halt scalars after every step."""
        if config != self.config:
            raise ValueError("up-front traced denoise config changed after controller construction")
        if self.released:
            raise RuntimeError("up-front traced denoise controller has been released")
        self._bind_adapter(adapter)

        captured_this_block = not self.captured
        if captured_this_block:
            if not getattr(adapter, "_upfront_capture_phase", False):
                raise RuntimeError("up-front denoise traces may only be captured during the startup trace warmup phase")
            self.capture(
                adapter,
                init_canvas,
                gumbel_noise_fn=gumbel_noise_fn,
                noise_tokens_fn=noise_tokens_fn,
            )
        else:
            start_pos = int(getattr(adapter, "q_rope_offset", 0) or 0)
            prompt_len = int(getattr(adapter, "prompt_len", 0) or 0)
            if prompt_len + self.config.canvas_length > int(self.reveal_pmax):
                raise ValueError(
                    f"rebound prefix plus canvas exceeds fixed reveal span: "
                    f"{prompt_len} + {self.config.canvas_length} > {self.reveal_pmax}"
                )
            if prompt_len != self._last_prompt_len:
                self.adapter_rebinds += 1
                self._last_prompt_len = prompt_len
            adapter.update_canvas_rope_buffers(start_pos)
            adapter.update_reveal_mask_buffer(prompt_len)

        ttnn.copy(init_canvas, self.canvas_buf)
        _deallocate_tensor(init_canvas)
        adapter.reset_signal_buffer()
        ttnn.synchronize_device(self.mesh)

        self.last_halt_trace = []
        halted = False
        steps_run = 0
        for step, trace_id in enumerate(self.traces):
            if not (captured_this_block and step == 0):
                self._refresh_gumbel(gumbel_noise_fn, step)
                self._refresh_noise(noise_tokens_fn, step)
            ttnn.execute_trace(self.mesh, trace_id, blocking=False)
            # No explicit device drain here: read_halt_scalars enqueues its ttnn.to_torch
            # reads on the same CQ0 after the trace and blocks the host until they land, so
            # the halt scalars already reflect this step's completed trace, and the next
            # step's ttnn.copy into gumbel_buf/noise_buf is CQ0-ordered after the trace's
            # read of those buffers. A per-step synchronize_device is therefore redundant
            # and only forecloses host/device overlap (host Gumbel prefetch).
            self.execute_trace_calls += 1
            self.halt_checks += 1
            steps_run = step + 1
            mean_entropy, mismatch = read_halt_scalars(self.halt_bufs)
            self.last_halt_trace.append((steps_run, mean_entropy, mismatch))
            if eval_halt(
                mean_entropy,
                mismatch,
                step,
                threshold=self.config.entropy_stop_threshold,
                n_stable=1,
            ):
                halted = True
                self.halted_blocks += 1
                break

        self.replay_blocks += 1
        _trace_metric(
            "upfront_replay",
            steps_run=steps_run,
            halted=halted,
            captured_this_block=captured_this_block,
            **_summarize_halt_trace(self.last_halt_trace, threshold=self.config.entropy_stop_threshold),
            **self.stats(),
        )
        return DenoiseTrajectory(_ids_to_torch(self.committed_buf), steps_run, halted, [])

    def release(self) -> None:
        """Best-effort terminal release of controller-owned traces and buffers."""
        if self.released:
            return
        final_stats = self.stats()
        cleanup_errors: list[str] = []

        def cleanup(label, fn) -> None:
            try:
                fn()
            except BaseException as cleanup_error:
                cleanup_errors.append(f"{label}: {cleanup_error}")
                logger.error(f"failed to release up-front denoise {label}: {cleanup_error}")

        try:
            for trace_id in list(self.traces):
                cleanup(
                    f"trace {trace_id}",
                    lambda trace_id=trace_id: ttnn.release_trace(self.mesh, trace_id),
                )
            pooled = {id(t) for t in _vocab_noise_pool(self.adapter).values() if t is not None}
            buffers = [
                ("canvas_buf", self.canvas_buf),
                ("committed_buf", self.committed_buf),
                ("gumbel_buf", self.gumbel_buf),
                ("gumbel_refresh_reserve", self._gumbel_refresh_reserve),
                ("noise_buf", self.noise_buf),
            ]
            # Pool-owned [canvas, vocab] buffers outlive every controller: the next
            # capture copies into them instead of reallocating 134 MB into whatever
            # fragments the rebuild left behind.
            buffers = [(label, t) for label, t in buffers if t is None or id(t) not in pooled]
            if self.halt_bufs is not None:
                buffers.extend(
                    (f"halt_bufs.{name}", tensor) for name, tensor in zip(self.halt_bufs._fields, self.halt_bufs)
                )
            for label, tensor in buffers:
                if tensor is not None and hasattr(tensor, "deallocate"):
                    cleanup(label, lambda tensor=tensor: tensor.deallocate(True))
            if self._owns_consts and self.consts is not None:
                for name, tensor in zip(self.consts._fields, self.consts):
                    if tensor is not None and hasattr(tensor, "deallocate"):
                        cleanup(f"consts.{name}", lambda tensor=tensor: tensor.deallocate(True))
        finally:
            self.traces = []
            self.canvas_buf = None
            self.committed_buf = None
            self.gumbel_buf = None
            self._gumbel_refresh_reserve = None
            self.noise_buf = None
            self.halt_bufs = None
            if self._owns_consts:
                self.consts = None
            self.captured = False
            self.released = True
            _trace_metric("upfront_release", cleanup_errors=cleanup_errors, **final_stats)


def upfront_traced_denoise_block(
    logits_fn,
    init_canvas,
    config: DiffusionConfig,
    *,
    gumbel_noise_fn=None,
    noise_tokens_fn=None,
) -> DenoiseTrajectory:
    """Denoise-block entry point for the model-lifetime up-front controller."""
    controller = getattr(logits_fn, _CONTROLLER_ATTR, None)
    if controller is None:
        if not getattr(logits_fn, "_upfront_capture_phase", False):
            raise RuntimeError("up-front denoise controller was not created during startup trace warmup")
        controller = UpfrontTracedDenoiseController(logits_fn.tt_model.mesh_device, config)
        setattr(logits_fn, _CONTROLLER_ATTR, controller)
    return controller.denoise_block(
        logits_fn,
        init_canvas,
        config,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
    )
