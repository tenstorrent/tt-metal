# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Production up-front Metal tracing for DiffusionGemma denoise.

The supported path is deliberately narrow:

* capture all 48 released denoise steps during model startup;
* capture one Metal trace per step so materialized Gumbel noise can be
  refreshed between replays and the host can check one halt scalar;
* reuse the traces across requests through a fixed ``p_max`` reveal-mask
  prefix reader and a model-lifetime logits adapter;
* require the released one-step stability criterion.

Legacy fixed-count, grouped-window, lazy-capture, frozen-prefix, prefix-growth
recapture, and trace-seeded chunked-RNG variants do not live in this module.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import ttnn
from loguru import logger

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
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


def upfront_capture_enabled() -> bool:
    """Return whether model-startup denoise capture is enabled."""
    return os.environ.get("DG_UPFRONT_CAPTURE", "0").lower() in ("1", "true", "yes", "on")


def _resolve_reveal_pmax(adapter) -> int:
    """Resolve and validate the fixed prefix span used by every captured trace."""
    raw = os.environ.get("DG_DENOISE_REVEAL_PMAX", "").strip()
    if not raw:
        raise RuntimeError("up-front denoise capture requires an explicit bounded DG_DENOISE_REVEAL_PMAX")
    try:
        p_max = int(raw)
    except ValueError as exc:
        raise RuntimeError("DG_DENOISE_REVEAL_PMAX must be a positive 32-token multiple") from exc
    if p_max <= 0 or p_max % ttnn.TILE_SIZE:
        raise RuntimeError("DG_DENOISE_REVEAL_PMAX must be a positive 32-token multiple")

    caches = getattr(getattr(adapter, "tt_model", None), "tt_kv_cache", None)
    if caches:
        allocated_span = min(int(k_cache.shape[-2]) for k_cache, _v_cache in caches)
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
    adapter.prepare_reveal_mask_buffers(
        canvas_len=canvas_len,
        p_max=p_max,
        prompt_len=prompt_len,
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


def _trace_metric(event: str, **fields) -> None:
    logger.info("DG_TRACE_METRIC " + json.dumps({"event": event, **fields}, sort_keys=True, default=str))


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
        self.noise_buf = _copy_materialized_noise(noise_tokens_fn(0), label="renoise step 0")

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
        self.gumbel_buf = _copy_materialized_noise(gumbel_noise_fn(0), label="Gumbel step 0")

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

    def _warm_persistent_outputs(self, adapter, init_canvas, sharded_terminal) -> None:
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
            sharded_terminal=sharded_terminal,
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
        self._initialize_noise(noise_tokens_fn)
        if self.consts is None:
            self.consts = make_denoise_constants(
                self.mesh,
                batch=1,
                canvas_len=cfg.canvas_length,
                budget=cfg.entropy_budget,
            )
        sharded_terminal = adapter.sharded_terminal_context()
        self._warm_persistent_outputs(adapter, init_canvas, sharded_terminal)

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
                        sharded_terminal=sharded_terminal,
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
            buffers = [
                ("canvas_buf", self.canvas_buf),
                ("committed_buf", self.committed_buf),
                ("gumbel_buf", self.gumbel_buf),
                ("noise_buf", self.noise_buf),
            ]
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
