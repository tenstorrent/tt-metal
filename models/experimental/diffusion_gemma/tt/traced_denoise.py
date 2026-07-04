# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced single-step denoise loop for serving (#47465, path to 30 t/s).

The eager serving denoise step is host-dispatch bound (~137 ms/step dispatch tax that
masks the OPT-004 sparse-MoE compute win). Capturing one Metal trace per denoise step and
replaying it once per step removes that tax and unmasks the compute win: at full 30L,
``DG_SPARSE_MOE=1 DG_DEDUP_ARGMAX=1 DG_SPARSE_MOE_TUNED=1`` gives 257.93 ms/step traced vs
~777 ms/step eager, i.e. **58.1 tokens/block/s @12 steps and 33.2 @24 steps** — both above
the 30 target (``probe_traced_serving.py``), verified bit-exact to the eager committed argmax
(CROSSBLOCK_OK: off0/off0_tvt/off1 all 100%).

Structure — one single-step trace per step index (session-8 architecture):
  * cross-step state (canvas + self-cond signal) lives in PERSISTENT device buffers threaded
    across replays (the KV-cache pattern; a single self-cond step traces bit-exactly);
  * per-step temperature ``T[i]`` and renoise tokens are baked/threaded per step index;
  * the canvas RoPE (the only per-BLOCK variation, since the denoise reads a FROZEN prompt
    prefix) is a constant-shape buffer whose content is refreshed per block OUTSIDE the trace,
    so ONE captured trace set replays for every block.

Correctness rule that makes this work (root cause of the 3-session phantom "self-cond race"):
**every persistent cross-replay buffer — canvas / committed / signal / rope / per-step noise —
is allocated BEFORE ``begin_trace_capture``.** A Metal trace bakes its intermediate-tensor
addresses at capture time; a buffer allocated into post-capture-freed memory overlaps that
trace scratch and is clobbered on every replay. The serving canvas init and per-step noise are
re-uploaded/refreshed into these pre-allocated buffers each block. See perf_progress.md session 8.

Scope: argmax (``gumbel_noise=None``) regime, contiguous cache + batched commit (the serving_smoke
default). Opt-in via ``DG_DENOISE_TRACED``; falls back to the eager loop for the gumbel regime.
Requires a large trace region (≈168 MB/single-step trace at 30L → ~2 GB @12, ~4 GB @24); set
``DG_TRACE_REGION_SIZE`` when opening the mesh. Kept opt-in (not default) precisely because those
prerequisites are not universal — paged/vLLM caches, the gumbel regimes, and a 0-byte trace region
would all break a default-on traced path.

Lifecycle: the controller is cached on the logits adapter, which a serving session rebuilds per
prompt at prefill — so each prompt gets a fresh controller (the traces bake the prompt's frozen
prefix). A long-lived multi-prompt server should call ``controller.release()`` on the prior
adapter before re-prefill to free its traces/buffers; the single-prompt serving_smoke path does not
need it (the mesh close frees everything).
"""
from __future__ import annotations

import os
from typing import List

import ttnn

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt.denoise_loop import (
    _deallocate_logits_if_unowned,
    _ids_to_torch,
    denoise_step_next_canvas,
    make_denoise_constants,
    temperature_at_step,
)


def traced_denoise_enabled() -> bool:
    """True when the traced single-step denoise loop is opted in (``DG_DENOISE_TRACED``)."""
    return os.environ.get("DG_DENOISE_TRACED", "0").lower() in ("1", "true", "yes", "on")


class TracedDenoiseController:
    """Stateful traced denoise loop bound to one serving session's logits adapter.

    Captures N single-step Metal traces on the first block and replays them for every
    subsequent block, carrying cross-step state in persistent buffers and refreshing the
    per-block canvas RoPE + per-step noise OUTSIDE the trace. Behaviourally identical to the
    eager fixed-step loop (``run_fixed_denoise_steps``) whenever early-halt does not fire —
    which is the serving contract (fixed budget; early-halt is a no-op under #48291).
    """

    def __init__(self, mesh_device, config: DiffusionConfig, consts=None):
        self.mesh = mesh_device
        self.config = config
        self.consts = consts
        self.captured = False
        self.traces: List = []
        self.canvas_buf = None
        self.committed_buf = None
        self.noise_bufs: List = []

    # -- capture (first block only) ------------------------------------------------
    def _fill_noise_buffers_from(self, noise_tokens_fn) -> None:
        """Allocate persistent per-step noise buffers from THIS block's noise stream.

        Consumes ``noise_tokens_fn(step)`` for step 0..N-1 (same order/count as the eager
        ``run_fixed_denoise_steps``), so the seeded generator stream stays identical to eager."""
        self.noise_bufs = []
        for step in range(self.config.max_denoise_steps):
            fresh = noise_tokens_fn(step)
            self.noise_bufs.append(ttnn.clone(fresh))
            if hasattr(fresh, "deallocate"):
                fresh.deallocate(True)

    def _refresh_noise_buffers_from(self, noise_tokens_fn) -> None:
        """Overwrite the persistent per-step noise buffers with THIS block's noise (in place)."""
        for step in range(self.config.max_denoise_steps):
            fresh = noise_tokens_fn(step)
            ttnn.copy(fresh, self.noise_bufs[step])
            if hasattr(fresh, "deallocate"):
                fresh.deallocate(True)

    def _capture(self, adapter, init_canvas, noise_tokens_fn, start_pos: int) -> None:
        cfg = self.config
        canvas_len = cfg.canvas_length
        # Constant-shape trace-safe state: persistent self-cond signal buffer + canvas RoPE
        # buffers (both allocated BEFORE capture). The denoise reads a frozen prompt prefix,
        # so canvas RoPE (refreshed per block) is the only per-block variation inside a step.
        adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_len)
        adapter.prepare_canvas_rope_buffers(canvas_len=canvas_len)
        adapter.update_canvas_rope_buffers(start_pos)
        adapter.use_canvas_rope = True
        if self.consts is None:
            self.consts = make_denoise_constants(self.mesh, batch=1, canvas_len=canvas_len, budget=cfg.entropy_budget)
        # Persistent per-step noise buffers (BEFORE capture), from block 0's noise stream.
        self._fill_noise_buffers_from(noise_tokens_fn)

        # WARM the persistent trace-write-target buffers by cloning the REAL first-step outputs
        # (exact spec match) and running the in-trace copies once eagerly, else the cold copy
        # compiled inside begin_trace_capture enqueues a host write -> "Writes not supported
        # during trace capture". Use a CLONE of init_canvas so the caller's tensor is untouched.
        adapter.reset_signal_buffer()
        warm_canvas = ttnn.clone(init_canvas)
        t0 = temperature_at_step(0, cfg.max_denoise_steps, cfg.temperature_start, cfg.temperature_end)
        logits0 = adapter(warm_canvas, 0)
        nc0, am0 = denoise_step_next_canvas(
            logits0,
            temperature=t0,
            entropy_budget=cfg.entropy_budget,
            gumbel_noise=None,
            noise_tokens=self.noise_bufs[0],
            constants=self.consts,
        )
        _deallocate_logits_if_unowned(adapter, logits0)
        self.canvas_buf = ttnn.clone(nc0)
        self.committed_buf = ttnn.clone(am0)
        ttnn.copy(nc0, self.canvas_buf)
        ttnn.copy(am0, self.committed_buf)
        nc0.deallocate(True)
        am0.deallocate(True)
        warm_canvas.deallocate(True)
        ttnn.synchronize_device(self.mesh)

        # Capture one single-step trace per step index (each bakes its T[i] + reads noise_bufs[i]).
        adapter.reset_signal_buffer()
        self.traces = []
        for step in range(cfg.max_denoise_steps):
            temperature = temperature_at_step(step, cfg.max_denoise_steps, cfg.temperature_start, cfg.temperature_end)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            logits = adapter(self.canvas_buf, step)
            next_canvas, argmax = denoise_step_next_canvas(
                logits,
                temperature=temperature,
                entropy_budget=cfg.entropy_budget,
                gumbel_noise=None,
                noise_tokens=self.noise_bufs[step],
                constants=self.consts,
            )
            _deallocate_logits_if_unowned(adapter, logits)
            ttnn.copy(next_canvas, self.canvas_buf)
            ttnn.copy(argmax, self.committed_buf)
            next_canvas.deallocate(True)
            argmax.deallocate(True)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces.append(tid)
        ttnn.synchronize_device(self.mesh)
        self.captured = True

    # -- per-block denoise ---------------------------------------------------------
    def denoise_block(
        self, adapter, init_canvas, config: DiffusionConfig, *, gumbel_noise_fn=None, noise_tokens_fn=None
    ) -> DenoiseTrajectory:
        # Argmax regime only: the argmax gumbel hook is a non-None callable whose per-step
        # value is None (temperature-scaled argmax, no gumbel tensor). A hook that yields a
        # real gumbel tensor would need per-step gumbel buffers threaded like the noise, which
        # is not built — fall back by unsetting DG_DENOISE_TRACED for that regime.
        if gumbel_noise_fn is not None:
            probe = gumbel_noise_fn(0)
            if probe is not None:
                if hasattr(probe, "deallocate"):
                    probe.deallocate(True)
                raise NotImplementedError(
                    "traced denoise supports the argmax (gumbel_noise=None) regime only; "
                    "unset DG_DENOISE_TRACED for the gumbel regime"
                )
        if noise_tokens_fn is None:
            raise ValueError("traced denoise requires a per-step noise_tokens_fn")
        cfg = self.config
        start_pos = getattr(adapter, "q_rope_offset", None)

        if not self.captured:
            # Block 0: capture consumes block-0 noise into the persistent buffers.
            self._capture(adapter, init_canvas, noise_tokens_fn, start_pos)
        else:
            # Block N: refresh per-step noise + canvas RoPE (both OUTSIDE the trace).
            self._refresh_noise_buffers_from(noise_tokens_fn)
            adapter.update_canvas_rope_buffers(start_pos)

        # Reset the threaded state from THIS block's fresh canvas init + zeroed signal, replay.
        ttnn.copy(init_canvas, self.canvas_buf)
        if hasattr(init_canvas, "deallocate"):
            init_canvas.deallocate(True)  # consume init_canvas (matches run_fixed_denoise_steps)
        adapter.reset_signal_buffer()
        ttnn.synchronize_device(self.mesh)
        for tid in self.traces:
            ttnn.execute_trace(self.mesh, tid, blocking=False)
        ttnn.synchronize_device(self.mesh)

        committed_host = _ids_to_torch(self.committed_buf)
        return DenoiseTrajectory(committed_host, cfg.max_denoise_steps, False, [])

    def release(self) -> None:
        for tid in self.traces:
            ttnn.release_trace(self.mesh, tid)
        self.traces = []
        for buf in [self.canvas_buf, self.committed_buf, *self.noise_bufs]:
            if buf is not None and hasattr(buf, "deallocate"):
                buf.deallocate(True)
        self.canvas_buf = None
        self.committed_buf = None
        self.noise_bufs = []
        self.captured = False


def traced_denoise_block(
    logits_fn, init_canvas, config: DiffusionConfig, *, gumbel_noise_fn=None, noise_tokens_fn=None
) -> DenoiseTrajectory:
    """``denoise_block``-compatible entry that lazily binds a :class:`TracedDenoiseController`.

    The controller is stateful across blocks, so it is cached on the ``logits_fn`` adapter
    (built once per serving session at prefill and reused for every block)."""
    controller = getattr(logits_fn, "_traced_denoise_controller", None)
    if controller is None:
        mesh_device = logits_fn.tt_model.mesh_device
        controller = TracedDenoiseController(mesh_device, config)
        logits_fn._traced_denoise_controller = controller
    return controller.denoise_block(
        logits_fn, init_canvas, config, gumbel_noise_fn=gumbel_noise_fn, noise_tokens_fn=noise_tokens_fn
    )
