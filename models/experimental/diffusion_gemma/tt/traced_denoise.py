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

MULTI-STEP TRACE BATCHING (``DG_DENOISE_TRACED_MULTISTEP``, path to 100 t/s lever 10). The
single-step controller replays **N single-step traces per block** (one ``execute_trace`` per denoise
step), so the per-block FIXED dispatch overhead scales with the step count: measured
``block(K) ≈ 0.275·K + 1.09 s`` (58.3 t/s @12, 33.3 @24), where the ~1.09 s fixed term is the
per-replay dispatch of the single-step trace paid ``K`` times/block. 100 t/s ⇔ ``block ≤ 2.56 s`` —
only reachable at ``K ≤ ~5`` with single-step replays, below a quality-safe budget.
:class:`MultiStepTracedDenoiseController` captures a **window of ``G`` denoise steps into ONE Metal
trace** (default ``G = max_denoise_steps`` → the whole fixed-K block in ONE capture + ONE replay),
so a block does ``ceil(K/G)`` replays instead of ``K``. That removes ``(K − ceil(K/G))`` per-replay
dispatch bubbles/block, cutting the effective per-step cost so 100 t/s holds at a HIGHER
(quality-safe) step budget rather than only at ~5 steps. ``DG_DENOISE_MULTISTEP_GROUP=G`` caps the
window (bounds trace-region memory when the whole-block capture is too large); unset ⇒ whole block.
It is **bit-exact to the ``K`` single-step replays** (see :class:`MultiStepTracedDenoiseController`).

Lifecycle: the controller is cached on the logits adapter, which a serving session rebuilds per
prompt at prefill — so each prompt gets a fresh controller (the traces bake the prompt's frozen
prefix). A long-lived multi-prompt server should call ``controller.release()`` on the prior
adapter before re-prefill to free its traces/buffers; the single-prompt serving_smoke path does not
need it (the mesh close frees everything).
"""

from __future__ import annotations

import json
import os
from typing import List

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
    denoise_step_next_canvas,
    denoise_step_next_canvas_and_halt,
    eval_halt,
    make_denoise_constants,
    read_halt_scalars,
    temperature_at_step,
    write_halt_scalars,
)


def traced_denoise_enabled() -> bool:
    """True when the traced single-step denoise loop is opted in (``DG_DENOISE_TRACED``)."""
    return os.environ.get("DG_DENOISE_TRACED", "0").lower() in ("1", "true", "yes", "on")


def _trace_metric(event: str, **fields) -> None:
    """Emit a stable, machine-readable live-serving trace marker."""
    logger.info("DG_TRACE_METRIC " + json.dumps({"event": event, **fields}, sort_keys=True, default=str))


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
        self.capture_events = 0
        self.traces_captured = 0
        self.replay_blocks = 0
        self.execute_trace_calls = 0

    def stats(self) -> dict:
        """Return counters used by live-server evidence and request cleanup."""
        return {
            "controller": type(self).__name__,
            "capture_events": self.capture_events,
            "traces_captured": self.traces_captured,
            "replay_blocks": self.replay_blocks,
            "execute_trace_calls": self.execute_trace_calls,
            "trace_ids": [str(tid) for tid in self.traces],
        }

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
        # DG_TERMINAL_SHARDED context (None => replicated terminal). Prepared OUTSIDE capture by
        # the caller's adapter.prepare_sharded_terminal; constant across steps, so derive once.
        sharded_terminal = adapter.sharded_terminal_context()
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
            sharded_terminal=sharded_terminal,
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
                sharded_terminal=sharded_terminal,
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
        self.capture_events += 1
        self.traces_captured += len(self.traces)
        _trace_metric("capture", start_pos=start_pos, **self.stats())

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
        captured_this_block = not self.captured

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
        self.replay_blocks += 1
        self.execute_trace_calls += len(self.traces)
        _trace_metric(
            "replay",
            start_pos=start_pos,
            captured_this_block=captured_this_block,
            steady_replay=not captured_this_block,
            **self.stats(),
        )

        committed_host = _ids_to_torch(self.committed_buf)
        return DenoiseTrajectory(committed_host, cfg.max_denoise_steps, False, [])

    def release(self) -> None:
        final_stats = self.stats()
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
        _trace_metric("release", **final_stats)


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


# --- Multi-step trace batching (path to 100 t/s, lever 10) -------------------------


def traced_denoise_multistep_enabled() -> bool:
    """True when multi-step trace batching is opted in (``DG_DENOISE_TRACED_MULTISTEP``)."""
    return os.environ.get("DG_DENOISE_TRACED_MULTISTEP", "0").lower() in ("1", "true", "yes", "on")


def multistep_group_size(n_steps: int) -> int:
    """Steps captured per Metal trace window (``DG_DENOISE_MULTISTEP_GROUP``, default whole block).

    Unset / non-positive / non-integer ⇒ ``n_steps`` (the whole fixed-K block in ONE capture +
    ONE replay). A positive ``G`` is clamped to ``[1, n_steps]`` and yields ``ceil(n_steps / G)``
    traces + replays per block (a memory knob: a smaller window bounds the per-trace region so the
    capture fits ``DG_TRACE_REGION_SIZE`` when the whole-block trace is too large)."""
    raw = os.environ.get("DG_DENOISE_MULTISTEP_GROUP", "").strip()
    if not raw:
        return n_steps
    try:
        g = int(raw)
    except ValueError:
        return n_steps
    if g <= 0:
        return n_steps
    return min(g, n_steps)


class MultiStepTracedDenoiseController(TracedDenoiseController):
    """Traced denoise loop that captures ``G`` steps per Metal trace instead of one.

    Same persistent-buffer + per-block-refresh architecture as
    :class:`TracedDenoiseController` — it inherits noise-buffer fill/refresh, the block
    replay path (:meth:`TracedDenoiseController.denoise_block`), and :meth:`release`. It
    overrides ONLY :meth:`_capture`: instead of ``N`` single-step traces (one
    ``execute_trace`` per step) it captures ``ceil(N/G)`` window traces (default one
    whole-block trace) and the inherited ``denoise_block`` replays that shorter trace list
    per block — removing ``(N − ceil(N/G))`` per-replay dispatch bubbles/block.

    **Bit-exactness to the ``N`` single-step replays (by inspection).** Both paths commit a
    deterministic function of ``(block init canvas, per-step T[i], per-step noise[i], frozen
    prompt prefix, self-cond signal chain)`` — and every one of those is identical here:

    * ``T[i]`` is baked per step exactly as single-step (``temperature_at_step(i, …)`` — the
      same Python constant folded into the same ``logits/T`` / entropy ops);
    * ``noise[i]`` is the same persistent ``noise_bufs[i]`` (filled from block 0's stream at
      capture, overwritten in place per block by the SAME ``noise_tokens_fn`` order/count as
      the eager ``run_fixed_denoise_steps`` — the seeded stream is untouched);
    * **self-cond threads step-to-step ON DEVICE** through the persistent in-place
      ``signal_buf`` (re-zeroed per block by ``reset_signal_buffer``), so step ``i`` reads
      exactly what step ``i−1`` wrote — WITHIN a window that is a normal read-after-write graph
      edge captured in one trace; ACROSS a window boundary it is the same in-place buffer carry
      the single-step path already uses between every step (an in-order-CQ RAW dependency);
    * the **canvas** threads step-to-step: within a window as a pure graph edge (step ``i+1``
      consumes step ``i``'s ``next_canvas`` intermediate — the proven ``run_fixed_denoise_steps``
      shape); across a window boundary via the persistent ``canvas_buf`` carry (identical to the
      single-step per-step ``ttnn.copy(next_canvas, canvas_buf)``);
    * per-block **canvas RoPE** is refreshed OUTSIDE the trace into the constant-shape buffers
      (RoPE depends only on absolute position ⇒ bit-identical to the growing-slice path), and
      the whole threaded state is reset per block (fresh init canvas + zeroed signal).

    So the committed clean-argmax of the final step is byte-identical to both the single-step
    traced loop (verified ``CROSSBLOCK_OK`` 100%) and the eager ``run_fixed_denoise_steps``.
    The lone functional change is grouping ``execute_trace`` calls; no committed *decision*
    moves. The earlier "whole-loop trace" divergence (60.5% match,
    ``probe_traced_denoise_loop.py``) was root-caused (perf_progress.md session 8) as a probe
    bug — a cross-replay buffer allocated AFTER ``begin_trace_capture`` and clobbered by trace
    scratch — NOT a self-cond race; this controller allocates EVERY persistent cross-replay
    buffer (canvas / committed / signal / rope / per-step noise) BEFORE capture, which is what
    makes the multi-step (incl. whole-block) trace bit-exact.

    **Memory.** Each window trace records ``G`` steps of commands, but the heavy per-step
    intermediates (the [1,1,C,262144] logits + forward activations) are deallocated between
    steps inside the capture, so the window's PEAK intermediate footprint is ~1 step's — a
    whole-block window is expected to need roughly ONE step's scratch plus the N-step command
    stream, i.e. NOT ``N×`` the single-step trace (which reserves independent per-trace scratch
    ``N`` times). Confirm the actual ``DG_TRACE_REGION_SIZE`` on device via
    ``doc/optimize_perf/bench_multistep_trace.py``; use ``DG_DENOISE_MULTISTEP_GROUP`` to shrink
    the window if a whole-block capture overflows the region.
    """

    def __init__(self, mesh_device, config: DiffusionConfig, consts=None, group_size: int | None = None):
        super().__init__(mesh_device, config, consts=consts)
        self.group_size = group_size if group_size is not None else multistep_group_size(config.max_denoise_steps)

    def _capture(self, adapter, init_canvas, noise_tokens_fn, start_pos: int) -> None:
        cfg = self.config
        canvas_len = cfg.canvas_length
        n_steps = cfg.max_denoise_steps
        g = max(1, min(self.group_size, n_steps))
        self.group_size = g
        # All persistent cross-replay state allocated BEFORE begin_trace_capture (session-8 rule):
        # self-cond signal buffer + constant-shape canvas RoPE buffers + per-step noise buffers.
        adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_len)
        adapter.prepare_canvas_rope_buffers(canvas_len=canvas_len)
        adapter.update_canvas_rope_buffers(start_pos)
        adapter.use_canvas_rope = True
        # DG_TERMINAL_SHARDED context (None => replicated terminal), constant across steps/windows.
        sharded_terminal = adapter.sharded_terminal_context()
        if self.consts is None:
            self.consts = make_denoise_constants(self.mesh, batch=1, canvas_len=canvas_len, budget=cfg.entropy_budget)
        self._fill_noise_buffers_from(noise_tokens_fn)

        # WARM the program cache + the persistent write-target buffers (the window-boundary
        # canvas carry and the committed readback) by cloning the REAL first-step outputs (exact
        # spec match) and running the in-trace copies ONCE eagerly — else the cold copy compiled
        # inside begin_trace_capture enqueues a host write ("Writes not supported during trace
        # capture"). Every window runs the SAME forward+decision graph (T[i]/noise[i] are runtime
        # args of one warmed program set), so a single eager step warms all G-step windows.
        adapter.reset_signal_buffer()
        warm_canvas = ttnn.clone(init_canvas)
        t0 = temperature_at_step(0, n_steps, cfg.temperature_start, cfg.temperature_end)
        logits0 = adapter(warm_canvas, 0)
        nc0, am0 = denoise_step_next_canvas(
            logits0,
            temperature=t0,
            entropy_budget=cfg.entropy_budget,
            gumbel_noise=None,
            noise_tokens=self.noise_bufs[0],
            constants=self.consts,
            sharded_terminal=sharded_terminal,
        )
        _deallocate_logits_if_unowned(adapter, logits0)
        self.canvas_buf = ttnn.clone(nc0)  # step-0 input / cross-window carry (both [1,1,L,1] uint32)
        self.committed_buf = ttnn.clone(am0)  # final clean-argmax readback target
        ttnn.copy(nc0, self.canvas_buf)  # warm the canvas carry copy
        ttnn.copy(am0, self.committed_buf)  # warm the committed readback copy
        nc0.deallocate(True)
        am0.deallocate(True)
        warm_canvas.deallocate(True)
        ttnn.synchronize_device(self.mesh)

        # Capture one trace per G-step window. Window w reads canvas_buf as its step-0 input,
        # threads the intra-window canvas as pure graph edges (freeing each step's intermediates),
        # then writes its end canvas back into canvas_buf (carry to window w+1) and the end
        # committed argmax into committed_buf. After replaying all windows in order, committed_buf
        # holds the LAST window's final argmax = the block commit.
        adapter.reset_signal_buffer()
        self.traces = []
        for w_start in range(0, n_steps, g):
            w_end = min(w_start + g, n_steps)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            canvas = self.canvas_buf
            committed = None
            for step in range(w_start, w_end):
                temperature = temperature_at_step(step, n_steps, cfg.temperature_start, cfg.temperature_end)
                logits = adapter(canvas, step)
                next_canvas, argmax = denoise_step_next_canvas(
                    logits,
                    temperature=temperature,
                    entropy_budget=cfg.entropy_budget,
                    gumbel_noise=None,
                    noise_tokens=self.noise_bufs[step],
                    constants=self.consts,
                    sharded_terminal=sharded_terminal,
                )
                _deallocate_logits_if_unowned(adapter, logits)
                if committed is not None:
                    committed.deallocate(True)
                committed = argmax
                if canvas is not self.canvas_buf:
                    canvas.deallocate(True)  # free the superseded intra-window intermediate
                canvas = next_canvas
            ttnn.copy(canvas, self.canvas_buf)  # window-end canvas -> carry buffer for the next window
            ttnn.copy(committed, self.committed_buf)  # window-end committed argmax
            if canvas is not self.canvas_buf:
                canvas.deallocate(True)
            committed.deallocate(True)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces.append(tid)
        ttnn.synchronize_device(self.mesh)
        self.captured = True


def traced_denoise_multistep_block(
    logits_fn, init_canvas, config: DiffusionConfig, *, gumbel_noise_fn=None, noise_tokens_fn=None
) -> DenoiseTrajectory:
    """``denoise_block``-compatible entry that lazily binds a :class:`MultiStepTracedDenoiseController`.

    Cached on the ``logits_fn`` adapter under a distinct attribute from the single-step
    controller, so the two traced paths never share captured traces (the group sizes differ)."""
    controller = getattr(logits_fn, "_traced_denoise_multistep_controller", None)
    if controller is None:
        mesh_device = logits_fn.tt_model.mesh_device
        controller = MultiStepTracedDenoiseController(mesh_device, config)
        logits_fn._traced_denoise_multistep_controller = controller
    return controller.denoise_block(
        logits_fn, init_canvas, config, gumbel_noise_fn=gumbel_noise_fn, noise_tokens_fn=noise_tokens_fn
    )


# --- Data-dependent EARLY-HALT (path to 100 t/s, lever 8) --------------------------
#
# The traced loops above run a FIXED step budget: a static Metal trace fixes the step
# count, and early-halt is data-dependent, so the trace-safe fixed-budget shape lost the
# eager path's observed early halt. This controller RECOVERS early-halt while KEEPING the
# traced dispatch savings by NOT tracing the whole variable-length loop. It captures a
# fixed K-step WINDOW (K==1 ⇒ scheme A per-step; K>1 ⇒ scheme B chunked-halt), replays one
# window at a time, and after each replay reads ONE tiny on-device halt scalar
# (mean_entropy + argmax-stability mismatch — :func:`~...tt.denoise_loop.write_halt_scalars`)
# and branches continue/stop on the HOST. The only host interaction per window is the
# 2-scalar (8-byte-logical) read + the branch — NOT the retired 5-tensor/step readback that
# ``bench_loop_readback.py`` measured at 27.76 ms/step.
#
# Scheme A (K=1) halts at the EXACT eager step and commits the bit-identical argmax (the
# halt scalar is a read-only side computation over the same per-step argmax/entropy the
# fixed-48 path commits). Scheme B (K>1) halts at the first window boundary at-or-after the
# eager step; its commit equals the eager commit under argmax-convergence-stability (the same
# property the fixed-48 traced path already relies on to match eager). Both are byte-identical
# to the fixed-48 traced path when early-halt does NOT fire (Guard 1) — the ONLY difference
# from the multi-step traced loop is the per-window host sync+read+branch.


def traced_early_halt_enabled() -> bool:
    """True when the traced early-halt loop is opted in (``DG_DENOISE_EARLY_HALT``)."""
    return os.environ.get("DG_DENOISE_EARLY_HALT", "0").lower() in ("1", "true", "yes", "on")


def early_halt_window(default: int = 1) -> int:
    """Halt-check window K (``DG_DENOISE_EARLY_HALT_WINDOW``): 1 ⇒ scheme A, K>1 ⇒ scheme B.

    Scheme A checks the halt scalar every step (finest granularity, one host sync/step);
    scheme B checks once per K-step window (coarser halt, K× fewer host syncs). Unset /
    non-positive / non-integer ⇒ ``default`` (scheme A). The value is later clamped to
    ``[1, max_denoise_steps]`` by the window capture.
    """
    raw = os.environ.get("DG_DENOISE_EARLY_HALT_WINDOW", "").strip()
    if not raw:
        return default
    try:
        k = int(raw)
    except ValueError:
        return default
    return k if k > 0 else default


class EarlyHaltTracedDenoiseController(MultiStepTracedDenoiseController):
    """Traced denoise loop with data-dependent early-halt (scheme A K=1 / scheme B K>1).

    Extends :class:`MultiStepTracedDenoiseController`: same persistent-buffer +
    per-block-refresh architecture and the same K-step window capture, plus (a) a persistent
    :class:`~...tt.denoise_loop.HaltBuffers` written by EVERY step inside the trace (so at a
    window boundary they hold the window's last step's scalars) and (b) a
    :meth:`denoise_block` that syncs + reads the halt scalar + branches after EACH window
    replay instead of replaying the whole fixed budget unconditionally.

    ``group_size`` is the halt-check window K (1 = scheme A). The in-trace per-step halt-scalar
    ops are identical for A and B and are ~sub-ms (256-wide reductions), so A and B run the
    same per-step device compute as the fixed budget; they differ only in host sync cadence.
    Only ``stable_steps_to_halt == 1`` (the released config) is supported.
    """

    def __init__(self, mesh_device, config: DiffusionConfig, consts=None, group_size: int | None = None):
        super().__init__(mesh_device, config, consts=consts, group_size=group_size)
        self.halt_bufs: HaltBuffers | None = None
        # Per-window (w_end_step_count, mean_entropy, mismatch) for the LAST block replayed —
        # diagnostics for the verify harness and the realized halt-step distribution.
        self.last_halt_trace: list = []
        if config.stable_steps_to_halt != 1:
            raise NotImplementedError(
                "traced early-halt supports stable_steps_to_halt == 1 (released config); "
                f"got {config.stable_steps_to_halt}"
            )

    def _release_halt_bufs(self) -> None:
        if self.halt_bufs is not None:
            for buf in self.halt_bufs:
                if buf is not None and hasattr(buf, "deallocate"):
                    buf.deallocate(True)
            self.halt_bufs = None

    def release(self) -> None:
        self._release_halt_bufs()
        super().release()

    def _capture(self, adapter, init_canvas, noise_tokens_fn, start_pos: int) -> None:
        cfg = self.config
        canvas_len = cfg.canvas_length
        n_steps = cfg.max_denoise_steps
        g = max(1, min(self.group_size, n_steps))
        self.group_size = g
        # All persistent cross-replay state allocated BEFORE begin_trace_capture (session-8 rule):
        # self-cond signal + canvas RoPE buffers + per-step noise + the halt buffers.
        adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_len)
        adapter.prepare_canvas_rope_buffers(canvas_len=canvas_len)
        adapter.update_canvas_rope_buffers(start_pos)
        adapter.use_canvas_rope = True
        # DG_TERMINAL_SHARDED context (None => replicated terminal), constant across steps/windows.
        sharded_terminal = adapter.sharded_terminal_context()
        if self.consts is None:
            self.consts = make_denoise_constants(self.mesh, batch=1, canvas_len=canvas_len, budget=cfg.entropy_budget)
        self._fill_noise_buffers_from(noise_tokens_fn)

        # WARM the program cache + persistent write-target buffers (canvas carry, committed
        # readback, AND the three halt buffers) by running one REAL eager step and its in-trace
        # copies once — else the cold copy compiled inside begin_trace_capture enqueues a host
        # write ("Writes not supported during trace capture").
        adapter.reset_signal_buffer()
        warm_canvas = ttnn.clone(init_canvas)
        t0 = temperature_at_step(0, n_steps, cfg.temperature_start, cfg.temperature_end)
        logits0 = adapter(warm_canvas, 0)
        res0 = denoise_step(
            logits0,
            temperature=t0,
            entropy_budget=cfg.entropy_budget,
            gumbel_noise=None,
            noise_tokens=self.noise_bufs[0],
            constants=self.consts,
            sharded_terminal=sharded_terminal,
        )
        _deallocate_logits_if_unowned(adapter, logits0)
        self.canvas_buf = ttnn.clone(res0.canvas)
        self.committed_buf = ttnn.clone(res0.argmax)
        # Halt buffers cloned from the real first-step scalars (exact spec match). prev_argmax
        # is this step's argmax as TILE fp32; mean_entropy/mismatch are [1,1,1,1] fp32 scalars.
        cur_af0 = _argmax_to_tile_f32(res0.argmax)
        prev_argmax_buf = ttnn.clone(cur_af0)
        mean0, mm0, af0 = compute_halt_scalars(res0.argmax, res0.entropy, prev_argmax_buf, canvas_len=canvas_len)
        self.halt_bufs = HaltBuffers(
            prev_argmax=prev_argmax_buf,
            mean_entropy=ttnn.clone(mean0),
            mismatch=ttnn.clone(mm0),
        )
        # Warm the in-trace copies once eagerly.
        ttnn.copy(res0.canvas, self.canvas_buf)
        ttnn.copy(res0.argmax, self.committed_buf)
        write_halt_scalars(res0.argmax, res0.entropy, self.halt_bufs, canvas_len=canvas_len)
        for tmp in (cur_af0, mean0, mm0, af0, res0.canvas, res0.argmax, res0.accept_mask, res0.entropy, res0.sampled):
            if tmp is not None and hasattr(tmp, "deallocate"):
                tmp.deallocate(True)
        warm_canvas.deallocate(True)
        ttnn.synchronize_device(self.mesh)

        # Capture one trace per G-step window. Each step writes the halt buffers (overwriting),
        # so after a window trace they hold the window's LAST step's scalars (mean_entropy of
        # the last step + argmax mismatch of the last step vs its predecessor). The canvas +
        # committed argmax thread exactly as the multi-step base (window-end carry).
        adapter.reset_signal_buffer()
        self.traces = []
        for w_start in range(0, n_steps, g):
            w_end = min(w_start + g, n_steps)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            canvas = self.canvas_buf
            committed = None
            for step in range(w_start, w_end):
                temperature = temperature_at_step(step, n_steps, cfg.temperature_start, cfg.temperature_end)
                logits = adapter(canvas, step)
                next_canvas, argmax = denoise_step_next_canvas_and_halt(
                    logits,
                    temperature=temperature,
                    entropy_budget=cfg.entropy_budget,
                    gumbel_noise=None,
                    noise_tokens=self.noise_bufs[step],
                    halt_bufs=self.halt_bufs,
                    canvas_len=canvas_len,
                    constants=self.consts,
                    sharded_terminal=sharded_terminal,
                )
                _deallocate_logits_if_unowned(adapter, logits)
                if committed is not None:
                    committed.deallocate(True)
                committed = argmax
                if canvas is not self.canvas_buf:
                    canvas.deallocate(True)
                canvas = next_canvas
            ttnn.copy(canvas, self.canvas_buf)
            ttnn.copy(committed, self.committed_buf)
            if canvas is not self.canvas_buf:
                canvas.deallocate(True)
            committed.deallocate(True)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces.append(tid)
        ttnn.synchronize_device(self.mesh)
        self.captured = True

    def denoise_block(
        self, adapter, init_canvas, config: DiffusionConfig, *, gumbel_noise_fn=None, noise_tokens_fn=None
    ) -> DenoiseTrajectory:
        # Argmax regime only (same prerequisite as the traced base): the argmax gumbel hook
        # yields per-step None. A real gumbel tensor would need per-step gumbel buffers threaded
        # like the noise, which is not built.
        if gumbel_noise_fn is not None:
            probe = gumbel_noise_fn(0)
            if probe is not None:
                if hasattr(probe, "deallocate"):
                    probe.deallocate(True)
                raise NotImplementedError(
                    "traced early-halt supports the argmax (gumbel_noise=None) regime only; "
                    "unset DG_DENOISE_EARLY_HALT for the gumbel regime"
                )
        if noise_tokens_fn is None:
            raise ValueError("traced early-halt requires a per-step noise_tokens_fn")
        cfg = self.config
        n_stable = cfg.stable_steps_to_halt
        threshold = cfg.entropy_stop_threshold
        n_steps = cfg.max_denoise_steps
        start_pos = getattr(adapter, "q_rope_offset", None)

        if not self.captured:
            self._capture(adapter, init_canvas, noise_tokens_fn, start_pos)
        else:
            self._refresh_noise_buffers_from(noise_tokens_fn)
            adapter.update_canvas_rope_buffers(start_pos)

        # Reset threaded state from this block's fresh canvas init + zeroed self-cond signal.
        # halt_bufs.prev_argmax is intentionally NOT reset per block: step 0 (the only step that
        # would read a stale cross-block prev) is never a halt candidate (eval_halt requires
        # step >= n_stable == 1), and step 0 overwrites prev_argmax with this block's argmax_0
        # before step 1 reads it — so the carried value only ever feeds the never-evaluated
        # step-0 mismatch.
        ttnn.copy(init_canvas, self.canvas_buf)
        if hasattr(init_canvas, "deallocate"):
            init_canvas.deallocate(True)
        adapter.reset_signal_buffer()
        ttnn.synchronize_device(self.mesh)

        # Replay one window at a time; after each, read the tiny halt scalar and branch. The
        # window's LAST step index (0-indexed) is w_end - 1; halting there means w_end steps ran.
        g = self.group_size
        self.last_halt_trace = []
        halted = False
        steps_run = 0
        for w, tid in enumerate(self.traces):
            w_end = min((w + 1) * g, n_steps)
            ttnn.execute_trace(self.mesh, tid, blocking=False)
            ttnn.synchronize_device(self.mesh)
            steps_run = w_end
            mean_entropy, mismatch = read_halt_scalars(self.halt_bufs)
            self.last_halt_trace.append((w_end, mean_entropy, mismatch))
            if eval_halt(mean_entropy, mismatch, w_end - 1, threshold=threshold, n_stable=n_stable):
                halted = True
                break

        committed_host = _ids_to_torch(self.committed_buf)
        return DenoiseTrajectory(committed_host, steps_run, halted, [])


def traced_early_halt_block(
    logits_fn, init_canvas, config: DiffusionConfig, *, gumbel_noise_fn=None, noise_tokens_fn=None
) -> DenoiseTrajectory:
    """``denoise_block``-compatible entry that lazily binds an :class:`EarlyHaltTracedDenoiseController`.

    Cached on the ``logits_fn`` adapter under a distinct attribute from the fixed-budget traced
    controllers (its per-window host branch makes it a different captured/replay shape). The
    halt-check window K is read from ``DG_DENOISE_EARLY_HALT_WINDOW`` (1 = scheme A)."""
    controller = getattr(logits_fn, "_traced_early_halt_controller", None)
    if controller is None:
        mesh_device = logits_fn.tt_model.mesh_device
        controller = EarlyHaltTracedDenoiseController(mesh_device, config, group_size=early_halt_window())
        logits_fn._traced_early_halt_controller = controller
    return controller.denoise_block(
        logits_fn, init_canvas, config, gumbel_noise_fn=gumbel_noise_fn, noise_tokens_fn=noise_tokens_fn
    )
