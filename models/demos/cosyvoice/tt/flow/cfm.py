# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ConditionalCFM: the flow-matching ODE solver that drives the estimator.

Ten forward-Euler steps on a cosine-spaced grid, each evaluating
`TtConditionalDecoder` on a batch of 2 for classifier-free guidance:

    dphi_dt = (1 + w) * conditioned - w * unconditioned      w = 0.7
    x       = x + dt * dphi_dt

Two details that are easy to get wrong and produce plausible-but-wrong audio:

**The grid is cosine, not linear.** `t_span = 1 - cos(linspace(0,1,11) * pi/2)`,
so the steps start dense near t=0 and widen. A linear grid still integrates to
something, just not this model's trajectory.

**`dt` is recomputed from the grid, not held fixed.** The reference sets `dt`
once before the loop and then updates it at the *end* of each iteration with
`dt = t_span[step + 1] - t`, where `t` has already advanced. Reading that as a
constant `1/n` gives the right first step and drifts thereafter.

**The noise is injected, never drawn.** `x0` comes from the caller. Seeding cannot
align a device RNG with torch's stream, so the initial `z` is captured from the
reference as a golden array and passed in -- the same rule the vocoder's source
module follows. `ConditionalCFM.forward` draws it as `randn_like(mu) * temperature`.
"""
from __future__ import annotations

import math
import os

import torch
from loguru import logger

import ttnn

from .estimator import TtConditionalDecoder


def cosine_t_span(n_timesteps: int = 10, scheduler: str = "cosine") -> torch.Tensor:
    """The `n_timesteps + 1` node positions the solver steps between."""
    span = torch.linspace(0, 1, n_timesteps + 1, dtype=torch.float32)
    return 1 - torch.cos(span * 0.5 * math.pi) if scheduler == "cosine" else span


def euler_steps(t_span: torch.Tensor) -> list[tuple[float, float]]:
    """`(t, dt)` per step, reproducing the reference's update order exactly.

    Returned as plain floats so the schedule can be asserted on the host without a
    device, and so the solver does no host<->device round trip mid-loop.
    """
    t = float(t_span[0])
    dt = float(t_span[1] - t_span[0])
    out = []
    for step in range(1, len(t_span)):
        out.append((t, dt))
        t = t + dt
        if step < len(t_span) - 1:
            dt = float(t_span[step + 1]) - t
    return out


class TtConditionalCFM:
    """The solver. Tensors are channels-last `[B, T, 80]`, batch 1 outside the
    estimator and 2 inside it."""

    def __init__(
        self,
        device,
        bag,
        *,
        inference_cfg_rate: float = 0.7,
        n_timesteps: int = 10,
        t_scheduler: str = "cosine",
        dtype=ttnn.bfloat16,
    ):
        self.device, self.dtype = device, dtype
        self.cfg_rate = inference_cfg_rate
        # `COSYVOICE_FLOW_STEPS` overrides the checkpoint's solver depth. This is the
        # one knob in the port that trades **accuracy for time by construction** rather
        # than by numerical accident, so it is an explicit environment variable and not
        # a default: 10 is what the checkpoint ships with and what every accuracy figure
        # in PERF.md is measured at.
        #
        # It exists because the flow decoder is the largest single stage after the LLM
        # and its cost is linear in this number, and because the issue asks for exactly
        # this experiment -- "optimize iterative refinement process / consider
        # approximations for faster inference". Whether the approximation is acceptable
        # is a measured question, not a matter of taste; `probe_flow_steps.py` measures
        # it against the shipped 10-step result.
        self.n_timesteps = int(os.environ.get("COSYVOICE_FLOW_STEPS", n_timesteps))
        self.t_scheduler = t_scheduler
        self.estimator = TtConditionalDecoder(device, bag.sub("estimator"), dtype=dtype)
        # **Keep the captured trace across utterances of the same mel length.**
        #
        # `solve_euler` captured and released on every call, so a stage measured at
        # 0.675 s spent 0.314 s of it -- 46.6 % -- recording a graph it then threw
        # away. That is why cutting `n_timesteps` from 10 to 5 buys 1.43x and not 2x:
        # half the stage is not the ODE.
        #
        # Reuse needs one thing to be true: everything the trace bakes an address for
        # must be refillable in place. `_x_buf` already is. `_packed_const` is the
        # utterance's conditioning, so it changes per call -- but its *shape* depends
        # only on the mel length, so the trace stays valid if the contents are copied
        # in rather than reallocated. That is the whole mechanism, and the mel length
        # is therefore the cache key.
        #
        # One slot, not a dict: each entry pins a trace region allocation plus its
        # buffers, and TTS lengths vary continuously, so an unbounded cache would grow
        # for the length of a session and hit twice by luck.
        self._cache_trace = os.environ.get("COSYVOICE_CFM_TRACE_CACHE", "1") != "0"
        self._trace_key = None

    def _cfg_pair(self, x, zero_second_row: bool):
        """Stack a tensor into the 2-row CFG batch.

        Row 0 is the conditioned copy. Row 1 is either the same tensor (for `x`,
        which both rows evaluate) or zeros (for `mu`, `spks` and `cond`, which is
        what "unconditioned" means here).
        """
        if not zero_second_row:
            return ttnn.concat([x, x], dim=0)
        zeros = ttnn.zeros(x.shape, dtype=x.dtype, layout=x.layout, device=self.device)
        out = ttnn.concat([x, zeros], dim=0)
        ttnn.deallocate(zeros)
        return out

    def _reuse(self, x, mu2, spks2, cond2, t_len) -> bool:
        """Refill the cached trace's inputs in place. `True` if it was usable.

        The conditioning is copied into `_packed_const` rather than reassigned,
        because the trace holds that buffer's *address*. Reassigning the attribute
        would leave the replay reading the previous utterance's conditioning and
        produce fluent audio in the wrong voice -- a failure with no exception and no
        shape mismatch, which is the reason this is one method with one comment
        rather than three lines inlined at the call site.
        """
        if not self._cache_trace or self._trace_key != (t_len, int(x.shape[2])):
            return False
        if getattr(self, "_trace_id", None) is None or self._packed_const is None:
            return False
        packed = self.estimator.pack_const(mu2, spks2, cond2, t_len)
        ttnn.copy(packed, self._packed_const)
        ttnn.deallocate(packed)
        self._fill_x(x)
        return True

    def _capture(self, x, mu2, spks2, cond2, t0, dt0):
        """Trace one estimator evaluation and replay it for every Euler step.

        The solver calls the *same graph* ten times -- only `x` and `t` change,
        while `mu`, `spks` and `cond` are fixed for the utterance. That makes it a
        better trace candidate than the AR decoder, which at least has a growing
        cache to work around: here there is no state at all between steps.

        The CFG split and the Euler update stay outside the trace. They are a
        handful of elementwise ops on `[1, T, 80]`, and keeping them out means the
        traced region is exactly the 16 resnet + 64 transformer blocks that cost
        something.

        **Getting here took removing a host->device write the convolutions were
        issuing on every call.** Capture originally failed with
        `!trace_id_.has_value()` at `fd_mesh_command_queue.cpp:762`, and it was
        tempting to read that as "conv is not trace-compatible on this stack". It is
        not a silicon limit. `ttnn.conv1d` and `ttnn.conv_transpose2d` prepare their
        weights -- tilize, pad to the sharding scheme, move to device -- *on every
        call*, and that is host work a trace cannot contain. A host-resident weight
        fails on the write; a device-resident one fails on the **read back** at
        `:809`. Weight residency is therefore not the fix.

        `ttnn.prepare_conv_weights` / `prepare_conv_transpose2d_weights` hoist the
        transform out of the op, and both conv wrappers now cache the prepared
        weights per input geometry (the sharding scheme follows the input length, so
        geometry is the cache key). Output is bit-identical, `max|d| 0.000e+00`.

        One trap on the transpose side: `prepare_conv_transpose2d_weights` asserts
        `conv_config.weights_dtype.has_value()`, while the bare `conv_transpose2d`
        call is happy with no config at all. Omitting it makes preparation throw,
        the wrapper falls back to the unprepared path, and the only symptom is a
        trace that still fails -- several ops downstream, for no visible reason.
        That is why the fallback logs.
        """
        _, t_len, ch = x.shape
        # The buffer holds a single row and the CFG doubling happens *inside* the
        # traced body. That ordering is not cosmetic. The first working version kept
        # a `[2, T, 80]` buffer and refreshed it with
        # `ttnn.copy(ttnn.concat([x, x], dim=0), buf)`; capture succeeded and replay
        # scored **PCC 0.077**. `scripts/probe_cfm_trace.py` pinned it down: with a
        # plain device tensor as the source, replay is bit-exact (PCC 1.00000000,
        # for both the `x` and the `t` refresh); with a dim-0 `concat` output as the
        # source, the same copy lands at **PCC 0.768**. So `ttnn.copy` does not
        # faithfully transfer out of a dim-0 concat here.
        #
        # Moving the concat inside the trace sidesteps it entirely -- it becomes a
        # device op the trace records -- and the refresh source is then the solver's
        # own `x`, which is the case the probe proves exact. Both inputs are also
        # allocated explicitly in DRAM rather than inheriting a memory config from
        # whatever op produced them, since a trace bakes in addresses.
        self._x_buf = ttnn.from_torch(
            torch.zeros(1, t_len, ch),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._t_buf = ttnn.from_torch(
            torch.full((2, 1, 1), t0, dtype=torch.float32),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # `dt` varies per step, so it is a device tensor rather than the Python
        # float it was when the update lived on the host -- otherwise its value
        # would be baked into the trace and every step would use the first one.
        self._dt_buf = ttnn.from_torch(
            torch.full((1, 1, 1), dt0, dtype=torch.float32),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._fill_x(x)

        # `mu`, `spks` and `cond` are fixed for the utterance, so the estimator's input
        # assembly -- broadcasting `spks` over time and concatenating the three constant
        # blocks -- is loop-invariant across all ten Euler steps. Built here, before
        # capture, so the traced body is left with one two-way concat against `x`.
        self._packed_const = self.estimator.pack_const(mu2, spks2, cond2, t_len)

        def body():
            """One complete Euler step: CFG pair, estimator, guidance, update."""
            x2 = ttnn.concat([self._x_buf, self._x_buf], dim=0)
            d = self.estimator(x2, None, self._t_buf, batch=2, packed_const=self._packed_const)
            ttnn.deallocate(x2)
            c = ttnn.slice(d, [0, 0, 0], [1, t_len, ch])
            u = ttnn.slice(d, [1, 0, 0], [2, t_len, ch])
            ttnn.deallocate(d)
            guided = ttnn.subtract(ttnn.multiply(c, 1.0 + self.cfg_rate), ttnn.multiply(u, self.cfg_rate))
            ttnn.deallocate(c)
            ttnn.deallocate(u)
            step = ttnn.multiply(guided, self._dt_buf)
            ttnn.deallocate(guided)
            nxt = ttnn.add(self._x_buf, step)
            ttnn.deallocate(step)
            return nxt

        # Warm the program cache *and* the conv weight-preparation caches. Both have
        # to be populated before recording: a JIT compile or a weight tilize during
        # capture is host work, and that is precisely what a trace cannot contain.
        for _ in range(2):
            ttnn.deallocate(body())
        ttnn.synchronize_device(self.device)

        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            # The output is allocated *inside* the capture, so its address is baked
            # into the trace and every replay writes to this exact tensor.
            #
            # The first attempt instead pre-allocated a `[2, T, 80]` DRAM buffer and
            # ended the body with `ttnn.copy(d, buf)`. That captured cleanly and
            # replayed to **PCC 0.0017**: the copy never landed, so the solver read
            # back zeros, `x` never advanced, and the "mel" it returned was the
            # initial noise. Silent because the same copy is in the warm-up, so
            # nothing failed loudly at capture time -- the only signal was an output
            # that scored like noise, which is exactly what it was. Letting the trace
            # own its output removes the copy and the question with it.
            self._next_x = body()
        finally:
            ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)
        self._trace_key = (t_len, ch)

    def _fill_x(self, x):
        ttnn.copy(x, self._x_buf)

    def _release(self):
        if getattr(self, "_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
        self._trace_key = None
        # `_next_x` is allocated inside the capture, so it belongs to the trace
        # region; `release_trace` reclaims it and deallocating it here would be a
        # double free. Dropping the reference is all that is wanted.
        self._next_x = None
        # `_packed_const` is allocated *outside* the capture, so it is ours to free --
        # unlike `_next_x`, and after `release_trace` so nothing is reading it.
        for name in ("_x_buf", "_t_buf", "_dt_buf", "_packed_const"):
            t = getattr(self, name, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, name, None)

    def solve_euler(self, x, mu, spks, cond, t_span=None, use_trace=True):
        """x/mu/cond `[1, T, 80]`, spks `[1, 1, 80]` -> `[1, T, 80]`.

        `mu`, `spks` and `cond` do not change across steps, so their CFG pairs are
        built once. The reference refills them every iteration from the same
        source, which is the same arithmetic -- it is reusing preallocated buffers,
        not recomputing anything.
        """
        t_span = cosine_t_span(self.n_timesteps, self.t_scheduler) if t_span is None else t_span
        schedule = euler_steps(t_span)
        t_len = x.shape[1]

        mu2 = self._cfg_pair(mu, zero_second_row=True)
        spks2 = self._cfg_pair(spks, zero_second_row=True)
        cond2 = self._cfg_pair(cond, zero_second_row=True)
        # All step times uploaded up front: one H2D per step would put a host
        # round trip in the middle of the ODE for a 3-element tensor.
        ts = [
            ttnn.from_torch(
                torch.full((2, 1, 1), t, dtype=torch.float32),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for t, _ in schedule
        ]

        dts = [
            ttnn.from_torch(
                torch.full((1, 1, 1), dt, dtype=torch.float32),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _, dt in schedule
        ]

        traced = False
        if use_trace:
            try:
                if not self._reuse(x, mu2, spks2, cond2, t_len):
                    self._release()  # a stale trace of a different length must go first
                    self._capture(x, mu2, spks2, cond2, schedule[0][0], schedule[0][1])
                traced = True
            except Exception as e:  # noqa: BLE001
                # Needs the device opened with a `trace_region_size`. Tracing is an
                # optimisation, so fall back loudly rather than failing the solve.
                logger.warning(f"CFM trace capture unavailable, running untraced: {e}")
                self._release()

        if traced:
            # The traced body is a *whole* Euler step, so the loop allocates
            # nothing. That is the point, not a tidiness win: the earlier version
            # kept the guidance and the update on the host, and a dozen allocate /
            # free pairs per step let the allocator hand back addresses the replay
            # had already baked in. It scored PCC ~0.07 with a graph the probe
            # showed to be bit-exact.
            ttnn.deallocate(x)
            for t_dev, dt_dev in zip(ts, dts):
                ttnn.copy(t_dev, self._t_buf)
                ttnn.copy(dt_dev, self._dt_buf)
                ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
                ttnn.copy(self._next_x, self._x_buf)
            if self._cache_trace:
                # The trace is being kept, so `_x_buf` must be kept too -- it is the
                # buffer the replay writes through. Copy the result out instead of
                # handing the buffer over, which is the one place caching costs
                # something: a full-tensor copy against the 0.31 s it saves.
                x = ttnn.clone(self._x_buf)
            else:
                # Hand the buffer to the caller instead of copying out of it, and keep
                # `_release` from freeing what the caller now owns.
                x = self._x_buf
                self._x_buf = None
        else:
            for (_t_val, dt), t_dev in zip(schedule, ts):
                x2 = self._cfg_pair(x, zero_second_row=False)
                d = self.estimator(x2, mu2, t_dev, spks=spks2, cond=cond2, batch=2)
                ttnn.deallocate(x2)

                cond_part = ttnn.slice(d, [0, 0, 0], [1, t_len, 80])
                uncond_part = ttnn.slice(d, [1, 0, 0], [2, t_len, 80])
                ttnn.deallocate(d)
                guided = ttnn.subtract(
                    ttnn.multiply(cond_part, 1.0 + self.cfg_rate),
                    ttnn.multiply(uncond_part, self.cfg_rate),
                )
                ttnn.deallocate(cond_part)
                ttnn.deallocate(uncond_part)

                step = ttnn.multiply(guided, dt)
                ttnn.deallocate(guided)
                nxt = ttnn.add(x, step)
                ttnn.deallocate(step)
                ttnn.deallocate(x)
                x = nxt

        if traced and not self._cache_trace:
            self._release()

        for t_dev in ts:
            ttnn.deallocate(t_dev)
        for dt_dev in dts:
            ttnn.deallocate(dt_dev)
        ttnn.deallocate(mu2)
        ttnn.deallocate(spks2)
        ttnn.deallocate(cond2)
        return x

    # -- host reference, for separating "solver wrong" from "estimator drifted" --
    @staticmethod
    def torch_solve_euler(estimator_fn, x, mu, spks, cond, t_span, cfg_rate=0.7):
        """The same loop in torch, on `[1, 80, T]` channel-first tensors.

        `estimator_fn(x2, mu2, t2, spks2, cond2)` stands in for the network, so
        this can be driven by `tt/flow/reference.py` on the host or by the captured
        per-step outputs.
        """
        for (t, dt), _ in zip(euler_steps(t_span), range(len(t_span) - 1)):
            x2 = torch.cat([x, x], dim=0)
            mu2 = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            spks2 = torch.cat([spks, torch.zeros_like(spks)], dim=0)
            cond2 = torch.cat([cond, torch.zeros_like(cond)], dim=0)
            d = estimator_fn(x2, mu2, torch.full((2,), t), spks2, cond2)
            x = x + dt * ((1.0 + cfg_rate) * d[:1] - cfg_rate * d[1:])
        return x
