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
        self.n_timesteps = n_timesteps
        self.t_scheduler = t_scheduler
        self.estimator = TtConditionalDecoder(device, bag.sub("estimator"), dtype=dtype)

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

    def _capture(self, x, mu2, spks2, cond2, t0):
        """Trace one estimator evaluation and replay it for every Euler step.

        The solver calls the *same graph* ten times -- only `x` and `t` change,
        while `mu`, `spks` and `cond` are fixed for the utterance. That makes it a
        better trace candidate than the AR decoder, which at least has a growing
        cache to work around: here there is no state at all between steps.

        The CFG split and the Euler update stay outside the trace. They are a
        handful of elementwise ops on `[1, T, 80]`, and keeping them out means the
        traced region is exactly the 16 resnet + 64 transformer blocks that cost
        something.

        **This does not work, and `use_trace` defaults to False because of it.**
        Capture fails with `!trace_id_.has_value()` -- a host->device write while a
        trace is open. **`ttnn.conv1d` is not trace-compatible in this build**, and
        the estimator has ~37 convolutions.

        Measured directly (see the probe recorded in the notes): a bare `conv1d`
        captured on its own fails with a host weight at
        `fd_mesh_command_queue.cpp:762` and with a **device-resident** weight at
        `:809` -- a different write, but a write either way. Making the weights
        device-resident is therefore not the fix; it is accepted and gives
        bit-identical output (`max|d| 0.000e+00`), but the op writes internally
        regardless.

        **It is a software limit, not a silicon one, and it is half-fixed.**
        `ttnn.conv1d` prepares its weights -- tilize, pad to the sharding scheme,
        move to device -- on *every call*. That is host work, and a trace forbids
        host traffic either way: a host weight fails on the write, a device weight
        fails on the **read back** to prepare it. `ttnn.prepare_conv_weights` hoists
        the transform out, and a bare `conv1d` then captures cleanly with
        bit-identical output. `TtConv1d` now does that and caches per input
        geometry.

        What still blocks this particular graph is the rest of the estimator:
        `TtConvTranspose1d` in the up path has the same lazy preparation and has not
        been converted, and `pack_input`'s `ttnn.repeat` is a further candidate. So
        the *class* of problem is understood and demonstrated solvable; this graph
        needs the remaining writes chased out one at a time.

        Worth carrying into `03_plan.md`: any conv-based model on this stack is
        untraceable by default, and the fix is per-op weight preparation rather than
        anything in hardware.
        """
        b, t_len, ch = x.shape
        self._x_buf = ttnn.concat([x, x], dim=0)
        self._t_buf = ttnn.from_torch(
            torch.full((2, 1, 1), t0, dtype=torch.float32),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self._d_buf = ttnn.from_torch(
            torch.zeros(2, t_len, ch), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        def body():
            d = self.estimator(self._x_buf, mu2, self._t_buf, spks=spks2, cond=cond2, batch=2)
            ttnn.copy(d, self._d_buf)
            ttnn.deallocate(d)

        for _ in range(2):  # warm the program cache before recording
            body()
        ttnn.synchronize_device(self.device)
        self._trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            body()
        finally:
            ttnn.end_trace_capture(self.device, self._trace_id, cq_id=0)

    def _release(self):
        if getattr(self, "_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
        for name in ("_x_buf", "_t_buf", "_d_buf"):
            t = getattr(self, name, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, name, None)

    def solve_euler(self, x, mu, spks, cond, t_span=None, use_trace=False):
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
            )
            for t, _ in schedule
        ]

        traced = False
        if use_trace:
            try:
                self._capture(x, mu2, spks2, cond2, schedule[0][0])
                traced = True
            except Exception as e:  # noqa: BLE001
                # Needs the device opened with a `trace_region_size`. Tracing is an
                # optimisation, so fall back loudly rather than failing the solve.
                logger.warning(f"CFM trace capture unavailable, running untraced: {e}")
                self._release()

        for (t_val, dt), t_dev in zip(schedule, ts):
            if traced:
                # Refresh the two inputs that change, then replay. Both CFG rows
                # evaluate the same `x`, so one concat feeds the buffer.
                pair = ttnn.concat([x, x], dim=0)
                ttnn.copy(pair, self._x_buf)
                ttnn.deallocate(pair)
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        torch.full((2, 1, 1), t_val, dtype=torch.float32),
                        dtype=self.dtype,
                        layout=ttnn.TILE_LAYOUT,
                    ),
                    self._t_buf,
                )
                ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(self.device)
                d = self._d_buf
            else:
                x2 = self._cfg_pair(x, zero_second_row=False)
                d = self.estimator(x2, mu2, t_dev, spks=spks2, cond=cond2, batch=2)
                ttnn.deallocate(x2)

            cond_part = ttnn.slice(d, [0, 0, 0], [1, t_len, 80])
            uncond_part = ttnn.slice(d, [1, 0, 0], [2, t_len, 80])
            if not traced:
                ttnn.deallocate(d)
            guided = ttnn.subtract(
                ttnn.multiply(cond_part, 1.0 + self.cfg_rate), ttnn.multiply(uncond_part, self.cfg_rate)
            )
            ttnn.deallocate(cond_part)
            ttnn.deallocate(uncond_part)

            step = ttnn.multiply(guided, dt)
            ttnn.deallocate(guided)
            nxt = ttnn.add(x, step)
            ttnn.deallocate(step)
            ttnn.deallocate(x)
            x = nxt

        if traced:
            self._release()

        for t_dev in ts:
            ttnn.deallocate(t_dev)
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
