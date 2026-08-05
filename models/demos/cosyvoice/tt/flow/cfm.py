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

    def solve_euler(self, x, mu, spks, cond, t_span=None):
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

        for (_, dt), t_dev in zip(schedule, ts):
            x2 = self._cfg_pair(x, zero_second_row=False)
            d = self.estimator(x2, mu2, t_dev, spks=spks2, cond=cond2, batch=2)
            ttnn.deallocate(x2)

            cond_part = ttnn.slice(d, [0, 0, 0], [1, t_len, 80])
            uncond_part = ttnn.slice(d, [1, 0, 0], [2, t_len, 80])
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
