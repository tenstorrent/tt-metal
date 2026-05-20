# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""DPM-Solver++ (SDE, 2nd-order, singlestep) port of k-diffusion's
``sample_dpmpp_sde``, adapted to Wan2.2 flow-matching schedules.

Mirrors ``diffusers.DPMSolverSDEScheduler.step`` semantics. Each real
diffusion step is split into two pipeline iterations:

    even step (first-order):  model runs at the outer sigma s_i, the
        solver computes an intermediate latent at the midpoint sigma
        (geometric mean of s_i and s_{i+1}) and stashes the original.
    odd step (second-order):  model runs at the midpoint sigma, the
        solver takes a full ancestral step from the stashed latent to
        s_{i+1}.

The base flow-sigma schedule is unchanged; only the timesteps array is
doubled (outer timesteps interleaved with midpoint timesteps) so the
existing pipeline loop iterates 2N times for N effective diffusion steps.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin

import ttnn

from ..utils.tensor import from_torch
from .base import Solver


class WanDPMSolverSDEScheduler(UniPCMultistepScheduler):
    """Flow-matching scheduler that exposes a doubled timestep array for
    stochastic DPM-Solver++.

    Inherits UniPC's flow-sigma logic (``use_flow_sigmas=True``,
    ``flow_shift=...``). ``set_timesteps`` is overridden to leave
    ``self.sigmas`` (length N+1) untouched but interleave midpoint
    timesteps into ``self.timesteps``, doubling its length to 2N.
    """

    def set_timesteps(self, num_inference_steps: Optional[int] = None, device=None, **kwargs):
        super().set_timesteps(num_inference_steps, device=device, **kwargs)

        base_sigmas = self.sigmas.cpu().numpy()  # length N+1, sigmas[-1] = 0
        base_ts = self.timesteps.cpu().numpy()  # length N
        num_train = self.config.num_train_timesteps

        # Interleave each outer timestep with the midpoint timestep that
        # corresponds to the geometric-mean midpoint sigma. Wan's flow
        # scheduler maps t = sigma * num_train_timesteps, so the midpoint
        # timestep is num_train * sqrt(sigma_i * sigma_{i+1}).
        doubled_ts = []
        for i in range(len(base_ts)):
            sigma_cur = float(base_sigmas[i])
            sigma_next = float(base_sigmas[i + 1])
            doubled_ts.append(float(base_ts[i]))
            if sigma_next > 0:
                sigma_mid = math.sqrt(sigma_cur * sigma_next)
                doubled_ts.append(sigma_mid * num_train)
            else:
                # Final step lands at sigma=0; the solver collapses the
                # second-order half to a plain Euler step. Duplicate the
                # timestep so the loop still iterates.
                doubled_ts.append(float(base_ts[i]))

        self.timesteps = torch.tensor(doubled_ts, dtype=torch.float32)
        if device is not None:
            self.timesteps = self.timesteps.to(device)


class DPMSolverSDESolver(Solver):
    """Stochastic DPM-Solver++(2S) on Wan's flow sigma schedule.

    Args:
        scheduler: A ``WanDPMSolverSDEScheduler`` (doubled timestep array).
        mesh_device, mesh_axes, dtype: needed to upload per-step Gaussian
            noise onto the same sharding as the working latent.
        seed: optional torch RNG seed for the stochastic noise sampler.
        s_noise: noise multiplier (1.0 matches k-diffusion default).
    """

    def __init__(
        self,
        scheduler: Optional[WanDPMSolverSDEScheduler] = None,
        *,
        mesh_device,
        mesh_axes: Sequence[Optional[int]],
        dtype: ttnn.DataType,
        seed: Optional[int] = None,
        # Defaults to 0 (deterministic 2nd-order DPM++) because the per-step
        # noise upload currently mismatches the latent's sharded shape on
        # BH 2x4 mesh ("Invalid subtile broadcast type" in eltwise add).
        # Stochastic noise is a follow-up once the shape resolution is fixed.
        s_noise: float = 0.0,
    ) -> None:
        if scheduler is None:
            scheduler = WanDPMSolverSDEScheduler(use_flow_sigmas=True, prediction_type="flow_prediction")
        if not isinstance(scheduler, SchedulerMixin):
            raise ValueError(f"scheduler must be a SchedulerMixin, got {type(scheduler).__name__}")
        if not isinstance(scheduler, WanDPMSolverSDEScheduler):
            raise ValueError(
                "DPMSolverSDESolver requires a WanDPMSolverSDEScheduler " f"(got {type(scheduler).__name__})."
            )
        super().__init__(scheduler)
        self._mesh_device = mesh_device
        self._mesh_axes = tuple(mesh_axes)
        self._dtype = dtype
        self._s_noise = float(s_noise)
        self._seed = seed
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._stored_sample: Optional[ttnn.Tensor] = None

    def set_schedule(self, num_inference_steps=None, *, device=None, **kwargs) -> None:
        super().set_schedule(num_inference_steps, device=device, **kwargs)
        self._stored_sample = None
        if self._seed is not None:
            self._generator.manual_seed(self._seed)

    def step(self, *, step: int, latent: ttnn.Tensor, velocity_pred: ttnn.Tensor) -> ttnn.Tensor:
        self._assert_schedule()

        # Each pair of steps consumes one outer sigma transition.
        outer_idx = step // 2
        first_order = (step % 2) == 0
        sigma = self._sigmas[outer_idx]
        sigma_next = self._sigmas[outer_idx + 1]

        t = _t_fn(sigma)
        t_next = _t_fn(sigma_next)
        t_mid = t + 0.5 * (t_next - t)

        # On the first-order half, the model just ran at the outer sigma.
        # On the second-order half, the model ran at the midpoint sigma
        # (the scheduler emitted that timestep into the doubled array).
        sigma_input = sigma if first_order else _sigma_fn(t_mid)
        # Flow matching: pred_original = sample - sigma * velocity.
        pred_original = latent - sigma_input * velocity_pred

        if sigma_next == 0:
            # Final outer step: take a plain Euler step to sigma=0 on the
            # first-order half, and do nothing (return the same latent) on
            # the second-order half so the doubled-loop pattern is honored.
            if first_order:
                if sigma > 0:
                    prev_sample = latent + ((sigma_next - sigma) / sigma) * (latent - pred_original)
                else:
                    prev_sample = latent
                self._stored_sample = None
                return prev_sample
            self._stored_sample = None
            return latent

        if first_order:
            t_target = t_mid
            sample_ref = latent
        else:
            t_target = t_next
            sample_ref = self._stored_sample
            assert sample_ref is not None, "second-order step requires stored sample from first-order"

        sigma_from = sigma
        sigma_to = _sigma_fn(t_target)
        # k-diffusion get_ancestral_step (eta=1):
        # sigma_up = min(sigma_to, sqrt(sigma_to^2 * (sigma_from^2 - sigma_to^2) / sigma_from^2))
        # sigma_down = sqrt(sigma_to^2 - sigma_up^2)
        if sigma_from > 0:
            up_candidate_sq = sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
        else:
            up_candidate_sq = 0.0
        sigma_up = min(sigma_to, math.sqrt(max(up_candidate_sq, 0.0)))
        sigma_down = math.sqrt(max(sigma_to**2 - sigma_up**2, 0.0))
        ancestral_t = _t_fn(sigma_down) if sigma_down > 0 else float("inf")

        # diffusers/k-diffusion step:
        # prev = (sigma_down / sigma_from) * sample - expm1(t - ancestral_t) * pred_original
        scale_sample = sigma_down / sigma_from if sigma_from > 0 else 0.0
        coeff_pred = -math.expm1(t - ancestral_t)
        prev_sample = scale_sample * sample_ref + coeff_pred * pred_original

        # Stochastic ancestral noise. The k-diffusion reference uses a
        # BrownianTreeNoiseSampler (torchsde) which produces variance-
        # reduced noise; we substitute independent N(0,1) per step. The
        # visual signature of dpm++_sde is dominated by sigma_up's
        # ancestral split, so this is a faithful approximation.
        if sigma_up > 0 and self._s_noise > 0:
            noise_tt = self._make_noise_like(latent)
            prev_sample = prev_sample + (self._s_noise * sigma_up) * noise_tt

        if first_order:
            self._stored_sample = latent
        else:
            self._stored_sample = None
        return prev_sample

    def _make_noise_like(self, latent: ttnn.Tensor) -> ttnn.Tensor:
        shape = tuple(latent.shape)
        host_noise = torch.randn(shape, generator=self._generator, dtype=torch.float32)
        return from_torch(
            host_noise,
            device=self._mesh_device,
            mesh_axes=list(self._mesh_axes),
            dtype=self._dtype,
        )


def _t_fn(sigma: float) -> float:
    if sigma <= 0:
        return float("inf")
    return -math.log(sigma)


def _sigma_fn(t: float) -> float:
    if math.isinf(t):
        return 0.0
    return math.exp(-t)
