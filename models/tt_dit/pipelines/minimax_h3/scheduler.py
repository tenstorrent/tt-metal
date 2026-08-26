# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 rectified-flow Euler scheduler.

Three conventions differ from the usual flow-match scheduler and each one is a
silent-wrong-output bug if missed:

1. ``t = 1 - sigma`` in ``[0, 1]``, and ``t = 1`` means *clean* -- the opposite
   direction from schedulers that expose ``timesteps = sigma * num_train_timesteps``.
2. The velocity is data-ward, so ``x0 = x_t + sigma * v``. Note the ``+``.
3. The sigma grid is ``linspace(1, 0, num_inference_steps)`` pushed through the
   exponential shift, terminal ``0`` included. So ``num_inference_steps`` counts
   grid points and drives ``num_inference_steps - 1`` model evaluations.

``eta = 0``, so despite the reference file being named "euler_ancestral" there is
no ancestral noise term. A request runs two of these, one per modality:
``shift=12.0`` for video and ``shift=3.0`` for audio.
"""

from __future__ import annotations

import torch


class MiniMaxH3Scheduler:
    """Rectified-flow Euler scheduler (``eta = 0``) with an exponential sigma shift."""

    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self._shift = float(shift)
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_shift(self, shift: float) -> None:
        """Override the sigma shift. Call before :meth:`set_timesteps`."""
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        self._shift = float(shift)

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
        sigmas: list[float] | torch.Tensor | None = None,
    ) -> None:
        """Build the sigma and timestep schedule.

        An explicit ``sigmas`` is used verbatim -- no shift, no deduplication.
        """
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "set_timesteps requires either an explicit sigmas schedule or "
                    f"num_inference_steps >= 2, got {num_inference_steps}"
                )
            # The rectified-flow sigma range is fixed at [1.0, 0.0], and the shift
            # maps 0 to exactly 0 so the terminal point survives it.
            base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
            sigmas = self._shift * base / (1 + (self._shift - 1) * base)
            # The shift compresses the grid near sigma = 1; collapse the float32
            # collisions that creates rather than stepping with ratio == 1.
            sigmas = torch.unique_consecutive(sigmas)
        else:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            if sigmas.numel() < 2 or not bool((sigmas[1:] < sigmas[:-1]).all()) or sigmas[-1].item() != 0.0:
                raise ValueError("sigmas must hold at least two strictly decreasing values ending at 0.0")

        self.sigmas = sigmas.to(device=device)
        self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) == 0:
            raise ValueError(f"timestep {timestep} is not in the schedule")
        return int(indices[0].item())

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: float | torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Rectified-flow forward process ``x_t = t*x_0 + (1 - t)*noise``.

        H3 noises its conditioning anchors with this, where ``t`` is the
        ``noise_aug`` level rather than a schedule entry -- so ``timestep`` is
        taken at face value and never looked up in ``self.timesteps``.
        """
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype, device=sample.device)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler (``eta = 0``) step, returning the next sample."""
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()):
            raise ValueError("step() takes a scheduler.timesteps value, not an enumerate() index")

        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep) if self._begin_index is None else self._begin_index

        # The sigma for x0 comes from the *timestep* the transformer was
        # conditioned on, while the Euler ratio below reads the sigma grid. Below
        # sigma = 0.5 the float32 round trip 1 - (1 - sigma) is not exact, and the
        # reference keeps the two sources apart.
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        sigma_from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(dtype=compute_dtype)

        self._step_index += 1
        return prev_sample.to(dtype=sample.dtype)

    def step_coefficient(self, i: int) -> float:
        """The scalar ``c`` in ``next = sample + c * model_output``, for applying the step on device.

        ``step()`` computes ``ratio*sample + (1-ratio)*denoised`` with ``denoised = sample +
        (1-t)*v`` and ``ratio = sigma_next/sigma``. That factors to ``sample + (1-ratio)*(1-t)*v``,
        and since the schedule defines ``timesteps[i] = 1 - sigmas[i]`` (so ``1 - t == sigma``) the
        coefficient collapses to ``sigma - sigma_next``. ``i`` is the loop index into ``timesteps``,
        matching ``step()``'s internal ``_step_index``.

        The fp nuance ``step()`` guards -- deriving the x0 sigma from ``1 - t`` rather than the grid
        sigma -- differs from ``sigma - sigma_next`` only below sigma = 0.5 and only at the fp32 ulp
        level, which the on-device apply (bf16) dwarfs. The host ``step()`` stays the reference.
        """
        return float(self.sigmas[i] - self.sigmas[i + 1])
