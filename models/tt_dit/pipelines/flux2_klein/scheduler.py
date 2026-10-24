# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein Euler schedule: the diffusers ``FlowMatchEulerDiscreteScheduler`` on the empirical-mu path
(``scheduling_flow_match_euler_discrete.py``, cited as ``S``) as ``Flux2KleinPipeline.__call__`` drives it
(``pipeline_flux2_klein.py``, ``P``); BFL ``sampling.py`` (``SA``) is the second citation. diffusers is never
imported here. Facts that are easy to get subtly wrong:

1. The grid: ``sigmas = np.linspace(1.0, 1 / N, N)`` (``P:814``), ``mu = compute_empirical_mu(image_seq_len, N)``
   with ``image_seq_len`` the LATENT token count, references excluded (``P:817-818``); ``set_timesteps`` casts the
   sigmas to float32 (``S:351``) and, with ``use_dynamic_shifting`` true and ``time_shift_type`` exponential
   (scheduler/scheduler_config.json), shifts them as ``exp(mu) / (exp(mu) + (1 / t - 1) ** 1.0)`` in numpy float32
   (``S:355``, ``S:655-656``); ``timesteps = sigmas * 1000`` fp32 (``S:376``); a terminal ``0`` is appended to the
   sigmas (``S:386``). ``shift`` / ``base_shift`` / ``max_shift`` of the config are unused on this path.
   ``SA:244-248`` ``get_schedule`` gives the same ``N + 1`` points.
2. Per step the DiT sees ``timestep = t.to(bf16)`` (``P:841``), ``/ 1000`` in bf16 (``P:853``), then ``* 1000`` in
   bf16 inside the transformer (``transformer_flux2.py:1284``) before the fp32 sinusoid: :func:`embedding_timestep`.
3. The Euler update (``S:493-526``): the sample is upcast to fp32, ``dt = sigmas[i+1] - sigmas[i]`` is a 0-dim
   fp32 tensor (negative), ``prev = sample + dt * model_output`` where ``dt * v`` stays in the DiT's bf16 under torch
   promotion, and the result is cast BACK to the model output's dtype (bf16): the state is bf16 between steps.
   ``SA:305`` ``img + (t_prev - t_curr) * pred`` is the same arithmetic in bf16.
4. Guidance: the distilled model runs one forward per step with no guidance (``P:595-596``, ``is_distilled``);
   the base model encodes the empty negative prompt and combines ``neg + g * (pos - neg)`` (``P:877``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

# README / BFL util.py: 4 steps and no guidance for the distilled model, 50 steps and CFG 4.0 for base.
FLUX2_KLEIN_STEPS = {"distilled": 4, "base": 50}
FLUX2_KLEIN_GUIDANCE = {"distilled": None, "base": 4.0}
# scheduler_config.json num_train_timesteps (S:376 timesteps = sigmas * 1000; P:853 timestep / 1000).
FLUX2_KLEIN_NUM_TRAIN_TIMESTEPS = 1000
# P:841: the timestep is cast to the latents' dtype before the transformer sees it.
FLUX2_KLEIN_TIMESTEP_DTYPE = torch.bfloat16


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """``P:63-78`` verbatim (``SA:251-266``)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def sigmas(num_steps: int, image_seq_len: int) -> Tensor:
    """The scheduler's ``sigmas`` after ``set_timesteps(sigmas=linspace(1, 1/N, N), mu=...)``: fp32 ``[N + 1]``
    with the terminal 0 (module docstring, item 1), op for op."""
    base = np.linspace(1.0, 1 / num_steps, num_steps)  # P:814
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)  # P:818
    shifted = np.array(base).astype(np.float32)  # S:351
    shifted = math.exp(mu) / (math.exp(mu) + (1 / shifted - 1) ** 1.0)  # S:355 -> S:655-656, sigma = 1.0
    out = torch.from_numpy(shifted).to(dtype=torch.float32)  # S:372
    return torch.cat([out, torch.zeros(1)])  # S:386


def embedding_timestep(timestep: float) -> float:
    """The value the transformer embeds for the fp32 ``timestep = sigma * 1000``: ``bf16(bf16(bf16(t) / 1000) * 1000)``
    (``P:841``, ``P:853``, ``transformer_flux2.py:1284``; each op rounds to bf16 as torch does)."""
    t = torch.tensor(timestep, dtype=torch.float32).to(FLUX2_KLEIN_TIMESTEP_DTYPE)
    t = t / FLUX2_KLEIN_NUM_TRAIN_TIMESTEPS
    t = t.to(FLUX2_KLEIN_TIMESTEP_DTYPE) * FLUX2_KLEIN_NUM_TRAIN_TIMESTEPS
    return t.item()


@dataclass(frozen=True)
class Flux2KleinStep:
    """One Euler step. ``sigma`` / ``sigma_next`` / ``dt`` are the fp32 schedule values as python floats, exactly as
    the scheduler reads them; ``timestep`` is ``sigma * 1000`` (fp32); ``embedding_t`` the bf16-rounded value the
    DiT embeds (:func:`embedding_timestep`)."""

    index: int
    sigma: float
    sigma_next: float
    dt: float
    timestep: float
    embedding_t: float

    @property
    def dt_rounded(self) -> float:
        """``dt`` as the Euler multiply consumes it. ``S:513`` ``dt = sigmas[i + 1] - sigmas[i]`` is a 0-dim fp32
        TENSOR, and torch casts a 0-dim tensor operand to the binary op's common dtype (bf16, the velocity's) before
        ``dt * model_output``, so the product is ``bf16(bf16(dt) * v)``; a Python-float ``dt`` (the BFL ``SA:305`` /
        FLUX 3 form) is multiplied in fp32 instead. The device Euler chain must therefore hold THIS value in its fp32
        ``dt`` slot (measured 2026-10-23, man_dog 1024, 4 steps: the fp32 ``dt`` chain differed from
        :meth:`Flux2KleinScheduler.update` in 19k-128k of 524k state elements per step by one bf16 ulp; with
        ``bf16(dt)`` it is bit-equal)."""
        return torch.tensor(self.dt, dtype=torch.float32).to(FLUX2_KLEIN_TIMESTEP_DTYPE).item()


class Flux2KleinScheduler:
    """Fixed Euler schedule: ``num_steps`` model evaluations over ``num_steps + 1`` sigma points, shifted for the
    ``image_seq_len`` latent tokens of the canvas."""

    def __init__(self, num_steps: int, image_seq_len: int) -> None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}")
        if image_seq_len < 1:
            raise ValueError(f"image_seq_len must be at least 1, got {image_seq_len}")
        self.num_steps = int(num_steps)
        self.image_seq_len = int(image_seq_len)
        self.mu: float = compute_empirical_mu(image_seq_len=self.image_seq_len, num_steps=self.num_steps)
        self.sigmas: Tensor = sigmas(self.num_steps, self.image_seq_len)
        self.timesteps: Tensor = self.sigmas[:-1] * FLUX2_KLEIN_NUM_TRAIN_TIMESTEPS  # S:376

    def __len__(self) -> int:
        return self.num_steps

    def step(self, index: int) -> Flux2KleinStep:
        if not 0 <= index < self.num_steps:
            raise IndexError(f"step {index} outside 0..{self.num_steps - 1}")
        timestep = self.timesteps[index].item()
        return Flux2KleinStep(
            index=index,
            sigma=self.sigmas[index].item(),
            sigma_next=self.sigmas[index + 1].item(),
            dt=(self.sigmas[index + 1] - self.sigmas[index]).item(),  # S:513
            timestep=timestep,
            embedding_t=embedding_timestep(timestep),
        )

    def steps(self) -> list[Flux2KleinStep]:
        return [self.step(i) for i in range(self.num_steps)]

    @staticmethod
    def update(state: Tensor, velocity: Tensor, step: Flux2KleinStep) -> Tensor:
        """``scheduler.step`` (``S:493-526``): fp32 sample, ``dt * v`` in the velocity's dtype (bf16 from the DiT;
        the 0-dim fp32 ``dt`` is itself rounded to bf16 by torch's type promotion, :attr:`Flux2KleinStep.dt_rounded`),
        fp32 add, cast back to the velocity's dtype."""
        sample = state.to(torch.float32)
        dt = torch.tensor(step.dt, dtype=torch.float32)  # 0-dim, as the scheduler's sigmas[i + 1] - sigmas[i]
        prev = sample + dt * velocity
        return prev.to(velocity.dtype)

    @staticmethod
    def cfg_velocity(unconditional: Tensor, conditional: Tensor, guidance: float) -> Tensor:
        """Classifier-free guidance of the base model, in the DiT output dtype (``P:877``)."""
        return unconditional + guidance * (conditional - unconditional)


__all__ = [
    "FLUX2_KLEIN_GUIDANCE",
    "FLUX2_KLEIN_NUM_TRAIN_TIMESTEPS",
    "FLUX2_KLEIN_STEPS",
    "FLUX2_KLEIN_TIMESTEP_DTYPE",
    "Flux2KleinScheduler",
    "Flux2KleinStep",
    "compute_empirical_mu",
    "embedding_timestep",
    "sigmas",
]
