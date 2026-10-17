# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein Euler schedule (flow map for the distilled model, CFG for the base model).

Mirror of the vendored reference ``sampling.py`` (``models/tt_dit/reference/flux3_klein``), which is never
imported here. Facts that are easy to get subtly wrong, each pinned by ``test_scheduler_flux3_klein.py``:

1. The grid has ``steps + 1`` points, ``t = alpha * u / (1 + (alpha - 1) * u)`` over ``u = linspace(1, 0, steps + 1)``
   in fp32, terminal ``0`` included, no deduplication (``sampling.py:28-30``). ``alpha = 6.93`` (``pipeline.py:80``);
   16 steps for the distilled model, 50 for base (``pipeline.py:23-26``).
2. Per step the reference reads ``t = schedule[i].item()`` and ``dt = (schedule[i+1] - schedule[i]).item()``: the
   difference is taken in fp32 and is negative (``sampling.py:38,51``). The image state is fp32 and the update is
   ``x + dt * v`` where ``v`` is the DiT's bf16 output, so ``dt * v`` rounds to bf16 before the fp32 add
   (``sampling.py:42-43,54``).
3. The DiT never sees ``t`` exactly. ``pack()`` allocates the per-token timesteps in bf16 (``pipeline.py:136``) and
   ``denoise*`` fills the image rows with ``fill_(t)`` (``sampling.py:39,52``): that fill is the bf16 rounding. The
   flow-map target is ``schedule[i+1].reshape(1)`` and is rounded by ``timesteps_ref.to(timesteps["image"].dtype)``
   inside the DiT (``dit.py:202``). ``timestep_embedding`` then computes ``1000.0 * t`` still in bf16 before its fp32
   cos/sin (``dit.py:43-46``). The text and image_cond rows keep timestep 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

FLUX3_KLEIN_ALPHA = 6.93
FLUX3_KLEIN_STEPS = {"distilled": 16, "base": 50}
FLUX3_KLEIN_GUIDANCE = {"distilled": None, "base": 3.5}
FLUX3_KLEIN_TIMESTEP_DTYPE = torch.bfloat16
# timestep_embedding(t) scales by 1000 in the timestep dtype (dit.py:43).
FLUX3_KLEIN_TIMESTEP_EMBED_SCALE = 1000.0


def timesteps(steps: int, alpha: float) -> Tensor:
    """The reference schedule, op for op (``sampling.py:28-30``): fp32, ``steps + 1`` points from 1 to 0."""
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def model_timestep(t: float) -> float:
    """The value the DiT sees for a python-float ``t`` written into the bf16 timestep rows (``fill_``)."""
    return torch.zeros(1, dtype=FLUX3_KLEIN_TIMESTEP_DTYPE).fill_(t).item()


def model_timestep_ref(t_next: Tensor) -> float:
    """The flow-map target the DiT sees: an fp32 schedule entry cast to the timestep dtype (``dit.py:202``)."""
    return t_next.reshape(1).to(FLUX3_KLEIN_TIMESTEP_DTYPE).item()


def embedding_timestep(t_model: float) -> float:
    """``1000.0 * t`` as ``timestep_embedding`` computes it, in the timestep dtype (``dit.py:43``)."""
    t = torch.full((1,), t_model, dtype=FLUX3_KLEIN_TIMESTEP_DTYPE)
    return (FLUX3_KLEIN_TIMESTEP_EMBED_SCALE * t).item()


@dataclass(frozen=True)
class Flux3KleinStep:
    """One Euler step. ``t``/``t_next``/``dt`` are the fp32 schedule values as python floats, exactly as the
    reference loop reads them; ``t_model`` and ``t_next_model`` are the bf16-rounded values the DiT embeds."""

    index: int
    t: float
    t_next: float
    dt: float
    t_model: float
    t_next_model: float

    @property
    def embedding_t(self) -> float:
        return embedding_timestep(self.t_model)

    @property
    def embedding_t_next(self) -> float:
        return embedding_timestep(self.t_next_model)


class Flux3KleinScheduler:
    """Fixed Euler schedule: ``steps`` model evaluations over ``steps + 1`` grid points."""

    def __init__(self, num_steps: int = FLUX3_KLEIN_STEPS["distilled"], alpha: float = FLUX3_KLEIN_ALPHA):
        if num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.num_steps = int(num_steps)
        self.alpha = float(alpha)
        self.schedule: Tensor = timesteps(self.num_steps, self.alpha)

    def __len__(self) -> int:
        return self.num_steps

    def step(self, index: int) -> Flux3KleinStep:
        if not 0 <= index < self.num_steps:
            raise IndexError(f"step {index} outside 0..{self.num_steps - 1}")
        t = self.schedule[index].item()
        t_next = self.schedule[index + 1].item()
        dt = (self.schedule[index + 1] - self.schedule[index]).item()
        return Flux3KleinStep(
            index=index,
            t=t,
            t_next=t_next,
            dt=dt,
            t_model=model_timestep(t),
            t_next_model=model_timestep_ref(self.schedule[index + 1]),
        )

    def steps(self) -> list[Flux3KleinStep]:
        return [self.step(i) for i in range(self.num_steps)]

    def model_timesteps(self, num_rows: int, step: Flux3KleinStep, device=None) -> Tensor:
        """The image rows' timestep tensor for ``step``: bf16, filled the way the reference fills it."""
        return torch.zeros(num_rows, dtype=FLUX3_KLEIN_TIMESTEP_DTYPE, device=device).fill_(step.t)

    @staticmethod
    def update(state: Tensor, velocity: Tensor, step: Flux3KleinStep) -> Tensor:
        """``x_{i+1} = x_i + dt * v`` with the reference's dtypes: ``dt * v`` in ``velocity``'s dtype (bf16 from the
        DiT), added to the fp32 state (``sampling.py:42-43,54``)."""
        return state + step.dt * velocity

    @staticmethod
    def cfg_velocity(unconditional: Tensor, conditional: Tensor, guidance: float) -> Tensor:
        """Classifier-free guidance of the base model, in the DiT output dtype (``sampling.py:40-41``)."""
        return unconditional + guidance * (conditional - unconditional)
