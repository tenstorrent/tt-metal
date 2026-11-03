# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Euler schedule of FLUX 3 Klein: t' = alpha * t / (1 + (alpha - 1) * t), with the timesteps the model sees rounded
to bf16 as in the reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

ALPHA = 6.93
STEPS = {"distilled": 16, "base": 50}
GUIDANCE = {"distilled": None, "base": 3.5}
TIMESTEP_EMBED_SCALE = 1000.0


def _bf16(value: float) -> torch.Tensor:
    return torch.full((1,), value, dtype=torch.bfloat16)


@dataclass(frozen=True)
class Flux3KleinStep:
    dt: float
    embedding_t: float
    embedding_t_next: float


class Flux3KleinScheduler:
    def __init__(self, num_steps: int, alpha: float = ALPHA) -> None:
        t = torch.linspace(1.0, 0.0, num_steps + 1)
        self.schedule = alpha * t / (1.0 + (alpha - 1.0) * t)

    def __len__(self) -> int:
        return len(self.schedule) - 1

    def step(self, index: int) -> Flux3KleinStep:
        t, t_next = self.schedule[index], self.schedule[index + 1]
        t_model = _bf16(t.item()).item()
        t_next_model = t_next.reshape(1).to(torch.bfloat16).item()
        return Flux3KleinStep(
            dt=(t_next - t).item(),
            embedding_t=(TIMESTEP_EMBED_SCALE * _bf16(t_model)).item(),
            embedding_t_next=(TIMESTEP_EMBED_SCALE * _bf16(t_next_model)).item(),
        )
