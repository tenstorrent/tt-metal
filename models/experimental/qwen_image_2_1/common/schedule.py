# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FlowMatch Euler schedule (dynamic exponential shift, shift_terminal) and the DiT's timestep
conditioning (sinusoidal embedding -> TimestepEmbedding MLP -> shared modulation rows), all on host in fp32."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import DIT, SCHED, SchedulerConfig


def calculate_mu(image_seq_len: int, cfg: SchedulerConfig = SCHED) -> float:
    m = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len)
    b = cfg.base_shift - m * cfg.base_image_seq_len
    return image_seq_len * m + b


def make_sigmas(num_steps: int, image_seq_len: int, cfg: SchedulerConfig = SCHED) -> np.ndarray:
    """Replicates FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=linspace(1, 1/N, N), mu=...)
    with use_dynamic_shifting=True, time_shift_type='exponential', shift_terminal=0.02.
    Returns the N+1 sigmas (last is 0)."""
    if num_steps < 2:
        raise ValueError("the terminal-shift scheduler requires at least two steps")
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float64)
    mu = calculate_mu(image_seq_len, cfg)
    # exponential time shift: exp(mu) / (exp(mu) + (1/t - 1))
    sigmas = math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))
    # stretch_shift_to_terminal
    one_minus_z = 1.0 - sigmas
    scale = one_minus_z[-1] / (1.0 - cfg.shift_terminal)
    sigmas = 1.0 - (one_minus_z / scale)
    return np.concatenate([sigmas, [0.0]]).astype(np.float32)


def sigmas_to_timesteps(sigmas: np.ndarray, cfg: SchedulerConfig = SCHED) -> np.ndarray:
    return (sigmas[:-1] * cfg.num_train_timesteps).astype(np.float32)


def timestep_sinusoid(t: torch.Tensor, dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
    """QwenImage21TemporalTimesteps: cos first half, sin second half. t in [0,1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    args = (time_factor * t.float())[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


@dataclass
class TimeConditioning:
    """Host-side copies (fp32) of the tiny conditioning weights."""

    lin1: torch.Tensor  # [4096, 256]  timestep_embedder.linear_1.weight
    lin2: torch.Tensor  # [4096, 4096] timestep_embedder.linear_2.weight
    mod: torch.Tensor  # [16384, 4096] modulation.1.weight
    norm_out: torch.Tensor  # [4096, 4096] norm_out.linear.weight

    @classmethod
    def from_ckpt(cls, ckpt) -> "TimeConditioning":
        f = lambda k: ckpt.get(k, torch.float32)
        return cls(
            lin1=f("time_text_embed.timestep_embedder.linear_1.weight"),
            lin2=f("time_text_embed.timestep_embedder.linear_2.weight"),
            mod=f("modulation.1.weight"),
            norm_out=f("norm_out.linear.weight"),
        )

    def temb(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] in [0,1] -> temb [B, 4096] (fp32)."""
        x = timestep_sinusoid(t, DIT.timestep_dim)
        x = torch.nn.functional.silu(x @ self.lin1.t())
        return x @ self.lin2.t()

    def modulation(self, t: torch.Tensor) -> torch.Tensor:
        """[B, 16384] = [scale1 | gate1 | scale2 | gate2] (each 4096), i.e. modulation(SiLU(temb))."""
        return torch.nn.functional.silu(self.temb(t)) @ self.mod.t()

    def norm_out_scale(self, t: torch.Tensor) -> torch.Tensor:
        """[B, 4096] scale of the final AdaLayerNormContinuous (out = LN(x) * (1 + scale))."""
        return torch.nn.functional.silu(self.temb(t)) @ self.norm_out.t()


@dataclass
class StepConditioning:
    """Everything a DiT forward needs from the timestep, as device-ready fp32 rows."""

    one_plus_scale1: torch.Tensor  # [1, 4096]
    tanh_gate1: torch.Tensor  # [1, 4096]
    one_plus_scale2: torch.Tensor  # [1, 4096]
    tanh_gate2: torch.Tensor  # [1, 4096]
    one_plus_scale_out: torch.Tensor  # [1, 4096]

    @classmethod
    def make(cls, tc: TimeConditioning, t01: float, dtype=torch.float32) -> "StepConditioning":
        t = torch.tensor([t01], dtype=torch.float32)
        m = tc.modulation(t)[0]
        s1, g1, s2, g2 = m.chunk(4)
        so = tc.norm_out_scale(t)[0]
        r = lambda v: (v.reshape(1, -1)).to(dtype)
        return cls(r(1 + s1), r(torch.tanh(g1)), r(1 + s2), r(torch.tanh(g2)), r(1 + so))


def modulation_rows_bf16_like_reference(tc: TimeConditioning, t01: float) -> StepConditioning:
    """Same as StepConditioning.make but rounding temb/modulation to bf16 at the places the bf16
    diffusers reference does, for tighter PCC comparisons against bf16 goldens."""
    t = torch.tensor([t01], dtype=torch.bfloat16)
    x = timestep_sinusoid(t.float(), DIT.timestep_dim).to(torch.bfloat16)
    x = torch.nn.functional.silu((x @ tc.lin1.t().to(torch.bfloat16)))
    temb = x @ tc.lin2.t().to(torch.bfloat16)
    m = (torch.nn.functional.silu(temb) @ tc.mod.t().to(torch.bfloat16))[0].float()
    so = (torch.nn.functional.silu(temb) @ tc.norm_out.t().to(torch.bfloat16))[0].float()
    s1, g1, s2, g2 = m.chunk(4)
    r = lambda v: v.reshape(1, -1)
    return StepConditioning(r(1 + s1), r(torch.tanh(g1)), r(1 + s2), r(torch.tanh(g2)), r(1 + so))
