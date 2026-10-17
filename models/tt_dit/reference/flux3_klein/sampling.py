# SPDX-FileCopyrightText: © 2026 Black Forest Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/black-forest-labs/flux-3-image-klein (src/flux3/sampling.py) at e53f4ea,
# licensed under Apache-2.0. Modified by Tenstorrent as the PyTorch reference for the tt-metal port.

"""Euler sampling: CFG on a two-sample pack [unconditional, conditional], or a flow map on one sample."""

import functools

import torch
from torch import Tensor

from .dit import DiT, Packed


def autocast_bf16(fn):
    """bf16 autocast on the pack's device; upstream decorates with ``torch.autocast("cuda", dtype=torch.bfloat16)``."""

    @functools.wraps(fn)
    def wrapper(dit: DiT, packed: Packed, *args, **kwargs):
        with torch.autocast(packed.data["image"].device.type, dtype=torch.bfloat16):
            return fn(dit, packed, *args, **kwargs)

    return wrapper


def timesteps(steps: int, alpha: float) -> Tensor:
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


@torch.inference_mode()
@autocast_bf16
def denoise(dit: DiT, packed: Packed, schedule: Tensor, guidance: float) -> Tensor:
    """Returns the denoised image tokens of the conditional sample."""
    for step in range(len(schedule) - 1):
        t, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
        packed.timesteps["image"].fill_(t)
        unconditional, conditional = dit(packed).chunk(2)
        velocity = unconditional + guidance * (conditional - unconditional)
        x = packed.data["image"].chunk(2)[0] + dt * velocity
        packed.data["image"] = torch.cat((x, x))
    return packed.data["image"].chunk(2)[0]


@torch.inference_mode()
@autocast_bf16
def denoise_flow_map(dit: DiT, packed: Packed, schedule: Tensor) -> Tensor:
    for step in range(len(schedule) - 1):
        t, dt = schedule[step].item(), (schedule[step + 1] - schedule[step]).item()
        packed.timesteps["image"].fill_(t)
        target = schedule[step + 1].reshape(1).to(packed.data["image"].device)
        packed.data["image"] = packed.data["image"] + dt * dit(packed, timesteps_ref=target)
    return packed.data["image"]
