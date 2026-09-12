# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import ttnn


def _broadcast_t(t_value, batch, mesh_device, dtype=None):
    """Broadcast a scalar timestep to a [batch] ttnn tensor."""
    dtype = ttnn.bfloat16 if dtype is None else dtype
    t = torch.full((batch,), float(t_value), dtype=torch.float32)
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device)


def sample_discrete_euler(model_fn, x: ttnn.Tensor, steps: int, mesh_device, sigma_max: float = 1.0, **extra_args):
    """Discrete-Euler rectified-flow sampler. `model_fn(x, t, **extra_args)` must
    take a ttnn tensor `x` plus a [batch] ttnn timestep tensor and return a ttnn
    tensor with the same shape as `x`."""
    schedule = torch.linspace(sigma_max, 0, steps + 1).tolist()
    batch = x.shape[0]

    for t_curr, t_prev in zip(schedule[:-1], schedule[1:]):
        t_tensor = _broadcast_t(t_curr, batch, mesh_device, dtype=x.get_dtype())
        dt = t_prev - t_curr
        velocity = model_fn(x, t_tensor, **extra_args)
        x = ttnn.add(x, ttnn.multiply(velocity, dt))

    return x


def sample_rf(
    model_fn,
    noise: ttnn.Tensor,
    mesh_device,
    init_data: ttnn.Tensor | None = None,
    steps: int = 100,
    sigma_max: float = 1.0,
    **extra_args,
):
    """Rectified-flow entry point. Matches upstream `sample_rf` semantics:
    pure noise sampling when `init_data` is None, otherwise variation mode that
    interpolates `init_data` and `noise` by `sigma_max`."""
    if sigma_max > 1:
        sigma_max = 1

    if init_data is not None:
        x = ttnn.add(
            ttnn.multiply(init_data, 1.0 - sigma_max),
            ttnn.multiply(noise, sigma_max),
        )
    else:
        x = noise

    return sample_discrete_euler(model_fn, x, steps, mesh_device, sigma_max, **extra_args)
