# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


@torch.no_grad()
def sample_discrete_euler(model_fn, x, steps, sigma_max=1.0, **extra_args):
    """Discrete-Euler sampler used by AudioX rectified-flow models."""
    t = torch.linspace(sigma_max, 0, steps + 1)
    for t_curr, t_prev in zip(t[:-1], t[1:]):
        t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
        dt = (t_prev - t_curr).item()
        x = x + dt * model_fn(x, t_curr_tensor, **extra_args)
    return x


@torch.no_grad()
def sample_rf(model_fn, noise, init_data=None, steps=100, sigma_max=1.0, **extra_args):
    """Rectified-flow entry point. Mixes init_data with noise (variation mode)
    or uses pure noise (sampling mode), then runs `sample_discrete_euler`."""
    if sigma_max > 1:
        sigma_max = 1

    if init_data is not None:
        x = init_data * (1 - sigma_max) + noise * sigma_max
    else:
        x = noise

    return sample_discrete_euler(model_fn, x, steps, sigma_max, **extra_args)
