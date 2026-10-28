# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Seeded input generators for the FlowState goldens (CONTRACT §5 "Realistic inputs", PLAN v1.1).

Every series of the realistic mix is one of: random walk plus seasonality at SNR ~1, intermittent counts, level
shifts, multi-seasonal, smooth trend, spikes. A few series get interior NaNs (the model's missing flag). The kind
of each row is a fixed cycle over KINDS, so a batch of any size covers all of them. The seed of each golden is
recorded in MANIFEST.json; device tests load the saved inputs and never call this module.
"""

from __future__ import annotations

import math

import torch

KINDS = ("rw_seasonal", "intermittent", "level_shift", "multi_seasonal", "trend", "spikes")


def _season(n, period, g):
    t = torch.arange(n, dtype=torch.float64)
    phase = torch.rand((), generator=g, dtype=torch.float64) * 2 * math.pi
    return torch.sin(2 * math.pi * t / period + phase)


def _one(kind, n, g):
    noise = torch.randn(n, generator=g, dtype=torch.float64)
    if kind == "rw_seasonal":
        walk = torch.cumsum(torch.randn(n, generator=g, dtype=torch.float64) * 0.1, 0)
        x = walk + _season(n, 24, g) * 1.0 + noise * 1.0
    elif kind == "intermittent":
        rate = 0.05 + 0.3 * torch.rand((), generator=g, dtype=torch.float64)
        hit = torch.rand(n, generator=g, dtype=torch.float64) < rate
        size = torch.poisson(torch.full((n,), 3.0, dtype=torch.float64), generator=g)
        x = torch.where(hit, size + 1, torch.zeros_like(size))
    elif kind == "level_shift":
        x = noise * 0.5
        for _ in range(3):
            at = int(torch.randint(1, n, (), generator=g))
            x[at:] += torch.randn((), generator=g, dtype=torch.float64) * 3
    elif kind == "multi_seasonal":
        x = 2 * _season(n, 24, g) + _season(n, 168, g) + 0.5 * _season(n, 12, g) + noise * 0.7
    elif kind == "trend":
        t = torch.linspace(0, 1, n, dtype=torch.float64)
        x = 5 * t**2 - 3 * t + 0.1 * noise
    elif kind == "spikes":
        x = 0.3 * noise + _season(n, 24, g)
        idx = torch.randint(0, n, (max(1, n // 200),), generator=g)
        x[idx] += 10 + 5 * torch.rand(idx.shape, generator=g, dtype=torch.float64)
    else:
        raise ValueError(kind)
    level = 100 * torch.randn((), generator=g, dtype=torch.float64)
    scale = 10 ** (2 * torch.rand((), generator=g, dtype=torch.float64) - 1)
    return level + scale * x if kind != "intermittent" else x * scale.ceil()


def realistic(batch, length, seed, nan_rows=0, nan_frac=0.02):
    """(batch, length, 1) float32; the first ``nan_rows`` rows get ~nan_frac interior NaNs."""
    g = torch.Generator().manual_seed(seed)
    rows = [_one(KINDS[i % len(KINDS)], length, g) for i in range(batch)]
    x = torch.stack(rows).float().unsqueeze(-1)
    for i in range(min(nan_rows, batch)):
        hit = torch.rand(length, generator=g) < nan_frac
        hit[: length // 10] = False
        hit[-8:] = False
        x[i, hit, 0] = float("nan")
    return x


def nan_mix(length, seed):
    """`nan_mix` path case: interior NaNs, a leading NaN run (n_t = 0 -> 1), and one constant series."""
    x = realistic(4, length, seed, nan_rows=2, nan_frac=0.05)
    x[1, :37, 0] = float("nan")
    x[3, :, 0] = 7.25
    return x
