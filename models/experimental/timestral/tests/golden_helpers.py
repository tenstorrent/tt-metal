# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Timestral golden tests (upstream reference as the oracle)."""

from __future__ import annotations

import math

import pytest
import torch

from models.experimental.timestral.common.timestral_src import (
    checkpoint_path,
    checkpoint_ready,
    download_hint,
    require_timestral,
)

require_timestral()

from timestral.config import TimestralConfig  # noqa: E402
from timestral.loading import load_model  # noqa: E402
from timestral.model import TimestralModel  # noqa: E402

__all__ = [
    "QUANTILE_KNOTS",
    "SEED",
    "load_checkpoint",
    "log_golden",
    "series",
    "tiny_config",
    "tiny_model",
]

SEED = 0

# Published mistralai/Timestral-* quantile levels.
QUANTILE_KNOTS = (
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
)


def tiny_config(**overrides) -> TimestralConfig:
    """Small config with the released checkpoints' layout: patch 32, head_dim 64, both layer kinds."""
    kwargs = dict(
        patch_size=32,
        d_model=128,
        num_heads=2,
        mlp_hidden_dim=256,
        block_schedule=("time", "time_variate"),
        quantile_knots=QUANTILE_KNOTS,
        block_size=32,
        context_length=4096,
        quantile_width_factor=3.0,
    )
    kwargs.update(overrides)
    return TimestralConfig(**kwargs)


def tiny_model(config: TimestralConfig | None = None, seed: int = SEED) -> TimestralModel:
    """Randomly initialized model whose norms and forecast query are away from their trivial init."""
    torch.manual_seed(seed)
    model = TimestralModel(config or tiny_config())
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".scale"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name.endswith("forecast_query"):
                p.copy_(torch.randn_like(p))
    return model.eval()


def load_checkpoint(name: str) -> TimestralModel:
    """Load a downloaded ``mistralai/<name>`` checkpoint, skipping the test when it is absent."""
    if not checkpoint_ready(name):
        pytest.skip(download_hint(name))
    return load_model(checkpoint_path(name))


def series(batch: int, variates: int, length: int, seed: int = SEED) -> torch.Tensor:
    """`(batch, variates, length)` seasonal random walks, the upstream test's synthetic data."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    return 10 * torch.sin(2 * math.pi * t / 24) + torch.randn(batch, variates, length, generator=g).cumsum(-1)


def log_golden(name: str, t: torch.Tensor, *, max_vals: int = 4) -> dict:
    """Print golden stats for a tensor so tt/ output can be diffed against it."""
    flat = t.detach().to(torch.float32).flatten()
    finite = flat[torch.isfinite(flat)]
    stats = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": float(finite.mean()) if finite.numel() else float("nan"),
        "std": float(finite.std()) if finite.numel() > 1 else 0.0,
        "min": float(finite.min()) if finite.numel() else float("nan"),
        "max": float(finite.max()) if finite.numel() else float("nan"),
        "first": [float(v) for v in finite[:max_vals]],
    }
    print(f"\n[GOLDEN] {name}: {stats}")
    return stats
