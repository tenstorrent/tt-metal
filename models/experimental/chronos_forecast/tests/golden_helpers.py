# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Chronos-2 golden stub tests (reference oracle only)."""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.common.chronos_src import CHRONOS_SUBMODULE_ROOT
from models.experimental.chronos_forecast.reference.chronos2.config import Chronos2CoreConfig

DUMMY_MODEL_PATH = CHRONOS_SUBMODULE_ROOT / "test" / "dummy-chronos2-model"
SEED = 0


def tiny_config(**overrides) -> Chronos2CoreConfig:
    """Small CPU-friendly config matching the dummy checkpoint geometry."""
    kwargs = dict(
        d_model=6,
        d_kv=4,
        d_ff=8,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        attn_implementation="eager",
    )
    kwargs.update(overrides)
    return Chronos2CoreConfig(**kwargs)


def log_golden(name: str, t: torch.Tensor, *, max_vals: int = 4) -> dict:
    """Print golden stats for a tensor so future tt/ output can be diffed."""
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
