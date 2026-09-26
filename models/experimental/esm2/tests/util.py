# SPDX-License-Identifier: MIT
"""Small test helpers (CPU only)."""

from __future__ import annotations

import torch


def nrmse(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Normalized RMSE per contract gates: ||got-ref|| / ||ref||."""
    ref = ref.float()
    got = got.float()
    return ((got - ref).norm() / ref.norm()).item()
