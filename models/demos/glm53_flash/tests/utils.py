# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() != b.numel():
        raise ValueError(f"pcc shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if torch.isnan(b).any():
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])
