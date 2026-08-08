# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def max_abs_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(prediction - target))


def pcc(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = prediction.reshape(-1).to(torch.float32)
    ref = target.reshape(-1).to(torch.float32)
    pred = pred - pred.mean()
    ref = ref - ref.mean()
    denom = torch.sqrt(torch.sum(pred * pred) * torch.sum(ref * ref)).clamp_min(eps)
    return torch.sum(pred * ref) / denom


def summarize_regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mse": float(mse(prediction, target)),
        "mae": float(mae(prediction, target)),
        "max_abs_error": float(max_abs_error(prediction, target)),
        "pcc": float(pcc(prediction, target)),
    }
