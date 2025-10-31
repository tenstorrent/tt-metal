# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn
from loguru import logger


def assert_quality(
    a: ttnn.Tensor | torch.Tensor,
    b: ttnn.Tensor | torch.Tensor,
    *,
    pcc: float | None = None,
    ccc: float | None = None,
    mse: float | None = None,
    relative_rmse: float | None = None,
) -> None:
    if math.prod(a.shape) != math.prod(b.shape):
        msg = f"incompatible shapes: {a.shape} != {b.shape}"
        raise ValueError(msg)

    if a.shape != b.shape:
        logger.warning(f"shape mismatch: {a.shape} != {b.shape}")

    a = a.detach().flatten().to(torch.float64)
    b = b.detach().flatten().to(torch.float64)

    cov = torch.cov(torch.stack([a, b])).numpy()

    std_a = math.sqrt(cov[0, 0])
    std_b = math.sqrt(cov[1, 1])
    mean_a = a.mean().item()
    mean_b = b.mean().item()

    pcc_found = cov[0, 1] / (std_a * std_b)
    beta_found = cov[0, 1] / cov[0, 0]
    ccc_found = 2 * pcc_found * std_a * std_b / (std_a**2 + std_b**2 + (mean_a - mean_b) ** 2)
    relative_rmse_found = torch.nn.functional.mse_loss(a, b).sqrt().item() / std_a

    if mse is not None:
        relative_rmse = math.sqrt(mse) / std_a

    logger.info(f"μ₁ = {mean_a:.3g}, μ₂ = {mean_b:.3g}, σ₁ = {std_a:.3g}, σ₂ = {std_b:.3g}")
    logger.info(
        f"PCC = {pcc_found * 100:.4f} %, "
        f"β = {beta_found * 100:.1f} %, "
        f"CCC = {ccc_found * 100:.4f} %, "
        f"RMSE/σ₁ = {relative_rmse_found * 100:.1f} %"
    )

    if pcc is not None and (math.isnan(pcc_found) or pcc_found < pcc):
        msg = f"PCC = {pcc_found * 100:.4f} % >= {pcc * 100:.4f} %"
        raise Exception(msg)  # noqa: TRY002

    if ccc is not None and (math.isnan(ccc_found) or ccc_found < ccc):
        msg = f"CCC = {ccc_found * 100:.4f} % >= {ccc * 100:.4f} %"
        raise Exception(msg)  # noqa: TRY002

    if relative_rmse is not None and (math.isnan(relative_rmse_found) or relative_rmse_found > relative_rmse):
        msg = f"RMSE/σ₁ = {relative_rmse_found * 100:.1f} % <= {relative_rmse * 100:.1f} %"
        raise Exception(msg)  # noqa: TRY002
