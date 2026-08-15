# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import math
import os

import torch
from loguru import logger

import ttnn


def _record_gate_pcc(pcc_found: float) -> None:
    """Append this call's PCC to the DiffVAE gate ledger, when one is requested.

    Enabled only when ``DIFFVAE_GATES_LEDGER`` points at a path; a no-op otherwise, so normal
    test runs are untouched. Append-only JSONL (no read-modify-write race across parametrized
    cases) keyed by pytest's own ``PYTEST_CURRENT_TEST``; the compare step takes the last record
    per test. Recorded before any threshold raise, so a *failing* gate still logs its PCC.
    """
    ledger = os.environ.get("DIFFVAE_GATES_LEDGER")
    if not ledger:
        return
    try:
        with open(ledger, "a") as handle:
            handle.write(
                json.dumps({"test": os.environ.get("PYTEST_CURRENT_TEST", "?"), "pcc": float(pcc_found)}) + "\n"
            )
    except OSError:
        pass


def assert_quality(
    a: ttnn.Tensor | torch.Tensor,
    b: ttnn.Tensor | torch.Tensor,
    *,
    pcc: float | None = None,
    ccc: float | None = None,
    mse: float | None = None,
    relative_rmse: float | None = None,
) -> float:
    """Assert output quality (PCC/CCC/RMSE) and return the measured PCC."""
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
    # beta_found = cov[0, 1] / cov[0, 0]
    ccc_found = 2 * pcc_found * std_a * std_b / (std_a**2 + std_b**2 + (mean_a - mean_b) ** 2)
    relative_rmse_found = torch.nn.functional.mse_loss(a, b).sqrt().item() / std_a

    if mse is not None:
        relative_rmse = math.sqrt(mse) / std_a

    # logger.info(f"μ₁ = {mean_a:.3g}, μ₂ = {mean_b:.3g}, σ₁ = {std_a:.3g}, σ₂ = {std_b:.3g}")
    logger.info(
        f"PCC = {pcc_found * 100:.4f} %, "
        # f"β = {beta_found * 100:.1f} %, "
        f"CCC = {ccc_found * 100:.4f} %, "
        f"RMSE/σ₁ = {relative_rmse_found * 100:.1f} %"
    )

    _record_gate_pcc(pcc_found)

    if pcc is not None and (math.isnan(pcc_found) or pcc_found < pcc):
        msg = f"PCC = {pcc_found * 100:.4f} % >= {pcc * 100:.4f} %"
        raise Exception(msg)  # noqa: TRY002

    if ccc is not None and (math.isnan(ccc_found) or ccc_found < ccc):
        msg = f"CCC = {ccc_found * 100:.4f} % >= {ccc * 100:.4f} %"
        raise Exception(msg)  # noqa: TRY002

    if relative_rmse is not None and (math.isnan(relative_rmse_found) or relative_rmse_found > relative_rmse):
        msg = f"RMSE/σ₁ = {relative_rmse_found * 100:.1f} % <= {relative_rmse * 100:.1f} %"
        raise Exception(msg)  # noqa: TRY002

    return pcc_found
