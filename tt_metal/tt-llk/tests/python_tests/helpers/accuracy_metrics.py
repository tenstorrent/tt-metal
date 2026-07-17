# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import torch

from .format_config import DataFormat
from .llk_params import format_dict

# Formats with a defined torch floating dtype usable for true local ULP.
_ULP_FORMATS = (DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32)


def local_ulp(golden: np.ndarray, out_fmt: DataFormat) -> np.ndarray:
    """Gap from each golden value to the next representable number in *out_fmt*."""
    golden = np.asarray(golden, dtype=np.float64)
    if out_fmt not in _ULP_FORMATS:
        return np.full(golden.shape, np.nan, dtype=np.float64)

    torch_dtype = format_dict[out_fmt]
    abs_g = torch.tensor(np.abs(golden), dtype=torch_dtype)
    nxt = torch.nextafter(abs_g, torch.tensor(float("inf"), dtype=torch_dtype))
    return (nxt - abs_g).to(torch.float32).numpy().astype(np.float64)


def compute_pointwise_metrics(
    x: np.ndarray,
    golden: np.ndarray,
    hw: np.ndarray,
    out_fmt: DataFormat,
) -> dict[str, np.ndarray]:
    """Compare hardware vs golden element-by-element and return the error columns."""
    golden = np.asarray(golden, dtype=np.float64)
    hw = np.asarray(hw, dtype=np.float64)

    if golden.shape != hw.shape:
        raise ValueError(
            f"golden and hw must have the same shape, got "
            f"{golden.shape} vs {hw.shape}"
        )

    signed_error = hw - golden

    is_finite_golden = np.isfinite(golden)
    is_finite_hw = np.isfinite(hw)
    finite = is_finite_golden & is_finite_hw

    golden_nonzero = golden != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.where(
            golden_nonzero, np.abs(signed_error) / np.abs(golden), np.nan
        )

    ulp = local_ulp(golden, out_fmt)
    ulp_defined = finite & np.isfinite(ulp) & (ulp > 0)
    safe_ulp = np.where(ulp_defined, ulp, 1.0)
    signed_ulp_error = np.where(ulp_defined, signed_error / safe_ulp, np.nan)

    return {
        "signed_error": signed_error,
        "rel_error": rel_error,
        "signed_ulp_error": signed_ulp_error,
        "is_finite_hw": is_finite_hw,
        "is_finite_golden": is_finite_golden,
    }
