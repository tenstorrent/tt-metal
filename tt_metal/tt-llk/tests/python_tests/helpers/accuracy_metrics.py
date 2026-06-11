# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shared, pure per-element accuracy metrics for SFPU golden-vs-hardware results.

Single source of truth for the ULP / finite math used by both the accuracy
CSV harness and test_sfpu_plot.py. All functions are pure (numpy in, numpy
out) and do not touch hardware.

ULP semantics: ULP error normalizes the signed error by the local spacing of
the *golden* value in the output format. ULP columns are NaN (never 0) when
undefined — block-float output, non-finite golden/hw, golden == 0, or a
degenerate local ULP of 0. abs/signed/rel error and the finite flags are
always populated (rel_error is NaN only where golden == 0).
"""

from __future__ import annotations

import numpy as np
import torch

from .format_config import DataFormat
from .llk_params import format_dict

# Formats with a defined torch floating dtype usable for true local ULP.
_ULP_FORMATS = (DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32)


def local_ulp(golden: np.ndarray, out_fmt: DataFormat) -> np.ndarray:
    """Local ULP (spacing) of each golden value in *out_fmt*.

    Returns an all-NaN array for formats without a torch float dtype
    (e.g. the block-float Bfp8_b/Bfp4_b/Bfp2_b formats; Tf32 is also
    treated as unsupported here). For supported formats, computes
    nextafter(|golden|, +inf) - |golden| in the target dtype.
    """
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
    """Per-element error metrics for aligned (x, golden, hw) arrays.

    ``x`` is accepted for caller convenience and interface symmetry with the
    broader test harness; it is not consumed by this function.

    Returns a dict of equal-length numpy arrays:
        abs_error, signed_error, rel_error,
        signed_ulp_error, abs_ulp_error,
        is_finite_hw, is_finite_golden
    See module docstring for NaN semantics.
    """
    golden = np.asarray(golden, dtype=np.float64)
    hw = np.asarray(hw, dtype=np.float64)

    if golden.shape != hw.shape:
        raise ValueError(
            f"golden and hw must have the same shape, got "
            f"{golden.shape} vs {hw.shape}"
        )

    signed_error = hw - golden
    abs_error = np.abs(signed_error)

    is_finite_golden = np.isfinite(golden)
    is_finite_hw = np.isfinite(hw)
    finite = is_finite_golden & is_finite_hw

    golden_nonzero = golden != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.where(
            golden_nonzero, np.abs(signed_error) / np.abs(golden), np.nan
        )

    ulp = local_ulp(golden, out_fmt)
    ulp_defined = finite & golden_nonzero & np.isfinite(ulp) & (ulp > 0)
    safe_ulp = np.where(ulp_defined, ulp, 1.0)
    signed_ulp_error = np.where(ulp_defined, signed_error / safe_ulp, np.nan)
    abs_ulp_error = np.abs(signed_ulp_error)

    return {
        "abs_error": abs_error,
        "signed_error": signed_error,
        "rel_error": rel_error,
        "signed_ulp_error": signed_ulp_error,
        "abs_ulp_error": abs_ulp_error,
        "is_finite_hw": is_finite_hw,
        "is_finite_golden": is_finite_golden,
    }
