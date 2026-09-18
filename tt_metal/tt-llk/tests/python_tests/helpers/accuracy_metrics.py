# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import torch

from .format_config import DataFormat
from .ulp import has_ulp_gate, ulp_dtype


def local_ulp(golden: np.ndarray, out_fmt: DataFormat) -> np.ndarray:
    """Gap from each golden value to the next representable number in *out_fmt*."""
    golden = np.asarray(golden, dtype=np.float64)
    # Asked through helpers.ulp, so the proxy formats it gates are measured here too; a
    # private copy of the native set left the sweep writing NaN for exactly the format
    # the gate can judge. For Bfp8_b the step returned is a *bfloat16* one -- at least
    # twice the Bfp8_b step, and far more where a shared block exponent coarsens a small
    # element -- so the `signed_ulp_error` column reads in bf16 steps for that format.
    if not has_ulp_gate(out_fmt):
        return np.full(golden.shape, np.nan, dtype=np.float64)

    torch_dtype = ulp_dtype(out_fmt)
    abs_g = torch.tensor(np.abs(golden), dtype=torch_dtype)
    nxt = torch.nextafter(abs_g, torch.tensor(float("inf"), dtype=torch_dtype))
    step = (nxt - abs_g).to(torch.float32).numpy().astype(np.float64)
    # The same finfo.max fixup local_step carries: nextafter from the largest finite goes
    # to Inf, and the binade downward is the same size. Taken from the converted tensor
    # rather than the float64 input, because a golden that *rounds* to the format maximum
    # is at the top of the range too -- an fp16 65503 rounds to 65504, which a float64
    # compare misses, leaving the gap at infinity.
    largest = float(torch.finfo(torch_dtype).max)
    at_max = (abs_g == largest).numpy()
    if at_max.any():
        top = torch.tensor(largest, dtype=torch_dtype)
        below = torch.nextafter(top, torch.tensor(0.0, dtype=torch_dtype))
        step = np.where(at_max, float((top - below).to(torch.float32)), step)
    return step


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
