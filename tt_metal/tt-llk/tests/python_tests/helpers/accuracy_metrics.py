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
    # Asked through helpers.ulp rather than against a local tuple, so the proxy formats
    # it gates (Bfp8_b in bfloat16 space) are measured here too. Probing a private copy
    # of the native set left the sweep writing NaN for exactly the format the gate can
    # judge.
    #
    # For Bfp8_b the value returned is a *bfloat16* step, not a Bfp8_b one, so the
    # docstring's "gap to the next representable number in out_fmt" is the proxy's gap
    # rather than the format's: at least 2x the Bfp8_b step (its 7 magnitude bits include
    # an explicit leading 1, leaving 6 fractional against bfloat16's 7) and far more where
    # the shared block exponent coarsens a small element. So the signed_ulp_error column
    # for Bfp8_b reads in bf16 steps -- worth knowing, since the CSV is where someone
    # picks a budget. See _ULP_PROXY_DTYPES in helpers.ulp.
    if not has_ulp_gate(out_fmt):
        return np.full(golden.shape, np.nan, dtype=np.float64)

    torch_dtype = ulp_dtype(out_fmt)
    abs_g = torch.tensor(np.abs(golden), dtype=torch_dtype)
    nxt = torch.nextafter(abs_g, torch.tensor(float("inf"), dtype=torch_dtype))
    step = (nxt - abs_g).to(torch.float32).numpy().astype(np.float64)
    # Same finfo.max fixup local_step carries: nextafter from the largest finite goes to
    # Inf, so the gap is infinite where the binade downward is the same size. Without it
    # the docstring claim that the two share one nextafter definition fails at the top of
    # the range.
    largest = float(torch.finfo(torch_dtype).max)
    # From the converted tensor, not the float64 input: abs_g is the value whose spacing
    # is being measured, and a golden that *rounds* to the format maximum is at the top of
    # the range even though the input is not equal to it. An fp16 65503 rounds to 65504,
    # which the float64 compare misses, leaving the upward nextafter gap at infinity.
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
