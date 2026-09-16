#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ttpoly.precision.bf16 — BF16/FP16 quantization, FTZ, and the BF16 grid.

SINGLE OWNER of every float-format conversion in the precision model.
Faithfully absorbs:
  - competition/precision_utils.py: to_bf16 (RNE/RTZ), _apply_ftz_bf16, to_fp16
  - bf16_grid.py: bf16 grid enumeration + subnormal filter

`horner_fp32` from bf16_grid.py is intentionally NOT ported here — eval.py is
the only Horner in the new tree.

Rounding-mode policy (the silicon convention, see spec/units.py RoundingMode):
  - x -> bf16 reference downcast uses RNE, and the modeled hardware output cast
    is also RNE (eval.py:_cast_output passes rounding_mode="rne"). The RTZ enum
    is a legacy SFPU-CAST label retained for any path that selects it explicitly;
    callers choose the mode explicitly rather than defaulting.
"""

import numpy as np

try:
    import ml_dtypes

    HAS_ML_DTYPES = True
except ImportError:  # pragma: no cover - environment without ml_dtypes
    HAS_ML_DTYPES = False

# Bit masks shared across this module (FP32 layout).
_EXP_MASK = np.uint32(0x7F800000)  # bits 30-23
_ABS_MASK = np.uint32(0x7FFFFFFF)  # all but sign
_SIGN_MASK = np.uint32(0x80000000)  # sign bit
# Smallest BF16/FP32 normal magnitude (shared exponent range): 2**-126.
MIN_NORMAL = 2.0**-126


def bh_bf16_ingress_inputs(raw_bf16_bits: np.ndarray) -> np.ndarray:
    """Model the production BH Float16_b SrcA/datacopy ingress quotient."""
    raw = np.asarray(raw_bf16_bits, dtype=np.uint16)
    effective = raw.astype(np.uint32) << np.uint32(16)
    exponent = raw & np.uint16(0x7F80)
    mantissa = raw & np.uint16(0x007F)
    sign = effective & np.uint32(0x80000000)
    # Float16_b SrcA flattens both signed zero and DAZ subnormals to +0.
    effective[exponent == 0] = np.uint32(0)
    # The datacopy path narrows NaN encodings to signed infinity before SFPU
    # predicates or the selected evaluator can observe them.
    nan = (exponent == np.uint16(0x7F80)) & (mantissa != 0)
    effective[nan] = sign[nan] | np.uint32(0x7F800000)
    return effective.view(np.float32)


def to_bf16(x, flush_to_zero=False, rounding_mode="rne"):
    """Round float32 values to BF16 precision and back to float32.

    Faithful port of competition/precision_utils.py:to_bf16.

    Args:
        x: scalar or numpy array of float32 values.
        flush_to_zero: flush BF16 subnormals to signed zero (FTZ).
        rounding_mode: 'rne' (round-to-nearest-even, IEEE default) or
            'rtz' (round-toward-zero / truncation, the SFPU CAST behavior).
    """
    rounding_mode = rounding_mode.lower()
    was_scalar = np.isscalar(x)

    if rounding_mode == "rtz":
        # Truncate the low 16 bits — matches Tenstorrent SFPU CAST.
        x_fp32 = np.atleast_1d(np.asarray(x, dtype=np.float32))
        x_uint32 = x_fp32.view(np.uint32)
        result_arr = (x_uint32 & np.uint32(0xFFFF0000)).view(np.float32)
        result = result_arr.item() if was_scalar else result_arr
    elif rounding_mode == "rne":
        if HAS_ML_DTYPES:
            result_arr = np.array(x, dtype=ml_dtypes.bfloat16).astype(np.float32)
            result = result_arr.item() if was_scalar else result_arr
        else:  # pragma: no cover - manual RNE fallback
            x_fp32 = np.atleast_1d(np.asarray(x, dtype=np.float32))
            x_uint32 = x_fp32.view(np.uint32)
            bias = np.uint32(0x7FFF) + ((x_uint32 >> np.uint32(16)) & np.uint32(1))
            x_uint32_rounded = (x_uint32 + bias) & np.uint32(0xFFFF0000)
            result_arr = x_uint32_rounded.view(np.float32)
            result = result_arr.item() if was_scalar else result_arr
    else:
        raise ValueError(f"Unknown rounding mode: {rounding_mode!r}. Use 'rne' or 'rtz'.")

    if flush_to_zero:
        result = apply_ftz(result)
    return result


def apply_ftz(x):
    """Flush subnormal float32 values to signed zero (FTZ).

    Faithful port of competition/precision_utils.py:_apply_ftz_bf16. Operates on
    the FP32 container; a value is subnormal iff exponent==0 and mantissa!=0.
    """
    was_scalar = np.isscalar(x)
    x_fp32 = np.atleast_1d(np.asarray(x, dtype=np.float32))
    x_uint32 = x_fp32.view(np.uint32)

    is_subnormal = ((x_uint32 & _EXP_MASK) == 0) & ((x_uint32 & _ABS_MASK) != 0)
    x_uint32 = np.where(is_subnormal, x_uint32 & _SIGN_MASK, x_uint32)

    result = x_uint32.view(np.float32)
    return result.item() if was_scalar else result


def to_fp16(x):
    """Round float32 values to FP16 precision and back to float32."""
    return np.array(x, dtype=np.float16).astype(np.float32)


def quantize(x, precision, flush_to_zero=False, rounding_mode="rne"):
    """Quantize to a named precision ('fp32' | 'bf16' | 'fp16')."""
    p = precision.lower()
    if p in ("fp32", "float32"):
        out = np.asarray(x, dtype=np.float32)
        return apply_ftz(out) if flush_to_zero else out
    if p in ("bf16", "bfloat16"):
        return to_bf16(x, flush_to_zero=flush_to_zero, rounding_mode=rounding_mode)
    if p in ("fp16", "half", "float16"):
        return to_fp16(x)
    raise ValueError(f"Unknown precision: {precision!r}")


def bf16_grid(lo, hi, include_subnormals=False):
    """Enumerate every representable BF16 value in [lo, hi].

    Absorbs bf16_grid.py's enumeration + subnormal filter. Returns an ascending
    float32 array of the exact BF16 grid points in range.
    """
    # Every BF16 value is an FP32 value whose low 16 bits are zero. Enumerate the
    # 16-bit BF16 patterns, expand to FP32, and filter to [lo, hi].
    codes = np.arange(0, 1 << 16, dtype=np.uint32)
    fp32 = (codes << np.uint32(16)).view(np.float32)
    finite = np.isfinite(fp32)
    grid = fp32[finite]
    if not include_subnormals:
        u = grid.view(np.uint32)
        is_sub = ((u & _EXP_MASK) == 0) & ((u & _ABS_MASK) != 0)
        grid = grid[~is_sub]
    grid = grid[(grid >= np.float32(lo)) & (grid <= np.float32(hi))]
    return np.sort(grid)
