# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TT B-format rounding, with 16 values per shared exponent.

Numerical behavior follows tt-metal/blockfloat_common.cpp (Apache-2.0,
Copyright 2026 Tenstorrent USA, Inc.). This NumPy implementation adds an
offline exponent search; it does not serialize TT tiles or change kernels.
"""
import numpy as np


def validate_options(bits, deltas):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8 (TT bfloat4_b / bfloat8_b)")
    deltas = tuple(deltas)
    if not deltas or deltas[0] != 0 or len(set(deltas)) != len(deltas):
        raise ValueError("exponent_deltas must start with 0 and contain unique candidates")
    if any(type(d) is not int or d not in (-2, -1, 0, 1) for d in deltas):
        raise ValueError("supported exponent deltas are -2, -1, 0, +1")
    return deltas


def search_numpy(x, bits=4, exponent_deltas=(0, -1)):
    """Reference implementation; groups follow the last axis. No padding here.

    Score each group by float64 sum of squared weight errors. Keep the first
    candidate on an exact tie. TT truncates exponent alignment before RNE.
    """
    deltas = validate_options(bits, exponent_deltas)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim < 1 or not x.size or x.shape[-1] % 16:
        raise ValueError("last dimension must be a positive multiple of 16")
    if not np.isfinite(x).all():
        raise ValueError("weights must be finite float32 values")
    shape = x.shape
    groups = x.reshape(-1, 16)
    raw = groups.view(np.uint32)
    exponents = ((raw >> 23) & 255).astype(np.int32)
    top = exponents.max(-1, keepdims=True)
    mantissa = np.where(exponents == 0, 0, (raw & 0x7FFFFF) | 0x800000).astype(np.uint32)
    best = np.empty_like(groups)
    best_error = np.full(len(groups), np.inf)
    selected = np.zeros(len(groups), dtype=np.int32)
    shift = 24 - (bits - 1)
    for candidate, delta in enumerate(deltas):
        shared = top if delta == 0 else np.clip(top + delta, 1, 254)
        alignment = shared - exponents
        aligned = np.where(
            alignment >= 0, mantissa >> np.clip(alignment, 0, 31), mantissa << np.clip(-alignment, 0, 2)
        ).astype(np.uint32)
        magnitude = aligned >> shift
        remainder = aligned & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        magnitude += ((remainder > half) | ((remainder == half) & ((magnitude & 1) != 0))).astype(np.uint32)
        magnitude = np.minimum(magnitude, (1 << (bits - 1)) - 1)
        q = np.ldexp(magnitude.astype(np.float32), shared - 127 - (bits - 2))
        q = np.where((raw >> 31) != 0, -q, q)
        error = ((groups.astype(np.float64) - q) ** 2).sum(-1)
        better = error < best_error
        best[better], best_error[better], selected[better] = q[better], error[better], candidate
    return best.reshape(shape), {str(d): int(np.count_nonzero(selected == i)) for i, d in enumerate(deltas)}
