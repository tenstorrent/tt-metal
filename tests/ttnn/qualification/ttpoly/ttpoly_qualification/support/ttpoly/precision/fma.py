#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ttpoly.precision.fma — bit-exact SFPU fused multiply-add.

SINGLE OWNER of the multiply-add primitive. `fma_bh`
computes ``x*y + z`` exactly as the Blackhole Tensix Vector Unit (SFPU) does.

Provenance: faithfully ported from competition/precision_utils.py:fma_model_bh,
which is itself ported from the tt-isa-documentation FMA reference (fma.c). The
bit-level uint32->uint32 form below mirrors craq-sim's
``src/tensix.cpp:fma_model`` (the SFPMAD 0x84 model) so that
ttpoly.precision.reciprocal can compose it the same way the SFPARECIP load-macro
does in silicon.

Validation status: byte-identical to the existing Python reference
(tests/test_precision_phase1.py asserts fma_bh == precision_utils.fma_model_bh
over a randomized + edge-case vector set). Golden-vector validation against
craq-sim SFPMAD per-arch remains a Phase-1 verification item (risk:
"ttsim fidelity is the foundation").
"""

import struct

import numpy as np


def _f2u(v):
    return struct.unpack("I", struct.pack("f", np.float32(v)))[0]


def _u2f(u):
    return struct.unpack("f", struct.pack("I", u & 0xFFFFFFFF))[0]


def _clz(val):
    """Count leading zeros in a 32-bit value."""
    if val == 0:
        return 32
    count = 0
    mask = 0x80000000
    while (val & mask) == 0:
        count += 1
        mask >>= 1
    return count


def fma_bits_bh(x, y, z):
    """SFPU FMA on raw FP32 bit patterns: returns bits of ``x*y + z``.

    All three args and the result are uint32 FP32 bit patterns. This mirrors
    craq-sim ``fma_model`` exactly and is what reciprocal.py composes.
    """
    x &= 0xFFFFFFFF
    y &= 0xFFFFFFFF
    z &= 0xFFFFFFFF

    def unpack_no_denorms(var):
        e = (var >> 23) & 255
        m = (var & 0x7FFFFF) ^ 0x800000  # implicit bit
        if e == 0:  # flush denormals
            m = 0
        return e, m

    x_e, x_m = unpack_no_denorms(x)
    y_e, y_m = unpack_no_denorms(y)
    z_e, z_m = unpack_no_denorms(z)
    z_sign = z & 0x80000000

    # p = x * y
    p_sign = (x ^ y) & 0x80000000
    p_m = x_m * y_m
    p_e = x_e + y_e - 23 - 127

    # Three extra precision bits (G, R, S).
    p_m <<= 3
    z_m <<= 3

    # Realign p_m to z_m (remove 23 bits, keep sticky).
    p_m = (p_m >> 23) | (1 if (p_m & 0x7FFFFF) != 0 else 0)
    p_e += 23

    # NaN / Inf handling.
    if x_e == 255 or y_e == 255 or p_e >= 255 or z_e == 255:
        if (
            (x_e == 255 and (x_m != 0x800000 or y_m == 0))
            or (y_e == 255 and (y_m != 0x800000 or x_m == 0))
            or (z_e == 255 and z_m != 0x4000000)
            or (z_e == 255 and (x_e == 255 or y_e == 255) and (z_sign != p_sign))
        ):
            return 0x7FC00000  # NaN
        elif z_e == 255:
            return z  # z Inf
        else:
            return p_sign | 0x7F800000  # Inf

    # Shortcut: p == 0 or multiply underflow.
    if p_m == 0 or p_e < 0:
        return z if z_m else (z_sign & p_sign)

    def semi_sticky_shift(var, amount):
        if amount >= 32:
            return 0
        orig = var
        var >>= amount
        if var and ((var << amount) != orig):
            var |= 1
        return var

    # r = z + p
    r_e = max(p_e, z_e)
    if p_e < r_e:
        p_m = semi_sticky_shift(p_m, r_e - p_e)
    if z_e < r_e:
        z_m = semi_sticky_shift(z_m, r_e - z_e)

    r_sign = p_sign if p_m >= z_m else z_sign
    if z_sign != r_sign:
        z_m = (~z_m) & 0xFFFFFFFF
    if p_sign != r_sign:
        p_m = (~p_m) & 0xFFFFFFFF

    r_m = (z_m + p_m + (1 if p_sign != z_sign else 0)) & 0xFFFFFFFF

    if r_m == 0:
        return z_sign & p_sign

    # Normalize to 5 zero bits, 1 one bit, 26 fractional bits.
    n = 5 - _clz(r_m)
    r_e += n

    if r_e >= 255:
        return r_sign | 0x7F800000  # Inf

    if r_e <= 0:
        n += 1
        r_e = 0

    if n <= 0:
        r_m <<= -n
    else:
        # NOTE: ``(n | 1)`` mask is reproduced verbatim from the C reference /
        # precision_utils.py. Do NOT "fix" to ((1<<n)-1) during the port — the
        # goal is bit-exact reproduction; any change is a separate, validated PR
        # (normalization is silent-corruption-prone).
        r_m = (r_m >> n) | (1 if (r_m & (n | 1)) != 0 else 0)

    r_m &= 0xFFFFFFFF

    r = (r_e << 23) + ((r_m >> 3) & 0x7FFFFF)
    # Round to nearest even.
    r += 1 if (((r_m & 7) + (r & 1)) > 4) else 0
    # Flush denormals after rounding (preserve sign).
    if not (r >> 23):
        r = 0

    return (r_sign | r) & 0xFFFFFFFF


def fma_bh(x_val, y_val, z_val):
    """Blackhole SFPU FMA: ``x*y + z`` as float32 (bit-exact).

    Faithful port of competition/precision_utils.py:fma_model_bh, verified
    bit-for-bit against craq-sim src/fma.cpp:fma_model_bh.
    """
    return np.float32(_u2f(fma_bits_bh(_f2u(x_val), _f2u(y_val), _f2u(z_val))))


def fma_bits_wh(x, y, z):
    """Wormhole SFPU FMA on raw FP32 bit patterns: bits of ``x*y + z``.

    Faithful port of craq-sim src/fma.cpp:fma_model_wh. WH is NOT the same as BH:
    NaN results leak mantissa bits (0x7f800001) rather than returning a canonical
    qNaN, the underflow shortcut returns +0 (not z_sign&p_sign), the r_e<0 path
    flushes instead of denormalizing, and the normalize sticky mask is
    ``(r_m & 1)`` rather than BH's ``(r_m & (n|1))``. Validated bit-for-bit
    against craq-sim (tests/craq_sim_golden).
    """
    x &= 0xFFFFFFFF
    y &= 0xFFFFFFFF
    z &= 0xFFFFFFFF

    def unpack_no_denorms(var):
        e = (var >> 23) & 255
        m = (var & 0x7FFFFF) ^ 0x800000
        if e == 0:
            m = 0
        return e, m

    x_e, x_m = unpack_no_denorms(x)
    y_e, y_m = unpack_no_denorms(y)
    z_e, z_m = unpack_no_denorms(z)
    z_sign = z & 0x80000000

    p_sign = (x ^ y) & 0x80000000
    p_m = x_m * y_m
    p_e = x_e + y_e - 23 - 127

    p_m <<= 3
    z_m <<= 3

    p_m = (p_m >> 23) | (1 if (p_m & 0x7FFFFF) != 0 else 0)
    p_e += 23

    # NaN/Inf: WH may continue with a nan_result that leaks mantissa bits.
    nan_result = 0
    if x_e == 255 or y_e == 255 or p_e >= 255 or z_e == 255:
        if (
            (x_e == 255 and (x_m != 0x800000 or y_m == 0))
            or (y_e == 255 and (y_m != 0x800000 or x_m == 0))
            or (z_e == 255 and z_m == 0x4000000 and (x_e == 255 or y_e == 255 or p_e >= 255) and (z_sign != p_sign))
        ):
            nan_result = p_sign | 0x7F800001
        elif z_e == 255 and z_m != 0x4000000:  # z NaN
            nan_result = z_sign | 0x7F800001
        elif z_e == 255:  # z Inf
            return z
        else:  # (x*y) Inf
            return p_sign | 0x7F800000
        if p_e > 255:
            p_e = 255

    if p_m == 0 or p_e < 0:
        if nan_result:
            p_m = 0
            p_e = 0
        else:
            return z if z_m else 0

    def semi_sticky_shift(var, amount):
        if amount >= 32:
            return 0
        orig = var
        var >>= amount
        if var and ((var << amount) != orig):
            var |= 1
        return var

    r_e = max(p_e, z_e)
    if p_e < r_e:
        p_m = semi_sticky_shift(p_m, r_e - p_e)
    if z_e < r_e:
        z_m = semi_sticky_shift(z_m, r_e - z_e)

    r_sign = p_sign if p_m >= z_m else z_sign
    if z_sign != r_sign:
        z_m = (~z_m) & 0xFFFFFFFF
    if p_sign != r_sign:
        p_m = (~p_m) & 0xFFFFFFFF

    r_m = (z_m + p_m + (1 if p_sign != z_sign else 0)) & 0xFFFFFFFF

    if r_m == 0:
        return nan_result

    n = 5 - _clz(r_m)
    r_e += n
    if r_e >= 255:
        return nan_result if nan_result else (r_sign | 0x7F800000)
    if r_e < 0:  # flush blatant denormals (WH: < 0, not <= 0)
        return nan_result
    if n <= 0:
        r_m <<= -n
    else:
        r_m = (r_m >> n) | (r_m & 1)  # WH sticky mask is (r_m & 1), NOT (n|1)

    r_m &= 0xFFFFFFFF
    r = (r_e << 23) + ((r_m >> 3) & 0x7FFFFF)
    r += 1 if (((r_m & 7) + (r & 1)) > 4) else 0
    if not (r >> 23):  # flush denormals after rounding
        return nan_result
    return ((nan_result if nan_result else r_sign) | r) & 0xFFFFFFFF


def fma_wh(x_val, y_val, z_val):
    """Wormhole SFPU FMA: ``x*y + z`` as float32 (bit-exact, distinct from BH)."""
    return np.float32(_u2f(fma_bits_wh(_f2u(x_val), _f2u(y_val), _f2u(z_val))))


_FMA_BY_ARCH = {"bh": fma_bh, "wh": fma_wh}


def fma(x_val, y_val, z_val, arch="bh"):
    """Dispatch the SFPU FMA for a given arch ('bh' | 'wh')."""
    try:
        return _FMA_BY_ARCH[arch](x_val, y_val, z_val)
    except KeyError:
        raise ValueError(f"Unknown arch for FMA: {arch!r}. Use 'bh' or 'wh'.")
