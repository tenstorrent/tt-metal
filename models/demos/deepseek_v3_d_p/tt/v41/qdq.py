# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device quantize-dequantize (QDQ) of DeepSeek-V4.1 activations (dev-spec D-H, D-I; bead F6.1a).

The reference quantizes activations in groups along the last dimension and immediately dequantizes
them back to bf16 (``kernel_cpu.act_quant`` / ``fp4_act_quant`` with ``inplace=True``). These functions
return the same bf16 values, bit for bit, given the same bf16 input:

* :func:`fp8_qdq`        -- FP8 e4m3, group 32, power-of-two (ue8m0) scale: window KV, DSpark rings and
  the activations entering quantized GEMMs.
* :func:`fp4_ue8m0_qdq`  -- FP4 E2M1, group 32, power-of-two scale: index keys and indexer queries.
* :func:`fp4_e4m3_qdq`   -- FP4 E2M1, group 16, e4m3 scale: compressed KV.

Every op is local to a row (a token) and to its device: no collectives, so replicated or
sequence-sharded mesh tensors work unchanged.

Exactness argument (all intermediates fp32; the device's fp32 mul/sub/floor/compare are exact whenever
the true result is representable, which is all this code relies on except where noted):

* Group amax is a max of bf16 magnitudes, hence exact.
* Power-of-two scales are built from the amax bit pattern. ``2^ceil(log2(fl(amax * fp32(1/M))))``
  (``fast_round_scale``) equals ``2^(e - k + [mantissa(amax) > m*])`` for bf16 amax, where
  ``M = 448`` (``k = 8``, ``m* = 1.75``) or ``M = 6`` (``k = 2``, ``m* = 1.5``), clamped from below by the
  scale of the amax floor (monotone). Checked on CPU for every positive finite bf16.
  Dividing by a power of two is exact, so the reference's ``x / s`` equals ``x * (1/s)`` here.
* e4m3 round-to-nearest-even (RNE) scales the magnitude by the inverse quantum
  ``2^max(e - 3, -9)`` (exact), splits it with ``floor`` and resolves exact halves to the even integer.
  Saturation (clamp to 448 before the cast) equals rounding then ``min(., 448)``.
* The e4m3 scale ``e4m3(amax_f / 6)`` of the group-16 FP4 path starts from ``amax_f * fp32(1/6)``,
  which may differ from the reference's correctly rounded ``amax_f / 6`` by an ulp. That only matters
  at an exact e4m3 half-way point (all other values lie >= 2^-9 relative away from one), and a tie is
  detected exactly as ``6 * midpoint == amax_f`` (7-bit product, exact).
* E2M1 rounding of ``v = x / s`` compares ``|x|`` against ``midpoint * s`` (products of <= 3 and <= 4
  significant bits, exact) instead of forming ``v``: for the non-power-of-two e4m3 scale ``x / s`` is
  inexact, but it lands on a midpoint exactly when ``x == midpoint * s`` and otherwise stays far from
  one, so the comparison reproduces the reference's rounding, ties included; ``> 5 s`` gives the
  clamp to 6.
* Dequantized values have <= 6 significant bits and a normal exponent, so the final bf16 cast is exact.

Not reproduced: bf16 subnormal inputs where the reference's result depends on them. The device flushes
subnormals to zero; the reference keeps them. This affects only :func:`fp4_ue8m0_qdq` groups whose
amax is below ~2^-121 (elsewhere a subnormal input quantizes to zero in the reference too).
"""

import ttnn

FP8_GROUP = 32
FP4_UE8M0_GROUP = 32
FP4_E4M3_GROUP = 16

_E4M3_MAX = 448.0
_FP4_MAX = 6.0

_EXP_MASK = 0x7F800000
_MANT_MASK = 0x007FFFFF
_IMPLICIT_ONE = 0x00800000
_ONE_EXP = 127 << 23  # 2^k has bits (k + 127) << 23; its inverse has bits (2 * 127 << 23) - bits.

# Scales of the amax floors: fast_round_scale(fp32(1e-4) * fp32(1/448)) = 2^-22 (act_quant's
# clamp_min(1e-4)); fast_round_scale(6 * 2^-126 * fp32(1/6)) = 2^-126 (fp4_act_quant, E8M0 scale).
_FP8_SCALE_FLOOR_BITS = (-22 + 127) << 23
_FP4_SCALE_FLOOR_BITS = (-126 + 127) << 23
# fp4_act_quant with an e4m3 scale floors amax at 6 * 2^-9 so that the scale is at least 2^-9.
_FP4_E4M3_AMAX_FLOOR = 6.0 * 2.0**-9

# E2M1 rounding boundaries between consecutive magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}, whether an exact
# tie rounds up (the upper code is even), and the magnitude step taken when crossing the boundary.
_E2M1_STEPS = (
    (0.25, False, 0.5),
    (0.75, True, 0.5),
    (1.25, False, 0.5),
    (1.75, True, 0.5),
    (2.5, False, 1.0),
    (3.5, True, 1.0),
    (5.0, False, 2.0),
)


def fp8_qdq(x: ttnn.Tensor) -> ttnn.Tensor:
    """``act_quant(x, 32, "ue8m0", inplace=True)``: FP8 e4m3 QDQ, group 32, power-of-two scale."""
    xg, xf = _groups(x, FP8_GROUP)
    s_bits = _pow2_scale_bits(_group_amax(xg), mantissa_bias=8, mantissa_threshold=0x600000)
    s_bits = ttnn.maximum(s_bits, _FP8_SCALE_FLOOR_BITS)
    y = ttnn.abs(ttnn.multiply(xf, _inverse_pow2(s_bits)))
    q = _round_e4m3(y, is_tie=lambda midpoint: ttnn.eq(midpoint, y))
    return _dequant(q, ttnn.bitcast(s_bits, ttnn.float32), xf, x)


def fp4_ue8m0_qdq(x: ttnn.Tensor) -> ttnn.Tensor:
    """``fp4_act_quant(x, 32, inplace=True)``: FP4 E2M1 QDQ, group 32, power-of-two (E8M0) scale."""
    xg, xf = _groups(x, FP4_UE8M0_GROUP)
    s_bits = _pow2_scale_bits(_group_amax(xg), mantissa_bias=2, mantissa_threshold=0x400000)
    s_bits = ttnn.maximum(s_bits, _FP4_SCALE_FLOOR_BITS)
    s = ttnn.bitcast(s_bits, ttnn.float32)
    return _dequant(_round_e2m1(xf, s), s, xf, x)


def fp4_e4m3_qdq(x: ttnn.Tensor) -> ttnn.Tensor:
    """``fp4_act_quant(x, 16, inplace=True, scale_dtype=float8_e4m3fn)``: FP4 E2M1 QDQ, group 16,
    scale ``e4m3_satfinite(max(amax, 6 * 2^-9) / 6)``."""
    xg, xf = _groups(x, FP4_E4M3_GROUP)
    amax = ttnn.maximum(_group_amax(xg), _FP4_E4M3_AMAX_FLOOR)
    s_estimate = ttnn.multiply(amax, 1.0 / _FP4_MAX)
    s = _round_e4m3(s_estimate, is_tie=lambda midpoint: ttnn.eq(ttnn.multiply(midpoint, _FP4_MAX), amax))
    return _dequant(_round_e2m1(xf, s), s, xf, x)


def _groups(x: ttnn.Tensor, group: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """[..., W] bf16 -> ([N, group] bf16, [N, group] fp32) with one quantization group per row."""
    assert x.dtype == ttnn.bfloat16, f"QDQ input must be bf16 (the reference kernels' input dtype), got {x.dtype}"
    assert x.layout == ttnn.TILE_LAYOUT, "QDQ input must be TILE layout"
    width = x.shape[-1]
    assert width % group == 0, f"last dim {width} is not a multiple of the group size {group}"
    xg = ttnn.reshape(x, (1, 1, x.volume() // group, group))
    return xg, ttnn.typecast(xg, ttnn.float32)


def _group_amax(xg: ttnn.Tensor) -> ttnn.Tensor:
    """[N, group] bf16 -> [N, 1] fp32 max magnitude (exact: a max of bf16 values)."""
    return ttnn.typecast(ttnn.max(ttnn.abs(xg), dim=-1, keepdim=True), ttnn.float32)


def _pow2_scale_bits(amax: ttnn.Tensor, mantissa_bias: int, mantissa_threshold: int) -> ttnn.Tensor:
    """fp32 bits of 2^(exponent(amax) - bias + [mantissa(amax) > threshold]) as int32 (before flooring)."""
    bits = ttnn.bitcast(amax, ttnn.int32)
    exponent = ttnn.subtract(ttnn.bitwise_and(bits, _EXP_MASK), mantissa_bias << 23)
    # mantissa > threshold  <=>  mantissa + (2^23 - 1 - threshold) carries into bit 23.
    carry = ttnn.bitwise_and(
        ttnn.add(ttnn.bitwise_and(bits, _MANT_MASK), _MANT_MASK - mantissa_threshold), _IMPLICIT_ONE
    )
    return ttnn.add(exponent, carry)


def _inverse_pow2(pow2_bits: ttnn.Tensor) -> ttnn.Tensor:
    """fp32 1/2^k from the int32 bits of 2^k (exact)."""
    return ttnn.bitcast(ttnn.rsub(pow2_bits, 2 * _ONE_EXP), ttnn.float32)


def _round_e4m3(t: ttnn.Tensor, is_tie) -> ttnn.Tensor:
    """Non-negative fp32 -> nearest e4m3 magnitude (RNE, saturating at 448), as fp32.

    ``is_tie(midpoint)`` says whether the exact value being rounded equals ``midpoint``, the half-way
    point above ``floor(t / quantum) * quantum``; it lets ``t`` be an estimate within a few ulp (see
    the module docstring).
    """
    exponent = ttnn.bitwise_and(ttnn.bitcast(t, ttnn.int32), _EXP_MASK)
    # quantum: 3 mantissa bits for normal values, 2^-9 (the smallest subnormal) below 2^-6
    quantum_bits = ttnn.maximum(ttnn.subtract(exponent, 3 << 23), (-9 + 127) << 23)
    quantum = ttnn.bitcast(quantum_bits, ttnn.float32)
    z = ttnn.multiply(t, _inverse_pow2(quantum_bits))
    lower = ttnn.floor(z)
    fraction = ttnn.subtract(z, lower)
    lower_is_odd = ttnn.subtract(lower, ttnn.multiply(ttnn.floor(ttnn.multiply(lower, 0.5)), 2.0))
    midpoint = ttnn.multiply(ttnn.add(lower, 0.5), quantum)
    round_up = ttnn.where(is_tie(midpoint), lower_is_odd, ttnn.gt(fraction, 0.5))
    return ttnn.minimum(ttnn.multiply(ttnn.add(lower, round_up), quantum), _E4M3_MAX)


def _round_e2m1(xf: ttnn.Tensor, s: ttnn.Tensor) -> ttnn.Tensor:
    """Nearest E2M1 magnitude (RNE) of clamp(|x| / s, 0, 6), as fp32 [N, group]."""
    magnitude = ttnn.abs(xf)
    q = None
    for boundary, tie_up, step in _E2M1_STEPS:
        compare = ttnn.ge if tie_up else ttnn.gt
        crossed = compare(magnitude, ttnn.multiply(s, boundary))
        term = crossed if step == 1.0 else ttnn.multiply(crossed, step)
        q = term if q is None else ttnn.add(q, term)
    return q


def _dequant(q: ttnn.Tensor, s: ttnn.Tensor, xf: ttnn.Tensor, x: ttnn.Tensor) -> ttnn.Tensor:
    """sign(x) * q * s -> bf16 in the input's shape (exact: <= 6 significant bits)."""
    y = ttnn.multiply(ttnn.multiply(q, s), ttnn.sign(xf))
    return ttnn.reshape(ttnn.typecast(y, ttnn.bfloat16), x.shape)
