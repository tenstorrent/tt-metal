# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's activation quantizers on the device, bit-exact with reference/kernels.py (components.yaml header).

fp8_qdq(x): reference/kernels.py:act_quant_dequant(x, 32), what the inference act_quant(..., 32, 'ue8m0') followed by a
dequantizing GEMM sees. Per 32 consecutive values of the last dim:

    amax  = max(|x|) clamped below at 1e-4
    s     = 2^ceil(log2(fp32(amax * fp32(1/448))))       (ue8m0; from the exponent bits: (bits + 0x7FFFFF) & 0x7F800000)
    y     = clamp(x / s, +-448)                          (x * 2^-k: exact; 1/s from the bits: 0x7F000000 - bits(s))
    q     = e4m3fn(y), round to nearest even             normals (|y| >= 2^-6): keep 3 mantissa bits with integer
                                                         ops on the bits ((b + 0x7FFFF + lsb) & ~0xFFFFF); below
                                                         2^-6: multiples of 2^-9, round(|y| 2^9) 2^-9 (ttnn.round: half
                                                         to even)
    out   = q * s                                        (exact; bf16-exact: e4m3 x 2^k)

Every arithmetic step is exact or a single IEEE fp32 operation on the SFPU (fp32 operands never pass the FPU, which
reads them at about TF32, K-30); the group max is checked on the device against torch (probe in BREADCRUMBS). The
input is fp32 TILE [..., R, W] with W a multiple of 32 (groups never straddle a chip's TP columns when the width is a
chip's H/TP slice); the output has the input's shape, ``dtype`` bf16 by default (exact). No host work.
"""

from __future__ import annotations

import ttnn

from .common import DRAM
from .settings import settings

FP8_MAX = 448.0
GROUP = 32
INV_FP8_MAX = 0.002232142724096775  # float32(1 / 448), the kernels' constant (exactly representable as a double)
AMAX_MIN = 1e-4  # clamped amax (cast to fp32 by the op, as torch's clamp_min on an fp32 tensor does)
E4M3_MIN_NORMAL = 2.0**-6
E4M3_SUB_STEP = 2.0**9  # subnormal step 2^-9
EXP_MASK = 0x7F800000
MANT_ALL = 0x7FFFFF
INV_POW2 = 0x7F000000  # bits(2^-k) = INV_POW2 - bits(2^k)
KEEP3_ROUND = 0x7FFFF  # half an e4m3 ulp minus one (fp32 mantissa: 23 bits, e4m3 keeps 3: drop 20)
KEEP3_MASK = -(1 << 20)  # 0xFFF00000 as int32


def _to_int(t: ttnn.Tensor) -> ttnn.Tensor:
    u = ttnn.bitcast(t, ttnn.uint32)
    i = ttnn.bitcast(u, ttnn.int32)
    return i


def _to_float(t: ttnn.Tensor) -> ttnn.Tensor:
    u = ttnn.bitcast(t, ttnn.uint32)
    return ttnn.bitcast(u, ttnn.float32)


def _free(*ts):
    for t in ts:
        ttnn.deallocate(t)


def pow2_ceil(t: ttnn.Tensor) -> ttnn.Tensor:
    """2^ceil(log2 t) for positive normal fp32 t (reference kernels.pow2_ceil), from the bits; and 2^-ceil(log2 t)."""
    b = _to_int(t)
    b1 = ttnn.add(b, MANT_ALL, memory_config=DRAM)
    sb = ttnn.bitwise_and(b1, EXP_MASK, memory_config=DRAM)
    _free(b1)
    ib = ttnn.rsub(sb, INV_POW2, memory_config=DRAM)  # INV_POW2 - sb
    # bitcast may alias its input's buffer: the int intermediates are left to the garbage collector, never freed here
    return _to_float(sb), _to_float(ib)


def round_e4m3(y: ttnn.Tensor) -> ttnn.Tensor:
    """fp32 y with |y| <= 448 -> its nearest e4m3fn value (ties to even), fp32. Exact."""
    a = ttnn.abs(y, memory_config=DRAM)
    # normals: round the fp32 mantissa to 3 bits on the integer bits (no carry beyond 448: 448 is the largest value)
    ab = _to_int(a)
    lsb = ttnn.bitwise_and(ttnn.bitwise_right_shift(ab, 20, memory_config=DRAM), 1, memory_config=DRAM)
    r = ttnn.add(ab, lsb, memory_config=DRAM)
    _free(lsb)
    r2 = ttnn.add(r, KEEP3_ROUND, memory_config=DRAM)
    _free(r)
    nb = ttnn.bitwise_and(r2, KEEP3_MASK, memory_config=DRAM)
    _free(r2)
    normal = _to_float(nb)
    # subnormals (|y| < 2^-6): multiples of 2^-9, round half to even
    v = ttnn.multiply(a, E4M3_SUB_STEP, dtype=ttnn.float32, memory_config=DRAM)
    vr = ttnn.round(v, memory_config=DRAM)
    _free(v)
    sub = ttnn.multiply(vr, 1.0 / E4M3_SUB_STEP, dtype=ttnn.float32, memory_config=DRAM)
    _free(vr)
    is_sub = ttnn.lt(a, E4M3_MIN_NORMAL, dtype=ttnn.float32, memory_config=DRAM)
    mag = ttnn.where(is_sub, sub, normal, memory_config=DRAM)
    _free(is_sub, sub)
    neg = ttnn.lt(y, 0.0, dtype=ttnn.float32, memory_config=DRAM)
    nmag = ttnn.neg(mag, memory_config=DRAM)
    out = ttnn.where(neg, nmag, mag, memory_config=DRAM)
    _free(neg, nmag, mag)
    return out


TF32_DROP = 13  # fp32 mantissa 23 bits, TF32 keeps 10
TF32_MASK = -(1 << TF32_DROP)


def round_tf32(x: ttnn.Tensor) -> ttnn.Tensor:
    """fp32 x -> x rounded to TF32 (10 mantissa bits), to nearest even, fp32; on the integer bits (sign-magnitude:
    the same add rounds the magnitude of either sign). The FPU reads fp32 operands at TF32 by truncation (K-30); a
    pre-rounded operand makes that read exact and the rounding unbiased (DS41-31). Does not consume x."""
    i = _to_int(x)
    lsb = ttnn.bitwise_and(ttnn.bitwise_right_shift(i, TF32_DROP, memory_config=DRAM), 1, memory_config=DRAM)
    r = ttnn.add(i, lsb, memory_config=DRAM)
    _free(lsb)
    r2 = ttnn.add(r, (1 << (TF32_DROP - 1)) - 1, memory_config=DRAM)
    _free(r)
    nb = ttnn.bitwise_and(r2, TF32_MASK, memory_config=DRAM)
    _free(r2)
    return _to_float(nb)


def fp8_qdq(x: ttnn.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """act_quant_dequant(x, 32) of an fp32 TILE tensor [..., R, W] (W % 32 == 0), groups of 32 along the last dim.
    Returns the same shape in ``dtype`` (bf16 holds every value exactly). Does not consume ``x``."""
    assert x.dtype == ttnn.float32, x.dtype
    shape = list(x.shape)
    w = shape[-1]
    assert w % GROUP == 0, f"last dim {w} is not a multiple of {GROUP}"
    rows = 1
    for d in shape[:-1]:
        rows *= d
    xg = ttnn.reshape(x, [1, 1, rows * (w // GROUP), GROUP])
    a = ttnn.abs(xg, memory_config=DRAM)
    amax = ttnn.max(a, dim=-1, keepdim=True, memory_config=DRAM)
    _free(a)
    am = ttnn.maximum(amax, AMAX_MIN, memory_config=DRAM)
    _free(amax)
    t = ttnn.multiply(am, INV_FP8_MAX, dtype=ttnn.float32, memory_config=DRAM)
    _free(am)
    s, inv = pow2_ceil(t)
    _free(t)
    y0 = ttnn.multiply(xg, inv, dtype=ttnn.float32, memory_config=DRAM)
    _free(inv)
    y = ttnn.clamp(y0, -FP8_MAX, FP8_MAX, memory_config=DRAM)
    _free(y0)
    q = round_e4m3(y)
    _free(y)
    out = ttnn.multiply(q, s, dtype=ttnn.float32, memory_config=DRAM)
    _free(q)
    res = ttnn.reshape(out, shape)  # (xg / res may be views: never freed here)
    if dtype != ttnn.float32:
        res2 = ttnn.typecast(res, dtype, memory_config=DRAM)
        _free(res)
        res = res2
    return res


# ------------------------------------------------------------------------------------------------- fp4 (e2m1)
FP4_MAX = 6.0
INV_FP4_MAX = 0.1666666716337204  # float32(1 / 6), the kernels' constant (exactly representable as a double)
FP4_E4M3_AMAX_MIN = FP4_MAX * 2.0**-9  # e4m3-scale groups: amax clamped at 6 x the smallest e4m3 subnormal
FP4_UE8M0_AMAX_MIN = FP4_MAX * 2.0**-126  # ue8m0-scale groups: amax clamped at 6 x the smallest normal
# e2m1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6: the midpoints between neighbours, the step up past each, and whether a
# value equal to the midpoint rounds up (ties to the even code: midpoint k sits between codes k and k + 1, and
# rounds up iff k + 1 is even, i.e. k odd)
FP4_STEPS = (
    (0.25, 0.5, False),
    (0.75, 0.5, True),
    (1.25, 0.5, False),
    (1.75, 0.5, True),
    (2.5, 1.0, False),
    (3.5, 1.0, True),
    (5.0, 2.0, False),
)


def fp4_qdq(x: ttnn.Tensor, block: int, e4m3_scale: bool = False, dtype=ttnn.float32) -> ttnn.Tensor:
    """reference/kernels.py:fp4_quant_dequant(x, block, e4m3_scale) of an fp32 TILE tensor [..., R, W]
    (W % block == 0), groups of ``block`` along the last dim, bit-exact:

        amax = max |x| per group, clamped below (6 x 2^-9 for an e4m3 scale, 6 x 2^-126 for ue8m0)
        s    = e4m3(amax / 6)                       e4m3 scale: one IEEE fp32 division by the scalar 6 (exact on the
                                                    SFPU: probe in BREADCRUMBS), tt/quant.round_e4m3
             = 2^ceil(log2(amax x fp32(1/6)))       ue8m0 scale: tt/quant.pow2_ceil on the bits
        q    = e2m1(clamp(x / s, +-6)) x s          without the division: x / s > m  <=>  |x| > m s for each e2m1
                                                    midpoint m, and m s is exact in fp32 (m has 3 significant bits,
                                                    s 4), so the code is a sum of 7 exact compares (> for a midpoint
                                                    that rounds down at a tie, >= for one that rounds up: ties to
                                                    the even code); |q| = s x the summed steps (exact), the sign of x

    Every step is a single exact fp32 SFPU operation (no FPU, K-30). Returns the input's shape in ``dtype`` (bf16
    holds every value exactly: e2m1 x an e4m3 or power-of-two scale). Does not consume ``x``."""
    assert x.dtype == ttnn.float32, x.dtype
    shape = list(x.shape)
    w = shape[-1]
    assert w % block == 0, f"last dim {w} is not a multiple of {block}"
    rows = 1
    for d in shape[:-1]:
        rows *= d
    xg = ttnn.reshape(x, [1, 1, rows * (w // block), block])
    a = ttnn.abs(xg, memory_config=DRAM)
    amax = ttnn.max(a, dim=-1, keepdim=True, memory_config=DRAM)
    if e4m3_scale:
        am = ttnn.maximum(amax, FP4_E4M3_AMAX_MIN, memory_config=DRAM)
        t = ttnn.divide(am, FP4_MAX, memory_config=DRAM)
        _free(am)
        tc = ttnn.clamp(t, 0.0, FP8_MAX, memory_config=DRAM)
        _free(t)
        s = round_e4m3(tc)
        _free(tc)
    else:
        am = ttnn.maximum(amax, FP4_UE8M0_AMAX_MIN, memory_config=DRAM)
        t = ttnn.multiply(am, INV_FP4_MAX, dtype=ttnn.float32, memory_config=DRAM)
        _free(am)
        s, _inv = pow2_ceil(t)
        _free(t)
    _free(amax)
    steps = None
    for mid, step, up in FP4_STEPS:
        thr = ttnn.multiply(s, mid, dtype=ttnn.float32, memory_config=DRAM)
        cmp = (ttnn.ge if up else ttnn.gt)(a, thr, dtype=ttnn.float32, memory_config=DRAM)
        _free(thr)
        if steps is None:
            steps = ttnn.multiply(cmp, step, dtype=ttnn.float32, memory_config=DRAM)
        else:
            nxt = ttnn.add(steps, ttnn.multiply(cmp, step, dtype=ttnn.float32, memory_config=DRAM), memory_config=DRAM)
            _free(steps)
            steps = nxt
        _free(cmp)
    _free(a)
    mag = ttnn.multiply(steps, s, dtype=ttnn.float32, memory_config=DRAM)
    _free(steps)
    neg = ttnn.lt(xg, 0.0, dtype=ttnn.float32, memory_config=DRAM)
    nmag = ttnn.neg(mag, memory_config=DRAM)
    out = ttnn.where(neg, nmag, mag, memory_config=DRAM)
    _free(neg, nmag, mag)
    res = ttnn.reshape(out, shape)  # (xg / res may be views: never freed here)
    if dtype != ttnn.float32:
        res2 = ttnn.typecast(res, dtype, memory_config=DRAM)
        _free(res)
        res = res2
    return res


# ------------------------------------------------------------------------------------------------- NUMERICS (P.8)
def native() -> bool:
    """NUMERICS=native (P.8, agent rule 17): every parity-only rounding of the model is off. The fp8 / fp4
    quant-dequant of activations the device keeps in bf16 / fp32 (fp8_qdq, fp4_qdq) and the TF32 rounding of fp32
    matmul inputs (round_tf32) emulate the reference GPU's formats; natively a matmul input is just cast to the dtype
    the matmul reads (bf16), a cache row written in bf16 is just cast. official (default): the released code's
    arithmetic, bit-exact."""
    return settings.get("NUMERICS") == "native"


def act_q(x: ttnn.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """The fp8 act quant a matmul input (or a bf16 cache row) gets. official: fp8_qdq (an fp32 x; a bf16 x is cast to
    fp32 first, exactly); native: x in ``dtype`` (a typecast, or x itself when it already is ``dtype``). Does not
    consume x; the caller frees the result only when it is not x."""
    if native():
        return x if x.dtype == dtype else ttnn.typecast(x, dtype, memory_config=DRAM)
    if x.dtype == ttnn.float32:
        return fp8_qdq(x, dtype=dtype)
    xf = ttnn.typecast(x, ttnn.float32, memory_config=DRAM)
    out = fp8_qdq(xf, dtype=dtype)
    ttnn.deallocate(xf)
    return out


def fp4_q(x: ttnn.Tensor, block: int, e4m3_scale: bool = False, dtype=ttnn.float32) -> ttnn.Tensor:
    """fp4_qdq (official) or x in ``dtype`` (native: a typecast, or x itself). Does not consume x; the caller frees
    the result only when it is not x."""
    if native():
        return x if x.dtype == dtype else ttnn.typecast(x, dtype, memory_config=DRAM)
    return fp4_qdq(x, block, e4m3_scale=e4m3_scale, dtype=dtype)


def tf32_q(x: ttnn.Tensor) -> ttnn.Tensor:
    """round_tf32 (official) or x itself (native). Does not consume x; the caller frees the result only when it is
    not x."""
    return x if native() else round_tf32(x)
