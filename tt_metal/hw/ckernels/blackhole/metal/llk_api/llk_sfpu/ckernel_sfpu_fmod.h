// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

// Round-to-integer magic constant shared by init_fmod / init_remainder: 1.5 * 2^23.
// (q + MAGIC) - MAGIC returns the integer nearest to q for |q| < 2^22 (the sum lands in
// [2^23, 2^24), where the fp32 ulp is exactly 1), and an integer within ulp(q + MAGIC)/2 of q
// above that. Unlike 2^23 it also works for the (small) negative q the residual step below
// produces, so one programmable constant serves both rounding steps.
constexpr float FMOD_ROUND_MAGIC = 12582912.0f;

template <bool APPROXIMATION_MODE>
inline void init_fmod(const uint value, const uint recip) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = Converter::as_float(value);
    sfpi::vConstFloatPrgm1 = Converter::as_float(recip);
    sfpi::vConstFloatPrgm2 = FMOD_ROUND_MAGIC;
}

/**
 * @brief |x| mod |s| for a non-negative dividend magnitude v.
 *
 * @param v          |x|
 * @param s_abs      |s|
 * @param recip_abs  |fl(1/s)|, the host-supplied reciprocal (vConstFloatPrgm1) without its sign
 * @param s_hi       |s| with the mantissa below its top 12 significant bits cleared
 * @param s_lo       |s| - s_hi (exact)
 *
 * Everything here is a magnitude: with q = v*recip_abs >= 0 the sum q + 1.5*2^23 never drops below
 * 2^23, where the fp32 ulp would be 0.5 and the "rounding" would return half-integers (that is what
 * a signed q does for s < 0 and |q| >= 2^22). The divisor's sign is applied by the callers.
 *
 * The result is exactly v - k*|s| (k = floor(v/|s|)) whenever the true quotient is below 2^24 - 1:
 *   1. qr = round(v*recip_abs) is within a few units of k (host reciprocal error plus one
 *      rounding at ulp 1 or 2).
 *   2. r = v - qr*|s| is that small residual, so t = round(r*recip_abs) recovers the integer
 *      correction k - qr, or k - qr + 1 when the true remainder rounds up; qr + t is k or k + 1.
 *   3. v - (qr + t)*|s| is r_true or r_true - |s|, both multiples of ulp(s) below |s| in magnitude
 *      and hence representable -- but Blackhole's SFPMAD is not a fused multiply-add: its product
 *      is kept to about 28 bits, so a single MAD is exact only while bits(k) + bits(s) <= 28. The
 *      residual is therefore formed as (v - p) - e with p = fl(qr*|s|) and e = qr*|s| - p recovered
 *      by Dekker's two-product from the 12-bit halves of qr and |s|: every partial product has at
 *      most 24 bits and every intermediate is representable, so each step is exact on this unit.
 *      The +|s| fix-up for a negative residual is exact for the same reason.
 * From 2^24 the integer quotient no longer fits an fp32 mantissa: qr + t is off by up to one unit
 * below 2^25 (result within one ulp(s) after the two fix-ups) and the result is inexact beyond,
 * as it was with the previous shift-truncate + subtract-loop form. Note that ulp(x) >= 2|s| in
 * that whole regime, so the exact remainder of the fp32 dividend is below the input's resolution.
 *
 * NaN / Inf propagate through the arithmetic (Inf * 0 -> NaN in the residual MAD), and a zero
 * divisor arrives as recip = Inf, so v*recip is Inf or NaN and the result is NaN without a
 * separate test. Denormal dividends are flushed to zero by the SFPU. An exact cancellation
 * returns +0; callers apply copysgn afterwards.
 */
sfpi_inline sfpi::vFloat calculate_fmod_magnitude(
    sfpi::vFloat v, sfpi::vFloat s_abs, sfpi::vFloat recip_abs, sfpi::vFloat s_hi, sfpi::vFloat s_lo) {
    sfpi::vFloat qr = (v * recip_abs + sfpi::vConstFloatPrgm2) - sfpi::vConstFloatPrgm2;
    sfpi::vFloat r = v - qr * s_abs;
    const sfpi::vFloat t = (r * recip_abs + sfpi::vConstFloatPrgm2) - sfpi::vConstFloatPrgm2;
    qr = qr + t;
    // Dekker two-product: p + e == qr * |s| exactly. qh keeps the top 12 significant bits of qr
    // (an integer below 2^24), ql the remaining ones; s_hi / s_lo split |s| the same way.
    const sfpi::vFloat p = qr * s_abs;
    const sfpi::vFloat d = v - p;  // exact: p is within a factor of two of v
    const sfpi::vFloat qh = sfpi::as<sfpi::vFloat>((sfpi::as<sfpi::vUInt>(qr) >> 12) << 12);
    const sfpi::vFloat ql = qr - qh;
    sfpi::vFloat e = qh * s_hi - p;
    e = e + qh * s_lo;
    e = e + ql * s_hi;
    e = e + ql * s_lo;
    r = d - e;
    // q < 2^24: r is r_true or r_true - |s|, so only the first fix-up can fire. From 2^24 the fp32
    // quotient has ulp 2 and qr + t can also land on k - 1; the second fix-up keeps |r| < |s|
    // there (result within one ulp(s) up to 2^25, inexact beyond).
    v_if(r < 0.0f) { r = r + s_abs; }
    v_endif;
    v_if(r >= s_abs) { r = r - s_abs; }
    v_endif;
    return r;
}

/**
 * @brief Top 12 significant bits of the divisor magnitude, for the Dekker split above
 *        (s_hi = fmod_split_hi(|s|), s_lo = |s| - s_hi, both exact). Loop-invariant.
 */
sfpi_inline sfpi::vFloat fmod_split_hi(sfpi::vFloat s_abs) {
    return sfpi::as<sfpi::vFloat>((sfpi::as<sfpi::vUInt>(s_abs) >> 12) << 12);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_fmod() {
    // SFPU microcode
    const sfpi::vFloat s_abs = sfpi::abs(sfpi::vConstFloatPrgm0);
    const sfpi::vFloat s_hi = fmod_split_hi(s_abs);
    const sfpi::vFloat s_lo = s_abs - s_hi;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat v = calculate_fmod_magnitude(
            sfpi::abs(val), s_abs, sfpi::abs(sfpi::vConstFloatPrgm1), s_hi, s_lo);
        // fmod takes the dividend's sign (torch.fmod), including for a zero or NaN result.
        v = sfpi::copysgn(v, val);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
