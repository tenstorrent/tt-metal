// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"  // _sfpu_round_to_nearest_int32_

namespace ckernel {
namespace sfpu {

// fp32 core shared by pow(tensor, tensor), pow(tensor, scalar) and rpow, identical on Wormhole and
// Blackhole (plain SFPI only).
//
// pow(x, y) = 2^(y * log2|x|). The error of z = y*log2|x| reaches the result as |dz| * ln2 relative,
// so for |z| up to 126 log2|x| has to be good to ~2^-29 of |y*log2 m|, not to one fp32 ulp. Here
// 2*log2|x| is carried as th + tl with th rounded to 8 significant bits, and y/2 is split the same
// way (yh + yl), so z_hi = yh*th is exact (8x8 bits) and z_lo = yl*th + (y/2)*tl only carries the
// remainder (|z_lo| < 2^-5 |z_hi|). Carrying 2*log2|x| and y/2 rather than log2|x| and y is exact and
// keeps yh finite (see _sfpu_pow_z_df_). No Dekker products or Veltkamp splits are needed.
//
// The 8-bit splits are SFPSTOCHRND FP32->FP16B with round-to-nearest (convert<vFloat16b>, already used
// by the bf16 paths): one instruction and no mask register, |x - rnd8(x)| <= 2^-8 |x|. 8 bits are
// enough for yh*th to be exact (a 16-bit product; FP16A's 11 bits would fit as well, 22 bits).
//
// Reused from earlier kernels:
//  - the bit-level range reduction of calculate_log_body (#44419): e = bits(x) - bits(c0),
//    setman(e, 0), m = bits(x) - e; with c0 = 0.70703125 (bf16 of sqrt(2)/2) m lands in
//    [0.707, 1.414) without a v_if, as the sqrt(2) reduction did before.
//  - the 2^f minimax of _sfpu_exp2_fp32_accurate_ (#46024) for the final step; f is already reduced,
//    so there is no f*ln2 product (the binary path used to go through _sfpu_exp_fp32_accurate_).
// The programmable constants are not used: the bf16 paths share the init and read vConstFloatPrgm0..2.
//
// Registers: the binary path keeps base and pow live across the core, and at its peak (inside
// _sfpu_log2x2_df_) the core needs the other 6 LRegs, so all 8 are in use. The order of operations
// below is what keeps it there.
namespace pow_df {
// 1/(m+1) ~ SEED_A - SEED_B*(m+1) on [1.707, 2.414]: 1.6% error, 2^-23.9 after two Newton steps (fp16).
constexpr float SEED_A = 0.9853515625f;
constexpr float SEED_B = 0.239013671875f;
// The constants below give 2*log2|x| (see _sfpu_pow_z_df_): each is twice the one for log2|x|, which
// changes no rounding.
// 2*log2 m = K*atanh(s), K = 4/ln2 = KH + KL. KH = 5.75 has 5 bits, so KH*sh is exact (bf16 immediate).
constexpr float KH = 5.75f;
// P(w) = KL + w*(P1 + w*(P2 + w*(P3 + w*P4))) ~ KL + K*(atanh(s)/s - 1) for w = s^2 <= 0.02946;
// minimax, error 2^-33 relative to K*s.
constexpr float KL = 0x1.547652p-6f;  // fp32(4/ln2 - 5.75) = 0x3CAA3B29
constexpr float P1 = 0x1.ec709cp+0f;  // 0x3FF6384E
constexpr float P2 = 0x1.2777d8p+0f;  // 0x3F93BBEC
constexpr float P3 = 0x1.a58p-1f;     // 0x3F52C000 (fp16)
constexpr float P4 = 0x1.5c8p-1f;     // 0x3F2E4000 (fp16)
// 2^f for |f| <= 0.5: the coefficients of _sfpu_exp2_fp32_accurate_ (#46024).
constexpr float E6 = 0x1.41cp-13f;
constexpr float E5 = 0x1.5f4p-10f;
constexpr float E4 = 0x1.3b4p-7f;
constexpr float E3 = 0x1.c6afd8p-5f;
constexpr float E2 = 0x1.ebfba0p-3f;
constexpr float E1 = 0x1.62e42ep-1f;
}  // namespace pow_df

sfpi_inline sfpi::vFloat _pow_df_rnd8_(sfpi::vFloat a) {
    return sfpi::convert<sfpi::vFloat16b>(a, sfpi::RoundMode::Nearest);
}

// 2*log2|x| = th + tl with th rounded to 8 significant bits. abs_base >= 0; 0, inf and NaN give twice
// the k + log2 m of the previous sqrt(2) reduction, and the callers handle the special cases as before.
sfpi_inline void _sfpu_log2x2_df_(sfpi::vFloat abs_base, sfpi::vFloat& th, sfpi::vFloat& tl) {
    using namespace pow_df;
    // |x| = 2^k * m, m in [0.70703125, 1.4140625): integer subtraction on the bits (#44419).
    sfpi::vInt e = sfpi::as<sfpi::vInt>(abs_base) - sfpi::as<sfpi::vInt>(sfpi::vFloat(0.70703125f));
    e = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(abs_base) - e);
    // 2k as an exact float: |e| = |k| << 23, convert, restore the sign, scale by 2^-22 (as in
    // calculate_log_body; addexp would turn k = 0 into 2^-22 * 2^127, so it is a multiply).
    sfpi::vFloat kf = sfpi::convert<sfpi::vFloat>(sfpi::abs(e), sfpi::RoundMode::Nearest);
    kf = sfpi::copysgn(kf, sfpi::as<sfpi::vFloat>(e));
    kf = kf * 0x1.0p-22f;

    // s = (m - 1)/(m + 1) = sh + sl. r ~ 1/(m+1) to 2^-23.9; its error only reaches sl (<= 2^-8 s).
    sfpi::vFloat d = m + 1.0f;
    sfpi::vFloat r = SEED_A - SEED_B * d;
    r = r * (2.0f - d * r);
    r = r * (2.0f - d * r);
    // m + 1 = dh + dl exactly: dh - 1 and m - (dh - 1) are exact (Sterbenz).
    sfpi::vFloat dh = _pow_df_rnd8_(d);
    sfpi::vFloat dl = m - (dh - 1.0f);
    sfpi::vFloat u = m - 1.0f;  // exact (Sterbenz)
    sfpi::vFloat sh = _pow_df_rnd8_(u * r);
    // u - sh*dh is exact: sh*dh has 16 bits and equals u*(1 +- 2^-7), Sterbenz. The second remainder
    // term is 2^-8 smaller, so its rounding does not matter.
    sfpi::vFloat rem = u - sh * dh;
    rem = rem - sh * dl;
    sfpi::vFloat sl = rem * r;
    sfpi::vFloat zc = sh + sl;  // s rounded once: feeds the series tail without the error of r

    // 2*log2 m = KH*sh (exact) + KH*sl + zc*P(zc^2).
    // T = 2k + 2*log2 m = th + tl: th = rnd8(2k + KH*sh). 2k - th is exact (both are multiples of
    // ulp(th) and |2k - th| <= |th|), and adding KH*sh is exact or off by 2^-24 of a value <= 2^-7 |T|,
    // so tl carries no error that scales with |T|. tl also takes KH*sl and the whole series tail (KL*s
    // and the atanh terms from s^3 on), so |tl| < 2^-5 |T| (2^-5.8 at |s| = 0.17 with |k| <= 1).
    sfpi::vFloat a = sh * KH;
    th = _pow_df_rnd8_(kf + a);
    tl = (kf - th) + a;
    tl = sl * KH + tl;
    sfpi::vFloat w = zc * zc;
    sfpi::vFloat p = w * P4 + P3;
    p = p * w + P2;
    p = p * w + P1;
    p = p * w + KL;
    tl = zc * p + tl;
}

// y*log2|x| as z_hi + z_lo with z_hi exact: z = (y/2) * (2*log2|x|), y/2 = yh + yl with yh = rnd8(y/2)
// and yl = y/2 - yh exact. Halving y keeps yh finite: rnd8(y) would round |y| >= 0x7F7F8000
// (~3.3962e38) up to inf and make z NaN, e.g. pow(1, FLT_MAX). Both scalings are exact, so z_hi and
// z_lo are the same as for splitting y and log2|x| directly.
sfpi_inline void _sfpu_pow_z_df_(sfpi::vFloat abs_base, sfpi::vFloat pow, sfpi::vFloat& z_hi, sfpi::vFloat& z_lo) {
    sfpi::vFloat th, tl;
    _sfpu_log2x2_df_(abs_base, th, tl);
    sfpi::vFloat h = pow * 0.5f;
    sfpi::vFloat yh = _pow_df_rnd8_(h);
    sfpi::vFloat yl = h - yh;
    z_hi = yh * th;
    z_lo = yl * th + h * tl;
}

// 2^f for |f| <= 0.5 (plus a fraction of an ulp): the minimax of _sfpu_exp2_fp32_accurate_ (#46024).
sfpi_inline sfpi::vFloat _sfpu_pow_df_exp2_poly_(sfpi::vFloat f) {
    using namespace pow_df;
    sfpi::vFloat r = E6 * f + E5;
    r = r * f + E4;
    r = r * f + E3;
    r = r * f + E2;
    r = r * f + E1;
    r = r * f + 1.0f;
    return r;
}

// 2^(z_hi + z_lo) with |z_lo| < 2^-5 |z_hi| or z_hi == 0 (so the FastTwoSum is exact), shared by
// the binary, rpow and unary paths.
// Lower clamp: if s < -126.5 then s = -126.5 and e = 0. Below that the result is subnormal and is
// flushed on store anyway; with the clamp k >= -126, so exexp(2^f) + k >= 0 and setexp never wraps.
// e has to be cleared too because for huge |z| the residual is huge (half an ulp of s). The previous
// clamps (max(s, -126.99999) without clearing e in the unary path, s < -127 in the binary one) let a
// negative residual push 2^f below 1, the exponent 126 - 127 = -1 wrapped to 255 in setexp and the
// result came out NaN instead of 0 (pow(0.1, 1000) in the unary path, #57446).
// Overflow is detected on the exponent about to be written (>= 255), as the binary path already did;
// it is equivalent to z >= 128, and it also catches z = +inf (k_int > 0, 2^f NaN).
sfpi_inline sfpi::vFloat _sfpu_pow2_df_(sfpi::vFloat z_hi, sfpi::vFloat z_lo) {
    sfpi::vFloat s = z_hi + z_lo;
    sfpi::vFloat e = z_lo - (s - z_hi);
    v_if(s < -126.5f) {
        s = -126.5f;
        e = 0.0f;
    }
    v_endif;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(s, k_int);
    sfpi::vFloat y = _sfpu_pow_df_exp2_poly_((s - k) + e);
    sfpi::vInt out_exp = sfpi::exexp(y, sfpi::ExponentMode::Biased) + k_int;
    y = sfpi::setexp(y, out_exp);
    v_if(out_exp >= 255) { y = std::numeric_limits<float>::infinity(); }
    v_endif;
    return y;
}

}  // namespace sfpu
}  // namespace ckernel
