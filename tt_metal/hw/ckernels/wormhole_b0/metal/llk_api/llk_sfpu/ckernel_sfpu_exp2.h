// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu_mode.h"

namespace ckernel::sfpu {

// 2**x for fp32 Dest, one instruction sequence per (APPROXIMATE, STRICT):
//   APPROXIMATE=false: degree-6 minimax polynomial for 2**f, ~2 ulp. Coefficients: three fp16 immediates
//                      plus vConstFloatPrgm2..0, loaded by exp2_init<Precise*>.
//   APPROXIMATE=true:  degree-4 minimax polynomial, max relative error 2.9e-6 (43 ulp on BH). One fp16
//                      immediate plus vConstFloatPrgm2..0, loaded by exp2_init<Approximate*>.
//   STRICT=true:  NaN -> NaN, x >= 128 or +inf -> +inf, results below 2**-126 (incl. -inf) -> 0.
//   STRICT=false: the caller vouches -126 <= x < 128 (a normal result); the guards are dropped.
template <bool APPROXIMATE, bool STRICT>
sfpi_inline sfpi::vFloat _sfpu_exp2_fp32_(sfpi::vFloat x) {
    sfpi::vFloat f, j, r, y, abs_x;
    sfpi::vInt i;
    sfpi::vSMag16 sm;

    // Convert x to sign-magnitude 16-bit integer (round to nearest with ties
    // away from zero), and convert back to floating point.
    sm = sfpi::convert<sfpi::vSMag16>(x, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);

    // Range reduced value in [-0.5, 0.5].
    f = x - j;

    // Minimax polynomial approximation for exp2(f), f in [-0.5, 0.5].
    // The first three coefficients are rounded to fp16 (one sfploadi per coefficient).
    // The subsequent three coefficients are fp32 values stored in constant registers.
    // Interleaved with conversion of sign-magnitude integer in [-32767, 32767] to two's complement.
    // Interleaved with calculation of the overflow case y = y * inf, giving inf or NaN.
    // Interleaved with calculation of abs_x = abs(x), which is >= 0.0 unless x is -NaN.
    if constexpr (APPROXIMATE) {
        // a4 is fp16-exact; a3..a1 are in vConstFloatPrgm2..0.
        r = 0x1.3ap-7f;
    } else {
        r = 0x1.41cp-13f;
        r = r * f + 0x1.5f4p-10f;
        r = r * f + 0x1.3b4p-7f;
    }
    i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
    y = r * f + sfpi::vConstFloatPrgm2;
    i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
    r = y * f + sfpi::vConstFloatPrgm1;
    if constexpr (STRICT) {
        y *= std::numeric_limits<float>::infinity();
    }
    r = r * f + sfpi::vConstFloatPrgm0;
    if constexpr (STRICT) {
        abs_x = sfpi::abs(x);
    }
    r = r * f + 1.0f;

    if constexpr (!STRICT) {
        // In range by contract: the result is normal, so only the exponent needs adjusting.
        sfpi::vInt e = sfpi::exexp(r, sfpi::ExponentMode::Biased);
        e += i;
        return sfpi::setexp(r, e);
    }

    // Exclude -NaN: abs(-NaN) remains negative.
    v_if(abs_x >= 0.0f) {
        sfpi::vInt e = sfpi::exexp(r, sfpi::ExponentMode::Biased);
        e += i;
        v_if(sfpi::nearby(e < 255)) {
            y = sfpi::setexp(r, e);
            v_if(sfpi::nearby(e < 1)) {
                // Underflow, including subnormals.
                y = 0.0f;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;

    return y;
}

// The legacy entry point: the precise, guarded sequence.
sfpi_inline sfpi::vFloat _sfpu_exp2_fp32_accurate_(sfpi::vFloat x) { return _sfpu_exp2_fp32_<false, true>(x); }

// BF16 path: branch-free saturation using the mantissa-as-fractional-part trick
// from the production `_sfpu_exp_21f_bf16_` kernel — but specialised for exp2 by
// skipping the `* (1/ln2)` multiply (we are already in base-2). vec_min_max clamps
// xlog2 to [0, 255], so the natural saturation of setexp + the final bf16 round
// give the correct overflow → +inf and underflow → 0 boundary encodings for free.
//
// NaN: ttnn.bfloat16 host-side packing already collapses NaN → +inf before the
// tensor ever reaches the SFPU (see the "NaN is packed as inf for ttnn.bfloat16"
// xfails on fmod / remainder / where / rdiv), so a device-side NaN guard here
// would be dead code.
//
// The overload taking c0, c1 and c2 is for a caller that evaluates it once per element
// (logaddexp2): as with _sfpu_exp_21f_bf16_unsafe_, it can load the fractional-part
// coefficients once, before its loop, and pass the registers. Each may be a float or a
// vFloat, so the one-argument form below, which passes floats, compiles as before.
//
// STRICT=false drops the clamp: the caller vouches -126 <= x < 128, so xlog2 is already in range.
template <bool STRICT = true, typename C0, typename C1, typename C2>
sfpi_inline sfpi::vFloat _sfpu_exp2_bf16_(sfpi::vFloat x, C0 c0, C1 c1, C2 c2) {
    // Map x → xlog2 such that 2^x has biased exponent floor(xlog2) and the
    // fractional mantissa supplies the (xlog2 - floor) refinement.
    sfpi::vFloat xlog2 = x + 127.f;

    // Clamp to [0, 255]. Boundary inputs land on the +inf / +0 encodings after
    // setexp + bf16 round.
    if constexpr (STRICT) {
        xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);
    }

    // Decompose xlog2 in [0, 255] into:
    //   exponential_part = floor(xlog2)             (integer in [0, 255])
    //   fractional_part  = (xlog2 - floor) * 2^23   (integer in [0, 2^23))
    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

    sfpi::vInt exponential_part = sfpi::exexp(sfpi::as<sfpi::vFloat>(z), sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(sfpi::as<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);

    // Refine 2^x_f on x_f to [0, 2^23). Same minimax coefficients as the
    // production exp_21f kernel (≤ 3 fp32 ULP, well under 1 bf16 ULP).
    frac = PolynomialEvaluator::eval(frac, c0, c1, c2);

    // Recombine: 2^x = (1.frac_mantissa) * 2^(exponential_part - 127).
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    // SFPSTORE truncates fp32→bf16; round explicitly so the bf16 result matches
    // a faithful nearest-even rounding of the fp32 mathematical value, and so
    // that the saturation tricks above (overflow → +inf, underflow → 0) land
    // on the correct bf16 encoding.
    return sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
}

template <bool STRICT = true>
sfpi_inline sfpi::vFloat _sfpu_exp2_bf16_(sfpi::vFloat x) {
    return _sfpu_exp2_bf16_<STRICT>(x, EXP_21F_BF16_C0, EXP_21F_BF16_C1, EXP_21F_BF16_C2);
}

// exp2 per SfpuMode:
//   Mode                 fp32 Dest                              bf16 Dest
//   PreciseStrict        degree-6 polynomial, guarded           exp_21f in base 2, clamped (legacy)
//   PreciseRelaxed       degree-6 polynomial, no guards         exp_21f, no clamp
//   ApproximateStrict    degree-4 polynomial, guarded           alias of PreciseStrict
//   ApproximateRelaxed   degree-4 polynomial, no guards         alias of PreciseRelaxed
// On bf16 Dest the precise sequence is already the cheap one (well under 1 bf16 ulp), so the approximate
// modes alias it. Strict: NaN -> NaN, x >= 128 or +inf -> +inf, x < -126 or -inf -> 0, +-0 -> 1.
// Relaxed: the caller vouches -126 <= x < 128.
template <SfpuMode MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_exp2() {
    if constexpr (is_fp32_dest_acc_en) {
        // fp32 path is a hand-scheduled ILP-interleaved body — leave it rolled so
        // the compiler keeps its per-row schedule (unrolling only bloats it).
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_exp2_fp32_<is_approximate(MODE), is_strict(MODE)>(v);
            sfpi::dst_reg++;
        }
    } else {
        // bf16 body is a dependent format-conversion chain (convert/exexp/setexp);
        // unrolling overlaps independent iterations to hide that latency.
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];
            sfpi::dst_reg[0] = _sfpu_exp2_bf16_<is_strict(MODE)>(v);
            sfpi::dst_reg++;
        }
    }
}

template <SfpuMode MODE, bool is_fp32_dest_acc_en>
inline void exp2_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (is_fp32_dest_acc_en) {
        if constexpr (is_approximate(MODE)) {
            // a1..a3 of the degree-4 minimax polynomial (a4 is an immediate in _sfpu_exp2_fp32_).
            sfpi::vConstFloatPrgm0 = 0x1.62e12cp-1f;
            sfpi::vConstFloatPrgm1 = 0x1.ec038p-3f;
            sfpi::vConstFloatPrgm2 = 0x1.c9fc5ap-5f;
        } else {
            // Coefficients for minimax polynomial.
            sfpi::vConstFloatPrgm0 = 0x1.62e42ep-1f;
            sfpi::vConstFloatPrgm1 = 0x1.ebfba0p-3f;
            sfpi::vConstFloatPrgm2 = 0x1.c6afd8p-5f;
        }
    }
}

// Legacy entry points. APPROXIMATION_MODE has never changed exp2's code, so both values map to the
// precise, guarded sequence they always ran.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_exp2() {
    calculate_exp2<SfpuMode::PreciseStrict, is_fp32_dest_acc_en, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void exp2_init() {
    exp2_init<SfpuMode::PreciseStrict, is_fp32_dest_acc_en>();
}

}  // namespace ckernel::sfpu
