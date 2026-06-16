// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat _sfpu_exp2_fp32_accurate_(sfpi::vFloat x) {
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
    r = 0x1.41cp-13f;
    r = r * f + 0x1.5f4p-10f;
    r = r * f + 0x1.3b4p-7f;
    i = sfpi::abs(sfpi::reinterpret<sfpi::vInt>(sm));
    y = r * f + sfpi::vConstFloatPrgm2;
    i = sfpi::reinterpret<sfpi::vInt>(sfpi::copysgn(sfpi::reinterpret<sfpi::vFloat>(i), j));
    r = y * f + sfpi::vConstFloatPrgm1;
    y *= std::numeric_limits<float>::infinity();
    r = r * f + sfpi::vConstFloatPrgm0;
    abs_x = sfpi::abs(x);
    r = r * f + 1.0f;

    // Exclude -NaN: abs(-NaN) remains negative.
    v_if(abs_x >= 0.0f) {
        sfpi::vInt e = sfpi::exexp(r, sfpi::ExponentMode::NoDebias);
        e += i;
        // e < 255
        v_block {
            sfpi::vInt e_lt_255 = __builtin_rvtt_sfpiadd_i(e.get(), -255, sfpi::SFPIADD_MOD1_CC_LT0);
            y = sfpi::setexp(r, e);
            // e < 1
            v_if(e_lt_255 < -254) {
                // Underflow, including subnormals.
                y = 0.0f;
            }
            v_endif;
        }
        v_endblock;
    }
    v_endif;

    return y;
}

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
sfpi_inline sfpi::vFloat _sfpu_exp2_bf16_(sfpi::vFloat x) {
    // Map x → xlog2 such that 2^x has biased exponent floor(xlog2) and the
    // fractional mantissa supplies the (xlog2 - floor) refinement.
    sfpi::vFloat xlog2 = x + 127.f;

    // Clamp to [0, 255]. Boundary inputs land on the +inf / +0 encodings after
    // setexp + bf16 round.
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    // Decompose xlog2 in [0, 255] into:
    //   exponential_part = floor(xlog2)             (integer in [0, 255])
    //   fractional_part  = (xlog2 - floor) * 2^23   (integer in [0, 2^23))
    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

    sfpi::vInt exponential_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z), sfpi::ExponentMode::NoDebias);
    sfpi::vMag fractional_part = sfpi::exman(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);

    // Refine 2^x_f on x_f to [0, 2^23). Same minimax coefficients as the
    // production exp_21f kernel (≤ 3 fp32 ULP, well under 1 bf16 ULP).
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: 2^x = (1.frac_mantissa) * 2^(exponential_part - 127).
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    // SFPSTORE truncates fp32→bf16; round explicitly so the bf16 result matches
    // a faithful nearest-even rounding of the fp32 mathematical value, and so
    // that the saturation tricks above (overflow → +inf, underflow → 0) land
    // on the correct bf16 encoding.
    return sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_exp2() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        if constexpr (is_fp32_dest_acc_en) {
            sfpi::dst_reg[0] = _sfpu_exp2_fp32_accurate_(v);
        } else {
            sfpi::dst_reg[0] = _sfpu_exp2_bf16_(v);
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
inline void exp2_init() {
    if constexpr (is_fp32_dest_acc_en) {
        // Coefficients for minimax polynomial.
        sfpi::vConstFloatPrgm0 = 0x1.62e42ep-1f;
        sfpi::vConstFloatPrgm1 = 0x1.ebfba0p-3f;
        sfpi::vConstFloatPrgm2 = 0x1.c6afd8p-5f;
    }
}

}  // namespace ckernel::sfpu
