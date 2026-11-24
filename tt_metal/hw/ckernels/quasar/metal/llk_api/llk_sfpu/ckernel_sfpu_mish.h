// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

// One Newton step on y ~ 1/d (Prgm0 = 2.0). No 0 / inf / NaN guard: d is in [2, ~8.9e6] for x < 8.
sfpi_inline sfpi::vFloat _mish_recip_nr_(const sfpi::vFloat d, const sfpi::vFloat y) {
    sfpi::vFloat t = sfpi::vConstFloatPrgm0 - d * y;
    return y * t;
}

// _sfpu_exp_21f_bf16_ for x < 8, where its overflow handling can't trigger, with the polynomial
// coefficients in Prgm1..3 (set by mish_init) instead of loaded every pass.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _mish_exp_21f_(const sfpi::vFloat x) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = sfpi::max(x * ONE_LN2 + 127.f, 0.0f);
    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
    sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(z);
    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
    frac = PolynomialEvaluator::eval(frac, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2, sfpi::vConstFloatPrgm3);
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);
    if constexpr (!is_fp32_dest_acc_en) {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }
    return y;
}

// _sfpu_exp_fp32_accurate_ for -88 <= a < 8: no overflow handling, and three coefficients in
// Prgm1..3 (set by mish_init). The underflow case (result 0) stays.
sfpi_inline sfpi::vFloat _mish_exp_fp32_(const sfpi::vFloat a) {
    sfpi::vInt i;
    sfpi::vFloat j = _sfpu_round_to_nearest_int32_(1.442695f * a, i);
    sfpi::vFloat f = j * -6.93145752e-1f + a;
    f = j * -1.42860677e-6f + f;

    sfpi::vFloat r = 1.37805939e-3f;
    r = r * f + 8.37312452e-3f;
    r = r * f + sfpi::vConstFloatPrgm1;  // 4.16695364e-2f
    r = r * f + sfpi::vConstFloatPrgm2;  // 1.66664720e-1f
    r = r * f + sfpi::vConstFloatPrgm3;  // 4.99999851e-1f
    r = r * f + 1.0f;
    r = r * f + 1.0f;

    sfpi::vInt e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
    sfpi::vFloat y = sfpi::setexp(r, e);
    v_if(e < 1) { y = 0.0f; }
    v_endif;
    return y;
}

/**
 * Mish activation function:  mish(x) = x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
 *
 * With u = exp(x), n = u (u + 2) and d = (1 + u)^2 + 1 = n + 2:
 *
 *     tanh(softplus(x)) = n / d = 1 - 2 / d
 *
 *     x >= 0:  mish(x) = x * (1 - 2 / d)
 *     x <  0:  mish(x) = x * (n / d)
 * The x >= 0 form avoids cancellation for large x, where the reciprocal's error would be scaled by x.
 * Both forms are computed and selected by sign, and x >= 8 returns x.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_mish() {
    constexpr float SAT_HI = 8.0f;
    constexpr bool exp_fp32_accurate = is_fp32_dest_acc_en && !APPROXIMATION_MODE;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat u;
        if constexpr (exp_fp32_accurate) {
            // Below about -8.7e6 the exp's rounding trick breaks and gives NaN; exp(-88) is already 0.
            u = _mish_exp_fp32_(sfpi::max(x, -88.0f));
        } else {
            u = _mish_exp_21f_<is_fp32_dest_acc_en>(x);
        }
        sfpi::vFloat denom = u + 1.0f;
        denom = denom * denom + 1.0f;
        // n goes before approx_recip so it fills the wait after the last SFPMAD.
        sfpi::vFloat n = u * (u + sfpi::vConstFloatPrgm0);
        sfpi::vFloat inv_denom = sfpi::approx_recip(denom);
        if constexpr (!APPROXIMATION_MODE) {
            inv_denom = _mish_recip_nr_(denom, inv_denom);
        }
        if constexpr (exp_fp32_accurate) {
            inv_denom = _mish_recip_nr_(denom, inv_denom);
        }
        sfpi::vFloat t = n * inv_denom;
        v_if(x >= 0.0f) { t = 1.0f - sfpi::vConstFloatPrgm0 * inv_denom; }
        v_endif;
        sfpi::vFloat result = x * t;
        v_if(x >= SAT_HI) { result = x; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false>
inline void mish_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    // 2.0 for u + 2, 1 - 2 / d and the Newton steps.
    sfpi::vConstFloatPrgm0 = 2.0f;
    if constexpr (is_fp32_dest_acc_en && !APPROXIMATION_MODE) {
        // _mish_exp_fp32_ coefficients.
        sfpi::vConstFloatPrgm1 = 4.16695364e-2f;
        sfpi::vConstFloatPrgm2 = 1.66664720e-1f;
        sfpi::vConstFloatPrgm3 = 4.99999851e-1f;
    } else {
        // _mish_exp_21f_ coefficients.
        sfpi::vConstFloatPrgm1 = EXP_21F_BF16_C0;
        sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C1;
        sfpi::vConstFloatPrgm3 = EXP_21F_BF16_C2;
    }
}

}  // namespace ckernel::sfpu
