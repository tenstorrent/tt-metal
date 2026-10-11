// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: MIT

#pragma once

#include "ckernel_sfpu.h"
#include "ckernel_sfpu_constants.h"
#include "ckernel_sfpu_math.h"
#include "ckernel_sfpu_types.h"

namespace ckernel {
namespace sfpu {

template <uint32_t src0_id, uint32_t dst_id>
inline void calculate_erfc() {
    v_read_reg_d(src0_id, 0);
    v_mov_nop();

    // Load constants
    constexpr float erfc_coeff_a[] = {
        0.254829592f, -0.284496736f, 1.421413741f, -1.453152027f, 1.061405429f};
    constexpr float erfc_coeff_b[] = {
        0.3275911f, 0.254829592f, -0.284496736f, 1.421413741f, -1.453152027f};

    // x is in reg 0
    vFloat x = v_load_d_reg(0);

    // Handle NaN early: erfc(NaN) should be NaN
    if (sfpi::is_nan(x)) {
        r = std::numeric_limits<float>::quiet_NaN();
        v_store_d_reg(dst_id, r);
        return;
    }

    // Clamp |x| to 5.0 for large inputs to avoid overflow/underflow issues
    // Note: sfpi::min uses sign-magnitude comparison, so +NaN would sort above +Inf.
    // However, we already handled NaN above. For +Inf, abs(+Inf) is +Inf, min(+Inf, 5.0) is 5.0.
    // For finite x, it works as expected.
    vFloat ax = sfpi::min(sfpi::abs(x), 5.0f);

    // Determine sign for erfc(-x) = 2 - erfc(x) approximation or similar logic
    // Actually, standard rational approximation is for erfc(|x|).
    // If x < 0, erfc(x) = 2 - erfc(-x). But usually approximations are for positive arguments.
    // Let's look at the original logic structure.
    
    // Original logic seems to compute erfc for positive argument then adjust?
    // The issue highlights that `ax` becomes 5.0 for +NaN because min(abs(NaN), 5.0) -> 5.0.
    // Since we handled NaN above, we proceed.

    bool neg_x = (x < 0.0f);
    
    // Rational approximation for erfc(ax) where ax >= 0
    // t = 1 / (1 + p * ax)
    // erfc(ax) ≈ t * exp(-ax*ax + sum(c_i * t^i))
    
    // Using simplified polynomial/rational form common in these kernels
    // The exact coefficients and method might vary, but the key is handling the input correctly.
    
    // Calculate t = 1 / (1 + 0.3275911 * ax)
    vFloat t = sfpi::recip(1.0f + 0.3275911f * ax);
    
    // Polynomial evaluation for the error function part
    // erf(x) = 1 - (a1*t + a2*t^2 + ... + a5*t^5) * exp(-x^2)
    // erfc(x) = 1 - erf(x) = (a1*t + ... + a5*t^5) * exp(-x^2)
    
    // Coefficients from Abramowitz and Stegun 7.1.26
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float P = 0.3275911f;

    // Compute polynomial: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    vFloat poly = sfpi::fmad(a5, t, a4);
    poly = sfpi::fmad(poly, t, a3);
    poly = sfpi::fmad(poly, t, a2);
    poly = sfpi::fmad(poly, t, a1);
    poly = sfpi::mul(poly, t);

    // exp(-ax * ax)
    vFloat exp_term = sfpi::exp(sfpi::neg(sfpi::mul(ax, ax)));

    // erfc(ax) = poly * exp_term
    vFloat erfc_val = sfpi::mul(poly, exp_term);

    // If original x was negative, erfc(x) = 2 - erfc(-x) = 2 - erfc(|x|)
    // Wait, erfc(-x) = 2 - erfc(x) for x>0? 
    // erfc(x) = 1 - erf(x). erf(-x) = -erf(x).
    // erfc(-x) = 1 - erf(-x) = 1 + erf(x) = 2 - (1 - erf(x)) = 2 - erfc(x).
    // So if x < 0, we want 2 - erfc(|x|).
    
    if (neg_x) {
        erfc_val = sfpi::sub(2.0f, erfc_val);
    }

    r = erfc_val;
    v_store_d_reg(dst_id, r);
}

} // namespace sfpu
} // namespace ckernel
