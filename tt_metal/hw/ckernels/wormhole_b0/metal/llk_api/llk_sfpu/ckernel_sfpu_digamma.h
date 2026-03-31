// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"    // for _calculate_log_body_no_init_
#include "ckernel_sfpu_recip.h"  // for _sfpu_reciprocal_<2> and _init_reciprocal_
#include "sfpi.h"

namespace ckernel::sfpu {

//
// digamma(z) = log(z) - correction
//
// Stirling asymptotic series (valid for z > 0, increasingly accurate for z >= 3):
//   digamma(z) = log(z) - r*(0.5 + r*p(r^2))
// where r = 1/z and p(r2) is the Bernoulli polynomial evaluated via Horner.
//
// Upward recurrence is applied to shift z into z_c >= 3 before Stirling:
//
//   z < 2:  double recurrence  digamma(z) = digamma(z+2) - 1/z - 1/(z+1)   [z_c = z+2 >= 2]
//   z < 3:  single recurrence  digamma(z) = digamma(z+1) - 1/z              [z_c = z+1 >= 3]
//   z >= 3: direct Stirling                                                  [z_c = z   >= 3]
//
// For z >= 1 both thresholds guarantee z_c >= 3.  For z in (0,1) the double
// recurrence yields z_c = z+2 ∈ (2,3); the 7-term Stirling series is still
// within 1 BF16 ULP at these values (verified empirically: max ULP = 1 over
// z ∈ (0.01, 0.5)).
//
// Bernoulli coefficients (Horner evaluation on r2 = (1/z)^2), from innermost:
//   c7 = +1/12          = +0.083333333f
//   c6 = -691/32760     = -0.021092796f
//   c5 = +1/132         = +0.007575758f
//   c4 = -1/240         = -0.004166667f
//   c3 = +1/252         = +0.003968254f
//   c2 = -1/120         = -0.008333333f
//   c1 = +1/12          = +0.083333333f
//
// NOTE: Uses _calculate_log_body_no_init_ (hardcoded polynomial constants, no
// vConstFloatPrgm dependency) and _sfpu_reciprocal_<2> (Newton-Raphson reciprocal,
// requires vConstFloatPrgm0/1/2 set by _init_reciprocal_ in digamma_init).
//

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_digamma() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat z = sfpi::dst_reg[0];

        // Upward recurrence to shift argument into z_c >= 3 for accurate Stirling eval.
        sfpi::vFloat shift = sfpi::vFloat(0.0f);
        sfpi::vFloat z_c = z;
        v_if(z < 2.0f) {
            // Double recurrence: digamma(z) = digamma(z+2) - 1/z - 1/(z+1)
            shift = _sfpu_reciprocal_<2>(z) + _sfpu_reciprocal_<2>(z + 1.0f);
            z_c = z + 2.0f;
        }
        v_elseif(z < 3.0f) {
            // Single recurrence: digamma(z) = digamma(z+1) - 1/z
            shift = _sfpu_reciprocal_<2>(z);
            z_c = z + 1.0f;
        }
        v_endif;

        // log(z_c) using hardcoded polynomial constants (no vConstFloatPrgm dependency)
        sfpi::vFloat log_z = _calculate_log_body_no_init_(z_c);

        // r = 1/z_c,  r2 = r^2
        sfpi::vFloat r = _sfpu_reciprocal_<2>(z_c);
        sfpi::vFloat r2 = r * r;

        // Horner evaluation of Bernoulli poly p(r2), innermost first
        sfpi::vFloat poly = sfpi::vFloat(0.083333333f);  // c7 = +1/12
        poly = sfpi::vFloat(-0.021092796f) + r2 * poly;  // c6 = -691/32760
        poly = sfpi::vFloat(0.007575758f) + r2 * poly;   // c5 = +1/132
        poly = sfpi::vFloat(-0.004166667f) + r2 * poly;  // c4 = -1/240
        poly = sfpi::vFloat(0.003968254f) + r2 * poly;   // c3 = +1/252
        poly = sfpi::vFloat(-0.008333333f) + r2 * poly;  // c2 = -1/120
        poly = sfpi::vFloat(0.083333333f) + r2 * poly;   // c1 = +1/12

        // correction = r * (0.5 + r * p(r2))
        sfpi::vFloat correction = r * (sfpi::vFloat(0.5f) + r * poly);

        // digamma(z) = log(z_c) - correction - shift
        sfpi::vFloat result = log_z - correction - shift;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void digamma_init() {
    // Must use _init_sfpu_reciprocal_ (not _init_reciprocal_) because
    // calculate_digamma calls _sfpu_reciprocal_<2> as an inline helper function.
    //
    // On WH: _sfpu_reciprocal_<2> reads vConstFloatPrgm0/1/2 for its quadratic
    //        initial estimate — set by _init_sfpu_reciprocal_.
    // On BH: _sfpu_reciprocal_<2> reads vConstFloatPrgm0 as the NR constant 2.0f
    //        — also set by _init_sfpu_reciprocal_.
    //
    // _init_reciprocal_ (used by the dedicated reciprocal op kernel) sets up
    // SFPLOADMACRO instruction templates on BH for bulk _calculate_reciprocal_
    // iterations.  It does NOT set vConstFloatPrgm0 = 2.0f, so using it here
    // leaves vConstFloatPrgm0 uninitialised and corrupts the NR reciprocal,
    // producing garbage output on BH.
    _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
