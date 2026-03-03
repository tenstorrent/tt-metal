// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_lgamma_stirling() {
    constexpr float LOG_SQRT_2PI = 0.9189385332046727f;

    // Minimal coefficients for 0-3 ULP
    constexpr float r0 = 0.0833333333f;   // 1/12
    constexpr float r1 = -0.0027777777f;  // -1/360

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat z = in;

        // 1. Reflection for x < 0.5
        v_if(in < 0.5f) { z = 1.0f - in; }
        v_endif;

        // 2. Stirling base: (z - 0.5) * log(z) - z + log(sqrt(2*pi))
        sfpi::vFloat res = ((z - 0.5f) * _calculate_log_body_no_init_(z) - z + LOG_SQRT_2PI);

        // 3. High-Accuracy Correction (The "Bernoulli" series)
        // We use a minimax rational fit for 1/z.
        sfpi::vFloat inv_z2 = _sfpu_reciprocal_<2>(z * z);

        // correction = (1/z) * (r0 + r1/z^2)
        sfpi::vFloat correction = _sfpu_reciprocal_<2>(z) * (r0 + inv_z2 * r1);
        res = res + correction;

        // use a poly bridge here instead
        v_if(in == 1.0f || in == 2.0f) { res = 0.0f; }
        v_endif;

        // adjustment for inputs < 0.5 are done in composite.
        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(res, 0));
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void lgamma_stirling_init() {
    // log_init<false, false, is_fp32_dest_acc_en>();
    _init_reciprocal_<APPROXIMATION_MODE, is_fp32_dest_acc_en, false>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_lgamma_adjusted(
    const uint dst_index_in0,  // lgamma_stirling result
    const uint dst_index_in1,  // sin (x * M_PI) with integer adjustments
    const uint dst_index_in2,  // input x
    const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    constexpr float ln_pi = 1.1447298858f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat res_stirling = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat log_sin_pi_x = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat in = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];

        sfpi::vFloat z = 1.1447298858f;

        // ln(pi) - log|sin(pi*x)|
        sfpi::vFloat reflection_adj = z - log_sin_pi_x;

        sfpi::vFloat result = res_stirling;

        // For x < 0.5: lgamma(x) = reflection_adj - lgamma(1-x); otherwise use res_stirling.
        v_if(in < 0.5f) { result = reflection_adj - res_stirling; }
        v_endif;

        // adjustment for inputs < 0.5 are done in composite.
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void lgamma_adjusted_init() {
    log_init<APPROXIMATION_MODE, false, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
