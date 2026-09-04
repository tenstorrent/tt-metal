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
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

/**
 * Mish activation function:  mish(x) = x * tanh(softplus(x))
 *
 * Reducing to algebraic identity:
 *
 *     tanh(softplus(x)) = u (u + 2) / (u^2 + 2u + 2),  where u = exp(x)
 *
 * so that mish becomes:
 *
 *     mish(x) = x * u (u + 2) / (u^2 + 2u + 2)
 *
 * Note: BH uses rearranged form (x - 2x/denom) for x >= 0 in order to avoid
 * cancellation through its lower-precision approx_recip.
 *
 * Saturation: For x >= 8.0, mish(x) is approximated as x.
 */
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_mish() {
    constexpr float SAT_HI = 8.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // x >= SAT_HI: mish(x) ≈ x
        sfpi::vFloat result = x;

        v_if(x < SAT_HI) {
            sfpi::vFloat u;
            if constexpr (APPROXIMATION_MODE) {
                u = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(x);
            } else {
                u = _sfpu_exp_accurate_<is_fp32_dest_acc_en>(x);
            }

            // numerator = u * (u + 2)
            sfpi::vFloat numer = u * (u + 2.0f);

            // denominator = (1 + u)^2 + 1 = u^2 + 2u + 2
            sfpi::vFloat one_plus_u = u + 1.0f;
            sfpi::vFloat denom = one_plus_u * one_plus_u + 1.0f;

            sfpi::vFloat inv_denom;
            if constexpr (APPROXIMATION_MODE) {
                inv_denom = sfpu_reciprocal_iter<0>(denom);
            } else if constexpr (is_fp32_dest_acc_en) {
                inv_denom = sfpu_reciprocal_iter<2>(denom);
            } else {
                inv_denom = sfpu_reciprocal_iter<1>(denom);
            }

            result = x * (numer * inv_denom);
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void mish_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // exp does not need an init
    recip_init<APPROXIMATION_MODE, false, false>();
}

// ---------------------------------------------------------------------------------------------------
// Mish<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode) / init()
//   backs mish_tile / mish_tile_init (init_kernel -> mish_init).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Mish : SfpuUnaryOp<Mish<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_mish<APPROXIMATION_MODE, DST_ACCUM, ITERATIONS>(); }

    static void init_kernel() { mish_init<APPROXIMATION_MODE>(); }
};
}  // namespace ckernel::sfpu
