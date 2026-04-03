// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel::sfpu {

// atan2(y, x) implemented as a single SFPU operation
// Algorithm:
// 1. Handle special cases (origin, axis-aligned)
// 2. Compute r = atan(|y/x|) using existing sfpu_atan pattern
// 3. Adjust for quadrant based on signs of x and y
//
// Input: dst_reg[0] = y (first vector), LREG1 = x (second vector, loaded separately)
// Output: dst_reg[0] = atan2(y, x)

static const float ATAN2_PI = 3.14159274101257324f;
static const float ATAN2_PI_2 = 1.5707963705062866f;

// Compute atan(|t|) using minimax polynomial — same core as sfpu_atan but without reciprocal step
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat sfpu_atan2_core(sfpi::vFloat t) {
    sfpi::vFloat t0 = sfpi::abs(t);
    sfpi::vFloat absval_minus_1 = t0 - sfpi::vConst1;

    // Range reduction: if |t| > 1, use reciprocal and subtract from PI/2
    v_if(absval_minus_1 > 0.0f) {
        t0 = sfpu_reciprocal<false>(t0);
    }
    v_endif;

    sfpi::vFloat t1 = t0 * t0;

    if constexpr (!is_fp32_dest_acc_en) {
        // 4-term minimax for bf16
        t1 = PolynomialEvaluator::eval(
            t1,
            0.999787867069244384765625f,
            -0.325808584690093994140625f,
            0.1555790007114410400390625f,
            -4.4326744973659515380859375e-2f);
    } else {
        // 9-term minimax for fp32
        t1 = PolynomialEvaluator::eval(
            t1,
            sfpi::vConst1,
            -0.3333314359188079833984375f,
            0.19993579387664794921875f,
            -0.14209578931331634521484375f,
            0.1066047251224517822265625f,
            -7.5408883392810821533203125e-2f,
            4.3082617223262786865234375e-2f,
            -1.62907354533672332763671875e-2f,
            2.90188402868807315826416015625e-3f);
    }

    t1 = t1 * t0;

    // Undo range reduction
    v_if(absval_minus_1 > 0.0f) {
        t1 = ATAN2_PI_2 - t1;
    }
    v_endif;

    return t1;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atan2() {
    // atan2 uses binary input: y in dst_reg, x passed via second source
    // The binary operation framework provides both inputs
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat y = sfpi::dst_reg[0];      // First operand (y)
        sfpi::vFloat x = sfpi::dst_reg[1];      // Second operand (x) - binary SFPU layout
        sfpi::vFloat result;

        // Handle special cases
        sfpi::vFloat abs_x = sfpi::abs(x);
        sfpi::vFloat abs_y = sfpi::abs(y);

        // Check for zero inputs
        v_if(abs_x == 0.0f) {
            v_if(abs_y == 0.0f) {
                // atan2(0, 0) = 0 by convention
                result = sfpi::vConst0;
            }
            v_else {
                // atan2(y, 0) = sign(y) * PI/2
                result = sfpi::setsgn(ATAN2_PI_2, y);
            }
            v_endif;
        }
        v_else {
            // Normal case: compute atan(y/x) with quadrant adjustment
            sfpi::vFloat ratio = y * sfpu_reciprocal<2>(x);
            sfpi::vFloat atan_val = sfpu_atan2_core<is_fp32_dest_acc_en>(ratio);

            // Quadrant correction based on sign of x
            v_if(x < 0.0f) {
                v_if(y >= 0.0f) {
                    // Q2: atan + PI
                    result = atan_val + ATAN2_PI;
                }
                v_else {
                    // Q3: atan - PI
                    result = atan_val - ATAN2_PI;
                }
                v_endif;
            }
            v_else {
                // Q1 and Q4: atan(y/x) is correct
                result = atan_val;
            }
            v_endif;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void atan2_init() {
    _init_reciprocal_<false, false>();
}

}  // namespace ckernel::sfpu
