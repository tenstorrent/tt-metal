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

static const float ATAN2_PI = 3.14159274101257324f;
static const float ATAN2_PI_2 = 1.5707963705062866f;

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat sfpu_atan2_core(sfpi::vFloat t) {
    sfpi::vFloat t0 = sfpi::abs(t);
    sfpi::vFloat t1 = t0 * t0;

    if constexpr (!is_fp32_dest_acc_en) {
        // bf16: 4-term minimax
        t1 = PolynomialEvaluator::eval(
            t1,
            0.999787867069244384765625f,
            -0.325808584690093994140625f,
            0.1555790007114410400390625f,
            -4.4326744973659515380859375e-2f);
    } else {
        // fp32: 9-term minimax
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
    // Handle |t| > 1 case via identity: atan(t) = PI/2 - atan(1/t)
    v_if(t0 > 1.0f) { t1 = ATAN2_PI_2 - t1; }
    v_endif;

    return sfpi::setsgn(t1, t);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atan2() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat y = sfpi::dst_reg[0];
        sfpi::vFloat x = sfpi::dst_reg[1];

        sfpi::vFloat result;
        sfpi::vFloat abs_y = sfpi::abs(y);
        sfpi::vFloat abs_x = sfpi::abs(x);

        // Special cases
        v_if(abs_x == 0.0f && abs_y == 0.0f) {
            result = 0.0f;
        }
        v_elseif(abs_x == 0.0f) {
            // x=0: atan2 = sign(y) * PI/2
            result = sfpi::setsgn(ATAN2_PI_2, y);
        }
        v_elseif(abs_y == 0.0f) {
            // y=0: atan2 = 0 or PI depending on sign of x
            v_if(x < 0.0f) { result = ATAN2_PI; }
            v_else { result = 0.0f; }
            v_endif;
        }
        v_else {
            sfpi::vFloat ratio = y * _sfpu_reciprocal_<2>(x);
            sfpi::vFloat atan_val = sfpu_atan2_core<is_fp32_dest_acc_en>(ratio);

            v_if(x < 0.0f) {
                v_if(y >= 0.0f) {
                    result = atan_val + ATAN2_PI;
                }
                v_else {
                    result = atan_val - ATAN2_PI;
                }
                v_endif;
            }
            v_else {
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
    _init_sfpu_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
