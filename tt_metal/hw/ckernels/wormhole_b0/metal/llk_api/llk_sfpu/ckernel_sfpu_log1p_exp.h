// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// log(1 + exp(x))
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_log1p_exp() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat input = dst_reg[0];
        vFloat result = _sfpu_exp_21f_<is_fp32_dest_acc_en>(input);
        result = result + 1.0f;
        result = _calculate_log_body_no_init_(result);
        // result = calculate_tanh_accurate_no_init<is_fp32_dest_acc_en>(result);

        // v_if (input > 20.0f){
        //     result = input;
        // }
        // v_elseif (input < -20.0f){
        //     result = 0.0f;
        // }
        // v_endif;
        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX>
inline void log1p_exp_init() {
    exp_init<false, false>();
    // _init_reciprocal_<APPROXIMATION_MODE, false>();
}

}  // namespace sfpu
}  // namespace ckernel
