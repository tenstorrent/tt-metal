// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_sfpu_recip.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat denominator = sfpi::vConst1 + ckernel::sfpu::_sfpu_exp_21f_<true>(-x);

    constexpr int recip_mode = is_fp32_acc_to_dest_mode ? 2 : 1;
    sfpi::vFloat result = ckernel::sfpu::_sfpu_reciprocal_<recip_mode>(denominator);

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

// sigmoid is anti-symmetric and offset by 1
// sigmoid[-x] = 1 - sigmoid[x]
sfpi_inline sfpi::vFloat _sfpu_sigmoid_legacy_(sfpi::vFloat val) {
    vFloat result = 0.0f;

    vFloat x = sfpi::setsgn(val, 0);
    result = _sigmoid_piecewise_linear_positive_(x);

    v_if(val < 0.0f) { result = 1.0f - result; }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sigmoid() {
    if constexpr (!APPROXIMATION_MODE) {
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat val = dst_reg[0];

            vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(val);

            dst_reg[0] = result;
            dst_reg++;
        }
    } else {
        calculate_sigmoid_appx<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    if constexpr (!APPROXIMATION_MODE) {
        ckernel::sfpu::_init_reciprocal_<false, false>();
    } else {
        sigmoid_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
