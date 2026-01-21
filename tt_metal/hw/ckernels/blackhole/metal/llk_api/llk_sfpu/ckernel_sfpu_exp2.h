// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_exp2.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_exp2() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v = v * sfpi::vConstFloatPrgm0;
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en) {
            result = _sfpu_exp_f32_accurate_(v);
        } else {
            result = _sfpu_exp_21f_<true>(v);
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void exp2_init() {
    sfpi::vConstFloatPrgm0 = 0.6931471805f;
}

}  // namespace ckernel::sfpu
