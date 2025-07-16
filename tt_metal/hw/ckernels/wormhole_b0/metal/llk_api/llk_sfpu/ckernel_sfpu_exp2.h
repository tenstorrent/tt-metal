// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_exp2.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_exp2() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        // log(2) = 0.6931471805;
        v = v * 0.6931471805f;
        // exp = e^(v)
        vFloat exp = _sfpu_exp_21f_(v);
        dst_reg[0] = exp;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void exp2_init() {
    _init_exp2_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
