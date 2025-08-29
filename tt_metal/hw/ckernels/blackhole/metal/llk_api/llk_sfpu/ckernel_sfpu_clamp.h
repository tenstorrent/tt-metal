// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel::sfpu {

// out = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint param0, uint param1) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(param0);
        calculate_unary_max_min_float_body<true>();
        load_value_param_float(param1);
        calculate_unary_max_min_float_body<false>();
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
