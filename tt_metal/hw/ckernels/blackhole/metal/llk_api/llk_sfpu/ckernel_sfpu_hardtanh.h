// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel::sfpu {

// Hardtanh(x) = max_val if x > max_val, min_val if x < min_val, else x
// Equivalent to: clamp(x, min_val, max_val) = min(max(x, min_val), max_val)
// Uses hardware SFPSWAP instruction for branchless min/max operations
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardtanh(uint param0, uint param1) {
    // param0 = min_val, param1 = max_val
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(param0);               // Load min_val to LREG2
        calculate_unary_max_min_float_body<true>();   // x = max(x, min_val)
        load_value_param_float(param1);               // Load max_val to LREG2
        calculate_unary_max_min_float_body<false>();  // x = min(x, max_val)
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
