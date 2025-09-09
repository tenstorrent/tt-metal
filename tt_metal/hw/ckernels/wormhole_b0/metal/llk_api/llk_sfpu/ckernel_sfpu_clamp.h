// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel::sfpu {

enum { Max = true, Min = false };  // Clamp Mode

// out = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint min_val, uint max_val) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_float(min_val);
        calculate_unary_max_min_float_body<Max>();
        load_value_param_float(max_val);
        calculate_unary_max_min_float_body<Min>();
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp_int32(uint min_val, uint max_val) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_int(min_val);
        calculate_unary_max_min_int32_body<Max>();
        load_value_param_int(max_val);
        calculate_unary_max_min_int32_body<Min>();
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
