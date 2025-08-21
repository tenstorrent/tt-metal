// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint param0, uint param1) {
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        v_if(min_val > max_val) { sfpi::dst_reg[0] = max_val; }
        v_elseif(val < min_val) { sfpi::dst_reg[0] = min_val; }
        v_elseif(val > max_val) { sfpi::dst_reg[0] = max_val; }
        v_endif;

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp_int32(uint param0, uint param1) {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param(param0);
        calculate_unary_max_min_int32_body(true);
        load_value_param(param1);
        calculate_unary_max_min_int32_body(false);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
