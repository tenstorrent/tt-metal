// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
// Hardtanh(x) = max_val if x > max_val, min_val if x < min_val, else x
inline void calculate_hardtanh(uint param0, uint param1) {
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        v_if(val < min_val) { sfpi::dst_reg[0] = min_val; }
        v_elseif(val > max_val) { sfpi::dst_reg[0] = max_val; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
