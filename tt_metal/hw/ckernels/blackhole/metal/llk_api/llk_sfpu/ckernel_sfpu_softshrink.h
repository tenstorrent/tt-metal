// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softshrink(uint32_t param0) {
    // Softshrink(x) = x - λ if x > λ, x + λ if x < -λ, else 0
    // SFPU microcode
    sfpi::vFloat lambda = Converter::as_float(param0);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::vConst0;
        v_if(v > lambda) { sfpi::dst_reg[0] = v - lambda; }
        v_elseif(v < (-lambda)) { sfpi::dst_reg[0] = v + lambda; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
