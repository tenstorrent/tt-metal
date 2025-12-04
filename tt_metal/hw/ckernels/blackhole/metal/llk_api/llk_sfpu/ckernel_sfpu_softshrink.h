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
    // Equivalently: x - sign(x)*λ if |x| > λ, else 0
    sfpi::vFloat lambda = Converter::as_float(param0);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpi::vConst0;

        v_if(sfpi::abs(v) > lambda) {
            result = v - sfpi::setsgn(lambda, v);  // v - sign(v)*λ
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
