// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardshrink(uint32_t param0) {
    // Hardshrink(x, λ) = x if |x| > λ, else 0
    // Single comparison using abs: setsgn(v, 0) clears sign bit
    sfpi::vFloat lambda = Converter::as_float(param0);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat abs_v = sfpi::setsgn(v, 0);
        v_if(abs_v <= lambda) { sfpi::dst_reg[0] = sfpi::vConst0; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
