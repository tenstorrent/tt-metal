// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(const int iterations, std::uint32_t param0, std::uint32_t param1) {
    // param0 = min_val as IEEE 754 float bits (bitcast uint32_t)
    // param1 = max_val as IEEE 754 float bits (bitcast uint32_t)
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        v_if(val < min_val) { val = min_val; }
        v_endif;

        v_if(val > max_val) { val = max_val; }
        v_endif;

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
