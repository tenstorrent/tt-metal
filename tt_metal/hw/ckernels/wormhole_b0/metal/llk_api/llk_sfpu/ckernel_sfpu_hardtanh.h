// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// hardtanh(x, min_val, max_val) = max(min_val, min(max_val, x))
// Clamps all elements in input into the range [min_val, max_val].
// Default: min_val = -1.0, max_val = 1.0
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardtanh(uint32_t param0, uint32_t param1) {
    // Reconstruct float parameters from bit-cast uint32_t
    sfpi::vFloat v_min = Converter::as_float(param0);
    sfpi::vFloat v_max = Converter::as_float(param1);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Clamp to max: if v > max_val, set v = max_val
        v_if(v > v_max) { v = v_max; }
        v_endif;

        // Clamp to min: if v < min_val, set v = min_val
        v_if(v < v_min) { v = v_min; }
        v_endif;

        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
