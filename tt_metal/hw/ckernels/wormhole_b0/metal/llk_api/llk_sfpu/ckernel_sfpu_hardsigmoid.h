// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_hardsigmoid() {
    // hardsigmoid(x) = max(0, min(1, x/6 + 0.5))
    //
    // Implementation strategy:
    // 1. Compute x/6 + 0.5
    // 2. Clamp result to [0, 1] using two v_if blocks
    //    - First: clamp to 1.0 if >= 1.0 (upper bound)
    //    - Second: clamp to 0.0 if <= 0.0 (lower bound)

    // Constants for the linear transformation x/6 + 0.5
    constexpr uint32_t one_sixth_fp32 = 0x3E2AAAAB;  // 1/6 in FP32 (≈ 0.16667)
    constexpr uint32_t half_fp32 = 0x3F000000;       // 0.5 in FP32
    constexpr uint32_t one_fp32 = 0x3F800000;        // 1.0 in FP32
    constexpr uint32_t zero_fp32 = 0x00000000;       // 0.0 in FP32

    sfpi::vFloat one_sixth = Converter::as_float(one_sixth_fp32);
    sfpi::vFloat half = Converter::as_float(half_fp32);
    sfpi::vFloat one = Converter::as_float(one_fp32);
    sfpi::vFloat zero = Converter::as_float(zero_fp32);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];  // Load 32 elements from current DEST row

        // Step 1: Apply linear transformation: x/6 + 0.5
        v = v * one_sixth + half;

        // Step 2: Clamp to upper bound (1.0)
        v_if(v >= one) {
            v = one;  // Clamp to 1.0 for values >= 1.0
        }
        v_endif;

        // Step 3: Clamp to lower bound (0.0)
        v_if(v <= zero) {
            v = zero;  // Clamp to 0.0 for values <= 0.0
        }
        v_endif;

        sfpi::dst_reg[0] = v;  // Store result back to DEST
        sfpi::dst_reg++;       // Advance to next DEST row (32 elements)
    }
}

template <bool APPROXIMATION_MODE>
inline void hardsigmoid_init() {
    // No special initialization required for hardsigmoid
    // The constants are loaded inline within the calculation function
}

}  // namespace sfpu
}  // namespace ckernel
