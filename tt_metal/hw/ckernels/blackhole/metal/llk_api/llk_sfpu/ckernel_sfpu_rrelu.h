// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

// rrelu(x) = x                         if x >= 0
//          = negative_slope * x         if x < 0
//
// Evaluation mode (training == 0):
//   negative_slope = (lower + upper) / 2   (constant for all elements)
//
// Training mode (training != 0):
//   negative_slope ~ Uniform(lower, upper) per element
//   Uses a simple LCG PRNG to generate per-element random slopes.

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint param_lower, uint param_upper, uint param_training) {
    // Bit-cast uint32 params to float
    union {
        uint32_t u;
        float f;
    } conv;

    conv.u = param_lower;
    float lower = conv.f;
    conv.u = param_upper;
    float upper = conv.f;
    conv.u = param_training;
    float training = conv.f;

    // Precompute eval-mode slope
    float neg_slope_eval = (lower + upper) * 0.5f;

    // PRNG state for training mode (LCG: state = state * a + c)
    static uint32_t prng = 2654435761u;
    constexpr float inv_2_32 = 2.3283064365386963e-10f;  // 1.0 / 2^32
    float range = upper - lower;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x;

        v_if(x < 0.0f) {
            if (training != 0.0f) {
                // Training mode: per-element random slope
                prng = prng * 1664525u + 1013904223u;
                float rand_01 = static_cast<float>(prng) * inv_2_32;
                float slope = lower + rand_01 * range;
                result = x * slope;
            } else {
                // Eval mode: fixed slope
                result = x * neg_slope_eval;
            }
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
