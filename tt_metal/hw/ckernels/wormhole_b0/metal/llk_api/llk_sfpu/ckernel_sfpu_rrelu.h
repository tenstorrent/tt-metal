// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// RReLU (Randomized Leaky ReLU):
//   f(x) = x                          if x >= 0
//   f(x) = a * x                      if x < 0
// where:
//   eval mode:     a = (lower + upper) / 2 = lower + range * 0.5
//   training mode: a ~ Uniform(lower, upper) per element
//
// Parameters (all bit-cast from float to uint32_t):
//   lower_u:  the 'lower' bound of the uniform distribution
//   range_u:  upper - lower (the range of the uniform distribution)
//   seed_u:   0 for eval mode, nonzero seed for training mode (PRNG)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_rrelu_(const uint32_t lower_u, const uint32_t range_u, const uint32_t seed_u) {
    // lower_u and range_u are bit-cast float32 representations.
    // s2vFloat16b(uint32_t) expects a 16-bit bfloat16 value, so shift right by 16
    // to extract the upper 16 bits (sign + exponent + 7 mantissa bits).
    sfpi::vFloat lower_v = sfpi::s2vFloat16b(lower_u >> 16);
    sfpi::vFloat range_v = sfpi::s2vFloat16b(range_u >> 16);

    if (seed_u != 0) {
        // Training mode: per-element random slope in [lower, upper)
        // On Wormhole, SFPI builtin doesn't support PRNG (mod1=8), so fall
        // back to deterministic eval-mode slope. True Wormhole PRNG training
        // would require raw TTI instructions (see dropout kernel pattern).
        sfpi::vFloat slope = lower_v + range_v * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];

            v_if(v < 0.0f) { v *= slope; }
            v_endif;

            sfpi::dst_reg[0] = v;
            sfpi::dst_reg++;
        }
    } else {
        // Eval mode: fixed slope = lower + range * 0.5 = (lower + upper) / 2
        sfpi::vFloat slope = lower_v + range_v * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];

            v_if(v < 0.0f) { v *= slope; }
            v_endif;

            sfpi::dst_reg[0] = v;
            sfpi::dst_reg++;
        }
    }
}

inline void _init_rrelu_(const uint32_t seed) {
    if (seed != 0) {
        init_prng_seed(seed);
    }
}

}  // namespace sfpu
}  // namespace ckernel
